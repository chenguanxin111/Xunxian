#!/usr/bin/env python3
"""
Standalone Line-Following Node based on last year's proven xunxian algorithm.

Features:
- Web Console on http://0.0.0.0:5002 with DISARMED, RUNNING, STOPPED, FAULT modes.
- High-speed line following (default linear velocity 0.25 - 0.30 m/s).
- Row-scan center extraction with single-edge fallback and historical memory.
- Dynamic PD angular control and lateral velocity (vy) fine-tuning.
"""
import os
import sys
import json
import time
import math
import threading
from urllib.parse import urlparse
from http.server import HTTPServer, BaseHTTPRequestHandler

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
HSV_PATH = os.path.join(PROJECT_DIR, 'config', 'white_lane.json')
FALLBACK_HSV_PATH = os.path.join(PROJECT_DIR, 'config', 'white_lane_right.json')

PORT = 5002
TRACK_ROI_TOP = 0.60
ALLOW_MOTION = False  # Unlock only after on-site IPM overlay/direction review.

DEFAULT_HSV = {
    'low_h': 42, 'high_h': 179,
    'low_s': 5, 'high_s': 71,
    'low_v': 116, 'high_v': 255,
    'blur_ksize': 4,
    'erode_iter': 0, 'erode_ksize': 3,
    'dilate_iter': 2, 'dilate_ksize': 3,
    'roi_top': TRACK_ROI_TOP, 'roi_bottom': 1.0,
    'roi_left': 0.0, 'roi_right': 1.0,
}

# 动态 PID 参数配置 (借鉴去年 xunxian.py 增益 + 入弯前瞻灵敏度提升)
PID_TABLE = {
    'straight': (0.015, 0.0025),
    'curve_small': (0.020, 0.0020),
    'curve_medium': (0.022, 0.0018),
    'curve_large': (0.024, 0.0015),
}


class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.mode = 'DISARMED'
        self.message = '巡线节点已启动 (仅感知模式)'
        self.last_image_time = 0.0
        self.hsv_params = dict(DEFAULT_HSV)
        
        # 巡线数据
        self.center_error = 0.0
        self.angle_error = 0.0
        self.last_center_points = []
        self.centers = []
        
        # 运行控制参数
        self.target_speed = 0.20
        self.command_linear_x = 0.0
        self.command_linear_y = 0.0
        self.command_angular_z = 0.0
        self.last_angular_z = 0.0
        self.delta_angular_z = 0.0
        self.vision_valid = False
        self.single_edge = False
        self.real_pair_ratio = 0.0
        self.lost_frames = 0
        self.last_valid_time = 0.0
        self.last_angle_error = 0.0
        self.last_control_time = 0.0
        self.ipm_observed_y_max = None

    def status(self):
        return {
            'mode': self.mode,
            'message': self.message,
            'center_error_px': round(self.center_error, 1) if self.center_error is not None else None,
            'angle_error_deg': round(self.angle_error, 1) if self.angle_error is not None else None,
            'center_count': len(self.centers),
            'target_speed': self.target_speed,
            'command_linear_x': round(self.command_linear_x, 3),
            'command_linear_y': round(self.command_linear_y, 3),
            'command_angular_z': round(self.command_angular_z, 3),
            'vision_valid': self.vision_valid,
            'single_edge': self.single_edge,
            'real_pair_ratio': round(self.real_pair_ratio, 2),
            'lost_frames': self.lost_frames,
            'ipm_observed_y_max': round(self.ipm_observed_y_max, 1) if self.ipm_observed_y_max is not None else None,
            'image_age_s': round(max(0.0, time.time() - self.last_image_time), 2),
        }


state = SharedState()
bridge = CvBridge()
cmd_pub = None
mask_pub = None
overlay_pub = None


def safe_cv2_to_imgmsg(cv_img, encoding="bgr8"):
    """安全转换为 ROS Image 消息，彻底规避 cv_bridge 底层 KeyError 16 缺陷"""
    try:
        return bridge.cv2_to_imgmsg(cv_img, encoding=encoding)
    except Exception:
        msg = Image()
        msg.height, msg.width = cv_img.shape[:2]
        if len(cv_img.shape) == 3:
            msg.encoding = encoding
            msg.step = msg.width * cv_img.shape[2]
        else:
            msg.encoding = encoding
            msg.step = msg.width
        msg.data = cv_img.tobytes()
        return msg


def load_hsv():
    path_to_use = HSV_PATH if os.path.exists(HSV_PATH) else (FALLBACK_HSV_PATH if os.path.exists(FALLBACK_HSV_PATH) else None)
    if path_to_use:
        try:
            with open(path_to_use, 'r') as stream:
                data = json.load(stream)
                state.hsv_params.update(data)
                # The proven tracking ROI is the lower ~40% of the image.
                state.hsv_params['roi_top'] = TRACK_ROI_TOP
                state.hsv_params['roi_bottom'] = 1.0
                rospy.loginfo("巡线节点载入 HSV 配置文件: %s", path_to_use)
        except Exception as err:
            rospy.logwarn("巡线节点读取 HSV 配置文件失败: %s", err)


def make_mask(frame, params):
    blur_k = int(params.get('blur_ksize', 3))
    if blur_k >= 3:
        if blur_k % 2 == 0:
            blur_k += 1
        frame_for_hsv = cv2.GaussianBlur(frame, (blur_k, blur_k), 0)
    else:
        frame_for_hsv = frame

    hsv = cv2.cvtColor(frame_for_hsv, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv,
        np.array([params['low_h'], params['low_s'], params['low_v']]),
        np.array([params['high_h'], params['high_s'], params['high_v']])
    )
    
    h, w = mask.shape
    y1 = int(h * min(params.get('roi_top', 0.55), params.get('roi_bottom', 1.0)))
    y2 = int(h * max(params.get('roi_top', 0.55), params.get('roi_bottom', 1.0)))
    x1 = int(w * min(params.get('roi_left', 0.0), params.get('roi_right', 1.0)))
    x2 = int(w * max(params.get('roi_left', 0.0), params.get('roi_right', 1.0)))

    y1 = max(0, min(h - 1, y1))
    y2 = max(y1 + 1, min(h, y2))
    x1 = max(0, min(w - 1, x1))
    x2 = max(x1 + 1, min(w, x2))

    roi = np.zeros_like(mask)
    roi[y1:y2, x1:x2] = mask[y1:y2, x1:x2]

    for operation in ('erode', 'dilate'):
        iterations = int(params.get(operation + '_iter', 1))
        size = max(1, int(params.get(operation + '_ksize', 3)))
        if size % 2 == 0:
            size += 1
        if iterations > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
            roi = getattr(cv2, operation)(roi, kernel, iterations=iterations)
    return roi, (x1, y1, x2, y2)


PERSP_PATH = os.path.join(PROJECT_DIR, 'config', 'perspective_params.json')

IPM_MATRIX = None
IPM_INV_MATRIX = None


def init_ipm():
    global IPM_MATRIX, IPM_INV_MATRIX
    if os.path.exists(PERSP_PATH):
        try:
            with open(PERSP_PATH, 'r') as f:
                data = json.load(f)
                src_pts = np.float32(data['src_points'])
                dst_pts = np.float32(data['dst_points'])
                IPM_MATRIX = cv2.getPerspectiveTransform(src_pts, dst_pts)
                IPM_INV_MATRIX = cv2.getPerspectiveTransform(dst_pts, src_pts)
                rospy.loginfo("巡线节点成功载入鸟瞰图 (IPM) 正向及反向矩阵: %s", PERSP_PATH)
        except Exception as err:
            rospy.logwarn("巡线节点读取 IPM 配置文件失败: %s", err)


def extract_line_centers_ipm(mask):
    """Fit real dual-edge centers in IPM and extend them to the vehicle."""
    h, w = mask.shape
    image_centers = []
    edge_pairs = []
    expected_center = w / 2.0
    expected_width = None
    candidate_rows = 0

    for y in range(h - 1, int(h * TRACK_ROI_TOP) - 1, -4):
        xs = np.flatnonzero(mask[y] > 0)
        if len(xs) < 2:
            continue
        groups = np.split(xs, np.where(np.diff(xs) > 1)[0] + 1)
        means = [float(np.mean(group)) for group in groups if len(group) >= 2]
        if not means:
            continue
        candidate_rows += 1

        candidates = []
        for i in range(len(means)):
            for j in range(i + 1, len(means)):
                width = means[j] - means[i]
                if 70.0 <= width <= w * 0.92:
                    pair_center = (means[i] + means[j]) / 2.0
                    width_penalty = (0.0 if expected_width is None else
                                     0.08 * abs(width - expected_width))
                    score = abs(pair_center - expected_center) + width_penalty
                    candidates.append((score, means[i], means[j], pair_center, width))

        if not candidates:
            continue
        _, left_x, right_x, center_x, width = min(candidates)
        if image_centers and abs(center_x - image_centers[-1][0]) > w / 8.0:
            continue
        image_centers.append((float(center_x), y))
        edge_pairs.append(((int(left_x), y), (int(right_x), y)))
        expected_width = width if expected_width is None else 0.75 * expected_width + 0.25 * width
        if len(image_centers) >= 2:
            expected_center = (image_centers[-1][0] +
                               image_centers[-1][0] - image_centers[-2][0])
        else:
            expected_center = center_x

    if IPM_MATRIX is None or IPM_INV_MATRIX is None or len(image_centers) < 4:
        return {
            'valid': False, 'centers': [], 'edge_pairs': edge_pairs,
            'angle_error': None, 'center_error': None,
            'real_pair_ratio': 0.0, 'single_edge': True,
            'observed_y_max': None, 'clean_mask': mask,
        }

    raw_points = np.float32([[[639.0 - x, y]] for x, y in image_centers])
    ipm = cv2.perspectiveTransform(raw_points, IPM_MATRIX).reshape(-1, 2)
    ipm = ipm[np.isfinite(ipm).all(axis=1)]
    ipm = ipm[(ipm[:, 0] > -300) & (ipm[:, 0] < 900) &
              (ipm[:, 1] > -100) & (ipm[:, 1] < 700)]
    if len(ipm) < 4 or np.ptp(ipm[:, 1]) < 70.0:
        return {
            'valid': False, 'centers': [], 'edge_pairs': edge_pairs,
            'angle_error': None, 'center_error': None,
            'real_pair_ratio': 0.0, 'single_edge': True,
            'observed_y_max': None, 'clean_mask': mask,
        }

    # A quadratic preserves bend curvature. Fall back to a line unless the
    # quadratic materially improves residuals, avoiding noisy extrapolation.
    ys, xs = ipm[:, 1], ipm[:, 0]
    line_coef = np.polyfit(ys, xs, 1)
    line_rmse = float(np.sqrt(np.mean((xs - np.polyval(line_coef, ys)) ** 2)))
    degree = 1
    coef = line_coef
    if len(ipm) >= 7 and np.ptp(ys) >= 140.0:
        quad_coef = np.polyfit(ys, xs, 2)
        quad_rmse = float(np.sqrt(np.mean((xs - np.polyval(quad_coef, ys)) ** 2)))
        if quad_rmse < line_rmse * 0.70 and abs(quad_coef[0]) < 0.003:
            degree, coef = 2, quad_coef

    y_vehicle = 590.0
    center_vehicle = float(np.polyval(coef, y_vehicle))
    slope_vehicle = float(np.polyval(np.polyder(coef), y_vehicle))
    center_error = center_vehicle - 300.0
    heading_error = math.degrees(math.atan2(-slope_vehicle, 1.0))
    if not (-260.0 <= center_error <= 260.0 and abs(heading_error) <= 65.0):
        return {
            'valid': False, 'centers': [], 'edge_pairs': edge_pairs,
            'angle_error': None, 'center_error': None,
            'real_pair_ratio': 0.0, 'single_edge': True,
            'observed_y_max': None, 'clean_mask': mask,
        }

    y_plot = np.linspace(max(-80.0, float(np.min(ys))), y_vehicle, 40)
    ipm_curve = np.float32([[np.polyval(coef, y), y] for y in y_plot]).reshape(-1, 1, 2)
    raw_curve = cv2.perspectiveTransform(ipm_curve, IPM_INV_MATRIX).reshape(-1, 2)
    overlay_centers = []
    for x, y in raw_curve:
        point = (int(round(639.0 - x)), int(round(y)))
        if 0 <= point[0] < w and 0 <= point[1] < h:
            overlay_centers.append(point)

    ratio = len(image_centers) / float(max(1, candidate_rows))
    observed_y_max = float(np.max(ys))
    return {
        'valid': len(overlay_centers) >= 3,
        'centers': overlay_centers, 'edge_pairs': edge_pairs,
        'angle_error': float(heading_error),
        'center_error': float(center_error),
        'real_pair_ratio': ratio,
        'single_edge': observed_y_max < 540.0,
        'observed_y_max': observed_y_max,
        'fit_degree': degree, 'clean_mask': mask,
    }


def image_cb(msg):
    try:
        frame = bridge.imgmsg_to_cv2(msg, 'passthrough')
        if msg.encoding.lower() == 'rgb8':
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        elif msg.encoding.lower() != 'bgr8':
            frame = bridge.imgmsg_to_cv2(msg, 'bgr8')

        # 保持与巡线环境完全一致的镜像翻转
        frame = cv2.flip(frame, 1)

        with state.lock:
            hsv_p = dict(state.hsv_params)

        mask, roi = make_mask(frame, hsv_p)
        result = extract_line_centers_ipm(mask)
        centers = result['centers']
        samples = result['edge_pairs']
        clean_mask = result['clean_mask']

        overlay = frame.copy()
        cv2.rectangle(overlay, (roi[0], roi[1]), (roi[2] - 1, roi[3] - 1), (255, 120, 0), 2)
        
        # 绘制检测出的边缘采样点与流畅绿色中线
        for p1, p2 in samples:
            cv2.circle(overlay, p1, 2, (0, 0, 255), -1)
            cv2.circle(overlay, p2, 2, (0, 0, 255), -1)
            
        if len(centers) > 1:
            cv2.polylines(overlay, [np.array(centers, np.int32)], False, (0, 255, 0), 4)
            cv2.putText(overlay, 'HIGH-SPEED LINE TRACKING', (200, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 0), 2)

        with state.lock:
            state.last_image_time = time.time()
            state.centers = centers
            state.vision_valid = result['valid']
            state.single_edge = result['single_edge']
            state.real_pair_ratio = result['real_pair_ratio']
            state.ipm_observed_y_max = result.get('observed_y_max')
            if result['valid']:
                state.angle_error = result['angle_error']
                state.center_error = result['center_error']
                state.last_valid_time = state.last_image_time
                state.last_center_points = list(centers)
                state.lost_frames = 0
            else:
                state.lost_frames += 1
            current_mode = state.mode

        angle_text = state.angle_error if result['valid'] else float('nan')
        center_text = state.center_error if result['valid'] else float('nan')
        cv2.putText(overlay, 'MODE: %s' % current_mode, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 255), 2)
        cv2.putText(overlay, 'angle_err=%.1f deg  center_err=%.1f px' %
                    (angle_text, center_text),
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)

        if mask_pub is not None:
            mask_pub.publish(safe_cv2_to_imgmsg(clean_mask, 'mono8'))
        if overlay_pub is not None:
            overlay_pub.publish(safe_cv2_to_imgmsg(overlay, 'bgr8'))

    except Exception as e:
        rospy.logerr_throttle(2, f"巡线图像回调异常: {e}")


def publish_stop():
    state.command_linear_x = 0.0
    state.command_linear_y = 0.0
    state.command_angular_z = 0.0
    state.last_angular_z = 0.0
    state.delta_angular_z = 0.0
    state.last_control_time = 0.0
    if cmd_pub is not None:
        cmd_pub.publish(Twist())


def control_timer(_event):
    with state.lock:
        now = time.time()
        mode = state.mode
        if mode != 'RUNNING':
            return
        if state.last_image_time == 0 or now - state.last_image_time > 0.6:
            state.mode = 'FAULT'
            state.message = '摄像头画面超时，已紧急停车'
            publish_stop()
            return

        if not state.vision_valid:
            if state.lost_frames <= 3 and now - state.last_valid_time <= 0.16:
                cmd = Twist()
                cmd.linear.x = 0.06
                cmd.angular.z = max(-0.25, min(0.25, state.last_angular_z))
                state.command_linear_x = cmd.linear.x
                state.command_linear_y = 0.0
                state.command_angular_z = cmd.angular.z
                cmd_pub.publish(cmd)
                return
            state.mode = 'FAULT'
            state.message = '连续丢失有效车道中线，已停车'
            publish_stop()
            return

        angle_err = state.angle_error
        center_err = state.center_error
        target_speed = state.target_speed

        # 根据角度误差选择动态 PID 增益
        abs_angle = abs(angle_err)
        if abs_angle < 15.0:
            kp_z, kd_z = PID_TABLE['straight']
        elif abs_angle < 30.0:
            kp_z, kd_z = PID_TABLE['curve_small']
        elif abs_angle < 45.0:
            kp_z, kd_z = PID_TABLE['curve_medium']
        else:
            kp_z, kd_z = PID_TABLE['curve_large']

        dt = now - state.last_control_time if state.last_control_time > 0 else 0.04
        dt = max(0.02, min(0.10, dt))
        error_rate = (angle_err - state.last_angle_error) / dt
        # IPM x increases toward physical right. A positive heading means the
        # lane points right and therefore requires negative ROS angular.z.
        angular_z = -kp_z * angle_err - kd_z * error_rate - 0.0008 * center_err
        state.last_angle_error = angle_err
        state.last_control_time = now

        # 弯道动态速度适应：当预判到前方弯道曲率增大 (>12°) 时，平滑适度降速给足转向抓地力
        current_speed = target_speed - 0.0035 * max(0.0, abs_angle - 8.0)
        current_speed = max(0.08, current_speed)
        if state.single_edge:
            current_speed = min(current_speed, 0.11)
        elif state.real_pair_ratio < 0.65:
            current_speed = min(current_speed, 0.14)

        # 限幅保护
        angular_z = max(-0.60, min(0.60, angular_z))
        state.last_angular_z = angular_z

        cmd = Twist()
        cmd.linear.x = current_speed
        cmd.linear.y = 0.0
        cmd.angular.z = angular_z

        state.command_linear_x = cmd.linear.x
        state.command_linear_y = 0.0
        state.command_angular_z = cmd.angular.z

        cmd_pub.publish(cmd)


PAGE = '''<!doctype html><meta charset="utf-8"><title>高速巡线独立控制台</title>
<style>body{font:16px sans-serif;background:#0f172a;color:#f8fafc;margin:20px}main{max-width:1100px;margin:auto}section{background:#1e293b;padding:16px;margin:12px 0;border-radius:10px}img{width:48%;background:#111;margin:1%;border-radius:6px}.start{background:#1677ff;color:white}.stop{background:#d00;color:white}button{padding:12px 22px;border:0;border-radius:6px;margin-right:10px;font-size:16px;cursor:pointer}pre{font-size:15px;background:#0f172a;padding:10px;border-radius:6px;color:#38bdf8}</style>
<main><h2>高速巡线独立控制台 (Port 5002)</h2>
<section><b>默认不运动。确认跑道清空后手动启动巡线。</b>
<p><button class="start" onclick="post('/api/start')">解锁并开始巡线</button><button class="stop" onclick="post('/api/stop')">立即停车</button><button onclick="post('/api/reset')">复位为仅感知</button></p>
<pre id="status">加载状态中...</pre></section>
<section><h3>识别叠加图 (/line_following/debug/overlay) 与 二值图 (/line_following/debug/mask)</h3>
<p><img id="img_overlay"><img id="img_mask"></p></section>
</main>
<script>
document.getElementById('img_overlay').src = 'http://' + window.location.hostname + ':8080/stream?topic=/line_following/debug/overlay';
document.getElementById('img_mask').src = 'http://' + window.location.hostname + ':8080/stream?topic=/line_following/debug/mask';
async function post(url){
    try {
        let r = await fetch(url,{method:'POST'});
        let d = await r.json();
        if(!d.ok){
            alert("⚠️ 操作失败: " + d.error);
        }
    } catch(e) {
        alert("⚠️ 请求失败: " + e);
    }
}
setInterval(async()=>{try{document.getElementById('status').textContent=JSON.stringify(await(await fetch('/api/status')).json(),null,2)}catch(e){}},400);
</script>'''


class Handler(BaseHTTPRequestHandler):
    def reply(self, obj):
        data = json.dumps(obj, ensure_ascii=False).encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if urlparse(self.path).path == '/':
            data = PAGE.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        elif urlparse(self.path).path == '/api/status':
            with state.lock:
                self.reply(state.status())
        else:
            self.send_error(404)

    def do_POST(self):
        path = urlparse(self.path).path
        with state.lock:
            if path == '/api/start':
                if not ALLOW_MOTION:
                    self.reply({'ok': False, 'error': 'IPM中线方向校验中，当前仅允许感知，禁止运动'})
                    return
                if state.mode == 'RUNNING':
                    self.reply({'ok': False, 'error': '巡线任务已经在运行中'})
                    return
                if time.time() - state.last_image_time > 0.6:
                    self.reply({'ok': False, 'error': '摄像头画面超时，请检查相机'})
                    return
                if not state.vision_valid or len(state.centers) < 8:
                    self.reply({'ok': False, 'error': '当前没有可靠车道中线，禁止启动'})
                    return
                if abs(state.center_error) > 100.0:
                    self.reply({'ok': False, 'error': '车辆初始位置偏离车道中心过大'})
                    return
                state.last_angle_error = state.angle_error
                state.last_control_time = 0.0
                state.last_angular_z = 0.0
                state.mode = 'RUNNING'
                state.message = '高速巡线模式运行中'
                self.reply({'ok': True})
            elif path == '/api/stop':
                state.mode = 'STOPPED'
                state.message = '用户手动立即停车'
                publish_stop()
                self.reply({'ok': True})
            elif path == '/api/reset':
                state.mode = 'DISARMED'
                state.message = '复位为仅感知模式'
                publish_stop()
                self.reply({'ok': True})
            else:
                self.send_error(404)


import socket

class ReusableHTTPServer(HTTPServer):
    allow_reuse_address = True

    def server_bind(self):
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        super().server_bind()


def shutdown():
    publish_stop()


def main():
    global cmd_pub, mask_pub, overlay_pub
    rospy.init_node('line_following_node', anonymous=False)
    init_ipm()
    load_hsv()

    cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
    mask_pub = rospy.Publisher('/line_following/debug/mask', Image, queue_size=1)
    overlay_pub = rospy.Publisher('/line_following/debug/overlay', Image, queue_size=1)

    rospy.Subscriber('/usb_cam/image_raw', Image, image_cb, queue_size=1, buff_size=2**24)
    rospy.Timer(rospy.Duration(0.04), control_timer)  # 25Hz 控制循环
    rospy.on_shutdown(shutdown)

    server = ReusableHTTPServer(('0.0.0.0', PORT), Handler)
    rospy.loginfo('独立高速巡线节点已启动: http://0.0.0.0:%d', PORT)
    server.serve_forever()


if __name__ == '__main__':
    main()
