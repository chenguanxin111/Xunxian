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
HSV_PATH = os.path.join(PROJECT_DIR, 'config', 'hsv_params.json')
FALLBACK_HSV_PATH = os.path.join(PROJECT_DIR, 'config', 'fallback_hsv_params.json')

PORT = 5002

DEFAULT_HSV = {
    'low_h': 0, 'high_h': 179,
    'low_s': 0, 'high_s': 60,
    'low_v': 60, 'high_v': 255,
    'blur_ksize': 3,
    'erode_iter': 1, 'erode_ksize': 1,
    'dilate_iter': 2, 'dilate_ksize': 5,
    'roi_top': 0.6, 'roi_bottom': 1.0,
    'roi_left': 0.0, 'roi_right': 1.0,
}

# 动态 PID 参数配置 (借鉴去年 xunxian.py 增益 + 入弯前瞻灵敏度提升)
PID_TABLE = {
    'straight': (0.028, 0.22),     # 提升入弯初始灵敏度 (Kp_z 从 0.015 提升至 0.028)
    'curve_small': (0.034, 0.18),  # 缓弯入弯前瞻预判
    'curve_medium': (0.038, 0.16), # 中弯
    'curve_large': (0.044, 0.12),  # 急弯大角度抱弯
}


class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.mode = 'DISARMED'
        self.message = '巡线节点已启动 (仅感知模式)'
        self.last_image_time = time.time()
        self.hsv_params = dict(DEFAULT_HSV)
        
        # 巡线数据
        self.center_error = 0.0
        self.angle_error = 0.0
        self.last_center_points = []
        self.centers = []
        
        # 运行控制参数
        self.target_speed = 0.26  # 默认巡线速度 0.26 m/s
        self.command_linear_x = 0.0
        self.command_linear_y = 0.0
        self.command_angular_z = 0.0
        self.last_angular_z = 0.0
        self.delta_angular_z = 0.0

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
            'image_age_s': round(max(0.0, time.time() - self.last_image_time), 2) if self.last_image_time > 0 else 0.0,
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
    """
    去年实车跑圈完全验证过的图像空间扫描算法：
    不做任何死板的鸟瞰图直线/抛物线拟合，直接在图像视距空间逐行扫描提取边线。
    利用前瞻割线角 (angle_first_middle) 在弯道前提前强力打方向！
    """
    height, width = mask.shape
    step = 4
    center_points = []
    edge_points = []

    for y in range(height - 1, -1, -step):
        white_indices = np.where(mask[y] == 255)[0]
        if len(white_indices) == 0:
            continue

        diff = np.diff(white_indices)
        breaks = np.where(diff > 1)[0] + 1
        clusters = np.split(white_indices, breaks)
        mean_indices = [np.mean(cluster) for cluster in clusters]

        current_edge_points = []
        if len(mean_indices) == 1:
            edge_x = int(mean_indices[0])
            current_edge_points.append(edge_x)

            if len(center_points) > 1:
                last_center_x = center_points[-1][0]
                second_last_center_x = center_points[-2][0]
                virtual_edge_x = 0 if last_center_x < second_last_center_x else width - 1
            else:
                virtual_edge_x = width - 1 if edge_x < width // 2 else 0

            current_edge_points.append(virtual_edge_x)
            avg_index = np.mean(current_edge_points)
            new_center_point = (int(avg_index), y)
            
        elif len(mean_indices) > 1:
            for idx in mean_indices:
                current_edge_points.append(int(idx))

            if len(current_edge_points) == 2:
                if abs(current_edge_points[0] - current_edge_points[1]) < width / 3:
                    if abs(current_edge_points[0] - width // 2) > abs(current_edge_points[1] - width // 2):
                        current_edge_points = [current_edge_points[1], width - 1 if current_edge_points[1] < width // 2 else 0]
                    else:
                        current_edge_points = [current_edge_points[0], width - 1 if current_edge_points[0] < width // 2 else 0]
                avg_index = np.mean(current_edge_points)
                new_center_point = (int(avg_index), y)
            else:
                mid_x = width // 2
                left_edge_points = [pt for pt in current_edge_points if pt < mid_x]
                right_edge_points = [pt for pt in current_edge_points if pt >= mid_x]
                left_nearest = min(left_edge_points, key=lambda x: abs(x - mid_x)) if left_edge_points else 0
                right_nearest = min(right_edge_points, key=lambda x: abs(x - mid_x)) if right_edge_points else width - 1
                current_edge_points = [left_nearest, right_nearest]
                avg_index = np.mean(current_edge_points)
                new_center_point = (int(avg_index), y)

        for rp in current_edge_points:
            edge_points.append((rp, y))

        if len(center_points) == 0 or abs(new_center_point[0] - center_points[-1][0]) < width / 8:
            center_points.append(new_center_point)

    if not center_points or len(center_points) < 3:
        return [], [], 0.0, 0.0, mask

    # 去年核心 calculate_slope 割线角度算法：
    first_point = center_points[0]
    middle_idx = min(len(center_points) - 1, int(round(len(center_points) / 3.5)))
    middle_point = center_points[middle_idx]
    
    angle_first_middle = np.degrees(np.arctan2(middle_point[1] - first_point[1], middle_point[0] - first_point[0]))
    
    # 图像空间角度误差公式：直上为 -90°，右歪 -90，左歪 90
    angle_error = -90.0 + angle_first_middle if angle_first_middle > 0 else 90.0 + angle_first_middle
    
    # 近端横向位置偏差
    center_error = float(first_point[0] - width // 2)

    return center_points, [], float(angle_error), float(center_error), mask


def extract_line_centers_fallback(mask):
    """退化模式降级备用"""
    h, w = mask.shape
    centers = []
    step = 4
    for y in range(h - 1, int(h * 0.45), -step):
        xs = np.flatnonzero(mask[y] > 0)
        if len(xs) > 0:
            centers.append((int(np.mean(xs)), y))
    return centers, [], 0.0, 0.0, mask


def image_cb(msg):
    rospy.loginfo_throttle(1, "巡线节点成功收到摄像头图像帧！")
    with state.lock:
        state.last_image_time = time.time()
        
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
        # 降采样到 (320, 240) 匹配去年算法的最佳性能 (避免 480 满分辨率循环导致 CPU 拖慢爆出画面超时)
        small_mask = cv2.resize(mask, (320, 240), interpolation=cv2.INTER_NEAREST)
        centers_small, samples, angle_error, center_error, clean_mask = extract_line_centers_ipm(small_mask)

        # 还原到原始 640x480 画面坐标进行叠加显示
        centers = [(x * 2, y * 2) for x, y in centers_small]
        center_error = center_error * 2.0  # 还原真实像素误差

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
            state.centers = centers
            state.angle_error = angle_error
            state.center_error = center_error
            current_mode = state.mode

        cv2.putText(overlay, 'MODE: %s' % current_mode, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 255), 2)
        cv2.putText(overlay, 'angle_err=%.1f deg  center_err=%.1f px' %
                    (angle_error, center_error),
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

        # 动态 PD 角速度控制算法 + 近处偏离修正 (修复横向偏差正反馈误打方向)
        angular_z = kp_z * angle_err - kd_z * state.delta_angular_z + 0.0012 * center_err
        state.delta_angular_z = angular_z - state.last_angular_z
        state.last_angular_z = angular_z

        # 弯道动态速度适应：当预判到前方弯道曲率增大 (>12°) 时，平滑适度降速给足转向抓地力
        if abs_angle > 12.0:
            current_speed = max(0.18, target_speed - 0.003 * (abs_angle - 12.0))
        else:
            current_speed = target_speed

        # 限幅保护
        angular_z = max(-0.85, min(0.85, angular_z))

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
                if state.mode == 'RUNNING':
                    self.reply({'ok': False, 'error': '巡线任务已经在运行中'})
                    return
                if time.time() - state.last_image_time > 1.2:
                    self.reply({'ok': False, 'error': '摄像头画面超时，请检查相机'})
                    return
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
        if hasattr(socket, 'SO_REUSEPORT'):
            try:
                self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
            except Exception:
                pass
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
