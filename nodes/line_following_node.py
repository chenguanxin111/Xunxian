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
    'roi_top': 0.55, 'roi_bottom': 1.0,
    'roi_left': 0.0, 'roi_right': 1.0,
}

# 动态 PID 参数配置 (衍生自去年 xunxian.py 实测增益)
PID_TABLE = {
    'straight': (0.015, 0.22),     # 小转角平稳直行 (Kp_z, Kd_z)
    'curve_small': (0.022, 0.18),  # 缓弯
    'curve_medium': (0.026, 0.16), # 中弯
    'curve_large': (0.030, 0.12),  # 急弯
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


def extract_line_centers(mask):
    """
    继承去年 xunxian.py 的逐行扫描算法：
    从图像最底部向上以 step=4 逐行扫描二值化连通块，提取多边/双边/单边中线点集。
    """
    h, w = mask.shape
    centers = []
    edge_samples = []
    
    # 清理掩膜主要轮廓，避免杂噪点干扰
    result = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = result[0] if len(result) == 2 else result[1]
    contours = [c for c in contours if cv2.contourArea(c) > 120]
    clean_mask = np.zeros_like(mask)
    if contours:
        cv2.drawContours(clean_mask, contours, -1, 255, -1)

    step = 4
    for y in range(h - 1, int(h * 0.45), -step):
        white_indices = np.where(clean_mask[y] == 255)[0]
        if len(white_indices) == 0:
            continue

        diff = np.diff(white_indices)
        breaks = np.where(diff > 1)[0] + 1
        clusters = np.split(white_indices, breaks)
        means = [int(np.mean(cluster)) for cluster in clusters if len(cluster) >= 2]

        if len(means) == 1:
            edge_x = means[0]
            # 单侧车道线情况：结合上一帧走向推算另一侧虚拟边缘
            if len(centers) > 1:
                last_x = centers[-1][0]
                prev_x = centers[-2][0]
                virtual_x = 0 if last_x < prev_x else w - 1
            else:
                virtual_x = w - 1 if edge_x < w // 2 else 0
            
            cand_x = int((edge_x + virtual_x) / 2)
            edge_samples.append(((edge_x, y), (virtual_x, y)))

        elif len(means) >= 2:
            left_x = max((x for x in means if x < w / 2), default=None)
            right_x = min((x for x in means if x >= w / 2), default=None)
            if left_x is not None and right_x is not None:
                cand_x = int((left_x + right_x) / 2)
                edge_samples.append(((left_x, y), (right_x, y)))
            else:
                cand_x = int(np.mean(means[:2]))
        else:
            continue

        # 防跳变约束：相邻中线点横向偏移不应超过 35 像素
        if len(centers) == 0 or abs(cand_x - centers[-1][0]) < 35:
            centers.append((cand_x, y))

    # 历史记忆 Fallback
    if len(centers) >= 3:
        state.last_center_points = list(centers)
    elif len(state.last_center_points) >= 3:
        centers = list(state.last_center_points)

    return centers, edge_samples, clean_mask


def calculate_line_errors(centers, width):
    """
    计算中线角度误差与近处横向偏离像素 (参照去年 calculate_slope 算法)
    """
    if len(centers) < 3:
        return 0.0, 0.0

    p_near = centers[0]
    p_far = centers[min(len(centers) - 1, int(len(centers) / 3.0))]
    
    # 近远处向向量夹角 (与正上方 -90 度的偏差)
    angle_rad = math.atan2(p_far[1] - p_near[1], p_far[0] - p_near[0])
    angle_deg = math.degrees(angle_rad)
    
    # 归一化航向误差 (正前方为 0°)
    if angle_deg > 0:
        angle_error = -90.0 + angle_deg
    else:
        angle_error = 90.0 + angle_deg

    # 近处横向偏离像素 (图片中心 320)
    near_points = centers[:min(5, len(centers))]
    center_error = float(np.mean([p[0] for p in near_points]) - width / 2.0)

    return angle_error, center_error


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
        centers, samples, clean_mask = extract_line_centers(mask)
        angle_error, center_error = calculate_line_errors(centers, frame.shape[1])

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

        # 去年 proven 的动态 PD 角速度控制算法
        angular_z = kp_z * angle_err - kd_z * state.delta_angular_z
        state.delta_angular_z = angular_z - state.last_angular_z
        state.last_angular_z = angular_z

        # 横向微调 (vy 修正)
        kp_y = 12.0
        linear_y = kp_y * center_err * 0.0005 if abs(center_err) > 10 else 0.0

        # 限幅保护
        angular_z = max(-0.8, min(0.8, angular_z))
        linear_y = max(-0.10, min(0.10, linear_y))

        cmd = Twist()
        cmd.linear.x = target_speed
        cmd.linear.y = linear_y
        cmd.angular.z = angular_z

        state.command_linear_x = cmd.linear.x
        state.command_linear_y = cmd.linear.y
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
                if time.time() - state.last_image_time > 0.6:
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


class ReusableHTTPServer(HTTPServer):
    allow_reuse_address = True


def shutdown():
    publish_stop()


def main():
    global cmd_pub, mask_pub, overlay_pub
    rospy.init_node('line_following_node', anonymous=False)
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
