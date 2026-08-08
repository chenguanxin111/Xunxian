#!/usr/bin/env python3
"""
巡线节点 3 (line_following_node3.py)
完全基于去年实车验证通过的 xunxian.py 算法移植重构：
1. 采用去年原汁原味的 get_ROI 与 find_center_edge_line 中线提取算法
2. 采用去年原汁原味的 calculate_slope 前瞻割线角与误差计算
3. 采用去年原汁原味的 get_pid_params 动态 PID 控制逻辑
4. 包含独立的 Web 控制台 (Port 5003)，实时绘制绿色中线与红色边缘点
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
ALT_HSV_PATH = os.path.join(PROJECT_DIR, 'config', 'white_line.json')
FALLBACK_HSV_PATH = os.path.join(PROJECT_DIR, 'config', 'white_lane_right.json')

PORT = 5003
ALLOW_MOTION = True

# 图片统一尺寸 (与去年 xunxian.py 完全一致: pw=640, ph=360)
pw = 640
ph = 360

DEFAULT_HSV = {
    'low_h': 0, 'high_h': 179,
    'low_s': 0, 'high_s': 75,
    'low_v': 70, 'high_v': 255,
    'blur_ksize': 3,
    'erode_iter': 1, 'erode_ksize': 1,
    'dilate_iter': 2, 'dilate_ksize': 5,
    'roi_top': 0.58, 'roi_bottom': 1.0,
}

PID_PARAMS = {
    # 比例系数 (kp_z, kp_y, kd_z) 调优：降低比例增益，防止转向打角过猛
    "small_curve_invisible": (0.014, 0.0005, 0.22),
    "medium_curve_invisible": (0.018, 0.0005, 0.20),
    "extreme_curve_invisible": (0.022, 0.0005, 0.15),
    "large_extreme_curve_invisible": (0.025, 0.0005, 0.18),
    "small_straight": (0.008, 0.0005, 0.25),
    "small_curve": (0.012, 0.0005, 0.22),
    "medium_curve": (0.015, 0.0005, 0.20),
    "large_curve": (0.018, 0.0005, 0.18),
    "large_straight": (0.008, 0.0005, 0.28)
}


class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.mode = 'DISARMED'
        self.message = '巡线节点3已启动 (去年原版算法模组)'
        self.last_image_time = 0.0
        self.hsv_params = dict(DEFAULT_HSV)
        
        # 巡线感知数据
        self.error = 0.0
        self.last_error = 0.0
        self.last_center_points = []
        self.centers = []
        self.edge_points = []
        self.current_edge_points_zuixiamian = []
        self.kanbujian = 0
        
        # 运动控制参数
        self.custom_speed = 0.26
        self.last_v_z = 0.0
        self.delat_v_z = 0.0
        self.command_linear_x = 0.0
        self.command_linear_y = 0.0
        self.command_angular_z = 0.0
        
        self.vision_valid = False
        self.lost_frames = 0
        self.last_valid_time = 0.0

    def status(self):
        return {
            'mode': self.mode,
            'message': self.message,
            'error_deg': round(self.error, 1) if self.error is not None else None,
            'center_count': len(self.centers),
            'target_speed': self.custom_speed,
            'kanbujian': bool(self.kanbujian),
            'command_linear_x': round(self.command_linear_x, 3),
            'command_linear_y': round(self.command_linear_y, 3),
            'command_angular_z': round(self.command_angular_z, 3),
            'vision_valid': self.vision_valid,
            'lost_frames': self.lost_frames,
            'image_age_s': round(max(0.0, time.time() - self.last_image_time), 2),
        }


state = SharedState()
bridge = CvBridge()
cmd_pub = None
mask_pub = None
overlay_pub = None


def safe_cv2_to_imgmsg(cv_img, encoding="bgr8"):
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
    target_path = None
    for p in [HSV_PATH, ALT_HSV_PATH, FALLBACK_HSV_PATH]:
        if os.path.exists(p):
            target_path = p
            break
    if target_path:
        try:
            with open(target_path, 'r') as stream:
                data = json.load(stream)
                state.hsv_params.update(data)
                if state.hsv_params.get('low_h', 0) > 10:
                    state.hsv_params['low_h'] = 0
                rospy.loginfo("巡线节点3成功载入 HSV 参数: %s", target_path)
        except Exception as err:
            rospy.logwarn("读取配置文件 %s 失败: %s", target_path, err)


def get_line_bin_img(img, params):
    """借鉴去年 xunxian.py 的二值化与尺寸缩放"""
    if img is None:
        raise ValueError("输入图像为空")
    
    blur_k = int(params.get('blur_ksize', 3))
    if blur_k >= 3:
        if blur_k % 2 == 0:
            blur_k += 1
        img_blur = cv2.GaussianBlur(img, (blur_k, blur_k), 0)
    else:
        img_blur = img

    hsv = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)
    lower_array = np.array([params['low_h'], params['low_s'], params['low_v']])
    upper_array = np.array([params['high_h'], params['high_s'], params['high_v']])
    
    img_mask = cv2.inRange(hsv, lowerb=lower_array, upperb=upper_array)
    resize_bin_img = cv2.resize(img_mask, (pw, ph), interpolation=cv2.INTER_AREA)
    resize_img = cv2.resize(img, (pw, ph), interpolation=cv2.INTER_AREA)
    return resize_img, resize_bin_img, img_mask


def get_ROI(resize_img, resize_bin_img, params):
    """去年 xunxian.py 原汁原味 ROI 获取函数"""
    line_up = float(params.get('roi_top', 0.58))
    line_low = float(params.get('roi_bottom', 1.0))
    H, W = resize_img.shape[:2]
    
    y1 = int(H * line_up)
    y2 = int(H * line_low)
    ROI_1 = resize_bin_img[y1:y2, :]
    
    e_k = max(1, int(params.get('erode_ksize', 1)))
    d_k = max(1, int(params.get('dilate_ksize', 5)))
    if e_k % 2 == 0: e_k += 1
    if d_k % 2 == 0: d_k += 1
    
    e_iter = int(params.get('erode_iter', 1))
    d_iter = int(params.get('dilate_iter', 2))
    
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (e_k, e_k))
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (d_k, d_k))
    
    if e_iter > 0:
        ROI_1 = cv2.erode(ROI_1, kernel_erode, iterations=e_iter)
    if d_iter > 0:
        ROI_1 = cv2.dilate(ROI_1, kernel_dilate, iterations=d_iter)
        
    return ROI_1, y1


def find_center_edge_line(img):
    """去年 xunxian.py 原汁原味车道线中心点与边缘检测算法"""
    if img is None:
        raise ValueError("输入图像为空")
    if len(img.shape) != 2 or img.dtype != np.uint8:
        raise ValueError("输入图像必须是单通道二值图像")
    
    height, width = img.shape
    current_edge_points_zuixiamian = []
    center_points = []
    edge_points = []
    sigle, double = 0, 0
    kanbujian = 0

    step = 4
    for y in range(height - 1, -1, -step):
        white_indices = np.where(img[y] == 255)[0]
        if len(white_indices) == 0:
            continue

        diff = np.diff(white_indices)
        breaks = np.where(diff > 1)[0] + 1
        clusters = np.split(white_indices, breaks)
        mean_indices = [np.mean(cluster) for cluster in clusters]

        current_edge_points = []
        if len(mean_indices) == 1:
            sigle += 1
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
            double += 1
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

        if current_edge_points:
            current_edge_points_zuixiamian = current_edge_points
        for rp in current_edge_points:
            edge_points.append((rp, y))

        if len(center_points) == 0 or abs(new_center_point[0] - center_points[-1][0]) < width / 8:
            center_points.append(new_center_point)

    if sigle + double > 0 and sigle / (sigle + double) > 0.9:
        kanbujian = 1

    return center_points, edge_points, current_edge_points_zuixiamian, kanbujian


def calculate_slope(center_points, shared_state):
    """去年 xunxian.py 原汁原味割线角度算法"""
    if not center_points or len(center_points) < 3:
        if not shared_state.last_center_points or len(shared_state.last_center_points) < 2:
            shared_state.last_center_points = [(320, 150), (320, 140), (320, 130), (320, 120), (320, 110), (320, 100)]
        center_points = shared_state.last_center_points

    if not center_points:
        center_points = [(320, 150), (320, 140), (320, 130)]

    first_point = center_points[0]
    last_point = center_points[-1]
    middle_idx = min(len(center_points) - 1, int(round(len(center_points) / 3.5)))
    middle_point = center_points[middle_idx]
    
    angle_first_last = np.degrees(np.arctan2(last_point[1] - first_point[1], last_point[0] - first_point[0]))
    angle_first_middle = np.degrees(np.arctan2(middle_point[1] - first_point[1], middle_point[0] - first_point[0]))
    avg_first_3_x = np.mean([point[0] for point in center_points[:3]])
    
    return angle_first_last, angle_first_middle, avg_first_3_x


def get_pid_params(error, kanbujian):
    """去年 xunxian.py 原汁原味 PID 决策参数组"""
    abs_error = abs(error)
    if kanbujian:
        if 33.5 < abs_error <= 51:
            return PID_PARAMS["small_curve_invisible"]
        elif 51 < abs_error <= 62:
            return PID_PARAMS["medium_curve_invisible"]
        elif 62 < abs_error <= 64:
            return PID_PARAMS["extreme_curve_invisible"]
        elif abs_error > 64:
            return PID_PARAMS["large_extreme_curve_invisible"]
        else:
            return PID_PARAMS["large_straight"]
    else:
        if 30 < abs_error <= 34:
            return PID_PARAMS["small_straight"]
        elif 34 < abs_error <= 55:
            return PID_PARAMS["small_curve"]
        elif 55 < abs_error <= 60:
            return PID_PARAMS["medium_curve"]
        elif abs_error > 60:
            return PID_PARAMS["large_curve"]
        return PID_PARAMS["large_straight"]


def image_cb(msg):
    try:
        frame = bridge.imgmsg_to_cv2(msg, 'passthrough')
        if msg.encoding.lower() == 'rgb8':
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        elif msg.encoding.lower() != 'bgr8':
            frame = bridge.imgmsg_to_cv2(msg, 'bgr8')

        # 镜像翻转
        frame = cv2.flip(frame, 1)

        with state.lock:
            hsv_p = dict(state.hsv_params)

        resize_img, resize_bin_img, img_mask = get_line_bin_img(frame, hsv_p)
        ROI_1, y_offset = get_ROI(resize_img, resize_bin_img, hsv_p)

        center_pts, edge_pts, edge_bottom, kanbujian = find_center_edge_line(ROI_1)

        if len(center_pts) > 3:
            with state.lock:
                state.last_center_points = center_pts

        angle_first_last, angle_first_middle, avg_first_3_x = calculate_slope(center_pts, state)

        # 去年原版误差计算公式
        calc_error = -90.0 + angle_first_middle if angle_first_middle > 0 else 90.0 + angle_first_middle

        # 还原到 resize_img (640x360) 叠加可视化坐标
        full_centers = [(x, y + y_offset) for (x, y) in center_pts]
        full_edges = [(x, y + y_offset) for (x, y) in edge_pts]

        overlay = resize_img.copy()
        
        # 绘制红色边缘点
        for pt in full_edges:
            cv2.circle(overlay, pt, 2, (0, 0, 255), -1)

        # 绘制绿色中线连线与中心点
        if len(full_centers) >= 1:
            for pt in full_centers:
                cv2.circle(overlay, pt, 3, (0, 255, 0), -1)
            if len(full_centers) > 1:
                cv2.polylines(overlay, [np.array(full_centers, np.int32)], False, (0, 255, 0), 3)

        with state.lock:
            state.last_image_time = time.time()
            state.centers = full_centers
            state.edge_points = full_edges
            state.current_edge_points_zuixiamian = edge_bottom
            state.kanbujian = kanbujian
            state.error = calc_error
            state.vision_valid = len(center_pts) >= 3
            if state.vision_valid:
                state.last_valid_time = state.last_image_time
                state.lost_frames = 0
            else:
                state.lost_frames += 1
            current_mode = state.mode

        cv2.putText(overlay, f"MODE: {current_mode} | KANBUJIAN: {kanbujian}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 255), 2)
        cv2.putText(overlay, f"error={calc_error:.1f} deg  centers={len(full_centers)}",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)

        if mask_pub is not None:
            mask_pub.publish(safe_cv2_to_imgmsg(ROI_1, 'mono8'))
        if overlay_pub is not None:
            overlay_pub.publish(safe_cv2_to_imgmsg(overlay, 'bgr8'))

    except Exception as e:
        rospy.logerr_throttle(2, f"巡线节点3图像回调异常: {e}")


def publish_stop():
    state.command_linear_x = 0.0
    state.command_linear_y = 0.0
    state.command_angular_z = 0.0
    state.last_v_z = 0.0
    state.delat_v_z = 0.0
    if cmd_pub is not None:
        cmd_pub.publish(Twist())


def control_timer(_event):
    with state.lock:
        now = time.time()
        mode = state.mode
        if mode != 'RUNNING':
            return

        if state.last_image_time == 0 or now - state.last_image_time > 0.8:
            state.mode = 'FAULT'
            state.message = '摄像头画面超时，已紧急停车'
            publish_stop()
            return

        if not state.vision_valid:
            if state.lost_frames <= 10 and now - state.last_valid_time <= 0.40:
                vel_lost = Twist()
                vel_lost.linear.x = 0.08
                vel_lost.angular.z = max(-0.35, min(0.35, state.last_v_z))
                state.command_linear_x = vel_lost.linear.x
                state.command_linear_y = 0.0
                state.command_angular_z = vel_lost.angular.z
                cmd_pub.publish(vel_lost)
                return
            state.mode = 'FAULT'
            state.message = '连续丢失有效车道中线，已停车'
            publish_stop()
            return

        err = state.error
        kanbujian = state.kanbujian
        current_edge_bottom = state.current_edge_points_zuixiamian

        kp_z, kp_y, kd_z = get_pid_params(err, kanbujian)

        vel = Twist()
        target_wz = kp_z * err
        delta_wz = target_wz - state.last_v_z
        vel.angular.z = target_wz - kd_z * delta_wz
        state.last_v_z = vel.angular.z

        # 弯道平滑自适应降速：保证抓地力
        current_speed = state.custom_speed - 0.002 * max(0.0, abs(err) - 15.0)
        vel.linear.x = max(0.12, current_speed)

        # 计算横向平移误差 (error_y)
        error_y = 0
        if current_edge_bottom is not None and len(current_edge_bottom) >= 2:
            left, right = current_edge_bottom[:2]
            if left not in [0, 639] and right not in [0, 639]:
                error_y = (left + right) / 2 - 320

        vel.linear.y = kp_y * error_y * 0.0005 if error_y != 0 else 0.0

        # 安全限幅保护（角速度由 0.85 压低到 0.45 rad/s，平滑切断任何猛打方向）
        vel.angular.z = max(-0.45, min(0.45, vel.angular.z))
        vel.linear.y = max(-0.08, min(0.08, vel.linear.y))

        state.command_linear_x = vel.linear.x
        state.command_linear_y = vel.linear.y
        state.command_angular_z = vel.angular.z

        rospy.loginfo_throttle(1, "巡线节点3正在发布运动速度: vx=%.2f, vy=%.2f, wz=%.2f (err=%.1f)",
                               vel.linear.x, vel.linear.y, vel.angular.z, err)
        cmd_pub.publish(vel)


PAGE = '''<!doctype html><meta charset="utf-8"><title>巡线控制台 Node3 (去年原版组)</title>
<style>body{font:16px sans-serif;background:#0f172a;color:#f8fafc;margin:20px}main{max-width:1100px;margin:auto}section{background:#1e293b;padding:16px;margin:12px 0;border-radius:10px}img{width:48%;background:#111;margin:1%;border-radius:6px}.start{background:#1677ff;color:white}.stop{background:#d00;color:white}button{padding:12px 22px;border:0;border-radius:6px;margin-right:10px;font-size:16px;cursor:pointer}pre{font-size:15px;background:#0f172a;padding:10px;border-radius:6px;color:#38bdf8}</style>
<main><h2>独立巡线控制台 Node3 (Port 5003 - 去年原版算法)</h2>
<section><b>准备就绪。确认跑道清空后手动启动巡线。</b>
<p><button class="start" onclick="post('/api/start')">解锁并开始巡线</button><button class="stop" onclick="post('/api/stop')">立即停车</button><button onclick="post('/api/reset')">复位为仅感知模式</button></p>
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
                    self.reply({'ok': False, 'error': '动作解锁开关未开启'})
                    return
                if state.mode == 'RUNNING':
                    self.reply({'ok': False, 'error': '巡线任务已经在运行中'})
                    return
                if time.time() - state.last_image_time > 0.8:
                    self.reply({'ok': False, 'error': '摄像头画面超时，请检查相机'})
                    return
                if not state.vision_valid or len(state.centers) < 3:
                    self.reply({'ok': False, 'error': '当前没有可靠车道中线，禁止启动'})
                    return
                state.last_v_z = 0.0
                state.delat_v_z = 0.0
                state.lost_frames = 0
                state.last_valid_time = time.time()
                state.last_control_time = time.time()
                state.mode = 'RUNNING'
                state.message = '巡线节点3运行中'
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
    rospy.init_node('line_following_node3', anonymous=False)
    load_hsv()

    cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
    mask_pub = rospy.Publisher('/line_following/debug/mask', Image, queue_size=1)
    overlay_pub = rospy.Publisher('/line_following/debug/overlay', Image, queue_size=1)

    rospy.Subscriber('/usb_cam/image_raw', Image, image_cb, queue_size=1, buff_size=2**24)
    rospy.Timer(rospy.Duration(0.04), control_timer)  # 25Hz 控制循环
    rospy.on_shutdown(shutdown)

    server = ReusableHTTPServer(('0.0.0.0', PORT), Handler)
    rospy.loginfo('独立高速巡线节点3已启动: http://0.0.0.0:%d', PORT)
    server.serve_forever()


if __name__ == '__main__':
    main()
