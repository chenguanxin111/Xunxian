#!/usr/bin/env python3
"""
line_following_ss_pure.py

基于去年 ss.py 的纯巡线程序（去掉绕环岛/路口检测/两阶段转弯逻辑）。
保留：
  1. ss.py 原汁原味 HSV 二值化 + 动态 ROI（软启动/直道/标准视野切换）
  2. find_white_pixel_indices 逐行聚类中心点提取
  3. calculate_metrics 割线角度误差计算
  4. ss.py 原版 PID 参数矩阵 + 软启动增益削减 + 动态视野
  5. 横向平移控制（error_y）
删除：
  - calculate_blue_slopes / make_hybrid_junction_decision / has_horizontal_line（路口检测）
  - xunxian_2 中的两阶段转弯执行（里程计粗转 + PID 视觉精调）
  - 避障状态机、终点线检测、语音播报
可视化：node4 风格的 Web 控制台（Port 5007），显示 overlay / mask / 状态。

使用：
  rosrun line_following_ss_pure line_following_ss_pure.py  (或 python3 直接运行)
  浏览器打开 http://<小车IP>:5007 进行解锁/启动/停止/调速
"""

import os
import sys

# 彻底屏蔽 Qt xcb 桌面显示器连接，防止在 SSH 环境下触发 Qt qFatal (SIGABRT) 终止程序
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["OPENCV_UI_BACKEND"] = "HEADLESS"
if "DISPLAY" in os.environ:
    del os.environ["DISPLAY"]

import json
import time
import math
import threading
from urllib.parse import urlparse
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist

# 统一分辨率（与去年 ss.py pw=640, ph=360 一致）
pw = 640
ph = 360

PORT = 5007
ALLOW_MOTION = True

# 软启动相关（与 ss.py 一致）
GENTLE_START_DURATION = 2.0          # 软启动持续时间（秒）
LARGE_FOV_DURATION_AFTER_START = 2.5 # 软启动结束后，使用远视野持续的时间（秒）

# 摄像头无图像保护（秒）
CAMERA_TIMEOUT = 0.8

# 停止线检测参数（外接矩形法）
STOP_LINE_ROI_TOP_RATIO = 0.75     # 停止线检测区域限制为画面下方 25% (y 从 0.75*H 到 H)
STOP_LINE_WIDTH_RATIO = 0.70       # 外接矩形宽度占比阈值（相对画面宽度）
STOP_LINE_THIN_RATIO = 0.30        # 外接矩形高度/宽度比上限（保证细长）
CREEP_SPEED = 0.12                 # 检测到停止线后的蠕动速度 (m/s)
# 车道线斜率匹配参数
SLOPE_TOLERANCE = 0.3          # 左右两条车道线斜率差值上限（|slope_L - slope_R| <= tolerance）
SLOPE_HIST_LEN = 4             # 用于计算每条边斜率的最小历史行数

CREEP_DISTANCE = 0.05              # 蠕动前进距离 (m)，5cm


class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.mode = 'DISARMED'
        self.message = '纯巡线程序已启动 (基于去年 ss.py，去掉环岛逻辑)'
        self.last_image_time = 0.0

        # 巡线感知数据
        self.error = 0.0
        self.last_error = 0.0
        self.raw_error = 0.0
        self.kanbujian = 0
        self.centers = []
        self.red_points = []
        self.current_red_points_zuixiamian = []
        self.roi_1 = None
        self.roi_2 = None
        self.vis = None
        self.vis_overlay = None
        self.vis_mask = None
        self.roi_up = 0.69

        # 运动控制参数
        self.target_speed = 0.36  # ss.py 默认速度 0.36 m/s
        self.start_time = 0.0
        self.last_v_z = 0.0
        self.delat_v_z = 0.0
        self.chongci = 0.0
        self.command_linear_x = 0.0
        self.command_linear_y = 0.0
        self.command_angular_z = 0.0

        self.vision_valid = False
        self.lost_frames = 0
        self.last_valid_time = 0.0

        # 停止线检测状态
        self.stop_line_detected = False
        self.stop_line_y = -1
        self.stop_line_hits = 0
        self.stop_line_stopped = False
        self.creep_started = False
        self.creep_start_x = 0.0
        self.creep_start_y = 0.0

        # odom
        self.odom_x = 0.0
        self.odom_y = 0.0

    def status(self):
        return {
            'mode': self.mode,
            'message': self.message,
            'error_deg': float(round(self.error, 1)) if self.error is not None else None,
            'kanbujian': bool(self.kanbujian),
            'center_count': int(len(self.centers)),
            'roi_up': float(self.roi_up),
            'target_speed': float(self.target_speed),
            'command_linear_x': float(round(self.command_linear_x, 3)),
            'command_linear_y': float(round(self.command_linear_y, 3)),
            'command_angular_z': float(round(self.command_angular_z, 3)),
            'vision_valid': bool(self.vision_valid),
            'lost_frames': int(self.lost_frames),
            'image_age_s': float(round(max(0.0, time.time() - self.last_image_time), 2)),
            'stop_line_detected': bool(self.stop_line_detected),
            'stop_line_y': int(self.stop_line_y),
            'stop_line_hits': int(self.stop_line_hits),
            'stop_line_stopped': bool(self.stop_line_stopped),
            'creep_started': bool(self.creep_started),
            'odom_x': float(round(self.odom_x, 3)),
            'odom_y': float(round(self.odom_y, 3)),
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
        msg.encoding = encoding
        msg.step = msg.width * (cv_img.shape[2] if len(cv_img.shape) == 3 else 1)
        msg.data = cv_img.tobytes()
        return msg


CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config')

def load_hsv_params():
    paths = [
        os.path.join(CONFIG_DIR, 'white_lane.json'),
        os.path.join(CONFIG_DIR, 'white_lane_right.json'),
        os.path.join(CONFIG_DIR, 'hsv_params.json'),
        os.path.join(CONFIG_DIR, 'fallback_hsv_params.json')
    ]
    for p in paths:
        if os.path.exists(p):
            try:
                with open(p, 'r') as f:
                    data = json.load(f)
                    rospy.loginfo("纯巡线节点成功载入 HSV 配置文件 [%s]: H=(%d,%d), S=(%d,%d), V=(%d,%d)",
                                  p, data.get('low_h', 0), data.get('high_h', 179),
                                  data.get('low_s', 0), data.get('high_s', 71),
                                  data.get('low_v', 116), data.get('high_v', 255))
                    return data
            except Exception as e:
                pass
    return {'low_h': 42, 'high_h': 179, 'low_s': 5, 'high_s': 71, 'low_v': 116, 'high_v': 255,
            'blur_ksize': 4, 'erode_iter': 0, 'erode_ksize': 3, 'dilate_iter': 2, 'dilate_ksize': 3}

HSV_PARAMS = load_hsv_params()


def new_get_yellow_lane_bin_img(frame, params=None):
    if params is None:
        params = HSV_PARAMS
    
    blur_k = int(params.get('blur_ksize', 4))
    if blur_k >= 3:
        if blur_k % 2 == 0:
            blur_k += 1
        frame_for_hsv = cv2.GaussianBlur(frame, (blur_k, blur_k), 0)
    else:
        frame_for_hsv = frame

    low_h = params.get('low_h', 42)
    high_h = params.get('high_h', 179)
    low_s = params.get('low_s', 5)
    high_s = params.get('high_s', 71)
    low_v = params.get('low_v', 116)
    high_v = params.get('high_v', 255)

    lower_array = np.array([low_h, low_s, low_v])
    upper_array = np.array([high_h, high_s, high_v])
    hsv = cv2.cvtColor(frame_for_hsv, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lowerb=lower_array, upperb=upper_array)
    
    small_mask = cv2.resize(mask, (pw, ph), interpolation=cv2.INTER_AREA)
    small_mask = small_mask.astype(np.uint8)
    retval, bin_img = cv2.threshold(small_mask, 125, 255, cv2.THRESH_BINARY)
    origin_img = cv2.resize(frame, (pw, ph), interpolation=cv2.INTER_AREA)
    return origin_img, bin_img, mask


def find_white_pixel_indices(img, recent_error_sign=0):
    current_red_points_zuixiamian = []
    height, width = img.shape
    sigle, double = 0, 0
    green_points = []
    red_points = []
    kanbujian = 0
    # 斜率匹配：双边追踪
    edge_L_xs, edge_R_xs, edge_ys = [], [], []
    frozen_half_width = 0  # 最近一次通过斜率匹配确认的"半车道宽"
    for y in range(height - 1, -1, -4):
        white_indices = np.where(img[y] == 255)[0]
        if len(white_indices) == 0:
            continue
        diff = np.diff(white_indices)
        breaks = np.where(diff > 1)[0] + 1
        clusters = np.split(white_indices, breaks)
        mean_indices = [np.mean(cluster) for cluster in clusters]
        current_red_points = []
        new_green_point = None
        if len(mean_indices) == 1:
            sigle += 1
            red_x = int(mean_indices[0])
            current_red_points.append(red_x)
            # 单边：优先用冻结半宽补对面边（比补到0/639更合理）
            if len(green_points) > 1 and frozen_half_width > 0:
                last_green_x = green_points[-1][0]
                virtual_red_x = red_x + frozen_half_width if red_x < last_green_x else red_x - frozen_half_width
                virtual_red_x = max(0, min(width - 1, int(virtual_red_x)))
            elif len(green_points) > 1:
                last_green_x = green_points[-1][0]
                second_last_green_x = green_points[-2][0]
                virtual_red_x = 0 if last_green_x < second_last_green_x else width - 1
            else:
                virtual_red_x = width - 1 if red_x < width // 2 else 0
            current_red_points.append(virtual_red_x)
            avg_index = np.mean(current_red_points)
            new_green_point = (int(avg_index), y)
        elif len(mean_indices) > 1:
            double += 1
            real_edges = sorted([int(idx) for idx in mean_indices])
            # 选出左右各一个代表性边
            if len(real_edges) == 2:
                selected = real_edges
            elif len(real_edges) > 2:
                mid_x = width // 2
                left_edges = [e for e in real_edges if e < mid_x]
                right_edges = [e for e in real_edges if e >= mid_x]
                left_nearest = min(left_edges, key=lambda x: abs(x - mid_x)) if left_edges else 0
                right_nearest = min(right_edges, key=lambda x: abs(x - mid_x)) if right_edges else width - 1
                selected = [left_nearest, right_nearest]
            eL, eR = selected[0], selected[1]
            current_red_points = [eL, eR]
            # 如果两条都是真实边（非虚拟0/639），做斜率匹配
            if eL not in [0, width - 1] and eR not in [0, width - 1]:
                edge_L_xs.append(eL)
                edge_R_xs.append(eR)
                edge_ys.append(y)
                if len(edge_ys) >= SLOPE_HIST_LEN:
                    ys = np.array(edge_ys, dtype=np.float64)
                    xs_L = np.array(edge_L_xs, dtype=np.float64)
                    xs_R = np.array(edge_R_xs, dtype=np.float64)
                    slope_L = np.polyfit(ys, xs_L, 1)[0]
                    slope_R = np.polyfit(ys, xs_R, 1)[0]
                    if abs(slope_L - slope_R) <= SLOPE_TOLERANCE:
                        # 匹配成功：当前帧的可靠配对
                        frozen_half_width = abs(eL - eR) / 2.0
                        avg_index = np.mean(selected)
                        new_green_point = (int(avg_index), y)
                    else:
                        # 不匹配：置信度选边
                        trusted = _pick_edge_by_confidence(
                            eL, slope_L, eR, slope_R, recent_error_sign, green_points
                        )
                        # 转为单边模式，用冻结半宽推算虚拟边
                        current_red_points = [trusted]
                        sigle += 1
                        if frozen_half_width > 0:
                            v = trusted + frozen_half_width if trusted < width // 2 else trusted - frozen_half_width
                        else:
                            v = width - 1 if trusted < width // 2 else 0
                        current_red_points.append(v)
                        avg_index = np.mean(current_red_points)
                        new_green_point = (int(avg_index), y)
                else:
                    avg_index = np.mean(selected)
                    new_green_point = (int(avg_index), y)
            else:
                # 有虚拟边：窄簇（lane line pair 太近）→ 舍弃靠近中央的补虚拟边
                if len(current_red_points) == 2:
                    if abs(current_red_points[0] - current_red_points[1]) < width / 3:
                        if abs(current_red_points[0] - width // 2) > abs(current_red_points[1] - width // 2):
                            current_red_points = [current_red_points[1], width - 1 if current_red_points[1] < width // 2 else 0]
                        else:
                            current_red_points = [current_red_points[0], width - 1 if current_red_points[0] < width // 2 else 0]
                    avg_index = np.mean(current_red_points)
                    new_green_point = (int(avg_index), y)
                else:
                    mid_x = width // 2
                    left_red_points = [pt for pt in current_red_points if pt < mid_x]
                    right_red_points = [pt for pt in current_red_points if pt >= mid_x]
                    left_nearest = min(left_red_points, key=lambda x: abs(x - mid_x)) if left_red_points else 0
                    right_nearest = min(right_red_points, key=lambda x: abs(x - mid_x)) if right_red_points else width - 1
                    current_red_points = [left_nearest, right_nearest]
                    avg_index = np.mean(current_red_points)
                    new_green_point = (int(avg_index), y)
        if current_red_points:
            current_red_points_zuixiamian = current_red_points
        for rp in current_red_points:
            red_points.append((rp, y))
        if sigle + double > 0 and sigle / (sigle + double) > 0.9:
            kanbujian = 1
        else:
            kanbujian = 0
        if new_green_point and (len(green_points) == 0 or abs(new_green_point[0] - green_points[-1][0]) < width / 8):
            green_points.append(new_green_point)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for x, y in green_points:
        cv2.circle(vis, (x, y), 3, (0, 255, 0), -1)
    for x, y in red_points:
        cv2.circle(vis, (x, y), 3, (0, 0, 255), -1)
    return green_points, red_points, current_red_points_zuixiamian, vis, kanbujian


def _pick_edge_by_confidence(eL, slope_L, eR, slope_R, recent_error_sign, green_points):
    """两条边斜率不匹配时，根据历史转向方向选择更可信的一条。

    recent_error_sign > 0 → 正在左转 → 左侧边更可信
    recent_error_sign < 0 → 正在右转 → 右侧边更可信
    recent_error_sign == 0 → 根据中心线趋势选斜率更匹配的边
    """
    if recent_error_sign > 0:
        return eL
    if recent_error_sign < 0:
        return eR
    if len(green_points) >= 5:
        recent_ys = np.array([gp[1] for gp in green_points[-5:]], dtype=np.float64)
        recent_xs = np.array([gp[0] for gp in green_points[-5:]], dtype=np.float64)
        center_slope = np.polyfit(recent_ys, recent_xs, 1)[0]
        if abs(slope_L - center_slope) < abs(slope_R - center_slope):
            return eL
        return eR
    return eL


def extract_horizontal_bands(bin_img, kernel_w=31):
    """用宽 x1 的水平核做开运算，分离横向白带（保留横向结构、消除纵向车道线）"""
    if bin_img is None or bin_img.size == 0:
        return bin_img
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 1))
    return cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel)


def remove_horizontal_white_bands(bin_img, width_ratio=0.50, thin_ratio=STOP_LINE_THIN_RATIO, kernel_w=31):
    """从二值图中剔除横向细长白色带（停止线），只保留纵向车道线。

    先用横向开运算分离出横向带，再按外接矩形宽度/细长条件清除，
    避免停止线与车道线相交连成一片导致过滤失效。
    """
    if bin_img is None or bin_img.size == 0:
        return bin_img
    h, w = bin_img.shape
    if h == 0 or w == 0:
        return bin_img
    horiz = extract_horizontal_bands(bin_img, kernel_w)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(horiz, 8)
    band_mask = np.zeros_like(bin_img)
    for label in range(1, num_labels):
        x, y, bw, bh, area = stats[label]
        if bw >= int(w * width_ratio) and bh <= bw * thin_ratio:
            band_mask[labels == label] = 255
    clean = bin_img.copy()
    clean[band_mask == 255] = 0
    return clean


def detect_stop_line(bin_img, top_ratio=STOP_LINE_ROI_TOP_RATIO, width_ratio=STOP_LINE_WIDTH_RATIO, thin_ratio=STOP_LINE_THIN_RATIO):
    """检测画面下方 25% 区域内的水平白线（停止线），外接矩形法。

    仅在 y >= height * top_ratio (默认画面最下方 25%) 区域内检测，
    先用横向开运算分离横向带，再对外接矩形按宽度占比 >= width_ratio 且高度/宽度 <= thin_ratio 判定。
    返回 (detected, rect_bottom_y)。
    """
    if bin_img is None or bin_img.size == 0:
        return False, -1
    height, width = bin_img.shape
    if height == 0 or width == 0:
        return False, -1

    roi_y_start = int(height * top_ratio)
    roi_bin = bin_img[roi_y_start:, :]
    if roi_bin.size == 0:
        return False, -1

    horiz = extract_horizontal_bands(roi_bin)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(horiz, 8)
    lowest_y = -1
    detected = False
    for label in range(1, num_labels):
        x, y, bw, bh, area = stats[label]
        if bw >= int(width * width_ratio) and bh <= bw * thin_ratio:
            detected = True
            y_in_full = y + bh + roi_y_start
            if y_in_full > lowest_y:
                lowest_y = int(y_in_full)
    return bool(detected), int(lowest_y)


def new_get_results(yuan_image, line_up_ratio=0.69, bin_img=None, recent_error_sign=0):
    line_low = 1.0
    if bin_img is None:
        origin_img, bin_img, mask = new_get_yellow_lane_bin_img(yuan_image, HSV_PARAMS)
    else:
        origin_img = cv2.resize(yuan_image, (pw, ph), interpolation=cv2.INTER_AREA)
    H = origin_img.shape[0]
    W = origin_img.shape[1]
    bin_img_rectangle_ROI = bin_img[int(H * line_up_ratio):int(H * line_low), :]
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    bin_img_rectangle_ROI = cv2.erode(bin_img_rectangle_ROI, kernel_erode, iterations=1)
    bin_img_rectangle_ROI = cv2.dilate(bin_img_rectangle_ROI, kernel_dilate, iterations=2)
    # 剔除横向停止线带，只保留纵向车道线，避免停止线被误匹配为车道线
    bin_img_rectangle_ROI = remove_horizontal_white_bands(bin_img_rectangle_ROI)
    green_points, red_points, current_red_points_zuixiamian, vis, kanbujian = find_white_pixel_indices(bin_img_rectangle_ROI, recent_error_sign)
    return green_points, red_points, bin_img_rectangle_ROI, bin_img_rectangle_ROI, current_red_points_zuixiamian, vis, kanbujian


def calculate_metrics(green_points):
    if len(green_points) < 3:
        return 0.0, 0.0, 0.0
    first_point = green_points[0]
    last_point = green_points[-1]
    middle_point = green_points[len(green_points) // 2]
    if first_point[0] != last_point[0]:
        slope_first_last = (last_point[1] - first_point[1]) / (last_point[0] - first_point[0])
        angle_first_last = np.degrees(np.arctan(slope_first_last))
    else:
        angle_first_last = 90.0 if (last_point[1] - first_point[1]) > 0 else -90.0
    middle_point = green_points[round(len(green_points) / 3.5)]
    if first_point[0] != middle_point[0]:
        slope_first_middle = (middle_point[1] - first_point[1]) / (middle_point[0] - first_point[0])
        angle_first_middle = np.degrees(np.arctan(slope_first_middle))
    else:
        angle_first_middle = 90.0 if (middle_point[1] - first_point[1]) > 0 else -90.0
    last_3_x_coords = [point[0] for point in green_points[:3]]
    avg_last_3_x = np.mean(last_3_x_coords)
    return angle_first_last, angle_first_middle, avg_last_3_x


# ------------------ PID（ss.py 原版） ------------------

def get_pid_message(error, kanbujian):
    """根据误差和是否看到单边线，生成 PID 状态文本（ss.py 原版）"""
    abs_error = abs(error)
    if kanbujian:
        if 33.5 < abs_error <= 51: return "看不见：小大弯！"
        if 51 < abs_error <= 62: return "看不见：中弯！"
        if 62 < abs_error <= 64: return "看不见：极弯"
        if abs_error > 64: return "看不见：大极弯！"
        return "看不见：直线！"
    else:
        if 30 < abs_error <= 34: return "小直线！"
        if 34 < abs_error <= 55: return "小弯，一般情况！"
        if 55 < abs_error <= 60: return "中弯！"
        if abs_error > 60: return "大弯！"
        return "大直线！"


def compute_pid(error, kanbujian, current_red_points_zuixiamian, state):
    """纯巡线 PID 控制（ss.py xunxian_2 的 else 分支，去掉转弯逻辑）"""
    pid_message = get_pid_message(error, kanbujian)
    if "看不见" in pid_message:
        if "小大弯" in pid_message: kp_z, kp_y, kd_z = 0.024, 0.00005, 0.22
        elif "中弯" in pid_message: kp_z, kp_y, kd_z = 0.026, 0.00005, 0.2
        elif "极弯" in pid_message: kp_z, kp_y, kd_z = 0.029, 0.00005, 0.1
        elif "大极弯" in pid_message: kp_z, kp_y, kd_z = 0.033, 0.00005, 0.25
        else: kp_z, kp_y, kd_z = 0.020, 0.00005, 0.22  # 看不见：直线默认增益
    else:
        if "小直线" in pid_message: kp_z, kp_y, kd_z = 0.016, 0.00005, 0.25
        elif "小弯" in pid_message: kp_z, kp_y, kd_z = 0.022, 0.00005, 0.2
        elif "中弯" in pid_message: kp_z, kp_y, kd_z = 0.024, 0.00005, 0.2
        elif "大弯" in pid_message: kp_z, kp_y, kd_z = 0.0265, 0.00005, 0.15
        else:
            state.chongci = 0.0
            kp_z, kp_y, kd_z = 0.013, 0.0005, 0.3

    vel = Twist()
    is_gentle_start = (time.time() - state.start_time) < GENTLE_START_DURATION
    if is_gentle_start:
        vel.linear.x = 0.15
        state.chongci = 0
        kp_z *= 0.45
        kd_z *= 0.5
    else:
        vel.linear.x = state.target_speed + state.chongci
        kp_z *= 1.22
        kd_z *= 1.2

    kp_y *= 15

    vel.angular.z = kp_z * error - kd_z * state.delat_v_z
    state.delat_v_z = vel.angular.z - state.last_v_z
    state.last_v_z = vel.angular.z

    error_y = 0
    if current_red_points_zuixiamian and all(p not in [0, 639] for p in current_red_points_zuixiamian):
        error_y = np.mean(current_red_points_zuixiamian)

    if error_y != 0:
        error_y_contorl = error_y - 320
        vel.linear.y = kp_y * error_y_contorl * 0.0005
    else:
        vel.linear.y = 0

    return vel, pid_message


# ------------------ 可视化辅助 ------------------

SHOW_GUI = '--show-gui' in sys.argv

def draw_overlay(resize_img, green_points, red_points, roi_up):
    """将 ROI 中的绿/红点映射回 640x360 全图并叠加显示"""
    overlay = resize_img.copy()
    H = ph
    roi_start_y = int(H * roi_up)

    for x, y in red_points:
        cv2.circle(overlay, (x, y + roi_start_y), 2, (0, 0, 255), -1)
    for x, y in green_points:
        cv2.circle(overlay, (x, y + roi_start_y), 3, (0, 255, 0), -1)
    if len(green_points) > 1:
        pts = np.array([(x, y + roi_start_y) for x, y in green_points], np.int32)
        cv2.polylines(overlay, [pts], False, (0, 255, 0), 3)
    return overlay


def image_cb(msg):
    try:
        frame = bridge.imgmsg_to_cv2(msg, 'passthrough')
        if msg.encoding.lower() == 'rgb8':
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        elif msg.encoding.lower() != 'bgr8':
            frame = bridge.imgmsg_to_cv2(msg, 'bgr8')

        with state.lock:
            start_time = state.start_time
            prev_error_sign = np.sign(state.error) if state.error != 0 else 0
        elapsed = time.time() - start_time if start_time > 0 else 9999.0

        # 动态 ROI 切换：默认根据用户要求设定为底部 40% (roi_up = 0.60)
        if elapsed < GENTLE_START_DURATION:
            roi_up = 0.40
        elif elapsed < GENTLE_START_DURATION + LARGE_FOV_DURATION_AFTER_START:
            roi_up = 0.50
        else:
            roi_up = 0.60

        # 停止线检测（基于全画面二值图）
        _, full_bin, _ = new_get_yellow_lane_bin_img(frame, HSV_PARAMS)
        stop_detected, stop_y = detect_stop_line(full_bin)

        green_points, red_points, roi_1, roi_2, current_red_points_zuixiamian, vis, kanbujian = new_get_results(frame, line_up_ratio=roi_up, bin_img=full_bin, recent_error_sign=prev_error_sign)
        angle_first_last, angle_first_middle, avg_last_3_x = calculate_metrics(green_points)

        if angle_first_middle > 0:
            error = -90 + angle_first_middle
        else:
            error = 90 + angle_first_middle

        overlay = draw_overlay(cv2.resize(frame, (pw, ph)), green_points, red_points, roi_up)
        cv2.putText(overlay, f"MODE: {state.mode} | KANBUJIAN: {kanbujian} | ROI: {roi_up}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 255), 2)
        cv2.putText(overlay, f"error={error:.1f} deg  centers={len(green_points)}",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
        # 停止线可视化：在检测到的矩形底边上画标记
        if stop_detected and stop_y >= 0:
            cv2.line(overlay, (0, stop_y), (pw - 1, stop_y), (0, 0, 255), 2)
        cv2.putText(overlay, f"STOPLINE: {stop_detected} y={stop_y}", (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255), 1)

        # 仅显示时翻转（检测仍用未翻转图，与 ss.py 符号约定一致）；
        # 先翻转图像再加文字，避免文字镜像。
        overlay_disp = cv2.flip(overlay, 1)
        mask_disp = cv2.flip(roi_1, 1)
        cv2.putText(overlay_disp, f"MODE: {state.mode} | KANBUJIAN: {kanbujian} | ROI: {roi_up}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 255), 2)
        cv2.putText(overlay_disp, f"error={error:.1f} deg  centers={len(green_points)}",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
        cv2.putText(overlay_disp, f"STOPLINE: {stop_detected} y={stop_y}", (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255), 1)

        with state.lock:
            state.last_image_time = time.time()
            state.roi_1 = roi_1
            state.roi_2 = roi_2
            state.vis = vis
            state.vis_overlay = overlay_disp
            state.vis_mask = mask_disp
            state.roi_up = roi_up
            state.centers = green_points
            state.red_points = red_points
            state.current_red_points_zuixiamian = current_red_points_zuixiamian
            state.kanbujian = kanbujian
            state.raw_error = error
            state.error = error
            state.vision_valid = len(green_points) >= 3
            if state.vision_valid:
                state.last_valid_time = state.last_image_time
                state.lost_frames = 0
            else:
                state.lost_frames += 1
            state.stop_line_detected = stop_detected
            state.stop_line_y = stop_y

        if mask_pub is not None:
            mask_pub.publish(safe_cv2_to_imgmsg(mask_disp, 'mono8'))
        if overlay_pub is not None:
            overlay_pub.publish(safe_cv2_to_imgmsg(overlay_disp, 'bgr8'))

    except Exception as e:
        rospy.logerr_throttle(2, f"纯巡线图像回调异常: {e}")


def publish_stop():
    state.command_linear_x = 0.0
    state.command_linear_y = 0.0
    state.command_angular_z = 0.0
    state.last_v_z = 0.0
    state.delat_v_z = 0.0
    if cmd_pub is not None:
        cmd_pub.publish(Twist())


def odom_cb(data):
    with state.lock:
        state.odom_x = data.pose.pose.position.x
        state.odom_y = data.pose.pose.position.y


def control_timer(_event):
    with state.lock:
        now = time.time()
        if state.mode != 'RUNNING':
            return

        if state.last_image_time == 0 or now - state.last_image_time > CAMERA_TIMEOUT:
            state.mode = 'FAULT'
            state.message = '摄像头画面超时，已紧急停车'
            publish_stop()
            return

        err = state.error
        kanbujian = bool(state.kanbujian)
        current_red_bottom = state.current_red_points_zuixiamian

        # 停止线检测：一旦检测到停止线，立即停止转向并进入蠕动状态
        if state.stop_line_detected:
            if not state.creep_started:
                state.creep_started = True
                state.creep_start_x = state.odom_x
                state.creep_start_y = state.odom_y
                rospy.loginfo("!!! 检测到停止线，开始蠕动 !!!")

            traveled = math.hypot(state.odom_x - state.creep_start_x,
                                  state.odom_y - state.creep_start_y)
            if traveled >= CREEP_DISTANCE:
                state.mode = 'STOPPED'
                state.message = '检测到停止线，蠕动到位，已刹停'
                state.stop_line_stopped = True
                publish_stop()
                rospy.loginfo("!!! 蠕动 %.2fcm 完成，已刹停 !!!", traveled * 100)
                return

            # 蠕动：保持直线低速前进，不转向
            vel_creep = Twist()
            vel_creep.linear.x = CREEP_SPEED
            vel_creep.linear.y = 0.0
            vel_creep.angular.z = 0.0
            state.command_linear_x = vel_creep.linear.x
            state.command_linear_y = vel_creep.linear.y
            state.command_angular_z = vel_creep.angular.z
            cmd_pub.publish(vel_creep)
            return

        vel, pid_message = compute_pid(err, kanbujian, current_red_bottom, state)

        state.command_linear_x = vel.linear.x
        state.command_linear_y = vel.linear.y
        state.command_angular_z = vel.angular.z

        rospy.loginfo_throttle(1, "纯巡线发布: vx=%.2f, vy=%.2f, wz=%.2f (err=%.1f, %s)",
                               vel.linear.x, vel.linear.y, vel.angular.z, err, pid_message)
        cmd_pub.publish(vel)


# ------------------ Web 控制台 ------------------

PAGE = '''<!doctype html><meta charset="utf-8"><title>纯巡线控制台 (ss_pure)</title>
<style>body{font:16px sans-serif;background:#0f172a;color:#f8fafc;margin:20px}main{max-width:1100px;margin:auto}section{background:#1e293b;padding:16px;margin:12px 0;border-radius:10px}img{width:48%;background:#111;margin:1%;border-radius:6px}.start{background:#1677ff;color:white}.stop{background:#d00;color:white}button{padding:10px 18px;border:0;border-radius:6px;margin-right:8px;font-size:15px;cursor:pointer}pre{font-size:15px;background:#0f172a;padding:10px;border-radius:6px;color:#38bdf8}</style>
<main><h2>纯巡线控制台 (Port 5007 - 去年 ss.py 去除环岛)</h2>
<section><b>准备就绪。确认跑道清空后手动启动巡线。</b>
<p><button class="start" onclick="post('/api/start')">解锁并开始巡线</button><button class="stop" onclick="post('/api/stop')">立即停车</button><button onclick="post('/api/reset')">复位为仅感知模式</button></p>
<p>设置巡线目标速度：
<button onclick="post('/api/set_speed?speed=0.18')">0.18 m/s (超慢试跑)</button>
<button onclick="post('/api/set_speed?speed=0.24')">0.24 m/s (中速赛道)</button>
<button onclick="post('/api/set_speed?speed=0.36')">0.36 m/s (去年原版速度)</button>
</p>
<pre id="status">加载状态中...</pre></section>
<section><h3>识别叠加图 (/stream/overlay) 与 二值图 (/stream/mask)</h3>
<p><img id="img_overlay"><img id="img_mask"></p></section>
</main>
<script>
document.getElementById('img_overlay').src = '/stream/overlay';
document.getElementById('img_mask').src = '/stream/mask';
async function post(url){
    try {
        let r = await fetch(url,{method:'POST'});
        let d = await r.json();
        if(!d.ok){ alert("⚠️ 操作失败: " + (d.error || '未知错误')); }
    } catch(e) { alert("⚠️ 请求失败: " + e); }
}
setInterval(async()=>{try{document.getElementById('status').textContent=JSON.stringify(await(await fetch('/api/status')).json(),null,2)}catch(e){}},400);
</script>'''


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def reply(self, obj):
        def default_converter(o):
            if isinstance(o, (np.integer, np.int32, np.int64)):
                return int(o)
            elif isinstance(o, (np.floating, np.float32, np.float64)):
                return float(o)
            elif isinstance(o, np.ndarray):
                return o.tolist()
            elif isinstance(o, np.bool_):
                return bool(o)
            raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

        data = json.dumps(obj, default=default_converter, ensure_ascii=False).encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == '/':
            data = PAGE.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        elif path == '/api/status':
            with state.lock:
                self.reply(state.status())
        elif path in ['/stream/overlay', '/stream/mask']:
            self.send_response(200)
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()
            while not rospy.is_shutdown():
                with state.lock:
                    img = state.vis_overlay if path == '/stream/overlay' else state.vis_mask
                if img is not None:
                    ret, jpeg = cv2.imencode('.jpg', img)
                    if ret:
                        try:
                            self.wfile.write(b'--frame\r\n')
                            self.wfile.write(b'Content-Type: image/jpeg\r\n\r\n')
                            self.wfile.write(jpeg.tobytes())
                            self.wfile.write(b'\r\n')
                        except Exception:
                            break
                time.sleep(0.04)
        else:
            self.send_error(404)

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path
        with state.lock:
            if path == '/api/start':
                if not ALLOW_MOTION:
                    self.reply({'ok': False, 'error': '动作解锁开关未开启'})
                    return
                if state.mode == 'RUNNING':
                    self.reply({'ok': False, 'error': '巡线任务已经在运行中'})
                    return
                if time.time() - state.last_image_time > CAMERA_TIMEOUT:
                    self.reply({'ok': False, 'error': '摄像头画面超时，请检查相机'})
                    return
                if not state.vision_valid or len(state.centers) < 3:
                    self.reply({'ok': False, 'error': '当前没有可靠车道中线，禁止启动'})
                    return
                state.last_v_z = 0.0
                state.delat_v_z = 0.0
                state.chongci = 0.0
                state.lost_frames = 0
                state.start_time = time.time()
                state.last_valid_time = time.time()
                state.stop_line_hits = 0
                state.stop_line_stopped = False
                state.creep_started = False
                state.mode = 'RUNNING'
                state.message = '纯巡线运行中'
                self.reply({'ok': True})
            elif path == '/api/set_speed':
                query = parsed.query
                params = dict(q.split('=') for q in query.split('&') if '=' in q)
                if 'speed' in params:
                    try:
                        sp = float(params['speed'])
                        state.target_speed = max(0.08, min(0.40, sp))
                        self.reply({'ok': True, 'target_speed': state.target_speed})
                        return
                    except Exception as e:
                        self.reply({'ok': False, 'error': str(e)})
                        return
                self.reply({'ok': False, 'error': '缺少 speed 参数'})
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


class ReusableHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True

    def server_bind(self):
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        super().server_bind()


def shutdown():
    publish_stop()


def main():
    global cmd_pub, mask_pub, overlay_pub
    rospy.init_node('line_following_ss_pure', anonymous=False)

    cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
    mask_pub = rospy.Publisher('/line_following_ss/debug/mask', Image, queue_size=1)
    overlay_pub = rospy.Publisher('/line_following_ss/debug/overlay', Image, queue_size=1)

    rospy.Subscriber('/usb_cam/image_raw', Image, image_cb, queue_size=1, buff_size=2**24)
    rospy.Subscriber('/odom', Odometry, odom_cb, queue_size=1)
    rospy.Timer(rospy.Duration(0.05), control_timer)  # 20Hz 控制循环
    rospy.on_shutdown(shutdown)

    server = ReusableHTTPServer(('0.0.0.0', PORT), Handler)
    rospy.loginfo('纯巡线程序已启动: http://0.0.0.0:%d', PORT)
    server.serve_forever()


if __name__ == '__main__':
    main()
