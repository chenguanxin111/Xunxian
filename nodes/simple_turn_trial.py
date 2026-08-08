#!/usr/bin/env python3
"""Minimal Junction Turn Node with Phase-1 Opposite Centerline Alignment.

Process Flow:
1. Phase 1 (F_ALIGN & F_YAW): Detect opposite junction centerline via IPM and translate
   laterally (linear.y) to correct center drift.
2. Micro-adjust vehicle yaw back to alignment origin (F_YAW).
3. Phase 2 (ADVANCE & ROTATE): Low-speed straight advance (e.g. 30 cm) -> Rotate (e.g. 65 deg) -> Stop.

Supports selecting Left or Right turn via Web UI (port 5002).
"""
import json
import math
import os
import sys
import threading
import time

# 彻底屏蔽 Qt xcb 桌面显示器连接，防止在 SSH 环境下触发 Qt qFatal终止程序
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["OPENCV_UI_BACKEND"] = "HEADLESS"
if "DISPLAY" in os.environ:
    del os.environ["DISPLAY"]

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from tf.transformations import euler_from_quaternion

BASE_DIR = '/home/ucar/Xunxian_standalone'
CONFIG_DIR = os.path.join(BASE_DIR, 'config')
HSV_PATH = os.path.join(CONFIG_DIR, 'white_lane.json')
FALLBACK_HSV_PATH = os.path.join(CONFIG_DIR, 'white_lane.json')
PERSP_PATH = os.path.join(CONFIG_DIR, 'perspective_params.json')
IMAGE_TOPIC = '/usb_cam/image_raw'
PORT = 5002

# ---- 路口对面中线提取参数 ----
OPP_ROI_BOTTOM = 0.85           # 只扫画面下半部分，剔除近场车道线干扰
OPP_ROI_TOP = 0.55              # 剔除顶部横向停止线/远处噪点
OPP_WIDTH_MIN = 120.0           # 配对宽度下界
OPP_WIDTH_MAX = 450.0           # 配对宽度上界
OPP_SLOPE_DIFF_MAX = 0.20       # 两线严格平行
OPP_MIN_OVERLAP = 30.0          # 两线最小重叠 Y 跨度
OPP_MIN_SPAN = 60.0             # 单线最小 IPM 纵向跨度
OPP_FIT_RESID = 30.0            # 单线拟合离群剔除阈值
IPM_CENTER_X = 300.0

# ---- 弧线车道中线检测参数（移植自 line_following_ss_pure_ipm.py）----
CAM_W = 640
CAM_H = 480
PROC_W = 320
PROC_H = 240
SCALE_X = PROC_W / CAM_W
SCALE_Y = PROC_H / CAM_H
CANVAS_X0 = 150.0
CANVAS_Y0 = 200.0
CANVAS_W = 300
CANVAS_H = 400
Y_NEAR = 600.0
Y_FAR = 260.0
ALPHA_POLY = 0.45

LANE_TUNE = {
    'roi_bottom_ratio': 0.30,
    'min_center_pts': 3,
    'min_center_span': 25.0,
    'poly_max_resid': 20.0,
    'lane_width_min': 110.0,
    'lane_width_max': 230.0,
    'lane_half_width': 84.0,
    'max_slope_diff': 0.55,
    'clean_min_h': 12,
    'clean_min_area': 40,
    'clean_min_ratio': 0.35,
    'lookahead_px': 120.0,
    'track_stale_frames': 10,
    'max_width_dev': 35.0,
    'max_cluster_w': 48,
    'match_min_overlap': 18.0,
}

# ---- 阶段一：路口对面平移对准参数 ----
F_ALIGN_SPEED = 0.04            # 平移对准前进速度 (m/s)
F_ALIGN_CENTER_TOL_PX = 18.0    # 收敛阈值 (IPM px)
F_ALIGN_HEADING_TOL_DEG = 12.0  # 航向角收敛阈值 (度)
F_ALIGN_CONFIRM_FRAMES = 10     # 连续确认帧数
F_ALIGN_KP_Y = 0.0022           # linear.y 平移增益
F_ALIGN_KP_Z = 0.0015           # center 对 angular.z 的耦合增益
F_ALIGN_WZ_CLAMP = 0.10         # 角速度限幅
F_ALIGN_MAX_DISTANCE = 0.40     # 平移阶段最远位移 (m)
F_ALIGN_TIMEOUT = 20.0          # 平移阶段超时 (s)

F_YAW_MAX_TIME = 0.6            # 微调最长耗时 (s)
F_YAW_TOL_DEG = 1.5             # 与对准起始 yaw 偏差收敛阈值 (度)
F_YAW_KP = 1.2                  # 微调 angular.z 增益
F_YAW_WZ = 0.15                 # 微调角速度限幅 (rad/s)

# ---- 阶段二：写死的前进 + 转弯参数 ----
ADVANCE_SPEED = 0.15            # m/s
ADVANCE_DISTANCE = 0.30         # m (30 cm)
ROTATE_SPEED = 0.35             # rad/s (~20 deg/s)
ROTATE_TARGET_DEG = 65.0        # deg

ADVANCE_TIMEOUT = 10.0          # s
ROTATE_TIMEOUT = 12.0           # s

# ---- 阶段三：弧线中线搜索（转弯后未看到车道线，同向慢速续转找中线）----
SEARCH_ROTATE_WZ = 0.15         # 慢速续转角速度 (rad/s)，左转继续左、右转继续右
SEARCH_ACCUM_LIMIT_DEG = 85.0   # 累计续转超限角 (deg)，超过则判定未转到位
SEARCH_TIMEOUT = 10.0           # 搜索超时 (s)
SEARCH_CONFIRM_FRAMES = 8       # 连续确认找到弧线中线的帧数
LANE_FOUND_TOPIC = '/simple_turn/lane_found'

DEFAULT_PARAMS = {
    'low_h': 0, 'high_h': 179,
    'low_s': 0, 'high_s': 45,
    'low_v': 170, 'high_v': 255,
    'roi_top': 0.45, 'roi_bottom': 1.0,
    'roi_left': 0.0, 'roi_right': 1.0,
    'blur_ksize': 3, 'erode_iter': 0, 'erode_ksize': 3, 'dilate_iter': 2, 'dilate_ksize': 3
}

IPM_MATRIX = None
IPM_INV_MATRIX = None
IPM_CANVAS_M = None


def normalize_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.mode = 'DISARMED'
        self.direction = 'LEFT'  # 'LEFT' or 'RIGHT'
        self.message = '仅感知模式：请在网页选择转向方向后点击解锁启动'

        self.odom = None         # (x, y, yaw)
        self.phase_start_pose = None
        self.phase_distance = 0.0
        self.advance_start_pose = None
        self.rotate_start_yaw = None

        self.advance_dist = 0.0
        self.turn_deg = 0.0
        self.state_started = time.time()
        self.last_image_time = 0.0

        # 对面中线对准状态
        self.hsv_params = dict(DEFAULT_PARAMS)
        self.opp_guide_valid = False
        self.opp_center_error = None
        self.opp_heading_error_deg = None
        self.opp_last_center_error = None
        self.opp_guide_good_frames = 0
        self.f_align_start_yaw = None
        self.f_yaw_started = None

        # 弧线中线检测（阶段三搜索）
        self.lane_valid = False
        self.lane_center_error = None
        self.lane_heading_error = None
        self.search_start_yaw = None
        self.search_start_time = 0.0
        self.search_good_frames = 0

        self.control_source = 'STOPPED'
        self.command_linear_x = 0.0
        self.command_linear_y = 0.0
        self.command_angular_z = 0.0

        self.vis_overlay = None
        self.vis_mask = None

    def status(self):
        return {
            'mode': self.mode,
            'direction': self.direction,
            'message': self.message,
            'opp_guide_valid': self.opp_guide_valid,
            'opp_center_error': round(self.opp_center_error, 1) if self.opp_center_error is not None else None,
            'opp_heading_error_deg': round(self.opp_heading_error_deg, 1) if self.opp_heading_error_deg is not None else None,
            'opp_guide_good_frames': self.opp_guide_good_frames,
            'lane_valid': self.lane_valid,
            'lane_center_error': round(self.lane_center_error, 1) if self.lane_center_error is not None else None,
            'search_good_frames': self.search_good_frames,
            'search_accum_deg': round(math.degrees(abs(normalize_angle(self.odom[2] - self.search_start_yaw))), 1)
                if self.odom is not None and self.search_start_yaw is not None else 0.0,
            'phase_distance_m': round(self.phase_distance, 3),
            'advance_dist_m': round(self.advance_dist, 3),
            'turn_deg': round(self.turn_deg, 1),
            'target_advance_m': ADVANCE_DISTANCE,
            'target_turn_deg': ROTATE_TARGET_DEG,
            'control_source': self.control_source,
            'command_linear_x': round(self.command_linear_x, 3),
            'command_linear_y': round(self.command_linear_y, 3),
            'command_angular_z': round(self.command_angular_z, 3),
            'odom_ready': self.odom is not None,
            'odom_x': round(self.odom[0], 3) if self.odom else None,
            'odom_y': round(self.odom[1], 3) if self.odom else None,
            'odom_yaw_deg': round(math.degrees(self.odom[2]), 1) if self.odom else None,
            'image_age_s': round(max(0.0, time.time() - self.last_image_time), 2),
        }


state = SharedState()
bridge = CvBridge()
cmd_pub = None
mask_pub = None
overlay_pub = None
lane_found_pub = None


class ArcLaneState:
    """弧线中线检测的跟踪状态（左右线身份 + 中线多项式滤波）。

    仅用于 SEARCH_ARC 阶段；进入该阶段前必须 reset() 以清空历史中线残留。
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.track_left = None
        self.track_right = None
        self.track_half_width = LANE_TUNE['lane_half_width']
        self.track_valid = False
        self.poly_filt = None
        self.poly_filt_mode = None
        self.poly_filt_y = (0.0, 0.0)


lane_state = ArcLaneState()


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


def init_ipm():
    global IPM_MATRIX, IPM_INV_MATRIX, IPM_CANVAS_M
    if os.path.exists(PERSP_PATH):
        try:
            with open(PERSP_PATH, 'r') as f:
                data = json.load(f)
                src_pts = np.float32(data['src_points'])
                dst_pts = np.float32(data['dst_points'])
                IPM_MATRIX = cv2.getPerspectiveTransform(src_pts, dst_pts)
                IPM_INV_MATRIX = cv2.getPerspectiveTransform(dst_pts, src_pts)
                small_src = src_pts.copy()
                small_src[:, 0] *= SCALE_X
                small_src[:, 1] *= SCALE_Y
                trans = np.float32([[1.0, 0.0, -CANVAS_X0],
                                    [0.0, 1.0, -CANVAS_Y0],
                                    [0.0, 0.0, 1.0]])
                IPM_CANVAS_M = trans @ cv2.getPerspectiveTransform(small_src, dst_pts)
                rospy.loginfo("成功载入 IPM 转换矩阵: %s", PERSP_PATH)
        except Exception as err:
            rospy.logwarn("读取 IPM 配置文件失败: %s", err)


def load_hsv():
    path_to_use = HSV_PATH if os.path.exists(HSV_PATH) else (FALLBACK_HSV_PATH if os.path.exists(FALLBACK_HSV_PATH) else None)
    if path_to_use:
        try:
            with open(path_to_use, 'r') as stream:
                data = json.load(stream)
                state.hsv_params.update(data)
                rospy.loginfo("已载入 HSV 配置文件: %s", path_to_use)
        except Exception as err:
            rospy.logwarn("读取 HSV 配置文件失败: %s", err)


def make_mask(frame, params):
    blur_k = int(params.get('blur_ksize', 0))
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
        np.array([params['high_h'], params['high_s'], params['high_v']]))

    h, w = mask.shape
    y1 = int(h * min(params.get('roi_top', 0.45), params.get('roi_bottom', 1.0)))
    y2 = int(h * max(params.get('roi_top', 0.45), params.get('roi_bottom', 1.0)))
    x1 = int(w * min(params.get('roi_left', 0.0), params.get('roi_right', 1.0)))
    x2 = int(w * max(params.get('roi_left', 0.0), params.get('roi_right', 1.0)))

    y1 = max(0, min(h - 1, y1))
    y2 = max(y1 + 1, min(h, y2))
    x1 = max(0, min(w - 1, x1))
    x2 = max(x1 + 1, min(w, x2))

    roi = np.zeros_like(mask)
    roi[y1:y2, x1:x2] = mask[y1:y2, x1:x2]

    for operation in ('erode', 'dilate'):
        iterations = int(params.get(operation + '_iter', 0))
        size = max(1, int(params.get(operation + '_ksize', 3)))
        if size % 2 == 0:
            size += 1
        if iterations > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
            roi = getattr(cv2, operation)(roi, kernel, iterations=iterations)
    return roi, (x1, y1, x2, y2)


def _opp_scan_side_points(mask_center, center_img_x):
    h = mask_center.shape[0]
    left_pts = []
    right_pts = []
    center_x = center_img_x
    for y in range(h - 1, -1, -4):
        xs = np.flatnonzero(mask_center[y])
        if len(xs) == 0:
            continue
        left_mask = xs < center_x - 10
        right_mask = xs > center_x + 10
        if left_mask.any():
            left_pts.append((int(xs[left_mask][-1]), y))
        if right_mask.any():
            right_pts.append((int(xs[right_mask][0]), y))
    return left_pts, right_pts


def _detect_opp_side_fit(pts, min_span):
    if len(pts) < 4:
        return None
    a = np.float32(pts).reshape(-1, 1, 2).copy()
    a[:, 0, 0] = 639.0 - a[:, 0, 0]  # 还原原始相机坐标
    ipm = cv2.perspectiveTransform(a, IPM_MATRIX).reshape(-1, 2)
    ipm = ipm[np.isfinite(ipm).all(axis=1)]
    if len(ipm) < 4:
        return None
    span = np.ptp(ipm[:, 1])
    if span < min_span:
        return None
    work = ipm.copy()
    for _ in range(3):
        k, b = np.polyfit(work[:, 1], work[:, 0], 1)
        resid = np.abs(work[:, 0] - (k * work[:, 1] + b))
        keep = resid <= OPP_FIT_RESID
        if keep.sum() < 4:
            return None
        if keep.sum() == len(work):
            break
        work = work[keep]
    if len(work) < 4 or np.ptp(work[:, 1]) < min_span:
        return None
    k, b = np.polyfit(work[:, 1], work[:, 0], 1)
    return {'k': float(k), 'b': float(b),
            'y_min': float(np.min(work[:, 1])), 'y_max': float(np.max(work[:, 1]))}


def detect_opposite_centerline(mask):
    """提取路口对面出口中线"""
    if IPM_MATRIX is None or IPM_INV_MATRIX is None:
        return None
    mask_center = mask.copy()
    cut_bot = int(mask.shape[0] * OPP_ROI_BOTTOM)
    mask_center[cut_bot:, :] = 0
    cut_top = int(mask.shape[0] * OPP_ROI_TOP)
    mask_center[:cut_top, :] = 0
    h, w = mask_center.shape
    center_x = w // 2

    left_pts, right_pts = _opp_scan_side_points(mask_center, center_x)
    left_fit = _detect_opp_side_fit(left_pts, OPP_MIN_SPAN)
    right_fit = _detect_opp_side_fit(right_pts, OPP_MIN_SPAN)
    if left_fit is None or right_fit is None:
        return None

    overlap_min = max(left_fit['y_min'], right_fit['y_min'], 20.0)
    overlap_max = min(left_fit['y_max'], right_fit['y_max'], 590.0)
    if overlap_max - overlap_min < OPP_MIN_OVERLAP:
        return None
    yc = (overlap_min + overlap_max) / 2.0
    ax = left_fit['k'] * yc + left_fit['b']
    bx = right_fit['k'] * yc + right_fit['b']
    width = abs(bx - ax)
    slope_diff = abs(left_fit['k'] - right_fit['k'])
    if not (OPP_WIDTH_MIN <= width <= OPP_WIDTH_MAX):
        return None
    if slope_diff > OPP_SLOPE_DIFF_MAX:
        return None

    k_c = (left_fit['k'] + right_fit['k']) / 2.0
    b_c = (left_fit['b'] + right_fit['b']) / 2.0
    y_near = overlap_max
    center_near = k_c * y_near + b_c
    heading_deg = math.degrees(math.atan2(-k_c, 1.0))

    best = {
        'score': slope_diff * 10.0 + abs(width - 220.0) / 220.0,
        'k': k_c, 'b': b_c,
        'center_error': float(center_near - IPM_CENTER_X),
        'heading_error_deg': float(heading_deg),
        'width': float(width), 'slope_diff': float(slope_diff),
        'y_min': overlap_min, 'y_max': y_near,
        'left_fit': left_fit, 'right_fit': right_fit,
    }

    y_values = np.linspace(best['y_min'], best['y_max'], 24)
    ipm_center = np.float32([
        [best['k'] * y + best['b'], y] for y in y_values
    ]).reshape(-1, 1, 2)
    raw = cv2.perspectiveTransform(ipm_center, IPM_INV_MATRIX).reshape(-1, 2)
    overlay_points = []
    for x, y in raw:
        pt = (int(round(639.0 - x)), int(round(y)))
        if -80 <= pt[0] < w + 80 and 0 <= pt[1] < h:
            overlay_points.append(pt)
    best['overlay_points'] = overlay_points

    def _side_overlay(fit):
        ys = np.linspace(fit['y_min'], fit['y_max'], 20)
        ipm_side = np.float32([[fit['k'] * y + fit['b'], y] for y in ys]
                              ).reshape(-1, 1, 2)
        raw_side = cv2.perspectiveTransform(ipm_side, IPM_INV_MATRIX).reshape(-1, 2)
        side_pts = []
        for x, y in raw_side:
            pt = (int(round(639.0 - x)), int(round(y)))
            if -80 <= pt[0] < w + 80 and 0 <= pt[1] < h:
                side_pts.append(pt)
        return side_pts

    best['left_fit']['overlay_points'] = _side_overlay(left_fit)
    best['right_fit']['overlay_points'] = _side_overlay(right_fit)
    return best


# ==================== 弧线车道中线检测（移植自 line_following_ss_pure_ipm.py）====================

def get_full_mask(frame, params):
    """在降采样 (PROC_W, PROC_H) 分辨率上做 HSV 二值化（弧线中线检测用）。"""
    small = cv2.resize(frame, (PROC_W, PROC_H), interpolation=cv2.INTER_AREA)
    blur_k = int(params.get('blur_ksize', 4))
    if blur_k >= 3:
        if blur_k % 2 == 0:
            blur_k += 1
        fh = cv2.GaussianBlur(small, (blur_k, blur_k), 0)
    else:
        fh = small
    hsv = cv2.cvtColor(fh, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv,
        np.array([params.get('low_h', 42), params.get('low_s', 5), params.get('low_v', 116)]),
        np.array([params.get('high_h', 179), params.get('high_s', 71), params.get('high_v', 255)]))
    return mask


def warp_to_ipm(mask):
    """将二值图变换到 IPM 画布，输出 (CANVAS_W, CANVAS_H)。"""
    return cv2.warpPerspective(mask, IPM_CANVAS_M, (CANVAS_W, CANVAS_H),
                               flags=cv2.INTER_NEAREST)


def clean_ipm_mask(warped):
    """保留竖直细长结构（车道线），剔除横向噪声/反射条带。"""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(warped, cv2.MORPH_CLOSE, kernel, iterations=2)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, 8)
    out = np.zeros_like(cleaned)
    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]
        if h >= LANE_TUNE['clean_min_h'] and area >= LANE_TUNE['clean_min_area'] and h >= w * LANE_TUNE['clean_min_ratio']:
            out[labels == label] = 255
    return out


def fit_lane_line(points):
    """对一组 IPM 采样点做直线拟合 x = k*y + b，两轮去离群。"""
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < LANE_TUNE['min_center_pts'] or np.ptp(pts[:, 1]) < LANE_TUNE['min_center_span']:
        return None
    for _ in range(2):
        k, b = np.polyfit(pts[:, 1], pts[:, 0], 1)
        resid = np.abs(pts[:, 0] - (k * pts[:, 1] + b))
        keep = resid <= LANE_TUNE['poly_max_resid']
        if keep.sum() < LANE_TUNE['min_center_pts']:
            return None
        if keep.sum() == len(pts):
            break
        pts = pts[keep]
    if len(pts) < LANE_TUNE['min_center_pts'] or np.ptp(pts[:, 1]) < LANE_TUNE['min_center_span']:
        return None
    k, b = np.polyfit(pts[:, 1], pts[:, 0], 1)
    return {'coeffs': (0.0, float(k), float(b)),
            'y_min': float(np.min(pts[:, 1])),
            'y_max': float(np.max(pts[:, 1]))}


def poly_x(coeffs, y):
    a, b, c = coeffs
    return a * y * y + b * y + c


def poly_k(coeffs, y):
    a, b, _ = coeffs
    return 2.0 * a * y + b


def lanes_matched(left_fit, right_fit):
    y_min = max(left_fit['y_min'], right_fit['y_min'])
    y_max = min(left_fit['y_max'], right_fit['y_max'])
    if y_max - y_min < LANE_TUNE['match_min_overlap']:
        return False
    y_mid = (y_min + y_max) / 2.0
    w = poly_x(right_fit['coeffs'], y_mid) - poly_x(left_fit['coeffs'], y_mid)
    if not (LANE_TUNE['lane_width_min'] - 15.0 <= w <= LANE_TUNE['lane_width_max'] + 15.0):
        return False
    kL = poly_k(left_fit['coeffs'], y_mid)
    kR = poly_k(right_fit['coeffs'], y_mid)
    return abs(kL - kR) <= LANE_TUNE['max_slope_diff']


def _find_sides_from_center(means, center):
    left = None
    for x in sorted((m for m in means if m < center - 10), reverse=True):
        left = x
        break
    right = None
    for x in sorted(m for m in means if m > center + 10):
        right = x
        break
    return left, right


def extract_raw_lanes(warped):
    """逐行扫描提取左右车道线采样点并各自直线拟合。返回 (left_fit, right_fit, width_samples)。"""
    h, w = warped.shape
    center_canvas_x = IPM_CENTER_X - CANVAS_X0
    left_pts, right_pts = [], []
    width_samples = []

    for yc in range(h - 1, -1, -4):
        xs = np.where(warped[yc] == 255)[0]
        if len(xs) == 0:
            continue
        diff = np.diff(xs)
        breaks = np.where(diff > 2)[0] + 1
        clusters = np.split(xs, breaks)
        means = [int(np.mean(c)) for c in clusters if 2 <= len(c) <= LANE_TUNE['max_cluster_w']]
        if not means:
            continue
        left_x, right_x = _find_sides_from_center(means, center_canvas_x)
        y_ipm = float(yc + CANVAS_Y0)
        if left_x is None and right_x is None:
            continue
        if left_x is not None and right_x is not None:
            lx = float(left_x + CANVAS_X0)
            rx = float(right_x + CANVAS_X0)
            width = rx - lx
            if LANE_TUNE['lane_width_min'] <= width <= LANE_TUNE['lane_width_max']:
                left_pts.append((lx, y_ipm))
                right_pts.append((rx, y_ipm))
                width_samples.append(width)
                continue
            if abs(left_x - center_canvas_x) <= abs(right_x - center_canvas_x):
                left_pts.append((lx, y_ipm))
            else:
                right_pts.append((rx, y_ipm))
        elif left_x is not None:
            left_pts.append((float(left_x + CANVAS_X0), y_ipm))
        elif right_x is not None:
            right_pts.append((float(right_x + CANVAS_X0), y_ipm))

    left_fit = fit_lane_line(left_pts)
    right_fit = fit_lane_line(right_pts)
    return left_fit, right_fit, width_samples


def _side_of(ls, near_x):
    if ls.track_left is not None and ls.track_right is not None:
        dL = abs(near_x - ls.track_left['near_x'])
        dR = abs(near_x - ls.track_right['near_x'])
        return 'L' if dL <= dR else 'R'
    if ls.track_left is not None:
        if abs(near_x - ls.track_left['near_x']) <= LANE_TUNE['lane_half_width']:
            return 'L'
        return 'L' if near_x < IPM_CENTER_X else 'R'
    if ls.track_right is not None:
        if abs(near_x - ls.track_right['near_x']) <= LANE_TUNE['lane_half_width']:
            return 'R'
        return 'L' if near_x < IPM_CENTER_X else 'R'
    return 'L' if near_x < IPM_CENTER_X else 'R'


def _update_tracker(ls, left_fit, right_fit, half_width):
    def upd(old, fit):
        if fit is None:
            if old is not None:
                old['miss'] += 1
            return old
        coeffs = ema_poly(old['coeffs'] if old else None, fit['coeffs'])
        return {'coeffs': coeffs,
                'near_x': poly_x(coeffs, min(fit['y_max'], Y_NEAR)),
                'miss': 0}

    ls.track_left = upd(ls.track_left, left_fit)
    ls.track_right = upd(ls.track_right, right_fit)
    if ls.track_left is not None and ls.track_left['miss'] > LANE_TUNE['track_stale_frames']:
        ls.track_left = None
    if ls.track_right is not None and ls.track_right['miss'] > LANE_TUNE['track_stale_frames']:
        ls.track_right = None
    if half_width is not None:
        ls.track_half_width = half_width
    ls.track_valid = (ls.track_left is not None or ls.track_right is not None)


def resolve_lane(ls, left_fit, right_fit, width_samples):
    """结合历史身份确定左右线、配对或单线，输出中线误差，并更新轨迹。"""
    cand = []
    for pos_side, fit in (('L', left_fit), ('R', right_fit)):
        if fit is not None:
            y_ref = min(fit['y_max'], Y_NEAR)
            cand.append({'pos_side': pos_side, 'fit': fit,
                         'near_x': poly_x(fit['coeffs'], y_ref)})

    lane_width = float(np.mean(width_samples)) if width_samples else None
    half_width = ls.track_half_width if ls.track_valid else LANE_TUNE['lane_half_width']
    pair_matched = False
    left_out = right_out = None
    coeffs = None
    kanbujian = False
    y_min = y_max = 0.0

    if len(cand) >= 2:
        a, b = cand[0], cand[1]
        if ls.track_left is not None and ls.track_right is not None:
            d_aL = abs(a['near_x'] - ls.track_left['near_x'])
            d_aR = abs(a['near_x'] - ls.track_right['near_x'])
            d_bL = abs(b['near_x'] - ls.track_left['near_x'])
            d_bR = abs(b['near_x'] - ls.track_right['near_x'])
            if d_aL + d_bR <= d_aR + d_bL:
                left_c, right_c = a, b
            else:
                left_c, right_c = b, a
        else:
            if a['pos_side'] != b['pos_side']:
                left_c, right_c = (a, b) if a['pos_side'] == 'L' else (b, a)
            else:
                left_c, right_c = (a, b) if a['near_x'] < b['near_x'] else (b, a)

        y_ref = min(left_c['fit']['y_max'], right_c['fit']['y_max'], Y_NEAR)
        lx = poly_x(left_c['fit']['coeffs'], y_ref)
        rx = poly_x(right_c['fit']['coeffs'], y_ref)
        width = rx - lx
        width_ok = LANE_TUNE['lane_width_min'] <= width <= LANE_TUNE['lane_width_max']
        hist_ok = (not ls.track_valid or
                   abs(width - 2.0 * ls.track_half_width) <= LANE_TUNE['max_width_dev'])
        if width_ok and hist_ok and lanes_matched(left_c['fit'], right_c['fit']):
            pair_matched = True
            left_out, right_out = left_c['fit'], right_c['fit']
            lc, rc = left_c['fit']['coeffs'], right_c['fit']['coeffs']
            coeffs = tuple((lc[i] + rc[i]) / 2.0 for i in range(3))
            y_min = max(left_c['fit']['y_min'], right_c['fit']['y_min'])
            y_max = min(left_c['fit']['y_max'], right_c['fit']['y_max'])
            half_width = width / 2.0

    if not pair_matched:
        single = None
        if len(cand) == 1:
            single = cand[0]
        elif len(cand) >= 2:
            if ls.track_valid and (ls.track_left is not None or ls.track_right is not None):
                def hist_d(c):
                    ds = []
                    if ls.track_left is not None:
                        ds.append(abs(c['near_x'] - ls.track_left['near_x']))
                    if ls.track_right is not None:
                        ds.append(abs(c['near_x'] - ls.track_right['near_x']))
                    return min(ds) if ds else 1e9
                single = min(cand, key=hist_d)
            else:
                single = min(cand, key=lambda c: abs(c['near_x'] - IPM_CENTER_X))
        if single is not None:
            cls = _side_of(ls, single['near_x'])
            a, b, c = single['fit']['coeffs']
            if cls == 'L':
                coeffs = (a, b, c + half_width)
                left_out, right_out = single['fit'], None
            else:
                coeffs = (a, b, c - half_width)
                left_out, right_out = None, single['fit']
            kanbujian = True
            y_min, y_max = single['fit']['y_min'], single['fit']['y_max']

    if coeffs is None or y_max <= y_min:
        _update_tracker(ls, None, None, None)
        return None

    center_y_range = (y_min, y_max)
    center_error_px = float(poly_x(coeffs, Y_NEAR) - IPM_CENTER_X)
    y_head = min(Y_NEAR - LANE_TUNE['lookahead_px'], y_max)
    kt = poly_k(coeffs, y_head)
    heading_deg = float(math.degrees(math.atan2(-kt, 1.0)))

    center_points = []
    for yv in np.linspace(y_min, y_max, 24):
        xc = poly_x(coeffs, yv)
        if 0 <= yv - CANVAS_Y0 < CANVAS_H:
            center_points.append((int(round(xc - CANVAS_X0)), int(round(yv - CANVAS_Y0))))

    _update_tracker(ls, left_out, right_out, half_width if pair_matched else None)

    return {
        'center_error_px': center_error_px,
        'heading_error_deg': heading_deg,
        'lane_width_px': lane_width,
        'kanbujian': kanbujian,
        'pair_matched': pair_matched,
        'center_points': center_points,
        'coeffs': coeffs,
        'center_y_range': center_y_range,
        'left_fit': left_out,
        'right_fit': right_out,
        'k_tangent': kt,
    }


def ema_poly(prev, new, alpha=ALPHA_POLY):
    if prev is None:
        return new
    return tuple(alpha * n + (1.0 - alpha) * p for p, n in zip(prev, new))


def extract_horizontal_bands(bin_img, kernel_w=31):
    if bin_img is None or bin_img.size == 0:
        return bin_img
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 1))
    return cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel)


def remove_horizontal_bands_ipm(warped, kernel_w=41):
    """在 IPM 画布上剔除横向长条（停止线/终点线），须在 clean_ipm_mask 之前。"""
    if warped is None or warped.size == 0:
        return warped
    horiz = extract_horizontal_bands(warped, kernel_w)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(horiz, 8)
    band = np.zeros_like(warped)
    for label in range(1, num_labels):
        x, y, bw, bh, area = stats[label]
        if bw >= LANE_TUNE['lane_width_min'] and bh <= bw * 0.4:
            band[labels == label] = 255
    clean = warped.copy()
    clean[band == 255] = 0
    return clean


def project_ipm_to_image(coeffs, y_range=None):
    """将 IPM 中线投影回原始相机坐标（640x480，未镜像）。"""
    if IPM_INV_MATRIX is None:
        return []
    a, b, c = coeffs
    if y_range is not None:
        y0, y1 = y_range
    else:
        y0, y1 = Y_FAR, Y_NEAR
    pts = []
    for yv in np.linspace(y0, y1, 24):
        ipm_x = a * yv * yv + b * yv + c
        src = cv2.perspectiveTransform(
            np.float32([[[ipm_x, yv]]]), IPM_INV_MATRIX).reshape(-1, 2)
        if len(src) > 0:
            x, y = src[0]
            if 0 <= y <= CAM_H:
                pts.append((int(round(x)), int(round(y))))
    return pts


def apply_poly_filter(ls, result):
    """对中线多项式做时间滤波，防止跳变。"""
    mode = 'S' if result['kanbujian'] else 'P'
    if ls.poly_filt is None or ls.poly_filt_mode != mode:
        ls.poly_filt = tuple(result['coeffs'])
        ls.poly_filt_mode = mode
        ls.poly_filt_y = tuple(result['center_y_range'])
    else:
        ls.poly_filt = ema_poly(ls.poly_filt, result['coeffs'])
        y0 = ALPHA_POLY * result['center_y_range'][0] + (1 - ALPHA_POLY) * ls.poly_filt_y[0]
        y1 = ALPHA_POLY * result['center_y_range'][1] + (1 - ALPHA_POLY) * ls.poly_filt_y[1]
        ls.poly_filt_y = (y0, y1)

    coeffs = ls.poly_filt
    y0, y1 = ls.poly_filt_y
    if y1 <= y0:
        return None
    result = dict(result)
    result['coeffs'] = coeffs
    result['center_y_range'] = ls.poly_filt_y
    result['center_error_px'] = float(poly_x(coeffs, Y_NEAR) - IPM_CENTER_X)
    y_head = min(Y_NEAR - LANE_TUNE['lookahead_px'], y1)
    kt = poly_k(coeffs, y_head)
    result['heading_error_deg'] = float(math.degrees(math.atan2(-kt, 1.0)))
    cpts = []
    for yv in np.linspace(y0, y1, 24):
        xc = poly_x(coeffs, yv)
        if 0 <= yv - CANVAS_Y0 < CANVAS_H:
            cpts.append((int(round(xc - CANVAS_X0)), int(round(yv - CANVAS_Y0))))
    result['center_points'] = cpts
    return result


def detect_arc_lane(frame, params):
    """SEARCH_ARC 阶段用的弧线车道中线检测（对应 line_following_ss_pure_ipm.image_cb 核心）。

    frame 为已水平翻转的显示画面；检测前翻回原始相机画面，使误差符号约定与
    ss_pure_ipm 一致（中心/朝向为正时中线在物理右侧）。
    """
    if IPM_CANVAS_M is None or IPM_INV_MATRIX is None:
        return None
    raw = cv2.flip(frame, 1)
    full_mask = get_full_mask(raw, params)
    mask_roi = full_mask.copy()
    y_cut = int(PROC_H * (1.0 - LANE_TUNE['roi_bottom_ratio']))
    mask_roi[:y_cut, :] = 0
    warped = warp_to_ipm(mask_roi)
    warped = remove_horizontal_bands_ipm(warped)
    cleaned = clean_ipm_mask(warped)
    left_fit, right_fit, width_samples = extract_raw_lanes(cleaned)
    result = resolve_lane(lane_state, left_fit, right_fit, width_samples)
    if result is not None:
        result = apply_poly_filter(lane_state, result)
        result['center_error_px'] = -result['center_error_px']
        result['heading_error_deg'] = -result['heading_error_deg']
    return result


def publish_stop():
    state.control_source = 'STOPPED'
    state.command_linear_x = 0.0
    state.command_linear_y = 0.0
    state.command_angular_z = 0.0
    if cmd_pub is not None:
        cmd_pub.publish(Twist())


def image_cb(msg):
    try:
        frame = bridge.imgmsg_to_cv2(msg, 'passthrough')
        if msg.encoding.lower() == 'rgb8':
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        elif msg.encoding.lower() != 'bgr8':
            frame = bridge.imgmsg_to_cv2(msg, 'bgr8')

        frame = cv2.flip(frame, 1)

        with state.lock:
            hsv_p = dict(state.hsv_params)

        mask, roi = make_mask(frame, hsv_p)
        opp_guide = detect_opposite_centerline(mask)
        lane_result = None
        if state.mode == 'SEARCH_ARC':
            lane_result = detect_arc_lane(frame, hsv_p)

        overlay = frame.copy()
        cv2.rectangle(overlay, (roi[0], roi[1]), (roi[2] - 1, roi[3] - 1), (255, 120, 0), 2)
        if opp_guide is not None:
            pts = opp_guide['overlay_points']
            for index in range(0, len(pts) - 1, 2):
                cv2.line(overlay, pts[index], pts[index + 1], (0, 255, 0), 5)
            for side in ('left_fit', 'right_fit'):
                lpts = opp_guide[side].get('overlay_points')
                if lpts and len(lpts) > 1:
                    cv2.polylines(overlay,
                                  [np.array(lpts, np.int32).reshape(-1, 1, 2)],
                                  False, (0, 0, 255) if side == 'left_fit' else (255, 0, 255), 4)
            cv2.putText(overlay, 'OPP GUIDE c=%.1f h=%.1f w=%.0f' % (
                opp_guide['center_error'], opp_guide['heading_error_deg'],
                opp_guide['width']), (175, 105),
                cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 2)

        if lane_result is not None:
            lpts = project_ipm_to_image(lane_result['coeffs'], lane_result['center_y_range'])
            disp = [((CAM_W - 1 - int(round(x))), int(round(y))) for x, y in lpts
                    if 0 <= int(round(y)) < CAM_H]
            if len(disp) > 1:
                cv2.polylines(overlay, [np.array(disp, np.int32).reshape(-1, 1, 2)],
                              False, (0, 255, 255), 4)
            cv2.putText(overlay, 'ARC LANE c=%.1f h=%.1f %s' % (
                lane_result['center_error_px'], lane_result['heading_error_deg'],
                'SINGLE' if lane_result['kanbujian'] else 'PAIR'),
                (175, 130), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255), 2)

        with state.lock:
            state.last_image_time = time.time()
            state.opp_guide_valid = opp_guide is not None
            state.opp_center_error = opp_guide['center_error'] if opp_guide else None
            state.opp_heading_error_deg = opp_guide['heading_error_deg'] if opp_guide else None
            if opp_guide is not None:
                state.opp_last_center_error = opp_guide['center_error']

            state.lane_valid = lane_result is not None
            state.lane_center_error = lane_result['center_error_px'] if lane_result else None
            state.lane_heading_error = lane_result['heading_error_deg'] if lane_result else None

            state.vis_overlay = overlay
            state.vis_mask = mask
            status = state.status()

        cv2.putText(overlay, 'STATE: %s  DIR: %s' % (status['mode'], status['direction']), (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 255, 255), 2)

        if mask_pub is not None:
            mask_pub.publish(safe_cv2_to_imgmsg(mask, 'mono8'))
        if overlay_pub is not None:
            overlay_pub.publish(safe_cv2_to_imgmsg(overlay, 'bgr8'))
    except Exception as exc:
        import traceback
        rospy.logwarn_throttle(2, f'image processing error: {exc}\n{traceback.format_exc()}')


def odom_cb(msg):
    q = msg.pose.pose.orientation
    yaw = euler_from_quaternion((q.x, q.y, q.z, q.w))[2]
    x = msg.pose.pose.position.x
    y = msg.pose.pose.position.y

    with state.lock:
        state.odom = (x, y, yaw)

        if state.phase_start_pose:
            p_dx = x - state.phase_start_pose[0]
            p_dy = y - state.phase_start_pose[1]
            state.phase_distance = math.hypot(p_dx, p_dy)

        if state.mode == 'ADVANCE' and state.advance_start_pose:
            dx = x - state.advance_start_pose[0]
            dy = y - state.advance_start_pose[1]
            state.advance_dist = math.hypot(dx, dy)

        elif state.mode == 'ROTATE' and state.rotate_start_yaw is not None:
            yaw_diff = normalize_angle(yaw - state.rotate_start_yaw)
            state.turn_deg = math.degrees(abs(yaw_diff))


def control_timer(_event):
    with state.lock:
        now = time.time()
        mode = state.mode

        if mode not in ('F_ALIGN', 'F_YAW', 'ADVANCE', 'ROTATE', 'SEARCH_ARC'):
            return

        if state.odom is None or now - state.last_image_time > 0.6:
            state.mode = 'FAULT'
            state.message = '里程计不可用或相机超时，已紧急停车'
            publish_stop()
            return

        cmd = Twist()

        if mode == 'F_ALIGN':
            # 平移对准路口对面中线：linear.y 平移 + 角速度耦合校正，车不前进
            if state.phase_distance >= F_ALIGN_MAX_DISTANCE or now - state.state_started > F_ALIGN_TIMEOUT:
                state.advance_start_pose = state.odom
                state.advance_dist = 0.0
                state.turn_deg = 0.0
                state.mode = 'ADVANCE'
                state.state_started = now
                state.message = '平移对准超时/超距，直接切换到直行阶段'
                publish_stop()
                return

            if not state.opp_guide_valid or state.opp_center_error is None:
                state.opp_guide_good_frames = 0
                if state.opp_last_center_error is not None:
                    # 短暂掉帧：沿用上次有效误差继续纠偏
                    center_error = state.opp_last_center_error
                    cmd.linear.x = 0.0
                    cmd.linear.y = F_ALIGN_KP_Y * center_error
                    cmd.angular.z = max(-F_ALIGN_WZ_CLAMP,
                                        min(F_ALIGN_WZ_CLAMP, F_ALIGN_KP_Z * center_error))
                    state.control_source = 'F_ALIGN_HOLD_LAST'
                else:
                    publish_stop()
                    state.control_source = 'F_ALIGN_WAIT'
                return

            center_error = state.opp_center_error
            heading_error = state.opp_heading_error_deg if state.opp_heading_error_deg is not None else 0.0

            if abs(center_error) <= F_ALIGN_CENTER_TOL_PX and abs(heading_error) <= F_ALIGN_HEADING_TOL_DEG:
                state.opp_guide_good_frames += 1
                if state.opp_guide_good_frames >= F_ALIGN_CONFIRM_FRAMES:
                    state.f_align_start_yaw = state.odom[2]
                    state.f_yaw_started = now
                    state.mode = 'F_YAW'
                    state.message = '阶段一：对准对面路口中线完成，轻微调整车头'
                    publish_stop()
                    return
            else:
                state.opp_guide_good_frames = 0

            cmd.linear.x = 0.0                 # 平移对准禁止前进
            cmd.linear.y = F_ALIGN_KP_Y * center_error
            cmd.angular.z = max(-F_ALIGN_WZ_CLAMP,
                                min(F_ALIGN_WZ_CLAMP, F_ALIGN_KP_Z * center_error))
            state.control_source = 'F_ALIGN_TRANSLATE'

        elif mode == 'F_YAW':
            # 轻微调整：原地转回对准起始 yaw (消除平移耦合带偏的车头角度)
            elapsed = now - (state.f_yaw_started or now)
            yaw_error = normalize_angle(state.f_align_start_yaw - state.odom[2])
            if abs(math.degrees(yaw_error)) <= F_YAW_TOL_DEG or elapsed > F_YAW_MAX_TIME:
                state.advance_start_pose = state.odom
                state.advance_dist = 0.0
                state.turn_deg = 0.0
                state.mode = 'ADVANCE'
                state.state_started = now
                state.message = '阶段二：车头微调完成，低速直行 30 厘米'
                publish_stop()
                return
            cmd.linear.x = 0.0
            cmd.linear.y = 0.0
            cmd.angular.z = max(-F_YAW_WZ, min(F_YAW_WZ, F_YAW_KP * yaw_error))
            state.control_source = 'F_YAW'

        elif mode == 'ADVANCE':
            # 超时保护
            if now - state.state_started > ADVANCE_TIMEOUT:
                state.mode = 'FAULT'
                state.message = '直行前进超时，已停车'
                publish_stop()
                return

            # 到达 30 cm 前进距离，切换到旋转阶段
            if state.advance_dist >= ADVANCE_DISTANCE:
                state.mode = 'ROTATE'
                state.state_started = now
                state.rotate_start_yaw = state.odom[2]
                state.turn_deg = 0.0
                dir_str = '左转' if state.direction == 'LEFT' else '右转'
                state.message = f'直行完成 {int(ADVANCE_DISTANCE*100)}cm，开始{dir_str} {ROTATE_TARGET_DEG} 度'
                publish_stop()
                return

            # 直行控制：直行速度 + 锁定航向角保持直线
            cmd.linear.x = ADVANCE_SPEED
            yaw_error = normalize_angle(state.advance_start_pose[2] - state.odom[2])
            cmd.angular.z = max(-0.08, min(0.08, 0.8 * yaw_error))
            state.control_source = 'ADVANCE_ODOM'

        elif mode == 'ROTATE':
            # 超时保护
            if now - state.state_started > ROTATE_TIMEOUT:
                state.mode = 'FAULT'
                state.message = '旋转超时，已停车'
                publish_stop()
                return

            # 到达 65 度旋转角，进入弧线中线搜索阶段
            if state.turn_deg >= ROTATE_TARGET_DEG:
                lane_state.reset()
                state.search_start_yaw = state.odom[2]
                state.search_start_time = now
                state.search_good_frames = 0
                state.mode = 'SEARCH_ARC'
                dir_str = '左转' if state.direction == 'LEFT' else '右转'
                state.message = f'{dir_str} {ROTATE_TARGET_DEG} 度完成，开始同向慢速搜索弧线车道中线'
                publish_stop()
                return

            # 旋转控制：左转 angular.z > 0，右转 angular.z < 0
            cmd.linear.x = 0.0
            sign = 1.0 if state.direction == 'LEFT' else -1.0
            cmd.angular.z = sign * ROTATE_SPEED
            state.control_source = 'ROTATE_ODOM'

        elif mode == 'SEARCH_ARC':
            # 超时 / 超限保护
            if now - state.search_start_time > SEARCH_TIMEOUT:
                state.mode = 'SEARCH_FAIL'
                state.message = '弧线中线搜索超时，未找到车道线，已停车'
                publish_stop()
                return
            accum_deg = math.degrees(abs(normalize_angle(state.odom[2] - state.search_start_yaw)))
            if accum_deg > SEARCH_ACCUM_LIMIT_DEG:
                state.mode = 'SEARCH_FAIL'
                state.message = f'弧线中线搜索超限 (> {SEARCH_ACCUM_LIMIT_DEG:.0f}°)，未找到车道线，已停车'
                publish_stop()
                return

            if state.lane_valid:
                state.search_good_frames += 1
                if state.search_good_frames >= SEARCH_CONFIRM_FRAMES:
                    state.mode = 'LANE_FOUND'
                    state.message = f'已找到弧线车道中线并连续确认 {SEARCH_CONFIRM_FRAMES} 帧，已停车并发出 lane_found 信号'
                    if lane_found_pub is not None:
                        lane_found_pub.publish(Bool(True))
                    publish_stop()
                    return
            else:
                state.search_good_frames = 0

            # 同向慢速续转：左转继续左 (z>0)，右转继续右 (z<0)
            sign = 1.0 if state.direction == 'LEFT' else -1.0
            cmd.linear.x = 0.0
            cmd.linear.y = 0.0
            cmd.angular.z = sign * SEARCH_ROTATE_WZ
            state.control_source = 'SEARCH_ARC'

        state.command_linear_x = cmd.linear.x
        state.command_linear_y = cmd.linear.y
        state.command_angular_z = cmd.angular.z
        if cmd_pub is not None:
            cmd_pub.publish(cmd)


PAGE = '''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>路口转向调试节点 (对面中线回正 + 写死转弯)</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 20px; }
  main { max-width: 1000px; margin: auto; }
  h2 { color: #38bdf8; margin-top: 0; }
  section { background: #1e293b; padding: 20px; margin-bottom: 20px; border-radius: 12px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.3); }
  .card-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-top: 15px; }
  .card { background: #0f172a; padding: 15px; border-radius: 8px; border: 1px solid #334155; }
  .card-title { font-size: 13px; color: #94a3b8; text-transform: uppercase; margin-bottom: 5px; }
  .card-value { font-size: 22px; font-weight: bold; color: #38bdf8; }
  .dir-selector { margin: 15px 0; display: flex; gap: 20px; align-items: center; background: #0f172a; padding: 12px; border-radius: 8px; border: 1px solid #334155; }
  .dir-selector label { font-size: 16px; font-weight: 500; cursor: pointer; display: flex; align-items: center; gap: 8px; }
  .dir-selector input[type="radio"] { width: 18px; height: 18px; accent-color: #38bdf8; }
  .btn-group { display: flex; gap: 12px; margin-top: 15px; flex-wrap: wrap; }
  button { padding: 12px 24px; border: 0; border-radius: 8px; font-size: 16px; font-weight: 600; cursor: pointer; transition: all 0.2s; }
  .btn-start { background: #0284c7; color: white; }
  .btn-start:hover { background: #0369a1; }
  .btn-stop { background: #dc2626; color: white; }
  .btn-stop:hover { background: #b91c1c; }
  .btn-reset { background: #475569; color: white; }
  .btn-reset:hover { background: #334155; }
  pre { font-family: monospace; background: #0f172a; padding: 12px; border-radius: 8px; border: 1px solid #334155; color: #7dd3fc; font-size: 14px; overflow-x: auto; }
  .status-badge { display: inline-block; padding: 4px 12px; border-radius: 20px; font-size: 14px; font-weight: 600; background: #334155; color: #f8fafc; }
  .status-DISARMED { background: #475569; }
  .status-F_ALIGN { background: #8b5cf6; }
  .status-F_YAW { background: #a855f7; }
  .status-ADVANCE { background: #0284c7; }
  .status-ROTATE { background: #d97706; }
  .status-SEARCH_ARC { background: #f59e0b; }
  .status-LANE_FOUND { background: #10b981; }
  .status-SEARCH_FAIL { background: #dc2626; }
  .status-COMPLETED { background: #16a34a; }
  .status-FAULT, .status-ESTOP { background: #dc2626; }
  .img-container { display: flex; gap: 15px; flex-wrap: wrap; margin-top: 15px; }
  img { width: 48%; min-width: 300px; background: #0f172a; border-radius: 8px; border: 1px solid #334155; }
</style>
</head>
<body>
<main>
  <h2>🚦 路口转向调试节点 (对面中线回正 + 写死转弯)</h2>

  <section>
    <div style="display:flex; justify-content:space-between; align-items:center;">
      <span><strong>当前状态：</strong> <span id="mode_badge" class="status-badge status-DISARMED">DISARMED</span></span>
      <span style="color:#94a3b8; font-size:14px;">流程：阶段1 (对面中线回正) ➔ 阶段2 (直行30cm + 旋转65°) ➔ 阶段3 (同向搜索弧线中线)</span>
    </div>

    <div class="dir-selector">
      <span style="font-weight:600; color:#e2e8f0;">转向方向选择：</span>
      <label><input type="radio" name="direction" value="LEFT" checked onchange="setDirection('LEFT')"> ⬅️ 左转 (LEFT)</label>
      <label><input type="radio" name="direction" value="RIGHT" onchange="setDirection('RIGHT')"> ➡️ 右转 (RIGHT)</label>
    </div>

    <div class="btn-group">
      <button class="btn-start" onclick="postAction('/api/start')">🚀 解锁并启动</button>
      <button class="btn-stop" onclick="postAction('/api/stop')">🛑 紧急停车</button>
      <button class="btn-reset" onclick="postAction('/api/reset')">🔄 复位模式</button>
    </div>
  </section>

  <section>
    <div class="card-grid">
      <div class="card">
        <div class="card-title">对面中线偏差 (px)</div>
        <div class="card-value" id="val_opp_err">N/A</div>
      </div>
      <div class="card">
        <div class="card-title">对准确认帧数</div>
        <div class="card-value" id="val_opp_confirm">0 / 10</div>
      </div>
      <div class="card">
        <div class="card-title">直行距离 (m)</div>
        <div class="card-value" id="val_dist">0.000 / 0.300</div>
      </div>
      <div class="card">
        <div class="card-title">旋转角度 (°)</div>
        <div class="card-value" id="val_turn">0.0 / 65.0</div>
      </div>
      <div class="card">
        <div class="card-title">指令速度 (x, y, z)</div>
        <div class="card-value" id="val_cmd" style="font-size:16px;">0.00, 0.00, 0.00</div>
      </div>
      <div class="card">
        <div class="card-title">弧线中线检测</div>
        <div class="card-value" id="val_lane" style="font-size:16px;">N/A</div>
      </div>
      <div class="card">
        <div class="card-title">搜索续转角 / 确认帧数</div>
        <div class="card-value" id="val_search" style="font-size:16px;">0.0° / 0</div>
      </div>
    </div>
  </section>

  <section>
    <h3>视觉识别与调试流 (/stream/overlay & /stream/mask)</h3>
    <div class="img-container">
      <img id="img_overlay" src="/stream/overlay" alt="Overlay Stream">
      <img id="img_mask" src="/stream/mask" alt="Mask Stream">
    </div>
  </section>

  <section>
    <h3>系统运行日志 / 状态响应</h3>
    <pre id="status_json">加载中...</pre>
  </section>
</main>

<script>
async function postAction(url, bodyData=null) {
  try {
    const opts = { method: 'POST' };
    if (bodyData) {
      opts.headers = { 'Content-Type': 'application/json' };
      opts.body = JSON.stringify(bodyData);
    }
    const res = await (await fetch(url, opts)).json();
    if (!res.ok) alert("⚠️ 操作失败: " + res.error);
  } catch(e) {
    alert("⚠️ 网络请求异常: " + e);
  }
}

function setDirection(dir) {
  postAction('/api/set_direction', { direction: dir });
}

setInterval(async () => {
  try {
    const data = await (await fetch('/api/status')).json();
    document.getElementById('status_json').textContent = JSON.stringify(data, null, 2);

    // Update badge
    const badge = document.getElementById('mode_badge');
    badge.textContent = data.mode;
    badge.className = 'status-badge status-' + data.mode;

    // Update radio buttons if not interacting
    const radios = document.getElementsByName('direction');
    for (let r of radios) {
      if (r.value === data.direction) r.checked = true;
    }

    // Update metric values
    document.getElementById('val_opp_err').textContent = data.opp_center_error !== null ? `${data.opp_center_error} px` : '未检测到';
    document.getElementById('val_opp_confirm').textContent = `${data.opp_guide_good_frames} / ${10}`;
    document.getElementById('val_dist').textContent = `${data.advance_dist_m.toFixed(3)} / ${data.target_advance_m.toFixed(3)}`;
    document.getElementById('val_turn').textContent = `${data.turn_deg.toFixed(1)}° / ${data.target_turn_deg.toFixed(1)}°`;
    document.getElementById('val_cmd').textContent = `${data.command_linear_x.toFixed(2)}, ${data.command_linear_y.toFixed(2)}, ${data.command_angular_z.toFixed(2)}`;
    document.getElementById('val_lane').textContent = data.lane_valid ? `已检测到 ${data.lane_center_error} px` : '未检测到';
    document.getElementById('val_search').textContent = `${data.search_accum_deg.toFixed(1)}° / ${data.search_good_frames}`;
  } catch(e) {}
}, 300);
</script>
</body>
</html>
'''


class Handler(BaseHTTPRequestHandler):
    def reply(self, obj):
        data = json.dumps(obj, ensure_ascii=False).encode('utf-8')
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
        path = urlparse(self.path).path
        content_len = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_len) if content_len > 0 else b'{}'
        try:
            req_data = json.loads(body.decode('utf-8'))
        except Exception:
            req_data = {}

        with state.lock:
            if path == '/api/set_direction':
                dir_val = req_data.get('direction', '').upper()
                if dir_val in ('LEFT', 'RIGHT'):
                    state.direction = dir_val
                    self.reply({'ok': True, 'direction': state.direction})
                else:
                    self.reply({'ok': False, 'error': '无效方向参数'})

            elif path == '/api/start':
                if state.mode in ('F_ALIGN', 'F_YAW', 'ADVANCE', 'ROTATE', 'SEARCH_ARC'):
                    self.reply({'ok': False, 'error': '任务已在运行中'})
                    return
                if state.odom is None:
                    self.reply({'ok': False, 'error': '里程计 (/odom) 未就绪，请检查 base_driver.launch'})
                    return
                if time.time() - state.last_image_time > 0.6:
                    self.reply({'ok': False, 'error': '摄像头画面超时'})
                    return
                if not state.opp_guide_valid or state.opp_center_error is None:
                    self.reply({'ok': False, 'error': '未检测到路口对面中线，请确保车头朝向路口'})
                    return

                state.advance_start_pose = state.odom
                state.phase_start_pose = state.odom
                state.phase_distance = 0.0
                state.rotate_start_yaw = None
                state.advance_dist = 0.0
                state.turn_deg = 0.0
                state.opp_guide_good_frames = 0
                state.f_align_start_yaw = None
                state.f_yaw_started = None

                state.mode = 'F_ALIGN'
                state.state_started = time.time()
                state.message = '阶段一：平移对准路口对面中线'
                self.reply({'ok': True})

            elif path == '/api/stop':
                state.mode = 'ESTOP'
                state.message = '网页紧急停车'
                publish_stop()
                self.reply({'ok': True})

            elif path == '/api/reset':
                state.mode = 'DISARMED'
                state.message = '已复位为仅感知模式'
                state.advance_start_pose = None
                state.phase_start_pose = None
                state.rotate_start_yaw = None
                state.advance_dist = 0.0
                state.turn_deg = 0.0
                state.opp_guide_good_frames = 0
                state.search_start_yaw = None
                state.search_good_frames = 0
                state.lane_valid = False
                lane_state.reset()
                publish_stop()
                self.reply({'ok': True})

            else:
                self.send_error(404)

    def log_message(self, *_args):
        pass


class ReusableHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True

    def server_bind(self):
        try:
            super().server_bind()
        except OSError as e:
            if getattr(e, 'errno', None) == 98:
                rospy.logwarn(f"端口 {PORT} 被占用，强制清理旧进程...")
                os.system(f"fuser -k {PORT}/tcp 2>/dev/null || pkill -9 -f simple_turn_trial.py 2>/dev/null")
                time.sleep(1)
                super().server_bind()
            else:
                raise


def shutdown():
    for _ in range(5):
        publish_stop()
        time.sleep(0.03)


if __name__ == '__main__':
    rospy.init_node('simple_turn_trial', anonymous=False)
    init_ipm()
    load_hsv()
    cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
    mask_pub = rospy.Publisher('/simple_turn/debug/mask', Image, queue_size=1)
    overlay_pub = rospy.Publisher('/simple_turn/debug/overlay', Image, queue_size=1)
    lane_found_pub = rospy.Publisher(LANE_FOUND_TOPIC, Bool, queue_size=1, latch=True)
    rospy.Subscriber(IMAGE_TOPIC, Image, image_cb, queue_size=1, buff_size=2 ** 24)
    rospy.Subscriber('/odom', Odometry, odom_cb, queue_size=1)
    rospy.Timer(rospy.Duration(0.05), control_timer)
    rospy.on_shutdown(shutdown)

    server = ReusableHTTPServer(('0.0.0.0', PORT), Handler)
    rospy.loginfo('极简转向调试节点已启动 (网页控制端口 http://0.0.0.0:%d)', PORT)
    server.serve_forever()
