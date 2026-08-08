#!/usr/bin/env python3
"""
polyline_following_ipm.py

第一阶段：穿过第一个十字路口并驶入对面直道巡线。

流程：
  DISARMED -> F_ALIGN(对面中线平移对准) -> ADVANCE(前进30cm 尝试巡线稳定) -> LINE_FOLLOW(巡线PID)

复用：
  - simple_turn_trial.py 的对面中线提取/平移对准（opp_guide，flip 显示坐标系 + 标准 IPM）
  - line_following_ss_pure_ipm.py 的巡线检测/PID（原始相机坐标系 + 降采样 IPM 画布）

坐标系约定（重要）：
  - opp_guide 在 cv2.flip 后的显示视角计算 center_error（相对 IPM_CENTER_X=300）
  - 巡线在原始相机坐标计算，最后统一取反（镜像），输出物理符号的 center_error_px/heading
  - 两者 dst 量纲实测一致（同一 perspective_params 标定），可直接各自独立工作

速度：最高 0.2 m/s（第一阶段调参用）。
"""
import collections
import json
import math
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["OPENCV_UI_BACKEND"] = "HEADLESS"
if "DISPLAY" in os.environ:
    del os.environ["DISPLAY"]

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from tf.transformations import euler_from_quaternion

BASE_DIR = '/home/ucar/Xunxian_standalone'
CONFIG_DIR = os.path.join(BASE_DIR, 'config')
HSV_PATH = os.path.join(CONFIG_DIR, 'white_lane.json')
FALLBACK_HSV_PATH = os.path.join(CONFIG_DIR, 'white_lane.json')
PERSP_PATH = os.path.join(CONFIG_DIR, 'perspective_params.json')
IMAGE_TOPIC = '/usb_cam/image_raw'
PORT = 5010

# ==================== 对面中线提取参数（simple_turn_trial） ====================
OPP_ROI_BOTTOM = 0.85
OPP_ROI_TOP = 0.55
OPP_WIDTH_MIN = 120.0
OPP_WIDTH_MAX = 450.0
OPP_SLOPE_DIFF_MAX = 0.20
OPP_MIN_OVERLAP = 30.0
OPP_MIN_SPAN = 60.0
OPP_FIT_RESID = 30.0
IPM_CENTER_X = 300.0

# ==================== 阶段一：对面中线平移对准 ====================
F_ALIGN_CENTER_TOL_PX = 18.0
F_ALIGN_HEADING_TOL_DEG = 12.0
F_ALIGN_CONFIRM_FRAMES = 10
F_ALIGN_KP_Y = 0.0022
F_ALIGN_KP_Z = 0.0015
F_ALIGN_WZ_CLAMP = 0.10
F_ALIGN_MAX_DISTANCE = 0.40
F_ALIGN_TIMEOUT = 20.0

# ==================== 阶段二：前进30cm 尝试巡线稳定 ====================
ADVANCE_SPEED = 0.15
ADVANCE_DISTANCE = 0.30
ADVANCE_TIMEOUT = 8.0
ADVANCE_KP_YAW = 0.8
ADVANCE_WZ_CLAMP = 0.08
LANE_STABLE_FRAMES = 10

# ==================== 阶段三：巡线 PID（line_following_ss_pure_ipm） ====================
CAMERA_TIMEOUT = 0.8
CREEP_SPEED = 0.10
CREEP_DISTANCE = 0.10
LOST_CREEP_TIMEOUT = 2.0

# ==================== 阶段四：巡线丢失后 前进中右转45° + 找可信右边界 ====================
TURN_ADVANCE_SPEED = 0.12
TURN_DRIVE_WZ = -0.30          # 前进中右转角速度（rad/s，负=右转）
TURN_YAW_DEG = 45.0            # 累计右转角目标（用角度判断结束）
TURN_TIMEOUT = 25.0

PROC_W = 320
PROC_H = 240
CAM_W = 640
CAM_H = 480
SCALE_X = PROC_W / 640.0
SCALE_Y = PROC_H / 480.0
CANVAS_X0 = 150.0
CANVAS_Y0 = 200.0
CANVAS_W = 300
CANVAS_H = 400
Y_NEAR = 600.0
Y_FAR = 260.0
ALPHA_POLY = 0.45

TUNE = {
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
    'kp_heading_lo': 0.8,
    'kp_heading_mid': 0.8,
    'kp_heading_hi': 0.75,
    'hk1_deg': 5.0,
    'hk2_deg': 10.0,
    'hk3_deg': 18.0,
    'kp_center': 0.018,
    'kd_heading': 0.10,
    'kd_center': 0.005,
    'heading_spike_max_deg': 5.0,
    'ema_h': 0.35,
    'ema_c': 0.55,
    'wz_slew': 3.0,
    'wz_max': 0.55,
    'speed_h_deg': 12.0,
    'speed_k_deg': 0.004,
    'kp_lat': 0.0,
    'heading_bias_deg': 0.0,
    'deadband_center_px': 3.0,
    'deadband_heading_deg': 1.2,
}

DEFAULT_PARAMS = {
    'low_h': 0, 'high_h': 179,
    'low_s': 0, 'high_s': 45,
    'low_v': 170, 'high_v': 255,
    'roi_top': 0.45, 'roi_bottom': 1.0,
    'roi_left': 0.0, 'roi_right': 1.0,
    'blur_ksize': 3, 'erode_iter': 0, 'erode_ksize': 3, 'dilate_iter': 2, 'dilate_ksize': 3,
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
        self.message = '待解锁启动'
        self.last_image_time = 0.0

        # opp_guide（对面中线）
        self.opp_guide_valid = False
        self.opp_center_error = None
        self.opp_heading_error_deg = None
        self.opp_last_center_error = None
        self.opp_guide_good_frames = 0

        # 巡线感知
        self.heading_error_deg = 0.0
        self.center_error_px = 0.0
        self.heading_filt = 0.0
        self.center_filt = 0.0
        self.heading_prev = None
        self.last_wz = 0.0
        self.heading_samples = collections.deque(maxlen=32)
        self.center_samples = collections.deque(maxlen=32)
        self.lane_width_px = None
        self.kanbujian = False
        self.vision_valid = False
        self.lost_frames = 0

        # 中线多项式时间滤波
        self.poly_filt = None
        self.poly_filt_mode = None
        self.poly_filt_y = (0.0, 0.0)

        # 车道线身份跟踪
        self.track_left = None
        self.track_right = None
        self.track_half_width = 84.0
        self.track_valid = False

        # 运动控制
        self.target_speed = 0.20
        self.command_linear_x = 0.0
        self.command_linear_y = 0.0
        self.command_angular_z = 0.0
        self.control_source = 'STOPPED'
        self.lost_creep_start = 0.0

        # 阶段状态
        self.odom = None
        self.state_started = time.time()
        self.phase_start_pose = None
        self.phase_distance = 0.0
        self.advance_start_pose = None
        self.advance_dist = 0.0
        self.lane_good_frames = 0
        self.turn_start_pose = None
        self.turn_dist = 0.0
        self.turn_start_yaw = 0.0
        self.turn_accum_deg = 0.0

        # 显示
        self.vis_overlay = None
        self.vis_mask = None

    def status(self):
        return {
            'mode': self.mode,
            'message': self.message,
            'opp_guide_valid': self.opp_guide_valid,
            'opp_center_error': round(self.opp_center_error, 1) if self.opp_center_error is not None else None,
            'opp_heading_error_deg': round(self.opp_heading_error_deg, 1) if self.opp_heading_error_deg is not None else None,
            'opp_confirm_frames': self.opp_guide_good_frames,
            'vision_valid': bool(self.vision_valid),
            'lane_good_frames': self.lane_good_frames,
            'center_error_px': round(self.center_error_px, 1),
            'heading_error_deg': round(self.heading_error_deg, 1),
            'lane_width_px': round(self.lane_width_px, 1) if self.lane_width_px is not None else None,
            'kanbujian': bool(self.kanbujian),
            'target_speed': float(self.target_speed),
            'phase_distance_m': round(self.phase_distance, 3),
            'advance_dist_m': round(self.advance_dist, 3),
            'turn_dist_m': round(self.turn_dist, 3),
            'turn_accum_deg': round(self.turn_accum_deg, 1),
            'control_source': self.control_source,
            'command_linear_x': round(self.command_linear_x, 3),
            'command_linear_y': round(self.command_linear_y, 3),
            'command_angular_z': round(self.command_angular_z, 3),
            'odom_ready': self.odom is not None,
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
        msg.encoding = encoding
        msg.step = msg.width * (cv_img.shape[2] if len(cv_img.shape) == 3 else 1)
        msg.data = cv_img.tobytes()
        return msg


def init_ipm():
    global IPM_MATRIX, IPM_INV_MATRIX, IPM_CANVAS_M
    if not os.path.exists(PERSP_PATH):
        rospy.logwarn("IPM 配置文件不存在: %s", PERSP_PATH)
        return
    try:
        with open(PERSP_PATH, 'r') as f:
            data = json.load(f)
        src_pts = np.float32(data['src_points'])
        dst_pts = np.float32(data['dst_points'])
        IPM_MATRIX = cv2.getPerspectiveTransform(src_pts, dst_pts)
        IPM_INV_MATRIX = cv2.getPerspectiveTransform(dst_pts, src_pts)
        # 巡线：降采样 + 画布平移组合矩阵
        small_src = src_pts.copy()
        small_src[:, 0] *= SCALE_X
        small_src[:, 1] *= SCALE_Y
        trans = np.float32([[1.0, 0.0, -CANVAS_X0],
                            [0.0, 1.0, -CANVAS_Y0],
                            [0.0, 0.0, 1.0]])
        IPM_CANVAS_M = trans @ cv2.getPerspectiveTransform(small_src, dst_pts)
        rospy.loginfo("IPM 标定载入成功: %s", PERSP_PATH)
    except Exception as err:
        rospy.logwarn("IPM 标定读取失败: %s", err)


def load_hsv():
    path = HSV_PATH if os.path.exists(HSV_PATH) else (FALLBACK_HSV_PATH if os.path.exists(FALLBACK_HSV_PATH) else None)
    if path:
        try:
            with open(path, 'r') as stream:
                state.hsv_params = dict(DEFAULT_PARAMS)
                state.hsv_params.update(json.load(stream))
                rospy.loginfo("HSV 配置载入: %s", path)
                return
        except Exception as err:
            rospy.logwarn("HSV 读取失败: %s", err)
    state.hsv_params = dict(DEFAULT_PARAMS)


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
    y1 = max(0, min(h - 1, y1)); y2 = max(y1 + 1, min(h, y2))
    x1 = max(0, min(w - 1, x1)); x2 = max(x1 + 1, min(w, x2))
    roi = np.zeros_like(mask)
    roi[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
    for op in ('erode', 'dilate'):
        iterations = int(params.get(op + '_iter', 0))
        size = max(1, int(params.get(op + '_ksize', 3)))
        if size % 2 == 0:
            size += 1
        if iterations > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
            roi = getattr(cv2, op)(roi, kernel, iterations=iterations)
    return roi, (x1, y1, x2, y2)


# ==================== 对面中线检测（simple_turn_trial） ====================
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
    a[:, 0, 0] = 639.0 - a[:, 0, 0]
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
    return best


# ==================== 巡线检测（line_following_ss_pure_ipm） ====================
def get_full_mask(frame, params):
    small = cv2.resize(frame, (PROC_W, PROC_H), interpolation=cv2.INTER_AREA)
    blur_k = int(params.get('blur_ksize', 4))
    if blur_k >= 3:
        if blur_k % 2 == 0:
            blur_k += 1
        fh = cv2.GaussianBlur(small, (blur_k, blur_k), 0)
    else:
        fh = small
    hsv = cv2.cvtColor(fh, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv,
                       np.array([params.get('low_h', 42), params.get('low_s', 5), params.get('low_v', 116)]),
                       np.array([params.get('high_h', 179), params.get('high_s', 71), params.get('high_v', 255)]))
    return mask


def warp_to_ipm(mask):
    return cv2.warpPerspective(mask, IPM_CANVAS_M, (CANVAS_W, CANVAS_H),
                               flags=cv2.INTER_NEAREST)


def clean_ipm_mask(warped):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(warped, cv2.MORPH_CLOSE, kernel, iterations=2)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, 8)
    out = np.zeros_like(cleaned)
    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]
        if h >= TUNE['clean_min_h'] and area >= TUNE['clean_min_area'] and h >= w * TUNE['clean_min_ratio']:
            out[labels == label] = 255
    return out


def extract_horizontal_bands(bin_img, kernel_w=31):
    if bin_img is None or bin_img.size == 0:
        return bin_img
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 1))
    return cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel)


def remove_horizontal_bands_ipm(warped, kernel_w=41):
    if warped is None or warped.size == 0:
        return warped
    horiz = extract_horizontal_bands(warped, kernel_w)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(horiz, 8)
    band = np.zeros_like(warped)
    for label in range(1, num_labels):
        x, y, bw, bh, area = stats[label]
        if bw >= TUNE['lane_width_min'] and bh <= bw * 0.4:
            band[labels == label] = 255
    clean = warped.copy()
    clean[band == 255] = 0
    return clean


def fit_lane_line(points):
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < TUNE['min_center_pts'] or np.ptp(pts[:, 1]) < TUNE['min_center_span']:
        return None
    for _ in range(2):
        k, b = np.polyfit(pts[:, 1], pts[:, 0], 1)
        resid = np.abs(pts[:, 0] - (k * pts[:, 1] + b))
        keep = resid <= TUNE['poly_max_resid']
        if keep.sum() < TUNE['min_center_pts']:
            return None
        if keep.sum() == len(pts):
            break
        pts = pts[keep]
    if len(pts) < TUNE['min_center_pts'] or np.ptp(pts[:, 1]) < TUNE['min_center_span']:
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
    if y_max - y_min < TUNE['match_min_overlap']:
        return False
    y_mid = (y_min + y_max) / 2.0
    w = poly_x(right_fit['coeffs'], y_mid) - poly_x(left_fit['coeffs'], y_mid)
    if not (TUNE['lane_width_min'] - 15.0 <= w <= TUNE['lane_width_max'] + 15.0):
        return False
    kL = poly_k(left_fit['coeffs'], y_mid)
    kR = poly_k(right_fit['coeffs'], y_mid)
    return abs(kL - kR) <= TUNE['max_slope_diff']


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
        means = [int(np.mean(c)) for c in clusters if 2 <= len(c) <= TUNE['max_cluster_w']]
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
            if TUNE['lane_width_min'] <= width <= TUNE['lane_width_max']:
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


def _side_of(state, near_x):
    if state.track_left is not None and state.track_right is not None:
        dL = abs(near_x - state.track_left['near_x'])
        dR = abs(near_x - state.track_right['near_x'])
        return 'L' if dL <= dR else 'R'
    if state.track_left is not None:
        if abs(near_x - state.track_left['near_x']) <= TUNE['lane_half_width']:
            return 'L'
        return 'L' if near_x < IPM_CENTER_X else 'R'
    if state.track_right is not None:
        if abs(near_x - state.track_right['near_x']) <= TUNE['lane_half_width']:
            return 'R'
        return 'L' if near_x < IPM_CENTER_X else 'R'
    return 'L' if near_x < IPM_CENTER_X else 'R'


def _update_tracker(state, left_fit, right_fit, half_width):
    def upd(old, fit):
        if fit is None:
            if old is not None:
                old['miss'] += 1
            return old
        coeffs = ema_poly(old['coeffs'] if old else None, fit['coeffs'])
        y_ref = min(fit['y_max'], Y_NEAR)
        near_x = poly_x(coeffs, y_ref)
        return {'coeffs': coeffs, 'near_x': near_x, 'miss': 0}

    state.track_left = upd(state.track_left, left_fit)
    state.track_right = upd(state.track_right, right_fit)
    if state.track_left is not None and state.track_left['miss'] > TUNE['track_stale_frames']:
        state.track_left = None
    if state.track_right is not None and state.track_right['miss'] > TUNE['track_stale_frames']:
        state.track_right = None
    if half_width is not None:
        state.track_half_width = half_width
    state.track_valid = (state.track_left is not None or state.track_right is not None)


def resolve_lane(state, left_fit, right_fit, width_samples, trust_right=False):
    # trust_right = 信任物理右线。摄像头原始画面为镜像，IPM x 与物理左右相反，
    # 故画面 left_fit（IPM 小 x）才是物理右线，right_fit（IPM 大 x）是物理左线。
    cand = []
    for pos_side, fit in (('L', left_fit), ('R', right_fit)):
        if fit is not None and not (trust_right and pos_side == 'R'):
            y_ref = min(fit['y_max'], Y_NEAR)
            cand.append({'pos_side': pos_side, 'fit': fit,
                         'near_x': poly_x(fit['coeffs'], y_ref)})

    lane_width = float(np.mean(width_samples)) if width_samples else None
    half_width = state.track_half_width if state.track_valid else TUNE['lane_half_width']
    pair_matched = False
    left_out = right_out = None
    coeffs = None
    kanbujian = False
    y_min = y_max = 0.0

    if len(cand) >= 2:
        a, b = cand[0], cand[1]
        if state.track_left is not None and state.track_right is not None:
            d_aL = abs(a['near_x'] - state.track_left['near_x'])
            d_aR = abs(a['near_x'] - state.track_right['near_x'])
            d_bL = abs(b['near_x'] - state.track_left['near_x'])
            d_bR = abs(b['near_x'] - state.track_right['near_x'])
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
        width_ok = TUNE['lane_width_min'] <= width <= TUNE['lane_width_max']
        hist_ok = (not state.track_valid or
                   abs(width - 2.0 * state.track_half_width) <= TUNE['max_width_dev'])
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
            if state.track_valid and (state.track_left is not None or state.track_right is not None):
                def hist_d(c):
                    ds = []
                    if state.track_left is not None:
                        ds.append(abs(c['near_x'] - state.track_left['near_x']))
                    if state.track_right is not None:
                        ds.append(abs(c['near_x'] - state.track_right['near_x']))
                    return min(ds) if ds else 1e9
                single = min(cand, key=hist_d)
            else:
                single = min(cand, key=lambda c: abs(c['near_x'] - IPM_CENTER_X))
        if single is not None:
            cls = 'L' if trust_right else _side_of(state, single['near_x'])
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
        _update_tracker(state, None, None, None)
        return None

    center_y_range = (y_min, y_max)
    center_error_px = float(poly_x(coeffs, Y_NEAR) - IPM_CENTER_X)
    y_head = min(Y_NEAR - TUNE['lookahead_px'], y_max)
    kt = poly_k(coeffs, y_head)
    heading_deg = float(math.degrees(math.atan2(-kt, 1.0)))

    center_points = []
    for yv in np.linspace(y_min, y_max, 24):
        xc = poly_x(coeffs, yv)
        if 0 <= yv - CANVAS_Y0 < CANVAS_H:
            center_points.append((int(round(xc - CANVAS_X0)), int(round(yv - CANVAS_Y0))))

    _update_tracker(state, left_out, right_out, half_width if pair_matched else None)

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


def apply_poly_filter(state, result):
    mode = 'S' if result['kanbujian'] else 'P'
    if state.poly_filt is None or state.poly_filt_mode != mode:
        state.poly_filt = tuple(result['coeffs'])
        state.poly_filt_mode = mode
        state.poly_filt_y = tuple(result['center_y_range'])
    else:
        state.poly_filt = ema_poly(state.poly_filt, result['coeffs'])
        y0 = ALPHA_POLY * result['center_y_range'][0] + (1 - ALPHA_POLY) * state.poly_filt_y[0]
        y1 = ALPHA_POLY * result['center_y_range'][1] + (1 - ALPHA_POLY) * state.poly_filt_y[1]
        state.poly_filt_y = (y0, y1)
    coeffs = state.poly_filt
    y0, y1 = state.poly_filt_y
    if y1 <= y0:
        return None
    result = dict(result)
    result['coeffs'] = coeffs
    result['center_y_range'] = state.poly_filt_y
    result['center_error_px'] = float(poly_x(coeffs, Y_NEAR) - IPM_CENTER_X)
    y_head = min(Y_NEAR - TUNE['lookahead_px'], y1)
    kt = poly_k(coeffs, y_head)
    result['heading_error_deg'] = float(math.degrees(math.atan2(-kt, 1.0)))
    cpts = []
    for yv in np.linspace(y0, y1, 24):
        xc = poly_x(coeffs, yv)
        if 0 <= yv - CANVAS_Y0 < CANVAS_H:
            cpts.append((int(round(xc - CANVAS_X0)), int(round(yv - CANVAS_Y0))))
    result['center_points'] = cpts
    return result


def project_ipm_to_image(coeffs, y_range=None):
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


# ==================== 巡线 PID（line_following_ss_pure_ipm） ====================
def compute_pid_ipm(heading_deg, center_px, state):
    heading_deg -= TUNE['heading_bias_deg']
    if state.heading_prev is not None and TUNE['heading_spike_max_deg'] > 0.0:
        if abs(heading_deg - state.heading_filt) > TUNE['heading_spike_max_deg']:
            heading_deg = state.heading_filt
    eh = TUNE['ema_h']
    ec = TUNE['ema_c']
    h = eh * heading_deg + (1 - eh) * state.heading_filt
    c = ec * center_px + (1 - ec) * state.center_filt
    state.heading_filt = h
    state.center_filt = c

    now = time.time()
    deriv_h = 0.0
    state.heading_samples.append((now, h))
    while len(state.heading_samples) >= 2 and now - state.heading_samples[0][0] > 0.25:
        state.heading_samples.popleft()
    if len(state.heading_samples) >= 2:
        t0, h0 = state.heading_samples[0]
        dt = now - t0
        if dt > 0.05:
            deriv_h = (h - h0) / dt

    deriv_c = 0.0
    state.center_samples.append((now, c))
    while len(state.center_samples) >= 2 and now - state.center_samples[0][0] > 0.25:
        state.center_samples.popleft()
    if len(state.center_samples) >= 2:
        t0, c0 = state.center_samples[0]
        dt = now - t0
        if dt > 0.05:
            deriv_c = (c - c0) / dt
    state.heading_prev = h

    ah = abs(h)
    if ah <= TUNE['hk1_deg']:
        kp = TUNE['kp_heading_lo']
    elif ah >= TUNE['hk2_deg']:
        t = (ah - TUNE['hk2_deg']) / max(1e-6, TUNE['hk3_deg'] - TUNE['hk2_deg'])
        kp = TUNE['kp_heading_mid'] + (TUNE['kp_heading_hi'] - TUNE['kp_heading_mid']) * min(1.0, t)
    else:
        t = (ah - TUNE['hk1_deg']) / max(1e-6, TUNE['hk2_deg'] - TUNE['hk1_deg'])
        kp = TUNE['kp_heading_lo'] + (TUNE['kp_heading_mid'] - TUNE['kp_heading_lo']) * t

    if abs(c) < TUNE['deadband_center_px'] and abs(h) < TUNE['deadband_heading_deg']:
        wz = 0.0
    else:
        wz = -(kp * math.radians(h) + TUNE['kp_center'] * c) \
             - TUNE['kd_heading'] * math.radians(deriv_h) - TUNE['kd_center'] * deriv_c
    wz = max(-TUNE['wz_max'], min(TUNE['wz_max'], wz))

    if TUNE['wz_slew'] > 0.0:
        slew_tick = TUNE['wz_slew'] * 0.05
        state.last_wz = max(state.last_wz - slew_tick, min(state.last_wz + slew_tick, wz))
        wz = state.last_wz

    speed = state.target_speed
    if ah > TUNE['speed_h_deg']:
        speed = max(0.18, state.target_speed - TUNE['speed_k_deg'] * (ah - TUNE['speed_h_deg']))

    vel = Twist()
    vel.angular.z = wz
    vel.linear.y = -TUNE['kp_lat'] * c
    vel.linear.x = speed
    return vel


# ==================== 感知回调 ====================
def image_cb(msg):
    global state
    try:
        frame = bridge.imgmsg_to_cv2(msg, 'passthrough')
        if msg.encoding.lower() == 'rgb8':
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        elif msg.encoding.lower() != 'bgr8':
            frame = bridge.imgmsg_to_cv2(msg, 'bgr8')
        raw_frame = frame.copy()
        frame = cv2.flip(frame, 1)

        with state.lock:
            hsv_p = dict(state.hsv_params)
            prev_heading = state.heading_error_deg
            cur_mode = state.mode

        # ---- 对面中线（显示视角，flip 后 mask） ----
        mask, roi = make_mask(frame, hsv_p)
        opp_guide = detect_opposite_centerline(mask)

        # ---- 巡线检测（原始相机坐标） ----
        full_mask = get_full_mask(raw_frame, hsv_p)
        mask_roi = full_mask.copy()
        y_cut = int(PROC_H * (1.0 - TUNE['roi_bottom_ratio']))
        mask_roi[:y_cut, :] = 0
        warped = warp_to_ipm(mask_roi)
        warped = remove_horizontal_bands_ipm(warped)
        cleaned = clean_ipm_mask(warped)
        left_fit, right_fit, width_samples = extract_raw_lanes(cleaned)
        # 直道巡线（LINE_FOLLOW）信任物理右线（画面 left_fit）；转弯（TURN_RIGHT）用正常双线判定
        trust = (cur_mode == 'LINE_FOLLOW')
        result = resolve_lane(state, left_fit, right_fit, width_samples, trust_right=trust)
        if result is None:
            result = resolve_lane(state, left_fit, right_fit, width_samples, trust_right=False)
        if result is not None:
            result = apply_poly_filter(state, result)
            # 摄像头原始画面为镜像（IPM x 与物理左右相反），取反得到物理符号，
            # 否则控制为正反馈会原地打转/打偏
            result['center_error_px'] = -result['center_error_px']
            result['heading_error_deg'] = -result['heading_error_deg']
            heading = result['heading_error_deg']
            center_px = result['center_error_px']
            lane_w = result['lane_width_px']
            kanbu = result['kanbujian']
        else:
            heading = prev_heading
            center_px = state.center_error_px
            lane_w = state.lane_width_px
            kanbu = state.kanbujian

        with state.lock:
            state.last_image_time = time.time()
            state.opp_guide_valid = opp_guide is not None
            state.opp_center_error = opp_guide['center_error'] if opp_guide else None
            state.opp_heading_error_deg = opp_guide['heading_error_deg'] if opp_guide else None
            if opp_guide is not None:
                state.opp_last_center_error = opp_guide['center_error']
            state.heading_error_deg = heading
            state.center_error_px = center_px
            state.lane_width_px = lane_w
            state.kanbujian = kanbu
            if result is not None:
                state.vision_valid = True
                state.lost_frames = 0
            else:
                state.vision_valid = False
                state.lost_frames += 1

        # ---- overlay ----
        overlay = frame.copy()
        cv2.rectangle(overlay, (roi[0], roi[1]), (roi[2] - 1, roi[3] - 1), (255, 120, 0), 2)
        if opp_guide is not None:
            pts = opp_guide['overlay_points']
            for index in range(0, len(pts) - 1, 2):
                cv2.line(overlay, pts[index], pts[index + 1], (0, 255, 0), 5)
            cv2.putText(overlay, 'OPP c=%.1f h=%.1f' % (
                opp_guide['center_error'], opp_guide['heading_error_deg']), (175, 105),
                cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 2)
        if result is not None:
            for fit, color, label in ((result.get('left_fit'), (0, 0, 255), 'L'),
                                      (result.get('right_fit'), (255, 0, 255), 'R')):
                if fit is None:
                    continue
                pts = project_ipm_to_image(fit['coeffs'], (fit['y_min'], fit['y_max']))
                if len(pts) > 1:
                    pts = [(CAM_W - 1 - x, y) for x, y in pts]
                    cv2.polylines(overlay, [np.array(pts, np.int32)], False, color, 3)
                    cv2.putText(overlay, label, (pts[-1][0] - 8, pts[-1][1] - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, .5, color, 2)
            pts = project_ipm_to_image(result['coeffs'], result['center_y_range'])
            if len(pts) > 1:
                pts = [(CAM_W - 1 - x, y) for x, y in pts]
                cv2.polylines(overlay, [np.array(pts, np.int32)], False, (0, 255, 0), 3)
                cv2.putText(overlay, 'C', (pts[-1][0] - 8, pts[-1][1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 2)
            cv2.putText(overlay, 'LANE c=%.1f h=%.1f pair=%d kanbu=%d' % (
                center_px, heading, int(result['pair_matched']), int(result['kanbujian'])),
                (175, 135), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 0, 255), 2)

        with state.lock:
            state.vis_overlay = overlay
            state.vis_mask = mask
            status = state.status()

        cv2.putText(overlay, 'STATE: %s' % status['mode'], (10, 25),
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
        if state.mode == 'TURN_RIGHT' and state.turn_start_pose:
            dx = x - state.turn_start_pose[0]
            dy = y - state.turn_start_pose[1]
            state.turn_dist = math.hypot(dx, dy)
            turn_deg = math.degrees(normalize_angle(state.turn_start_yaw - yaw))
            state.turn_accum_deg = max(0.0, turn_deg)


def publish_stop():
    state.control_source = 'STOPPED'
    state.command_linear_x = 0.0
    state.command_linear_y = 0.0
    state.command_angular_z = 0.0
    if cmd_pub is not None:
        cmd_pub.publish(Twist())


def control_timer(_event):
    with state.lock:
        now = time.time()
        mode = state.mode

        if mode not in ('F_ALIGN', 'ADVANCE', 'LINE_FOLLOW', 'TURN_RIGHT'):
            return
        if state.odom is None or now - state.last_image_time > CAMERA_TIMEOUT:
            state.mode = 'FAULT'
            state.message = '里程计不可用或相机超时，已紧急停车'
            publish_stop()
            return

        cmd = Twist()

        if mode == 'F_ALIGN':
            # 对面中线平移对准，禁止前进
            if state.phase_distance >= F_ALIGN_MAX_DISTANCE or now - state.state_started > F_ALIGN_TIMEOUT:
                state.mode = 'ADVANCE'
                state.advance_start_pose = state.odom
                state.advance_dist = 0.0
                state.state_started = now
                state.lane_good_frames = 0
                state.message = '对准超时/超距，直接前进30cm 尝试巡线'
                publish_stop()
                return

            if not state.opp_guide_valid or state.opp_center_error is None:
                state.opp_guide_good_frames = 0
                if state.opp_last_center_error is not None:
                    center_error = state.opp_last_center_error
                    cmd.linear.x = 0.0
                    cmd.linear.y = F_ALIGN_KP_Y * center_error
                    cmd.angular.z = max(-F_ALIGN_WZ_CLAMP,
                                        min(F_ALIGN_WZ_CLAMP, F_ALIGN_KP_Z * center_error))
                    state.control_source = 'F_ALIGN_HOLD_LAST'
                else:
                    publish_stop()
                    state.control_source = 'F_ALIGN_WAIT'
                state.command_linear_x = cmd.linear.x
                state.command_linear_y = cmd.linear.y
                state.command_angular_z = cmd.angular.z
                if cmd_pub is not None:
                    cmd_pub.publish(cmd)
                return

            center_error = state.opp_center_error
            heading_error = state.opp_heading_error_deg if state.opp_heading_error_deg is not None else 0.0

            if abs(center_error) <= F_ALIGN_CENTER_TOL_PX and abs(heading_error) <= F_ALIGN_HEADING_TOL_DEG:
                state.opp_guide_good_frames += 1
                if state.opp_guide_good_frames >= F_ALIGN_CONFIRM_FRAMES:
                    state.mode = 'ADVANCE'
                    state.advance_start_pose = state.odom
                    state.advance_dist = 0.0
                    state.state_started = now
                    state.lane_good_frames = 0
                    state.message = '对准对面中线完成，前进30cm 尝试巡线'
                    publish_stop()
                    return
            else:
                state.opp_guide_good_frames = 0

            cmd.linear.x = 0.0
            cmd.linear.y = F_ALIGN_KP_Y * center_error
            cmd.angular.z = max(-F_ALIGN_WZ_CLAMP,
                                min(F_ALIGN_WZ_CLAMP, F_ALIGN_KP_Z * center_error))
            state.control_source = 'F_ALIGN_TRANSLATE'

        elif mode == 'ADVANCE':
            # 前进30cm，同时检查巡线是否稳定
            if state.advance_dist >= ADVANCE_DISTANCE:
                state.mode = 'FAULT'
                state.message = '前进30cm 巡线仍未稳定，停车'
                publish_stop()
                return
            if now - state.state_started > ADVANCE_TIMEOUT:
                state.mode = 'FAULT'
                state.message = '前进30cm 超时，停车'
                publish_stop()
                return

            if state.vision_valid:
                state.lane_good_frames += 1
                if state.lane_good_frames >= LANE_STABLE_FRAMES:
                    state.mode = 'LINE_FOLLOW'
                    state.message = '巡线检测稳定，切换巡线PID'
                    publish_stop()
                    return
            else:
                state.lane_good_frames = 0

            cmd.linear.x = ADVANCE_SPEED
            yaw_error = normalize_angle(state.advance_start_pose[2] - state.odom[2])
            cmd.angular.z = max(-ADVANCE_WZ_CLAMP,
                                min(ADVANCE_WZ_CLAMP, ADVANCE_KP_YAW * yaw_error))
            state.control_source = 'ADVANCE'

        elif mode == 'LINE_FOLLOW':
            # 巡线 PID（右线优先判定已在 image_cb 完成，这里直接使用）
            if state.vision_valid:
                state.lost_creep_start = 0.0
                cmd = compute_pid_ipm(state.heading_error_deg, state.center_error_px, state)
                state.control_source = 'LINE_FOLLOW_PID'
            else:
                if state.lost_creep_start == 0.0:
                    state.lost_creep_start = now
                if now - state.lost_creep_start > LOST_CREEP_TIMEOUT:
                    state.mode = 'TURN_RIGHT'
                    state.turn_start_pose = state.odom
                    state.turn_dist = 0.0
                    state.turn_start_yaw = state.odom[2]
                    state.turn_accum_deg = 0.0
                    state.state_started = now
                    state.poly_filt = None
                    state.poly_filt_mode = None
                    state.poly_filt_y = (0.0, 0.0)
                    state.track_left = None
                    state.track_right = None
                    state.track_valid = False
                    state.message = '巡线丢失，前进中右转45°'
                    publish_stop()
                    return
                cmd.linear.x = CREEP_SPEED
                cmd.linear.y = 0.0
                cmd.angular.z = 0.0
                state.control_source = 'LINE_FOLLOW_LOST_CREEP'

        elif mode == 'TURN_RIGHT':
            # 转弯探路：必须转满 TURN_YAW_DEG 后再回巡线判定，
            # 转弯中途画面中线是错配的，不能提前切巡线
            if state.turn_accum_deg >= TURN_YAW_DEG:
                state.mode = 'LINE_FOLLOW'
                state.message = '右转45°完成，恢复巡线'
                publish_stop()
                return
            if now - state.state_started > TURN_TIMEOUT:
                state.mode = 'FAULT'
                state.message = '右转超时，停车'
                publish_stop()
                return
            cmd.linear.x = TURN_ADVANCE_SPEED
            cmd.linear.y = 0.0
            cmd.angular.z = TURN_DRIVE_WZ
            state.control_source = 'TURN_RIGHT_DRIVE_AND_TURN'

        state.command_linear_x = cmd.linear.x
        state.command_linear_y = cmd.linear.y
        state.command_angular_z = cmd.angular.z
        if cmd_pub is not None:
            cmd_pub.publish(cmd)


# ==================== Web 控制台 ====================
PAGE = '''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>polyline_following_ipm 调试节点</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 20px; }
  main { max-width: 1000px; margin: auto; }
  h2 { color: #38bdf8; margin-top: 0; }
  section { background: #1e293b; padding: 20px; margin-bottom: 20px; border-radius: 12px; }
  .card-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-top: 15px; }
  .card { background: #0f172a; padding: 15px; border-radius: 8px; border: 1px solid #334155; }
  .card-title { font-size: 13px; color: #94a3b8; text-transform: uppercase; margin-bottom: 5px; }
  .card-value { font-size: 22px; font-weight: bold; color: #38bdf8; }
  .btn-group { display: flex; gap: 12px; margin-top: 15px; flex-wrap: wrap; }
  button { padding: 12px 24px; border: 0; border-radius: 8px; font-size: 16px; font-weight: 600; cursor: pointer; }
  .btn-start { background: #0284c7; color: white; }
  .btn-stop { background: #dc2626; color: white; }
  .btn-reset { background: #475569; color: white; }
  .status-badge { display: inline-block; padding: 4px 12px; border-radius: 20px; font-weight: 600; background: #334155; }
  .status-F_ALIGN { background: #8b5cf6; }
  .status-ADVANCE { background: #d97706; }
  .status-LINE_FOLLOW { background: #0284c7; }
  .status-TURN_RIGHT { background: #7c3aed; }
  .status-FAULT, .status-ESTOP { background: #dc2626; }
  pre { background: #0f172a; padding: 12px; border-radius: 8px; color: #7dd3fc; font-size: 14px; overflow-x: auto; }
  .img-container { display: flex; gap: 15px; flex-wrap: wrap; margin-top: 15px; }
  img { width: 48%; min-width: 300px; background: #0f172a; border-radius: 8px; border: 1px solid #334155; }
</style>
</head>
<body>
<main>
  <h2>polyline_following_ipm 调试节点</h2>
  <section>
    <div style="display:flex; justify-content:space-between; align-items:center;">
      <span><strong>当前状态：</strong> <span id="mode_badge" class="status-badge">DISARMED</span></span>
      <span style="color:#94a3b8; font-size:14px;">流程：对面中线对准 ➔ 前进30cm尝试巡线 ➔ 巡线PID</span>
    </div>
    <div class="btn-group">
      <button class="btn-start" onclick="postAction('/api/start')">解锁并启动</button>
      <button class="btn-stop" onclick="postAction('/api/stop')">紧急停车</button>
      <button class="btn-reset" onclick="postAction('/api/reset')">复位模式</button>
    </div>
    <div class="card-grid">
      <div class="card"><div class="card-title">对面中线偏差</div><div class="card-value" id="val_opp">N/A</div></div>
      <div class="card"><div class="card-title">对准确认帧</div><div class="card-value" id="val_confirm">0</div></div>
      <div class="card"><div class="card-title">巡线有效</div><div class="card-value" id="val_vis">N/A</div></div>
      <div class="card"><div class="card-title">巡线中心误差</div><div class="card-value" id="val_center">N/A</div></div>
      <div class="card"><div class="card-title">前进距离</div><div class="card-value" id="val_adv">0.000</div></div>
      <div class="card"><div class="card-title">累计右转</div><div class="card-value" id="val_accum">0.0°</div></div>
      <div class="card"><div class="card-title">指令速度 (x,y,z)</div><div class="card-value" id="val_cmd" style="font-size:16px;">0,0,0</div></div>
    </div>
  </section>
  <section>
    <h3>视觉流</h3>
    <div class="img-container">
      <img src="/stream/overlay" alt="Overlay">
      <img src="/stream/mask" alt="Mask">
    </div>
  </section>
  <section>
    <h3>状态 JSON</h3>
    <pre id="status_json">加载中...</pre>
  </section>
</main>
<script>
async function postAction(url) {
  try {
    const res = await (await fetch(url, { method: 'POST' })).json();
    if (!res.ok) alert("操作失败: " + res.error);
  } catch(e) { alert("网络请求异常: " + e); }
}
setInterval(async () => {
  try {
    const data = await (await fetch('/api/status')).json();
    document.getElementById('status_json').textContent = JSON.stringify(data, null, 2);
    const badge = document.getElementById('mode_badge');
    badge.textContent = data.mode;
    badge.className = 'status-badge status-' + data.mode;
    document.getElementById('val_opp').textContent = data.opp_center_error !== null ? data.opp_center_error + ' px' : '未检测到';
    document.getElementById('val_confirm').textContent = data.opp_confirm_frames + ' / 10';
    document.getElementById('val_vis').textContent = data.vision_valid ? '有效' : '无效';
    document.getElementById('val_center').textContent = data.center_error_px + ' px';
    document.getElementById('val_adv').textContent = data.advance_dist_m.toFixed(3);
    document.getElementById('val_accum').textContent = data.turn_accum_deg.toFixed(1) + '°';
    document.getElementById('val_cmd').textContent = data.command_linear_x.toFixed(2) + ', ' + data.command_linear_y.toFixed(2) + ', ' + data.command_angular_z.toFixed(2);
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
        elif path in ('/stream/overlay', '/stream/mask'):
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
        with state.lock:
            if path == '/api/start':
                if state.mode in ('F_ALIGN', 'ADVANCE', 'LINE_FOLLOW'):
                    self.reply({'ok': False, 'error': '任务已在运行中'})
                    return
                if state.odom is None:
                    self.reply({'ok': False, 'error': '里程计未就绪'})
                    return
                if time.time() - state.last_image_time > 0.6:
                    self.reply({'ok': False, 'error': '摄像头画面超时'})
                    return
                if not state.opp_guide_valid or state.opp_center_error is None:
                    self.reply({'ok': False, 'error': '未检测到路口对面中线，请先对准'})
                    return
                state.phase_start_pose = state.odom
                state.phase_distance = 0.0
                state.opp_guide_good_frames = 0
                state.advance_start_pose = None
                state.advance_dist = 0.0
                state.lane_good_frames = 0
                state.lost_creep_start = 0.0
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
                state.phase_start_pose = None
                state.advance_start_pose = None
                state.advance_dist = 0.0
                state.opp_guide_good_frames = 0
                state.lane_good_frames = 0
                state.turn_start_pose = None
                state.turn_dist = 0.0
                state.turn_accum_deg = 0.0
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
                os.system(f"fuser -k {PORT}/tcp 2>/dev/null || pkill -9 -f polyline_following_ipm.py 2>/dev/null")
                time.sleep(1)
                super().server_bind()
            else:
                raise


def shutdown():
    for _ in range(5):
        publish_stop()
        time.sleep(0.03)


if __name__ == '__main__':
    rospy.init_node('polyline_following_ipm', anonymous=False)
    init_ipm()
    load_hsv()
    cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
    mask_pub = rospy.Publisher('/polyline/debug/mask', Image, queue_size=1)
    overlay_pub = rospy.Publisher('/polyline/debug/overlay', Image, queue_size=1)
    rospy.Subscriber(IMAGE_TOPIC, Image, image_cb, queue_size=1, buff_size=2 ** 24)
    rospy.Subscriber('/odom', Odometry, odom_cb, queue_size=1)
    rospy.Timer(rospy.Duration(0.05), control_timer)
    rospy.on_shutdown(shutdown)

    server = ReusableHTTPServer(('0.0.0.0', PORT), Handler)
    rospy.loginfo('polyline_following_ipm 已启动 (网页控制端口 http://0.0.0.0:%d)', PORT)
    server.serve_forever()
