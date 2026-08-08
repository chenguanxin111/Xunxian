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

# ==================== 阶段二：前进50cm 尝试巡线稳定 ====================
ADVANCE_SPEED = 0.15
ADVANCE_DISTANCE = 0.50
ADVANCE_TIMEOUT = 8.0
ADVANCE_KP_YAW = 0.8
ADVANCE_WZ_CLAMP = 0.08
LANE_STABLE_FRAMES = 10

# 巡线 IPM 检测 ROI（底部占比）。
# 进入 LINE_FOLLOW 前用较宽的 0.48 帮助稳定建立双线；进入 LINE_FOLLOW 前
# 宽_ROI_FRAMES 帧后切回 0.30，避免把远处对面车道线/噪声纳入。
ROI_WIDE_BOTTOM_RATIO = 0.48
ROI_TIGHT_BOTTOM_RATIO = 0.30
ROI_WIDE_FRAMES_AFTER_FOLLOW = 15
ADVANCE_ROI_SWITCH_DIST = 0.30   # ADVANCE 跑满 30cm 后才切宽 ROI 开始探索

# 探索阶段右偏偏移（px）：宽 ROI 已启用但仍只跟到物理左线（单线）时，
# 把预测半宽加大该值，让中心线右移、小车偏右，从而把物理右线拉进视野。
LANE_BIAS_RIGHT_PX = 10.0

# ==================== 阶段三：巡线 PID（line_following_ss_pure_ipm） ====================
CAMERA_TIMEOUT = 0.8
CREEP_SPEED = 0.10
CREEP_DISTANCE = 0.10
LOST_CREEP_TIMEOUT = 2.0

# 停止线检测参数（移植 line_following_ss_pure_ipm.py：原始透视图底部20% + 水平带连通域 + 细长判定）
# width_ratio=0.40 宽容度，配合 "高 <= 宽*0.40" 的细长条件排除转向时的粗横带误触。
# max_angle_deg：minAreaRect 绝对倾角上限，过滤非水平的斜矩形，进一步防误触。
# stop_line_delay_sec：转弯完成启用检测后，前 N 秒忽略（避开转弯结束瞬间的残留画面）。
STOP_LINE_ROI_TOP_RATIO = 0.85
STOP_LINE_WIDTH_RATIO = 0.55       #这个参数可能要调整
STOP_LINE_THIN_RATIO = 0.30
STOP_LINE_MAX_ANGLE_DEG = 20.0
STOP_LINE_ENABLE_DELAY_SEC = 1.0

# 左偏角保护：巡线期间相对起始 yaw 累计左转超过阈值，判定走错岔路
LEFT_YAW_LIMIT_DEG = 20.0   # 正值=左偏上限（度），超出则触发右转修正

# ==================== 阶段四：巡线丢失后 前进中右转47° + 找可信右边界 ====================
TURN_ADVANCE_SPEED = 0.131
TURN_DRIVE_WZ = -0.29          # 前进中右转角速度（rad/s，负=右转）
TURN_YAW_DEG = 47.0            # 累计右转角目标（用角度判断结束）
TURN_TIMEOUT = 25.0

# 可信右边界判定
RIGHT_TRUST_NX_MIN = 340.0     # 右线近端 IPM x 必须在此（中心300）右侧
RIGHT_TRUST_SPAN_MIN = 60.0    # 右线 y 跨度下限
RIGHT_TRUST_FRAMES = 5         # 连续可信帧数
RIGHT_TRUST_JITTER_PX = 25.0   # 帧间 near_x 抖动上限

# 前进转弯结束后的原地右转搜索入口（TURN_RIGHT 完成后若未找到物理左边界紫色线）
SEARCH_ROTATE_WZ = -0.25         # 原地缓慢右转角速度（rad/s，负=右转）
SEARCH_ACCUM_LIMIT_DEG = 70.0    # 从转弯开始累计右转达到此角度仍未找到 → 硬切巡线前进
SEARCH_TIMEOUT = 15.0            # 原地搜索超时上限

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
        self.pair_matched = False
        self.lost_frames = 0

        # 物理左右边界可见性（IPM 大 x = 物理左线=紫色；IPM 小 x = 物理右线=红色）
        self.phys_left_found = False
        self.phys_right_found = False

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
        self.target_speed = 0.2
        self.command_linear_x = 0.0
        self.command_linear_y = 0.0
        self.command_angular_z = 0.0
        self.control_source = 'STOPPED'
        self.lost_creep_start = 0.0
        self.line_follow_start_yaw = 0.0   # 进入 LINE_FOLLOW 时记录的 yaw

        # ROI 切换：进入 LINE_FOLLOW 后宽 ROI 的剩余帧数（<=0 用收紧 ROI）
        self.roi_wide_remaining = 0

        # 左偏保护触发的强制右转修正：为 True 时 TURN_RIGHT 不看 right_fit_ok，
        # 必须转满 TURN_YAW_DEG 才回巡线（否则会被"右边界可信"立即短路，修正失效）
        self.turn_forced = False

        # 探索阶段右偏偏移量（px）：宽 ROI 已启用但仍单线时由 image_cb 设置
        self.lane_bias_px = 0.0

        # 停止线（发现停止线 -> 蠕动 -> 停车）
        # 仅在 Y 型岔路转弯（TURN_RIGHT）完成进入 LINE_FOLLOW 后才启用检测，直道/转弯前不检测。
        self.stop_line_enabled = False
        self.stop_line_enable_time = 0.0
        self.stop_line_detected = False
        self.stop_line_y = -1
        self.stop_line_hits = 0
        self.stop_line_stopped = False
        self.creep_started = False
        self.creep_start_pose = None
        self.creep_angular_z = 0.0

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

        # 可信右边界（阶段四）
        self.right_fit_ok = False
        self.right_near_x = None
        self.right_last_near_x = None
        self.right_good_frames = 0

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
            'right_fit_ok': bool(self.right_fit_ok),
            'stop_line_detected': bool(self.stop_line_detected),
            'stop_line_y': int(self.stop_line_y),
            'stop_line_hits': int(self.stop_line_hits),
            'stop_line_enabled': bool(self.stop_line_enabled),
            'stop_line_stopped': bool(self.stop_line_stopped),
            'creep_started': bool(self.creep_started),
            'right_near_x': round(self.right_near_x, 1) if self.right_near_x is not None else None,
            'right_good_frames': self.right_good_frames,
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


def detect_stop_line(bin_img, top_ratio=STOP_LINE_ROI_TOP_RATIO, width_ratio=STOP_LINE_WIDTH_RATIO, thin_ratio=STOP_LINE_THIN_RATIO,
                     max_angle_deg=STOP_LINE_MAX_ANGLE_DEG):
    """检测停止线（原始透视图底部 ROI）。

    先用水平开运算核提取横向长条，再对连通域做"宽度 >= width_ratio*W 且
    高度 <= 宽度*thin_ratio"的细长判定，并用 minAreaRect 过滤倾斜的矩形
    （真正的停止线应接近水平，|angle| 很小），避免转向时画面里斜带误触发。
    返回 (detected, lowest_y)。
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
            band = (labels == label).astype(np.uint8) * 255
            contours, _ = cv2.findContours(band, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            rect = cv2.minAreaRect(contours[0])
            (cx, cy), (rw, rh), angle = rect
            # OpenCV angle: [-90,0)，转成 0~90 的绝对倾角（水平带 => 接近 0）
            ang_abs = abs(angle)
            if ang_abs > 45.0:
                ang_abs = 90.0 - ang_abs
            if ang_abs > max_angle_deg:
                continue
            detected = True
            y_in_full = y + bh + roi_y_start
            if y_in_full > lowest_y:
                lowest_y = int(y_in_full)
    return bool(detected), int(lowest_y)


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
    cand = []
    for pos_side, fit in (('L', left_fit), ('R', right_fit)):
        if fit is not None and not (trust_right and pos_side == 'L'):
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
            cls = 'R' if trust_right else _side_of(state, single['near_x'])
            a, b, c = single['fit']['coeffs']
            if cls == 'L':
                coeffs = (a, b, c + half_width)
                left_out, right_out = single['fit'], None
            else:
                # cls=='R' 即跟到物理左线(IPM 大 x)。探索阶段(lane_bias_px>0)把预测半宽
                # 加大，中心线右移、小车偏右，便于把物理右线拉进视野建立双线。
                w = half_width + state.lane_bias_px
                coeffs = (a, b, c - w)
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

        # ---- 对面中线（显示视角，flip 后 mask） ----
        mask, roi = make_mask(frame, hsv_p)
        opp_guide = detect_opposite_centerline(mask)

        # ---- 巡线检测（原始相机坐标） ----
        full_mask = get_full_mask(raw_frame, hsv_p)
        with state.lock:
            stop_enabled = state.stop_line_enabled
            stop_enable_time = state.stop_line_enable_time
        if stop_enabled and time.time() - stop_enable_time >= STOP_LINE_ENABLE_DELAY_SEC:
            stop_detected, stop_y = detect_stop_line(full_mask)
        else:
            stop_detected, stop_y = False, -1
        mask_roi = full_mask.copy()
        with state.lock:
            cur_mode = state.mode
            adv_dist = state.advance_dist
        # 宽 ROI 仅用于：ADVANCE 跑满 ADVANCE_ROI_SWITCH_DIST(30cm) 之后（开始探索对面）
        # 以及穿越路口后的第一次直道巡线（宽计数未耗尽时）。
        # ADVANCE 前 30cm 与转弯后的巡线(TURN_RIGHT/SEARCH_RIGHT)一律用紧 ROI。
        if cur_mode == 'ADVANCE' and adv_dist >= ADVANCE_ROI_SWITCH_DIST:
            roi_ratio = ROI_WIDE_BOTTOM_RATIO
            state.lane_bias_px = LANE_BIAS_RIGHT_PX
        elif cur_mode == 'LINE_FOLLOW' and state.roi_wide_remaining > 0:
            state.roi_wide_remaining -= 1
            roi_ratio = ROI_WIDE_BOTTOM_RATIO
            state.lane_bias_px = LANE_BIAS_RIGHT_PX
        else:
            roi_ratio = ROI_TIGHT_BOTTOM_RATIO
            state.lane_bias_px = 0.0
        y_cut = int(PROC_H * (1.0 - roi_ratio))
        mask_roi[:y_cut, :] = 0
        warped = warp_to_ipm(mask_roi)
        warped = remove_horizontal_bands_ipm(warped)
        cleaned = clean_ipm_mask(warped)
        left_fit, right_fit, width_samples = extract_raw_lanes(cleaned)
        result = resolve_lane(state, left_fit, right_fit, width_samples, trust_right=False)
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
            # 物理左右边界可见性：IPM 大 x=right_fit=物理左线(紫)，IPM 小 x=left_fit=物理右线(红)
            state.phys_left_found = right_fit is not None
            state.phys_right_found = left_fit is not None
            state.stop_line_detected = stop_detected
            state.stop_line_y = stop_y
            if stop_detected and not state.creep_started:
                state.stop_line_hits += 1
            else:
                state.stop_line_hits = 0
            if result is not None:
                state.vision_valid = True
                state.pair_matched = bool(result['pair_matched'])
                state.lost_frames = 0
            else:
                state.vision_valid = False
                state.pair_matched = False
                state.lost_frames += 1

            # ---- 可信右边界（阶段四）：跨度 + 位置 + 帧间抖动 + 连续帧 ----
            r_nx = None
            if right_fit is not None:
                span = right_fit['y_max'] - right_fit['y_min']
                y_ref = min(right_fit['y_max'], Y_NEAR)
                r_nx = poly_x(right_fit['coeffs'], y_ref)
                jitter_ok = (state.right_last_near_x is None or
                             abs(r_nx - state.right_last_near_x) <= RIGHT_TRUST_JITTER_PX)
                if (span >= RIGHT_TRUST_SPAN_MIN and r_nx >= RIGHT_TRUST_NX_MIN and jitter_ok):
                    state.right_good_frames += 1
                else:
                    state.right_good_frames = 0
            else:
                state.right_good_frames = 0
            state.right_near_x = r_nx
            state.right_last_near_x = r_nx
            state.right_fit_ok = state.right_good_frames >= RIGHT_TRUST_FRAMES

        # ---- overlay ----
        overlay = frame.copy()
        if stop_detected and stop_y >= 0:
            stop_y_full = int(stop_y * CAM_H / PROC_H)
            cv2.line(overlay, (0, stop_y_full), (CAM_W - 1, stop_y_full), (0, 0, 255), 2)
            cv2.putText(overlay, 'STOPLINE y=%d' % stop_y, (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 2)
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
        if state.mode in ('TURN_RIGHT', 'SEARCH_RIGHT') and state.turn_start_pose:
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

        if mode not in ('F_ALIGN', 'ADVANCE', 'LINE_FOLLOW', 'TURN_RIGHT', 'SEARCH_RIGHT'):
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
            # 前进30cm，同时检查巡线是否稳定（必须双线同时检测到才切换）
            # 超时或到距后不再停车，直接切巡线直行（开始时已对准，必然能看到双线）
            if state.advance_dist >= ADVANCE_DISTANCE:
                state.mode = 'LINE_FOLLOW'
                state.line_follow_start_yaw = state.odom[2] if state.odom else 0.0
                state.roi_wide_remaining = ROI_WIDE_FRAMES_AFTER_FOLLOW
                state.message = '前进30cm 到位，直接切换巡线直行'
                publish_stop()
                return
            if now - state.state_started > ADVANCE_TIMEOUT:
                state.mode = 'LINE_FOLLOW'
                state.line_follow_start_yaw = state.odom[2] if state.odom else 0.0
                state.roi_wide_remaining = ROI_WIDE_FRAMES_AFTER_FOLLOW
                state.message = '前进30cm 超时，直接切换巡线直行'
                publish_stop()
                return

            if state.pair_matched:
                state.lane_good_frames += 1
                if state.lane_good_frames >= LANE_STABLE_FRAMES:
                    state.mode = 'LINE_FOLLOW'
                    state.line_follow_start_yaw = state.odom[2] if state.odom else 0.0
                    state.roi_wide_remaining = ROI_WIDE_FRAMES_AFTER_FOLLOW
                    state.message = '巡线双线稳定，切换巡线PID'
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
            # 停止线：发现停止线后蠕动前进并刹车（启动后不依赖停止线持续可见）
            if state.creep_started or state.stop_line_detected:
                if not state.creep_started:
                    state.creep_started = True
                    state.creep_start_pose = state.odom
                    state.creep_angular_z = max(-0.15, min(0.15, state.last_wz))
                if state.odom is not None and state.creep_start_pose is not None:
                    traveled = math.hypot(state.odom[0] - state.creep_start_pose[0],
                                          state.odom[1] - state.creep_start_pose[1])
                else:
                    traveled = 0.0
                if traveled >= CREEP_DISTANCE:
                    state.mode = 'STOPPED'
                    state.message = '检测到停止线，蠕动到位，已刹停'
                    state.stop_line_stopped = True
                    publish_stop()
                    return
                cmd.linear.x = CREEP_SPEED
                cmd.linear.y = 0.0
                cmd.angular.z = state.creep_angular_z
                state.control_source = 'STOPLINE_CREEP'
            elif state.vision_valid:
                state.lost_creep_start = 0.0
                # 左偏角保护：累计 yaw 左偏超过阈值，判定走错岔路，立即右转修正
                if state.odom is not None:
                    delta_yaw_deg = math.degrees(normalize_angle(state.odom[2] - state.line_follow_start_yaw))
                    if delta_yaw_deg > LEFT_YAW_LIMIT_DEG:
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
                        state.right_good_frames = 0
                        state.right_near_x = None
                        state.right_last_near_x = None
                        state.right_fit_ok = False
                        state.turn_forced = True
                        state.message = '左偏角超限，右转修正'
                        publish_stop()
                        return
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
                    state.right_good_frames = 0
                    state.right_near_x = None
                    state.right_last_near_x = None
                    state.right_fit_ok = False
                    state.message = '巡线丢失，前进中右转45°'
                    publish_stop()
                    return
                cmd.linear.x = CREEP_SPEED
                cmd.linear.y = 0.0
                cmd.angular.z = 0.0
                state.control_source = 'LINE_FOLLOW_LOST_CREEP'

        elif mode == 'TURN_RIGHT':
            # 前进中右转47°（角度判断结束），转满后回巡线（右线优先判定在 image_cb）
            # 左偏保护触发的强制修正：忽略 right_fit_ok 短路，必须转满 TURN_YAW_DEG 才回巡线
            if not state.turn_forced and state.right_fit_ok:
                state.mode = 'LINE_FOLLOW'
                state.line_follow_start_yaw = state.odom[2] if state.odom else 0.0
                state.stop_line_enabled = True
                state.stop_line_enable_time = now
                state.message = '右转中已找到可信右边界，切入巡线'
                publish_stop()
                return
            if state.turn_accum_deg >= TURN_YAW_DEG:
                state.turn_forced = False
                # 前进转弯结束：必须找到物理左边界（紫色 right_fit）或双线才认为对准入口，
                # 否则说明未对准入口（仅见单线/右线），原地缓慢右转继续搜索。
                if state.phys_left_found:
                    state.mode = 'LINE_FOLLOW'
                    state.line_follow_start_yaw = state.odom[2] if state.odom else 0.0
                    state.stop_line_enabled = True
                    state.stop_line_enable_time = now
                    state.message = '右转47°完成且已找到左边界，恢复巡线'
                    publish_stop()
                    return
                # 未找到左边界 → 进入原地右转搜索入口
                state.mode = 'SEARCH_RIGHT'
                state.state_started = now
                state.message = '右转完成但未找到左边界，原地缓慢右转搜索入口'
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

        elif mode == 'SEARCH_RIGHT':
            # 原地缓慢右转搜索入口：双线（第一 if）或物理左边界（第二 if）出现即可进巡线
            if state.pair_matched:
                state.mode = 'LINE_FOLLOW'
                state.line_follow_start_yaw = state.odom[2] if state.odom else 0.0
                state.stop_line_enabled = True
                state.stop_line_enable_time = now
                state.message = '原地搜索已找到双线，切入巡线'
                publish_stop()
                return
            if state.phys_left_found:
                state.mode = 'LINE_FOLLOW'
                state.line_follow_start_yaw = state.odom[2] if state.odom else 0.0
                state.stop_line_enabled = True
                state.stop_line_enable_time = now
                state.message = '原地搜索已找到左边界，切入巡线'
                publish_stop()
                return
            if state.turn_accum_deg >= SEARCH_ACCUM_LIMIT_DEG:
                # 从转弯开始累计右转已达70°仍未找到入口 → 硬切巡线前进碰运气
                state.mode = 'LINE_FOLLOW'
                state.line_follow_start_yaw = state.odom[2] if state.odom else 0.0
                state.stop_line_enabled = True
                state.stop_line_enable_time = now
                state.message = '原地搜索达70°仍未找到入口，硬切巡线前进'
                publish_stop()
                return
            if now - state.state_started > SEARCH_TIMEOUT:
                state.mode = 'FAULT'
                state.message = '原地搜索超时，停车'
                publish_stop()
                return
            cmd.linear.x = 0.0
            cmd.linear.y = 0.0
            cmd.angular.z = SEARCH_ROTATE_WZ
            state.control_source = 'SEARCH_RIGHT_ODOM'

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
      <div class="card"><div class="card-title">可信右边界</div><div class="card-value" id="val_right">否</div></div>
      <div class="card"><div class="card-title">右线近端x</div><div class="card-value" id="val_rightx">-</div></div>
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
    document.getElementById('val_right').textContent = data.right_fit_ok ? '是' : '否';
    document.getElementById('val_rightx').textContent = data.right_near_x !== null ? data.right_near_x.toFixed(1) : '-';
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
                state.stop_line_enabled = False
                state.stop_line_enable_time = 0.0
                state.stop_line_hits = 0
                state.stop_line_stopped = False
                state.creep_started = False
                state.creep_start_pose = None
                state.creep_angular_z = 0.0
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
                state.stop_line_enabled = False
                state.stop_line_enable_time = 0.0
                state.stop_line_hits = 0
                state.stop_line_stopped = False
                state.creep_started = False
                state.creep_start_pose = None
                state.creep_angular_z = 0.0
                state.right_good_frames = 0
                state.right_near_x = None
                state.right_last_near_x = None
                state.right_fit_ok = False
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
