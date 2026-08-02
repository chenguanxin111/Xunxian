#!/usr/bin/env python3
"""
line_following_ss_pure_ipm.py

基于 line_following_ss_pure.py，新增 IPM（鸟瞰图）中线提取与线性控制律。

为什么用鸟瞰图：
  原 ss.py/ss_pure.py 在图像空间用"割线角"度量误差，车道线近竖直时
  Δx 很小，角度对横向偏移极敏感（噪声 → ±70~90° 剧烈震荡，见 diag 记录）。
    在 IPM 平面内车道线平行且近竖直，可用两个稳定量：
      center_error_px : 近端（IPM Y=600，约车头前 0.5m）中线相对 X=300 的横向偏移
      heading_error_deg: 中线局部切线（带前瞻距离）相对"正前方"的朝向角
   中线 = 左右直线（x=k*y+b）的平均，朝向角取该直线在近端的前瞻切线，
   过弯时随每帧近端线方向更新，跟随弯道。
   控制律（符号沿用 right_turn_trial.py，实车已验证）：
    wz = -(kp_h * heading_rad + kp_c * center_px) - kd_h * d(heading)/dt
    正向 IPM 误差 = 中线在物理右侧 -> 需右转（angular.z 为负）。

IPM 标定：
  config/perspective_params.json 由 calib_page.py 写入（S=400px/m，板 0.3x1.0m）。
  IPM 平面：X=300 为横向中线，Y 向下为正（600=车头近端，200=远端 1m 处）。
  实际车道宽 0.42m = 168px，本节点用放大画布覆盖 IPM X∈[150,450]、Y∈[200,600]。

其余逻辑（软启动/停车线蠕动/视觉丢失降级/Web 控制台）沿用 ss_pure。

使用：
  python3 line_following_ss_pure_ipm.py
  浏览器打开 http://<小车IP>:5007 进行解锁/启动/停止/调速
  诊断：/stream/overlay 原图叠加（含投影回原图的中线）
        /stream/mask    鸟瞰图二值（IPM 画布）
        /stream/ipm     鸟瞰图彩色叠加（含拟合线）
"""

import os
import sys

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
from tf.transformations import euler_from_quaternion

# 统一分辨率（显示用 640x360，与 ss.py 一致；IPM 处理用 640x480 原始相机分辨率）
pw = 640
ph = 360
CAM_W = 640
CAM_H = 480
# 掩码处理降采样（src_points 按比例缩放，单应保持等价，CPU 大幅下降）
PROC_W = 320
PROC_H = 240
SCALE_X = PROC_W / CAM_W
SCALE_Y = PROC_H / CAM_H

PORT = 5007
ALLOW_MOTION = True

GENTLE_START_DURATION = 2.0
CAMERA_TIMEOUT = 0.8

# 停止线检测参数（沿用 ss_pure，图像空间）
STOP_LINE_ROI_TOP_RATIO = 0.75
STOP_LINE_WIDTH_RATIO = 0.40
STOP_LINE_THIN_RATIO = 0.40
CREEP_SPEED = 0.10
CREEP_DISTANCE = 0.05

# ------------------ IPM（鸟瞰图）参数 ------------------
CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config')
PERSP_PATH = os.path.join(CONFIG_DIR, 'perspective_params.json')

IPM_MATRIX = None       # src(图像) -> dst(IPM 平面)
IPM_INV_MATRIX = None   # dst -> src
IPM_CANVAS_M = None     # 组合矩阵：平移后的大画布（含 -origin 平移）

IPM_CENTER_X = 300.0    # IPM 平面横向中线
# 画布覆盖 IPM X∈[150,450], Y∈[200,600]
CANVAS_X0 = 150.0
CANVAS_Y0 = 200.0
CANVAS_W = 300
CANVAS_H = 400
# 拟合用近端/远端 Y（IPM 坐标）
Y_NEAR = 600.0
Y_FAR = 260.0
ALPHA_POLY = 0.45        # 中线多项式时间滤波系数（0.45 新值，防跳变）

# 可网页在线调整的检测参数（/api/set_param），改后立即生效
TUNE = {
    'roi_bottom_ratio': 0.40,   # 检测 ROI：画面下方比例（越大看得越远，采样越足）
    'min_center_pts': 3,        # 拟合最少采样点数
    'min_center_span': 25.0,    # 拟合最短 IPM 纵向跨度 (px)
    'poly_max_resid': 20.0,     # 拟合离群剔除阈值 (px)
    'lane_width_min': 110.0,    # 配对宽度下限 (px)
    'lane_width_max': 230.0,    # 配对宽度上限 (px)
    'lane_half_width': 84.0,    # 单线回退半车道宽 (px)
    'max_slope_diff': 0.55,     # 配对平行度：局部切线斜率差上限
    'clean_min_h': 12,          # 鸟瞰连通域最小高度 (px)
    'clean_min_area': 40,       # 鸟瞰连通域最小面积
    'clean_min_ratio': 0.35,    # 连通域高/宽比下限（弯道倾斜线也能保留）
    'lookahead_px': 100.0,      # 朝向角前瞻距离 (px)
    'track_stale_frames': 10,   # 某侧车道线连续丢失多少帧后丢弃该侧身份
    'max_width_dev': 35.0,      # 配对宽度与跟踪到的半宽*2 的最大偏差 (px)
    'max_cluster_w': 48,        # 单行选中簇的最大横向宽度 (px)，超宽视为横条/合并噪点跳过
    'match_min_overlap': 18.0,  # 配对匹配要求左右线观测 Y 重叠区间最小长度 (px)
}

# ------------------ 控制参数（初始值，上车后微调） ------------------
KP_HEADING = 2.6        # rad/s / rad(朝向误差)
KP_CENTER = 0.016       # rad/s / px(横向误差)
KD_HEADING = 0.30       # 朝向微分阻尼（s）
WZ_MAX = 0.45
KP_LAT = 0.0            # 侧移 linear.y 增益（px->m/s，符号待实测；0=禁用）
EMA_ALPHA = 0.55        # 误差平滑系数
DEADBAND_CENTER_PX = 3.0
DEADBAND_HEADING_DEG = 1.2
HEADING_BIAS_DEG = 0.0  # 朝向系统偏差补偿（实测校准）


class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.mode = 'DISARMED'
        self.message = '纯巡线(IPM)已启动，等待解锁'
        self.last_image_time = 0.0

        # 巡线感知数据（IPM）
        self.error = 0.0
        self.heading_error_deg = 0.0
        self.center_error_px = 0.0
        self.heading_filt = 0.0
        self.center_filt = 0.0
        self.heading_prev = None
        self.heading_prev_t = 0.0
        self.lane_width_px = None
        self.kanbujian = False
        self.centers = []
        self.vis = None
        self.vis_overlay = None
        self.vis_mask = None
        self.vis_ipm = None

        # 中线多项式时间滤波（防跳变）
        self.poly_filt = None        # (a,b,c)
        self.poly_filt_mode = None   # 'P'=配对 'S'=单线
        self.poly_filt_y = (0.0, 0.0)  # (y_min, y_max) 观测范围
        self.left_fit_filt = None
        self.right_fit_filt = None

        # 车道线身份跟踪（防左右互换/中线跳变）
        self.track_left = None       # {'coeffs','near_x','miss'}
        self.track_right = None
        self.track_half_width = 84.0
        self.track_valid = False

        # 运动控制参数
        self.target_speed = 0.36
        self.start_time = 0.0
        self.last_v_z = 0.0
        self.delat_v_z = 0.0
        self.command_linear_x = 0.0
        self.command_linear_y = 0.0
        self.command_angular_z = 0.0

        self.vision_valid = False
        self.lost_frames = 0
        self.last_valid_time = 0.0

        # 停止线
        self.stop_line_detected = False
        self.stop_line_y = -1
        self.stop_line_hits = 0
        self.stop_line_stopped = False
        self.creep_started = False
        self.creep_start_x = 0.0
        self.creep_start_y = 0.0
        self.creep_angular_z = 0.0
        self.wz_history = []
        self.lost_creep_start = 0.0

        # odom
        self.odom_x = 0.0
        self.odom_y = 0.0
        self.odom_yaw = 0.0
        self.creep_start_yaw = 0.0

    def status(self):
        return {
            'mode': self.mode,
            'message': self.message,
            'error_deg': float(round(self.heading_error_deg, 1)),
            'center_error_px': float(round(self.center_error_px, 1)),
            'heading_error_deg': float(round(self.heading_error_deg, 1)),
            'lane_width_px': float(round(self.lane_width_px, 1)) if self.lane_width_px is not None else None,
            'kanbujian': bool(self.kanbujian),
            'center_count': int(len(self.centers)),
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
            'tune': dict(TUNE),
        }


state = SharedState()
bridge = CvBridge()
cmd_pub = None
mask_pub = None
overlay_pub = None
ipm_pub = None


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
                    rospy.loginfo("纯巡线(IPM)节点成功载入 HSV 配置文件 [%s]", p)
                    return data
            except Exception:
                pass
    return {'low_h': 42, 'high_h': 179, 'low_s': 5, 'high_s': 71, 'low_v': 116, 'high_v': 255,
            'blur_ksize': 4, 'erode_iter': 0, 'erode_ksize': 3, 'dilate_iter': 2, 'dilate_ksize': 3}


HSV_PARAMS = load_hsv_params()


def init_ipm():
    """载入新标定的 IPM 参数并构建放大画布变换矩阵。"""
    global IPM_MATRIX, IPM_INV_MATRIX, IPM_CANVAS_M
    if not os.path.exists(PERSP_PATH):
        rospy.logwarn("IPM 配置文件不存在: %s", PERSP_PATH)
        return False
    try:
        with open(PERSP_PATH, 'r') as f:
            data = json.load(f)
        src_pts = np.float32(data['src_points'])
        dst_pts = np.float32(data['dst_points'])
        IPM_MATRIX = cv2.getPerspectiveTransform(src_pts, dst_pts)
        IPM_INV_MATRIX = cv2.getPerspectiveTransform(dst_pts, src_pts)
        # 降采样处理用的单应：src 点按比例缩放到 (PROC_W, PROC_H) 平面
        small_src = src_pts.copy()
        small_src[:, 0] *= SCALE_X
        small_src[:, 1] *= SCALE_Y
        trans = np.float32([[1.0, 0.0, -CANVAS_X0],
                            [0.0, 1.0, -CANVAS_Y0],
                            [0.0, 0.0, 1.0]])
        IPM_CANVAS_M = trans @ cv2.getPerspectiveTransform(small_src, dst_pts)
        rospy.loginfo("纯巡线(IPM)载入标定: src=%s", src_pts.tolist())
        return True
    except Exception as err:
        rospy.logwarn("读取 IPM 配置失败: %s", err)
        return False


def get_full_mask(frame, params=None):
    """在降采样 (PROC_W, PROC_H) 分辨率上做 HSV 二值化（IPM 与停止线共用）。"""
    if params is None:
        params = HSV_PARAMS
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
    """将 640x480 二值图变换到 IPM 画布，输出 (CANVAS_W, CANVAS_H)。"""
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
        if h >= TUNE['clean_min_h'] and area >= TUNE['clean_min_area'] and h >= w * TUNE['clean_min_ratio']:
            out[labels == label] = 255
    return out


def fit_lane_line(points):
    """对一组 IPM 采样点做直线拟合 x = k*y + b，两轮去离群。

    参考 straight_intersection_pass.contour_ipm_fit：直线拟合无二次项，
    不会产生弧线爆炸，也不会因远端外推放大斜率误差。
    返回 dict {'coeffs': (0.0, k, b), 'y_min': float, 'y_max': float}，
    其中 y_min/y_max 为参与拟合的内点覆盖的 Y 范围（只画真实可见段）。
    点数或跨度不足返回 None。
    """
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
    """x(y) 的局部切线斜率 dx/dy。"""
    a, b, _ = coeffs
    return 2.0 * a * y + b


def lanes_matched(left_fit, right_fit):
    """配对匹配：只要求两线重叠区间的中点宽度与斜率一致。

    对齐 straight_intersection_pass.build_center_guide_ipm 的宽容思路：
      - 重叠区间小一点也行（match_min_overlap）；
      - 宽度只在重叠中点一个点校验，容忍两端（远端/近端）局部不匹配；
      - 直线拟合斜率 k 是常量，直接比较 |kL - kR|，不再逐点扫描。
    这样只有整条线才对不上才判不匹配，kanbujian 不会因一小段噪声误触发。
    """
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
    """从画面中心向两侧搜索：分别返回中心左、右侧最靠近中心的一个簇 x。

    从左向右、从右向左扫描簇列表，遇到第一条即停（不取远处杂物）。
    返回 (left_x, right_x)，找不到为 None。
    """
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
    """逐行扫描提取左右车道线采样点并各自直线拟合。

    每行以画面中心 (IPM X=300) 为原点，向两侧搜索第一条簇，
    成对行要求宽度在合理范围。返回 (left_fit, right_fit, width_samples)。
    左右身份由 resolve_lane 结合历史确定。
    """
    h, w = warped.shape
    center_canvas_x = IPM_CENTER_X - CANVAS_X0  # 150
    left_pts, right_pts = [], []
    width_samples = []

    for yc in range(h - 1, -1, -4):
        xs = np.where(warped[yc] == 255)[0]
        if len(xs) == 0:
            continue
        diff = np.diff(xs)
        breaks = np.where(diff > 2)[0] + 1
        clusters = np.split(xs, breaks)
        # 簇宽护栏：过宽簇（横向带/与横线连体）跳过，防止混入拟合
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
            # 宽度不合理：可能一侧是墙壁/杂物，仅保留靠近中央的一侧
            if abs(left_x - center_canvas_x) <= abs(right_x - center_canvas_x):
                left_pts.append((lx, y_ipm))
            else:
                right_pts.append((rx, y_ipm))
        elif left_x is not None:
            # 仅左侧可见：独立收集该侧采样（弯道丢线/虚线断裂也不丢行）
            left_pts.append((float(left_x + CANVAS_X0), y_ipm))
        elif right_x is not None:
            right_pts.append((float(right_x + CANVAS_X0), y_ipm))

    left_fit = fit_lane_line(left_pts)
    right_fit = fit_lane_line(right_pts)
    return left_fit, right_fit, width_samples


def _side_of(state, near_x):
    """结合历史身份给一条线定左右：上一帧是右线，这一帧仍按右线处理。

    双侧历史：按"离哪条历史线近"判定（抗墙壁干扰）。
    单侧历史：候选线必须落在该侧历史位置附近(±半车道宽)才沿用其身份，
              否则说明历史已失效/该线其实是另一侧，回退按画面物理位置判定，
              避免把右线误判成左线导致中线被推到赛道外。
    """
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
    """用本帧拟合更新左右身份轨迹（EMA），一侧缺失时保留旧值并计数。"""
    def upd(old, fit):
        if fit is None:
            if old is not None:
                old['miss'] += 1
            return old
        coeffs = ema_poly(old['coeffs'] if old else None, fit['coeffs'])
        # near_x 取该线自身近端观测处（短线不外推到 Y=600）
        return {'coeffs': coeffs,
                'near_x': poly_x(coeffs, min(fit['y_max'], Y_NEAR)),
                'miss': 0}

    state.track_left = upd(state.track_left, left_fit)
    state.track_right = upd(state.track_right, right_fit)
    if state.track_left is not None and state.track_left['miss'] > TUNE['track_stale_frames']:
        state.track_left = None
    if state.track_right is not None and state.track_right['miss'] > TUNE['track_stale_frames']:
        state.track_right = None
    if half_width is not None:
        state.track_half_width = half_width
    state.track_valid = (state.track_left is not None or state.track_right is not None)


def resolve_lane(state, left_fit, right_fit, width_samples):
    """结合历史身份确定左右线、配对或单线，输出中线误差，并更新轨迹。

    关键点：
      - 两条候选同时存在时，用与历史左右线的距离分配身份（而非仅凭画面位置），
        墙壁/杂物混入时不会把右线误判成左线。
      - 配对宽度同时受历史半宽约束（|w - 2*hw| <= max_width_dev），不一致则降级单线。
      - 单线时按身份决定外推方向，并用跟踪到的半宽而非固定值。
    """
    cand = []
    for pos_side, fit in (('L', left_fit), ('R', right_fit)):
        if fit is not None:
            # near_x 取在该线自身近端观测范围（不超出数据外推）
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

        # 宽度在公共参考 y（两条线都真实覆盖的最接近端）处测量，避免外推
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
        # 单线：候选多于 1 时选离历史任一侧最近（或离中央最近）的一条
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
            cls = _side_of(state, single['near_x'])
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


def project_ipm_to_image(coeffs, y_range=None):
    """将 IPM 直线/中线投影回原始相机坐标（640x480，未镜像）。"""
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


def ema_poly(prev, new, alpha=ALPHA_POLY):
    if prev is None:
        return new
    return tuple(alpha * n + (1.0 - alpha) * p for p, n in zip(prev, new))


def extract_horizontal_bands(bin_img, kernel_w=31):
    if bin_img is None or bin_img.size == 0:
        return bin_img
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 1))
    return cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel)


def remove_horizontal_white_bands(bin_img, width_ratio=0.50, thin_ratio=STOP_LINE_THIN_RATIO, kernel_w=31):
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


def remove_horizontal_bands_ipm(warped, kernel_w=41):
    """在 IPM 画布上剔除横向长条（停止线/终点线）。

    先用横向开运算分离横向结构，再按"宽度>=车道宽下限 且 细长"整条擦除。
    必须放在 clean_ipm_mask 之前：否则终点线会与竖直车道线连成一个连通域，
    宽度/高度过滤失效，横向带像素混进二次拟合导致弧线爆炸。
    """
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


def detect_stop_line(bin_img, top_ratio=STOP_LINE_ROI_TOP_RATIO, width_ratio=STOP_LINE_WIDTH_RATIO, thin_ratio=STOP_LINE_THIN_RATIO):
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


def compute_pid_ipm(heading_deg, center_px, state):
    """线性控制律：wz = -(kp_h*heading_rad + kp_c*center_px) - kd_h*heading_deriv"""
    heading_deg -= HEADING_BIAS_DEG
    h = EMA_ALPHA * heading_deg + (1 - EMA_ALPHA) * state.heading_filt
    c = EMA_ALPHA * center_px + (1 - EMA_ALPHA) * state.center_filt
    state.heading_filt = h
    state.center_filt = c

    now = time.time()
    deriv = 0.0
    if state.heading_prev is not None:
        dt = now - state.heading_prev_t
        if dt > 0.0:
            deriv = (h - state.heading_prev) / dt
    state.heading_prev = h
    state.heading_prev_t = now

    if abs(c) < DEADBAND_CENTER_PX and abs(h) < DEADBAND_HEADING_DEG:
        wz = 0.0
    else:
        wz = -(KP_HEADING * math.radians(h) + KP_CENTER * c) - KD_HEADING * deriv
    wz = max(-WZ_MAX, min(WZ_MAX, wz))

    vel = Twist()
    vel.angular.z = wz
    vel.linear.y = -KP_LAT * c
    vel.linear.x = state.target_speed
    return vel, h, c


def draw_overlay(frame, result):
    """把 IPM 拟合的左右线（红/紫）与中线（绿）投影回原图叠加。

    只画各自实际观测到的 Y 范围，不做无据外推。
    """
    overlay = frame.copy()
    if result is None:
        return overlay
    for fit, color, label in ((result.get('left_fit'), (0, 0, 255), 'L'),
                              (result.get('right_fit'), (255, 0, 255), 'R')):
        if fit is None:
            continue
        pts = project_ipm_to_image(fit['coeffs'], (fit['y_min'], fit['y_max']))
        if len(pts) > 1:
            cv2.polylines(overlay, [np.array(pts, np.int32)], False, color, 3)
            cv2.putText(overlay, label, (pts[-1][0] - 8, pts[-1][1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, .5, color, 2)
    pts = project_ipm_to_image(result['coeffs'], result['center_y_range'])
    if len(pts) > 1:
        cv2.polylines(overlay, [np.array(pts, np.int32)], False, (0, 255, 0), 3)
        cv2.putText(overlay, 'C', (pts[-1][0] - 8, pts[-1][1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 2)
    if result.get('kanbujian'):
        cv2.putText(overlay, 'SINGLE-LINE (kanbujian)', (10, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 255, 255), 2)
    return overlay


def draw_ipm_view(warped, result):
    """鸟瞰图可视化：左线红、右线紫、中线绿（只画观测到的线段）。"""
    vis = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
    cv2.line(vis, (int(IPM_CENTER_X - CANVAS_X0), 0),
             (int(IPM_CENTER_X - CANVAS_X0), CANVAS_H - 1), (255, 255, 0), 1)
    if result is None:
        return vis
    for fit, color in ((result.get('left_fit'), (0, 0, 255)),
                       (result.get('right_fit'), (255, 0, 255))):
        if fit is None:
            continue
        a, b, c = fit['coeffs']
        pts = []
        for yv in np.linspace(fit['y_min'], fit['y_max'], 40):
            xc = a * yv * yv + b * yv + c
            cx = int(round(xc - CANVAS_X0))
            cy = int(round(yv - CANVAS_Y0))
            if 0 <= cx < CANVAS_W and 0 <= cy < CANVAS_H:
                pts.append((cx, cy))
        if len(pts) > 1:
            cv2.polylines(vis, [np.array(pts, np.int32)], False, color, 3)
    a, b, c = result['coeffs']
    y0, y1 = result['center_y_range']
    pts = []
    for yv in np.linspace(y0, y1, 40):
        xc = a * yv * yv + b * yv + c
        cx = int(round(xc - CANVAS_X0))
        cy = int(round(yv - CANVAS_Y0))
        if 0 <= cx < CANVAS_W and 0 <= cy < CANVAS_H:
            pts.append((cx, cy))
    if len(pts) > 1:
        cv2.polylines(vis, [np.array(pts, np.int32)], False, (0, 255, 0), 3)
    for x, y in result.get('center_points', []):
        cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)
    cv2.putText(vis, f"PAIR={int(result['pair_matched'])} kanbu={int(result['kanbujian'])}  "
                      f"c={result['center_error_px']:+.1f}px  h={result['heading_error_deg']:+.1f}deg  "
                      f"w={result['lane_width_px'] if result['lane_width_px'] is not None else -1:.0f}",
                (5, 18), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255), 2)
    return vis


def apply_poly_filter(state, result):
    """对中线多项式做时间滤波，防止跳变。

    配对/单线模式切换时直接采用新值（避免跨模式混叠）；
    同模式下用 EMA 平滑系数与观测范围。
    """
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


def image_cb(msg):
    try:
        frame = bridge.imgmsg_to_cv2(msg, 'passthrough')
        if msg.encoding.lower() == 'rgb8':
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        elif msg.encoding.lower() != 'bgr8':
            frame = bridge.imgmsg_to_cv2(msg, 'bgr8')

        with state.lock:
            prev_heading = state.heading_error_deg

        full_mask = get_full_mask(frame, HSV_PARAMS)
        stop_detected, stop_y = detect_stop_line(full_mask)
        # ROI：只保留画面下方 TUNE['roi_bottom_ratio'] 作为中线检测区（近场可靠区）
        mask_roi = full_mask.copy()
        y_cut = int(PROC_H * (1.0 - TUNE['roi_bottom_ratio']))
        mask_roi[:y_cut, :] = 0

        warped = warp_to_ipm(mask_roi)
        warped = remove_horizontal_bands_ipm(warped)
        cleaned = clean_ipm_mask(warped)
        left_fit, right_fit, width_samples = extract_raw_lanes(cleaned)
        result = resolve_lane(state, left_fit, right_fit, width_samples)

        if result is not None:
            result = apply_poly_filter(state, result)
            heading = result['heading_error_deg']
            center_px = result['center_error_px']
        else:
            heading = prev_heading
            center_px = state.center_error_px

        overlay = draw_overlay(frame, result)
        if stop_detected and stop_y >= 0:
            stop_y_full = int(stop_y * CAM_H / PROC_H)
            cv2.line(overlay, (0, stop_y_full), (CAM_W - 1, stop_y_full), (0, 0, 255), 2)
        cv2.putText(overlay, f"STOPLINE: {stop_detected} y={stop_y}", (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255), 1)
        cv2.putText(overlay, f"IPM c={center_px:+.1f}px h={heading:+.1f}deg  vis={result is not None}",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)

        overlay_small = cv2.resize(overlay, (pw, ph), interpolation=cv2.INTER_AREA)
        overlay_disp = cv2.flip(overlay_small, 1)
        cv2.putText(overlay_disp, f"IPM c={center_px:+.1f}px h={heading:+.1f}deg  "
                                  f"MODE:{state.mode}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1)

        ipm_view = draw_ipm_view(cleaned, result)

        with state.lock:
            state.last_image_time = time.time()
            state.vis = cleaned
            state.vis_overlay = overlay_disp
            state.vis_mask = cleaned
            state.vis_ipm = ipm_view
            state.heading_error_deg = heading
            state.center_error_px = center_px
            state.error = heading
            state.kanbujian = bool(result['kanbujian']) if result else False
            state.lane_width_px = result['lane_width_px'] if result else None
            state.centers = list(result['center_points']) if result else []
            state.vision_valid = result is not None
            if state.vision_valid:
                state.last_valid_time = state.last_image_time
                state.lost_frames = 0
            else:
                state.lost_frames += 1
            state.stop_line_detected = stop_detected
            state.stop_line_y = stop_y

        if mask_pub is not None:
            mask_pub.publish(safe_cv2_to_imgmsg(cleaned, 'mono8'))
        if overlay_pub is not None:
            overlay_pub.publish(safe_cv2_to_imgmsg(overlay_disp, 'bgr8'))
        if ipm_pub is not None:
            ipm_pub.publish(safe_cv2_to_imgmsg(ipm_view, 'bgr8'))

    except Exception as e:
        rospy.logerr_throttle(2, f"纯巡线(IPM)图像回调异常: {e}")


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
        q = data.pose.pose.orientation
        euler = euler_from_quaternion((q.x, q.y, q.z, q.w))
        state.odom_yaw = euler[2]


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

        # 停止线蠕动（沿用 ss_pure）
        if state.stop_line_detected:
            if not state.creep_started:
                state.creep_started = True
                state.creep_start_x = state.odom_x
                state.creep_start_y = state.odom_y
                state.creep_start_yaw = state.odom_yaw
                recent = state.wz_history[-10:]
                avg_wz = float(np.mean(recent)) if recent else 0.0
                state.creep_angular_z = max(-0.15, min(0.15, avg_wz))
            traveled = math.hypot(state.odom_x - state.creep_start_x,
                                  state.odom_y - state.creep_start_y)
            if traveled >= CREEP_DISTANCE:
                state.mode = 'STOPPED'
                state.message = '检测到停止线，蠕动到位，已刹停'
                state.stop_line_stopped = True
                publish_stop()
                return
            vel_creep = Twist()
            vel_creep.linear.x = CREEP_SPEED
            vel_creep.linear.y = 0.0
            MAX_CREEP_YAW_DELTA = math.radians(5.0)
            d_yaw = state.odom_yaw - state.creep_start_yaw
            d_yaw = (d_yaw + math.pi) % (2 * math.pi) - math.pi
            if abs(d_yaw) >= MAX_CREEP_YAW_DELTA:
                vel_creep.angular.z = 0.0
            else:
                vel_creep.angular.z = max(-0.15, min(0.15, state.creep_angular_z))
            state.command_linear_x = vel_creep.linear.x
            state.command_linear_y = vel_creep.linear.y
            state.command_angular_z = vel_creep.angular.z
            cmd_pub.publish(vel_creep)
            return

        # 视觉丢失降级蠕动（沿用 ss_pure）
        if not state.vision_valid:
            if state.lost_creep_start == 0.0:
                state.lost_creep_start = now
            if now - state.lost_creep_start > 2.0:
                state.mode = 'STOPPED'
                state.message = '视觉持续丢失，已停车'
                publish_stop()
                return
            recent = state.wz_history[-10:]
            avg_wz = float(np.mean(recent)) if recent else 0.0
            vel_lost = Twist()
            vel_lost.linear.x = CREEP_SPEED
            vel_lost.linear.y = 0.0
            vel_lost.angular.z = max(-0.20, min(0.20, avg_wz))
            state.command_linear_x = vel_lost.linear.x
            state.command_linear_y = vel_lost.linear.y
            state.command_angular_z = vel_lost.angular.z
            cmd_pub.publish(vel_lost)
            return

        vel, h, c = compute_pid_ipm(state.heading_error_deg, state.center_error_px, state)

        # 软启动
        is_gentle = (now - state.start_time) < GENTLE_START_DURATION
        if is_gentle:
            vel.linear.x = 0.15
            vel.angular.z *= 0.5
        else:
            elapsed = now - state.start_time - GENTLE_START_DURATION
            ramp = 0.15 + elapsed * 0.15
            vel.linear.x = min(state.target_speed, max(0.15, ramp))

        state.lost_creep_start = 0.0
        state.wz_history.append(float(vel.angular.z))
        if len(state.wz_history) > 10:
            state.wz_history = state.wz_history[-10:]

        state.command_linear_x = vel.linear.x
        state.command_linear_y = vel.linear.y
        state.command_angular_z = vel.angular.z

        rospy.loginfo_throttle(1, "IPM巡线: vx=%.2f vy=%.2f wz=%.2f (h=%.1fdeg c=%.1fpx)",
                               vel.linear.x, vel.linear.y, vel.angular.z, h, c)
        cmd_pub.publish(vel)


# ------------------ Web 控制台 ------------------

PAGE = '''<!doctype html><meta charset="utf-8"><title>纯巡线控制台 (ss_pure_ipm)</title>
<style>body{font:16px sans-serif;background:#0f172a;color:#f8fafc;margin:20px}main{max-width:1100px;margin:auto}section{background:#1e293b;padding:16px;margin:12px 0;border-radius:10px}img{width:32%;background:#111;margin:.33%;border-radius:6px}.start{background:#1677ff;color:white}.stop{background:#d00;color:white}button{padding:10px 18px;border:0;border-radius:6px;margin-right:8px;font-size:15px;cursor:pointer}pre{font-size:15px;background:#0f172a;padding:10px;border-radius:6px;color:#38bdf8}</style>
<main><h2>纯巡线控制台 (IPM 鸟瞰版, Port 5007)</h2>
<section><b>准备就绪。确认跑道清空后手动启动巡线。</b>
<p><button class="start" onclick="post('/api/start')">解锁并开始巡线</button><button class="stop" onclick="post('/api/stop')">立即停车</button><button onclick="post('/api/reset')">复位为仅感知模式</button></p>
<p>设置巡线目标速度：
<button onclick="post('/api/set_speed?speed=0.18')">0.18 m/s (超慢试跑)</button>
<button onclick="post('/api/set_speed?speed=0.24')">0.24 m/s (中速赛道)</button>
<button onclick="post('/api/set_speed?speed=0.36')">0.36 m/s (去年原版速度)</button>
</p>
<pre id="status">加载状态中...</pre></section>
<section><h3>检测参数在线调节（改后立即生效，无需重启）</h3>
<p>格式 <code>参数名=数值</code>，可多组空格分隔，如：
<code>min_center_span=40 min_center_pts=4 clean_min_h=15 roi_bottom_ratio=0.45</code></p>
<p><input id="tune_input" style="width:560px;padding:8px;background:#0f172a;color:#f8fafc;border:1px solid #475569;border-radius:4px">
<button onclick="tuneApply()">应用</button></p>
<p>常用：<button onclick="setParam('min_center_span',40)">span=40</button>
<button onclick="setParam('min_center_span',55)">span=55</button>
<button onclick="setParam('clean_min_h',15)">clean_h=15</button>
<button onclick="setParam('clean_min_h',30)">clean_h=30</button>
<button onclick="setParam('lane_width_max',240)">width_max=240</button>
<button onclick="setParam('lane_width_min',100)">width_min=100</button>
<button onclick="setParam('roi_bottom_ratio',0.5)">ROI=0.5</button>
<button onclick="setParam('roi_bottom_ratio',0.6)">ROI=0.6</button>
<button onclick="setParam('poly_max_resid',20)">resid=20</button></p>
<pre id="tune_status">当前参数...</pre></section>
<section><h3>原图叠加 (/stream/overlay) · 鸟瞰二值 (/stream/mask) · 鸟瞰彩色 (/stream/ipm)</h3>
<p><img id="img_overlay"><img id="img_mask"><img id="img_ipm"></p></section>
</main>
<script>
document.getElementById('img_overlay').src = '/stream/overlay';
document.getElementById('img_mask').src = '/stream/mask';
document.getElementById('img_ipm').src = '/stream/ipm';
async function post(url){
    try {
        let r = await fetch(url,{method:'POST'});
        let d = await r.json();
        if(!d.ok){ alert("⚠️ 操作失败: " + (d.error || '未知错误')); }
    } catch(e) { alert("⚠️ 请求失败: " + e); }
}
async function setParam(name,value){
    try {
        let r = await fetch('/api/set_param?name='+name+'&value='+value,{method:'POST'});
        let d = await r.json();
        if(!d.ok){ alert("⚠️ 设置失败: " + (d.error||'')); }
    } catch(e) { alert("⚠️ 请求失败: " + e); }
}
async function tuneApply(){
    let s = document.getElementById('tune_input').value.trim();
    if(!s) return;
    for(let kv of s.split(/\s+/)){
        let [k,v] = kv.split('=');
        if(k && v) await setParam(k, v);
    }
}
setInterval(async()=>{try{
    let st = await (await fetch('/api/status')).json();
    document.getElementById('status').textContent=JSON.stringify(st,null,2);
    if(st.tune){ document.getElementById('tune_status').textContent='当前: '+JSON.stringify(st.tune); }
}catch(e){}},400);
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
        elif path in ['/stream/overlay', '/stream/mask', '/stream/ipm']:
            self.send_response(200)
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()
            while not rospy.is_shutdown():
                with state.lock:
                    if path == '/stream/overlay':
                        img = state.vis_overlay
                    elif path == '/stream/mask':
                        img = state.vis_mask
                    else:
                        img = state.vis_ipm
                if img is not None:
                    ret, jpeg = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if ret:
                        try:
                            self.wfile.write(b'--frame\r\n')
                            self.wfile.write(b'Content-Type: image/jpeg\r\n\r\n')
                            self.wfile.write(jpeg.tobytes())
                            self.wfile.write(b'\r\n')
                        except Exception:
                            break
                time.sleep(0.06)
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
                    self.reply({'ok': False, 'error': '当前没有可靠 IPM 车道中线，禁止启动'})
                    return
                state.last_v_z = 0.0
                state.delat_v_z = 0.0
                state.heading_filt = 0.0
                state.center_filt = 0.0
                state.heading_prev = None
                state.heading_prev_t = 0.0
                state.poly_filt = None
                state.poly_filt_mode = None
                state.lost_frames = 0
                state.start_time = time.time()
                state.last_valid_time = time.time()
                state.stop_line_hits = 0
                state.stop_line_stopped = False
                state.creep_started = False
                state.wz_history = []
                state.lost_creep_start = 0.0
                state.mode = 'RUNNING'
                state.message = '纯巡线(IPM)运行中'
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
            elif path == '/api/set_param':
                query = parsed.query
                params = dict(q.split('=') for q in query.split('&') if '=' in q)
                name = params.get('name', '')
                if name not in TUNE:
                    self.reply({'ok': False, 'error': f'未知参数: {name}，可用: {",".join(TUNE)}'})
                    return
                try:
                    value = float(params['value'])
                except Exception:
                    self.reply({'ok': False, 'error': 'value 需为数值'})
                    return
                cur = TUNE[name]
                # 简单范围保护：整型保持整型
                if isinstance(cur, int):
                    value = int(value)
                if value <= 0:
                    self.reply({'ok': False, 'error': 'value 需为正数'})
                    return
                TUNE[name] = value
                rospy.loginfo("调参: %s = %s", name, TUNE[name])
                self.reply({'ok': True, 'name': name, 'value': TUNE[name]})
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
    global cmd_pub, mask_pub, overlay_pub, ipm_pub
    rospy.init_node('line_following_ss_pure_ipm', anonymous=False)

    if not init_ipm():
        rospy.logerr("IPM 初始化失败，无法使用鸟瞰图中线提取，退出。")
        sys.exit(1)

    cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
    mask_pub = rospy.Publisher('/line_following_ss_ipm/debug/mask', Image, queue_size=1)
    overlay_pub = rospy.Publisher('/line_following_ss_ipm/debug/overlay', Image, queue_size=1)
    ipm_pub = rospy.Publisher('/line_following_ss_ipm/debug/ipm', Image, queue_size=1)

    rospy.Subscriber('/usb_cam/image_raw', Image, image_cb, queue_size=1, buff_size=2**24)
    rospy.Subscriber('/odom', Odometry, odom_cb, queue_size=1)
    rospy.Timer(rospy.Duration(0.05), control_timer)  # 20Hz
    rospy.on_shutdown(shutdown)

    server = ReusableHTTPServer(('0.0.0.0', PORT), Handler)
    rospy.loginfo('纯巡线(IPM)已启动: http://0.0.0.0:%d', PORT)
    server.serve_forever()


if __name__ == '__main__':
    main()
