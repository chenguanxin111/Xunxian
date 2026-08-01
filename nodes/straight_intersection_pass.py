#!/usr/bin/env python3
"""
straight_intersection_pass.py

第一段：穿过无车道线路口（先平移对准路口对面中线）→ 龟速直行（保持 y 方向）
→ 检测到红绿灯前脚下横向白线 → 刹停。

流程：
  DISARMED → ALIGN(平移对准) → STRAIGHT(龟速直行) → STOPPED(检测到白线)

核心视觉复用 right_turn_trial.py 的 IPM 中线提取：
  - contour_ipm_fit / build_entry_guide_ipm / detect_dual_lane_ipm
  - 在 IPM(600x600, X=100~500, Y=600脚下) 空间计算 center_error 与 heading_error
停车：在 IPM 空间检测脚下(Y>=300) 横向白带，外接矩形宽度占比 > 阈值且细长，连续确认。
可视化：仿 line_following_ss_pure.py 的 Web 控制台（Port 5008）。

用法：浏览器 http://<小车IP>:5008 解锁启动；/stream/overlay 与 /stream/mask 内嵌流。
"""
import json
import math
import os
import sys
import threading
import time
from urllib.parse import urlparse

# 屏蔽 Qt 桌面显示，防止 SSH 下 SIGABRT
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["OPENCV_UI_BACKEND"] = "HEADLESS"
if "DISPLAY" in os.environ:
    del os.environ["DISPLAY"]

from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

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
FALLBACK_HSV_PATH = os.path.join(CONFIG_DIR, 'white_lane_right.json')
PERSP_PATH = os.path.join(CONFIG_DIR, 'perspective_params.json')
IMAGE_TOPIC = '/usb_cam/image_raw'
PORT = 5008
ALLOW_MOTION = True

CAMERA_TIMEOUT = 0.8

# ---- 控制参数 ----
STRAIGHT_SPEED = 0.10            # 直行龟速 (m/s)
ALIGN_SPEED = 0.04               # 平移对准时的前进速度 (m/s)
HEADING_TOL_DEG = 5.0            # 直行允许的车头 y 方向偏差 (度)
ALIGN_CENTER_TOL_PX = 25.0       # 平移对准收敛阈值 (IPM 像素, 相对 X=300)
ALIGN_CONFIRM_FRAMES = 10        # 对准连续确认帧数
CENTER_KP_Y = -0.0012            # 平移对准 linear.y 增益 (IPM 像素 -> m/s)
HEADING_KP_Z = -0.008            # 直行 heading 修正 angular.z 增益 (角度->rad/s)
CENTER_KP_Z_ALIGN = -0.0009      # 对准阶段 center_error 对 angular.z 的耦合（小）
WZ_CLAMP = 0.10                  # 直行/对准角速度限幅
ALIGN_MAX_DISTANCE = 0.25        # 对准阶段最远前进距离
STRAIGHT_MAX_DISTANCE = 3.0      # 直行最远距离（保护）
STRAIGHT_TIMEOUT = 30.0

# ---- 停车线检测 (IPM 横向白带) ----
STOP_ROI_Y_TOP = 300             # IPM Y>=300 即画面下方 50%
STOP_LANE_X_MIN = 100            # IPM 车道有效宽左边界
STOP_LANE_X_MAX = 500            # IPM 车道有效宽右边界
STOP_WIDTH_RATIO = 0.65          # 外接矩形宽度相对车道有效宽占比 (>65%，400px底边即可触发)
STOP_THIN_RATIO = 0.40           # 高度/宽度比上限
STOP_CONFIRM_FRAMES = 4          # 连续确认帧数
STOP_MIN_AREA = 300
STOP_KERNEL_W = 41               # 水平开运算核宽度（分离纯横向底边，滤除梯形腰）

# ---- IPM ----
IPM_CENTER_X = 300.0
IPM_LANE_HALF_WIDTH = 200.0
IPM_LANE_WIDTH_MIN = 220.0
IPM_LANE_WIDTH_MAX = 520.0

DEFAULT_PARAMS = {
    'low_h': 42, 'high_h': 179,
    'low_s': 5, 'high_s': 71,
    'low_v': 116, 'high_v': 255,
    'roi_top': 0.45, 'roi_bottom': 1.0,
    'roi_left': 0.0, 'roi_right': 1.0,
    'blur_ksize': 4, 'erode_iter': 0, 'erode_ksize': 3,
    'dilate_iter': 2, 'dilate_ksize': 3,
}

IPM_MATRIX = None
IPM_INV_MATRIX = None


def normalize_angle(value):
    return math.atan2(math.sin(value), math.cos(value))


class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.mode = 'DISARMED'
        self.message = '仅感知模式：确认安全后在网页启动'
        self.last_image_time = 0.0
        self.hsv_params = dict(DEFAULT_PARAMS)

        self.odom = None          # (x, y, yaw)
        self.start_pose = None    # 本次任务起点
        self.phase_start_pose = None
        self.phase_distance = 0.0
        self.state_started = time.time()

        self.center_error = None          # IPM 中线横向误差 (px, 相对 X=300)
        self.heading_error_deg = None     # IPM 中线朝向 vs 车头
        self.lane_valid = False
        self.entry_guide_valid = False
        self.dual_lane_valid = False
        self.last_center_error = 0.0
        self.align_good_frames = 0
        self.straight_heading_offset = 0.0
        self.straight_history_wz = []     # 直行时视觉丢失的缓行方向

        self.stop_detected = False
        self.stop_y_ipm = -1
        self.stop_width_ratio = 0.0
        self.stop_hits = 0

        self.command_linear_x = 0.0
        self.command_linear_y = 0.0
        self.command_angular_z = 0.0
        self.control_source = 'STOPPED'
        self.vis_overlay = None
        self.vis_mask = None

    def status(self):
        return {
            'mode': self.mode,
            'message': self.message,
            'center_error_px': round(self.center_error, 1) if self.center_error is not None else None,
            'heading_error_deg': round(self.heading_error_deg, 1) if self.heading_error_deg is not None else None,
            'lane_valid': self.lane_valid,
            'entry_guide_valid': self.entry_guide_valid,
            'dual_lane_valid': self.dual_lane_valid,
            'align_good_frames': self.align_good_frames,
            'phase_distance_m': round(self.phase_distance, 3),
            'stop_detected': self.stop_detected,
            'stop_y_ipm': self.stop_y_ipm,
            'stop_width_ratio': round(self.stop_width_ratio, 3),
            'stop_hits': self.stop_hits,
            'control_source': self.control_source,
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
    global IPM_MATRIX, IPM_INV_MATRIX
    if os.path.exists(PERSP_PATH):
        try:
            with open(PERSP_PATH, 'r') as f:
                data = json.load(f)
                src_pts = np.float32(data['src_points'])
                dst_pts = np.float32(data['dst_points'])
                IPM_MATRIX = cv2.getPerspectiveTransform(src_pts, dst_pts)
                IPM_INV_MATRIX = cv2.getPerspectiveTransform(dst_pts, src_pts)
                rospy.loginfo("IPM 矩阵载入成功: %s", PERSP_PATH)
        except Exception as err:
            rospy.logwarn("IPM 配置读取失败: %s", err)


def load_hsv():
    path = HSV_PATH if os.path.exists(HSV_PATH) else (FALLBACK_HSV_PATH if os.path.exists(FALLBACK_HSV_PATH) else None)
    if path:
        try:
            with open(path, 'r') as stream:
                state.hsv_params.update(json.load(stream))
                rospy.loginfo("HSV 配置载入: %s", path)
        except Exception as err:
            rospy.logwarn("HSV 读取失败: %s", err)


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


def contour_ipm_fit(contour, min_y_span=55.0):
    """在 IPM 平面拟合 x = k*y + b。"""
    if contour is None or len(contour) < 5 or IPM_MATRIX is None:
        return None
    pts = contour.reshape(-1, 1, 2).astype(np.float32).copy()
    pts[:, 0, 0] = 639.0 - pts[:, 0, 0]
    ipm = cv2.perspectiveTransform(pts, IPM_MATRIX).reshape(-1, 2)
    ipm = ipm[np.isfinite(ipm).all(axis=1)]
    if len(ipm) < 5:
        return None
    ipm = ipm[(ipm[:, 0] > -250) & (ipm[:, 0] < 850) &
              (ipm[:, 1] > -100) & (ipm[:, 1] < 700)]
    if len(ipm) < 5 or np.ptp(ipm[:, 1]) < min_y_span:
        return None
    k, b = np.polyfit(ipm[:, 1], ipm[:, 0], 1)
    return {'k': float(k), 'b': float(b),
            'y_min': float(np.min(ipm[:, 1])), 'y_max': float(np.max(ipm[:, 1])),
            'contour': contour}


def build_center_guide_ipm(contours):
    """找到一对平行车道线并计算其中线（路口对面车道）。

    纵向线在 IPM 中 k≈0（竖直），横向线 k 巨大。这里要求：
      - 两条线斜率接近（平行）
      - 单线斜率 |k| <= 1.0（排除横线）
      - 宽度在合理车道宽范围内
    返回中线 center_error（相对 X=300）与 heading_error_deg。
    """
    if IPM_MATRIX is None or IPM_INV_MATRIX is None:
        return None
    fits = [fit for fit in (contour_ipm_fit(c, min_y_span=18.0) for c in contours[:8]) if fit]
    # 严格过滤横向线：纵向车道线 k 应接近 0
    fits = [f for f in fits if abs(f['k']) <= 1.0]
    if not fits:
        return None
    best = None
    for i in range(len(fits)):
        for j in range(i + 1, len(fits)):
            a, b = fits[i], fits[j]
            overlap_min = max(a['y_min'], b['y_min'], 20.0)
            overlap_max = min(a['y_max'], b['y_max'], 590.0)
            if overlap_max - overlap_min < 15.0:
                continue
            yc = (overlap_min + overlap_max) / 2.0
            ax = a['k'] * yc + a['b']
            bx = b['k'] * yc + b['b']
            width = abs(bx - ax)
            slope_diff = abs(a['k'] - b['k'])
            if not (IPM_LANE_WIDTH_MIN <= width <= IPM_LANE_WIDTH_MAX):
                continue
            if slope_diff > 0.30:
                continue
            score = (slope_diff + abs(width - 2.0 * IPM_LANE_HALF_WIDTH) / 400.0)
            if best is None or score < best['score']:
                k_c = (a['k'] + b['k']) / 2.0
                b_c = (a['b'] + b['b']) / 2.0
                y_near = overlap_max
                center_near = k_c * y_near + b_c
                heading_deg = math.degrees(math.atan2(-k_c, 1.0))
                best = {'score': score, 'k': k_c, 'b': b_c,
                        'center_error': float(center_near - IPM_CENTER_X),
                        'heading_error_deg': float(heading_deg),
                        'y_max': y_near, 'lane_width': float(width),
                        'left_fit': a if ax < bx else b, 'right_fit': b if ax < bx else a}
    if best is None:
        # 单条纵向线兜底：用固定半宽推算中线
        long_fits = [f for f in fits if f['y_max'] - f['y_min'] >= 90.0]
        if long_fits:
            f = max(long_fits, key=lambda x: x['y_max'] - x['y_min'])
            y_near = min(f['y_max'], 560.0)
            x_edge = f['k'] * y_near + f['b']
            # 判断是左线还是右线（相对 X=300 中心）
            is_right = x_edge >= IPM_CENTER_X
            x_mid = x_edge - IPM_LANE_HALF_WIDTH if is_right else x_edge + IPM_LANE_HALF_WIDTH
            heading_deg = math.degrees(math.atan2(-f['k'], 1.0))
            best = {'score': 99.0, 'k': f['k'], 'b': f['b'],
                    'center_error': float(x_mid - IPM_CENTER_X),
                    'heading_error_deg': float(heading_deg),
                    'y_max': y_near, 'lane_width': None,
                    'left_fit': None, 'right_fit': f if is_right else None,
                    'single': True}
    if best is None:
        return None
    # 生成 overlay 中线点（逆投影回图像）
    y_values = np.linspace(best.get('y_min', 40.0), best['y_max'], 28)
    ipm_center = np.float32([[best['k'] * y + best['b'], y] for y in y_values]).reshape(-1, 1, 2)
    raw = cv2.perspectiveTransform(ipm_center, IPM_INV_MATRIX).reshape(-1, 2)
    h, w = 480, 640
    overlay_points = []
    for x, y in raw:
        pt = (int(round(639.0 - x)), int(round(y)))
        if -80 <= pt[0] < w + 80 and 0 <= pt[1] < h:
            overlay_points.append(pt)
    best['overlay_points'] = overlay_points
    return best


def build_center_guide_pixels(mask, contours=None, img_w=640, img_h=480):
    """像素空间行扫描中线兜底（当 IPM 无法拟合时）。

    与 right_turn_trial.analyze_lanes 保持一致：
      - 取左右两侧最大的轮廓构建 clean_mask，再逐行扫描
      - center_error 直接由中线点计算（无纵向跨度门槛）
      - heading 仅在纵向跨度足够时计算，否则视为 0
    """
    if contours is None:
        clean_mask = mask
    else:
        left = [c for c in contours
                if cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2] / 2 < img_w / 2]
        right = [c for c in contours
                 if cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2] / 2 >= img_w / 2]
        clean_mask = np.zeros_like(mask)
        if left:
            cv2.drawContours(clean_mask, [left[0]], -1, 255, -1)
        if right:
            cv2.drawContours(clean_mask, [right[0]], -1, 255, -1)

    roi_top = int(img_h * 0.45)
    centers_px = []
    for y in range(img_h - 1, roi_top, -8):
        xs = np.flatnonzero(clean_mask[y] > 0)
        if len(xs) == 0:
            continue
        groups = np.split(xs, np.where(np.diff(xs) > 2)[0] + 1)
        means = [int(np.mean(g)) for g in groups if len(g) >= 2]
        left_x = max((x for x in means if x < img_w // 2), default=None)
        right_x = min((x for x in means if x >= img_w // 2), default=None)
        if left_x is not None and right_x is not None and 100 < right_x - left_x < img_w * 0.9:
            cand_x = int((left_x + right_x) / 2)
            if len(centers_px) == 0 or abs(cand_x - centers_px[-1][0]) < 40:
                centers_px.append((cand_x, y))

    if len(centers_px) < 3:
        return None

    near_count = min(5, len(centers_px))
    far_count = min(5, len(centers_px))
    near_x = float(np.mean([p[0] for p in centers_px[:near_count]]))
    far_x = float(np.mean([p[0] for p in centers_px[-far_count:]]))
    center_error = near_x - img_w / 2.0

    near_y = float(np.mean([p[1] for p in centers_px[:near_count]]))
    far_y = float(np.mean([p[1] for p in centers_px[-far_count:]]))
    forward_px = near_y - far_y
    if forward_px >= 20.0:
        heading_deg = math.degrees(math.atan2(far_x - near_x, forward_px))
    else:
        heading_deg = 0.0

    overlay_points = list(centers_px)
    return {
        'center_error': center_error,
        'heading_error_deg': heading_deg,
        'overlay_points': overlay_points,
        'pixel_centers': centers_px,
        'source': 'pixel_fallback',
    }


def detect_stop_line_ipm(bin_img_full, roi_y_top=STOP_ROI_Y_TOP):
    """在 IPM 空间检测脚下横向白带（停止线）。

    输入为原始相机二值图。先做 IPM 变换，再用水平开运算（宽 x1 核）分离
    出纯横向底边，滤除梯形两腰（斜向线）。最后统计外接矩形宽度相对
    车道有效宽 (X=100~500) 的占比，> 阈值且细长则判定为停止线。
    返回 (detected, bottom_y_ipm, width_ratio)。
    """
    if IPM_MATRIX is None or bin_img_full is None or bin_img_full.size == 0:
        return False, -1, 0.0
    h_ipm, w_ipm = 600, 600
    bird = cv2.warpPerspective(bin_img_full, IPM_MATRIX, (w_ipm, h_ipm),
                               flags=cv2.INTER_NEAREST)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (STOP_KERNEL_W, 1))
    horiz = cv2.morphologyEx(bird, cv2.MORPH_OPEN, kernel)
    roi = horiz[roi_y_top:, :]
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(roi, 8)
    lane_w = float(STOP_LANE_X_MAX - STOP_LANE_X_MIN)
    detected = False
    bottom_y = -1
    best_ratio = 0.0
    for label in range(1, num_labels):
        x, y, bw, bh, area = stats[label]
        if area < STOP_MIN_AREA:
            continue
        ratio = bw / lane_w
        if bw >= int(lane_w * STOP_WIDTH_RATIO) and bh <= bw * STOP_THIN_RATIO:
            if ratio > best_ratio:
                best_ratio = ratio
                bottom_y = y + bh + roi_y_top
                detected = True
    return detected, bottom_y, best_ratio


def image_cb(msg):
    global state
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
        result = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = result[0] if len(result) == 2 else result[1]
        contours = [c for c in contours if cv2.contourArea(c) > 120 and cv2.arcLength(c, False) > 50]
        contours.sort(key=lambda c: cv2.contourArea(c), reverse=True)

        guide = build_center_guide_ipm(contours)
        # 当 IPM 无法拟合时（路口对面无纵向车道线），使用像素空间行扫描兜底
        if guide is None:
            guide = build_center_guide_pixels(mask, contours)
        stop_det, stop_y, stop_ratio = detect_stop_line_ipm(mask)

        with state.lock:
            state.last_image_time = time.time()
            state.lane_valid = guide is not None
            state.center_error = guide['center_error'] if guide else None
            state.heading_error_deg = guide['heading_error_deg'] if guide else None
            state.dual_lane_valid = guide is not None and not guide.get('single', False) and guide.get('source') != 'pixel_fallback'
            state.entry_guide_valid = guide is not None
            state.stop_detected = stop_det
            state.stop_y_ipm = stop_y
            state.stop_width_ratio = stop_ratio

        overlay = frame.copy()
        cv2.rectangle(overlay, (roi[0], roi[1]), (roi[2] - 1, roi[3] - 1), (255, 120, 0), 2)
        if guide is not None:
            pts = guide['overlay_points']
            if guide.get('source') == 'pixel_fallback':
                # 像素空间兜底中线：画连续折线（绿色）
                if len(pts) > 1:
                    cv2.polylines(overlay, [np.array(pts, np.int32)], False, (0, 255, 0), 3)
                cv2.putText(overlay, 'PIXEL GUIDE centers=%d' % len(pts),
                            (160, 40), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 2)
            else:
                # IPM 中线：画分段线
                for i in range(0, len(pts) - 1, 2):
                    cv2.line(overlay, pts[i], pts[i + 1], (0, 255, 0), 4)
                for fit_key in ('left_fit', 'right_fit'):
                    fit = guide.get(fit_key)
                    if fit is not None and fit.get('contour') is not None:
                        cv2.drawContours(overlay, [fit['contour']], -1, (0, 255, 255), 3)
                cv2.putText(overlay, 'IPM GUIDE k=%.3f' % (guide['k'] if 'k' in guide else 0),
                            (180, 40), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 2)

        # 停止线 overlay：画检测到的 IPM 底边（逆投影到图像）
        if stop_det and stop_y > 0:
            pts_ipm = np.float32([[50.0, stop_y], [550.0, stop_y]]).reshape(-1, 1, 2)
            raw = cv2.perspectiveTransform(pts_ipm, IPM_INV_MATRIX).reshape(-1, 2)
            for x, y in raw:
                pt = (int(round(639.0 - x)), int(round(y)))
                cv2.circle(overlay, pt, 4, (0, 0, 255), -1)
            cv2.putText(overlay, 'STOPLINE ratio=%.2f' % stop_ratio, (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 2)

        with state.lock:
            state.vis_overlay = overlay
            state.vis_mask = mask
            mode = state.mode

        cv2.putText(overlay, 'STATE: %s' % mode, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 255, 255), 2)
        center_str = '%.1f' % state.center_error if state.center_error is not None else 'N/A'
        heading_str = '%.1f' % state.heading_error_deg if state.heading_error_deg is not None else 'N/A'
        cv2.putText(overlay, 'center=%s  heading=%s deg' % (center_str, heading_str),
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, .50, (0, 255, 255), 2)

        if mask_pub is not None:
            mask_pub.publish(safe_cv2_to_imgmsg(mask, 'mono8'))
        if overlay_pub is not None:
            overlay_pub.publish(safe_cv2_to_imgmsg(overlay, 'bgr8'))
    except Exception as exc:
        rospy.logwarn_throttle(2, f'image_cb error: {exc}')


def odom_cb(msg):
    q = msg.pose.pose.orientation
    yaw = euler_from_quaternion((q.x, q.y, q.z, q.w))[2]
    with state.lock:
        state.odom = (msg.pose.pose.position.x, msg.pose.pose.position.y, yaw)
        if state.start_pose:
            state.distance = math.hypot(state.odom[0] - state.start_pose[0],
                                        state.odom[1] - state.start_pose[1])
        if state.phase_start_pose:
            state.phase_distance = math.hypot(state.odom[0] - state.phase_start_pose[0],
                                              state.odom[1] - state.phase_start_pose[1])


def publish_stop():
    state.control_source = 'STOPPED'
    state.command_linear_x = 0.0
    state.command_linear_y = 0.0
    state.command_angular_z = 0.0
    if cmd_pub is not None:
        cmd_pub.publish(Twist())


def begin_phase(mode, message):
    state.mode = mode
    state.message = message
    state.state_started = time.time()
    state.phase_start_pose = state.odom
    state.phase_distance = 0.0


def control_timer(_event):
    with state.lock:
        now = time.time()
        mode = state.mode
        if mode not in ('ALIGN', 'STRAIGHT'):
            return
        if state.odom is None or now - state.last_image_time > CAMERA_TIMEOUT:
            state.mode = 'FAULT'
            state.message = '里程计不可用或相机超时，已停车'
            publish_stop()
            return

        cmd = Twist()
        if mode == 'ALIGN':
            # 平移对准：先用 linear.y 平移对准中线，车几乎不前冲
            if state.phase_distance >= ALIGN_MAX_DISTANCE or now - state.state_started > 15.0:
                state.mode = 'FAULT'
                state.message = '平移对准超距/超时'
                publish_stop()
                return
            if not state.lane_valid:
                # 未见中线：原地等待，不猛动
                state.align_good_frames = 0
                publish_stop()
                state.control_source = 'ALIGN_WAIT'
                return
            center_error = state.center_error
            heading_error = state.heading_error_deg
            if center_error is None:
                publish_stop()
                state.control_source = 'ALIGN_WAIT'
                return
            if abs(center_error) <= ALIGN_CENTER_TOL_PX and abs(heading_error or 0) <= 10.0:
                state.align_good_frames += 1
                if state.align_good_frames >= ALIGN_CONFIRM_FRAMES:
                    state.straight_heading_offset = state.odom[2]
                    begin_phase('STRAIGHT', '对准完成，开始龟速直行')
                    rospy.loginfo('!!! 平移对准完成，进入直行 !!!')
                    return
            else:
                state.align_good_frames = 0
            # linear.y 平移：center_error 正（中线在右）→ 向右平移（linear.y 负）
            cmd.linear.x = ALIGN_SPEED
            cmd.linear.y = CENTER_KP_Y * center_error
            cmd.angular.z = CENTER_KP_Z_ALIGN * center_error
            cmd.angular.z = max(-WZ_CLAMP, min(WZ_CLAMP, cmd.angular.z))
            state.control_source = 'ALIGN_TRANSLATE'

        elif mode == 'STRAIGHT':
            # 停止线检测：IPM 脚下横向白带
            if state.stop_detected:
                state.stop_hits += 1
                if state.stop_hits >= STOP_CONFIRM_FRAMES:
                    state.mode = 'STOPPED'
                    state.message = '检测到红绿灯前白线，已刹停'
                    publish_stop()
                    rospy.loginfo('!!! 检测到停止线，刹停 !!!')
                    return
            else:
                state.stop_hits = 0

            if state.phase_distance >= STRAIGHT_MAX_DISTANCE or now - state.state_started > STRAIGHT_TIMEOUT:
                state.mode = 'FAULT'
                state.message = '直行超距/超时，已停车'
                publish_stop()
                return

            # 车头保持 y 方向：相对任务起点的 yaw 偏差 <= 5°
            yaw_offset = normalize_angle(state.odom[2] - state.straight_heading_offset)
            cmd.linear.x = STRAIGHT_SPEED

            if state.lane_valid:
                # 中线可见：heading 修正优先（小），中线指向相悖则只修 heading
                heading_error = state.heading_error_deg if state.heading_error_deg is not None else 0.0
                center_error = state.center_error if state.center_error is not None else 0.0
                # 若中线指向与当前前进方向相反，仅保持原 heading，等中线更新
                if abs(yaw_offset) <= math.radians(HEADING_TOL_DEG):
                    cmd.angular.z = CENTER_KP_Z_ALIGN * center_error
                else:
                    cmd.angular.z = 0.0
                cmd.linear.y = CENTER_KP_Y * center_error * 0.5
            else:
                # 中线丢失：沿历史方向缓行，角度限幅
                recent = state.straight_history_wz[-10:]
                avg_wz = float(np.mean(recent)) if recent else 0.0
                cmd.angular.z = max(-0.06, min(0.06, avg_wz))
                cmd.linear.y = 0.0
            cmd.angular.z = max(-WZ_CLAMP, min(WZ_CLAMP, cmd.angular.z))
            state.control_source = 'STRAIGHT_VISION' if state.lane_valid else 'STRAIGHT_LOST'

        # 记录角速度历史（直行丢失时用）
        state.straight_history_wz.append(float(cmd.angular.z))
        if len(state.straight_history_wz) > 10:
            state.straight_history_wz = state.straight_history_wz[-10:]

        state.command_linear_x = cmd.linear.x
        state.command_linear_y = cmd.linear.y
        state.command_angular_z = cmd.angular.z
        cmd_pub.publish(cmd)


PAGE = '''<!doctype html><meta charset="utf-8"><title>直行穿路口</title>
<style>body{font:16px sans-serif;background:#0f172a;color:#f8fafc;margin:20px}main{max-width:1100px;margin:auto}section{background:#1e293b;padding:16px;margin:12px 0;border-radius:10px}img{width:48%;background:#111;margin:1%;border-radius:6px}.start{background:#1677ff;color:white}.stop{background:#d00;color:white}button{padding:12px 22px;border:0;border-radius:6px;margin-right:10px;font-size:16px;cursor:pointer}pre{font-size:15px;background:#0f172a;padding:10px;border-radius:6px;color:#38bdf8}</style>
<main><h2>直行穿路口 + 白线停车 (Port 5008)</h2>
<section><b>默认不运动。确认场地清空并准备急停后再启动。</b>
<p>流程：平移对准路口对面中线 → 龟速直行(保持y方向≤5°) → 检测脚下白线 → 刹停。</p>
<p><button class="start" onclick="post('/api/start')">解锁并开始</button><button class="stop" onclick="post('/api/stop')">立即停车</button><button onclick="post('/api/reset')">复位为仅感知</button></p>
<pre id="status">加载状态中...</pre></section>
<section><h3>识别叠加图 (/straight_pass/debug/overlay) 与 二值图 (/straight_pass/debug/mask)</h3>
<img id="img_overlay"><img id="img_mask">
</section></main>
<script>
document.getElementById('img_overlay').src = '/stream/overlay';
document.getElementById('img_mask').src = '/stream/mask';
async function post(p){
    try {
        let res = await (await fetch(p,{method:'POST'})).json();
        if(!res.ok && res.error){ alert("⚠️ 无法启动: " + res.error); }
    } catch(e) { alert("⚠️ 请求失败: " + e); }
}
setInterval(async()=>{try{document.getElementById('status').textContent=JSON.stringify(await(await fetch('/api/status')).json(),null,2)}catch(e){}},400);
</script>'''


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def reply(self, obj):
        def converter(o):
            if isinstance(o, (np.integer, np.int32, np.int64)):
                return int(o)
            elif isinstance(o, (np.floating, np.float32, np.float64)):
                return float(o)
            elif isinstance(o, np.ndarray):
                return o.tolist()
            elif isinstance(o, np.bool_):
                return bool(o)
            raise TypeError(str(type(o)))
        data = json.dumps(obj, default=converter, ensure_ascii=False).encode('utf-8')
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
        with state.lock:
            if path == '/api/start':
                if not ALLOW_MOTION:
                    self.reply({'ok': False, 'error': '动作解锁开关未开启'})
                    return
                if state.mode not in ('DISARMED', 'FAULT', 'STOPPED'):
                    self.reply({'ok': False, 'error': '任务已经运行'})
                    return
                if state.odom is None:
                    self.reply({'ok': False, 'error': '里程计未就绪'})
                    return
                if time.time() - state.last_image_time > CAMERA_TIMEOUT:
                    self.reply({'ok': False, 'error': '相机超时'})
                    return
                if not state.lane_valid:
                    self.reply({'ok': False, 'error': '当前未看到路口对面车道中线，无法对准'})
                    return
                state.start_pose = state.odom
                state.align_good_frames = 0
                state.stop_hits = 0
                state.straight_history_wz = []
                begin_phase('ALIGN', '平移对准路口对面中线')
                rospy.loginfo('!!! 开始平移对准 !!!')
                self.reply({'ok': True})
            elif path == '/api/stop':
                state.mode = 'STOPPED'
                state.message = '用户手动停车'
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
    rospy.init_node('straight_intersection_pass', anonymous=False)
    init_ipm()
    load_hsv()

    cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
    mask_pub = rospy.Publisher('/straight_pass/debug/mask', Image, queue_size=1)
    overlay_pub = rospy.Publisher('/straight_pass/debug/overlay', Image, queue_size=1)

    rospy.Subscriber(IMAGE_TOPIC, Image, image_cb, queue_size=1, buff_size=2**24)
    rospy.Subscriber('/odom', Odometry, odom_cb, queue_size=1)
    rospy.Timer(rospy.Duration(0.05), control_timer)
    rospy.on_shutdown(shutdown)

    server = ReusableHTTPServer(('0.0.0.0', PORT), Handler)
    rospy.loginfo('直行穿路口程序已启动: http://0.0.0.0:%d', PORT)
    server.serve_forever()


if __name__ == '__main__':
    main()
