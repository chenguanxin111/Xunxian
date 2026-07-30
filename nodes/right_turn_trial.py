#!/usr/bin/env python3
"""Low-speed right-turn entry trial with IPM lane alignment.

The node starts DISARMED and publishes no velocity until /api/start is called.
It reaches the junction, searches the right exit, enters it on a slow arc,
aligns between two current-frame lane boundaries, and then stops.
"""
import json
import math
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

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
HSV_PATH = os.path.join(CONFIG_DIR, 'white_lane_right.json')
FALLBACK_HSV_PATH = os.path.join(CONFIG_DIR, 'white_lane.json')
PERSP_PATH = os.path.join(CONFIG_DIR, 'perspective_params.json')
IMAGE_TOPIC = '/usb_cam/image_raw'
PORT = 5001

ADVANCE_DISTANCE = 0.25
ADVANCE_MAX_DISTANCE = 0.30
ADVANCE_SPEED = 0.08
ADVANCE_TIMEOUT = 12.0
ROTATE_SPEED = 0.20
SEARCH_START_DEG = 28.0
ROTATE_LIMIT_DEG = 60.0
ROTATE_TIMEOUT = 15.0
CENTER_TOLERANCE_PX = 120
EDGE_CONFIRM_FRAMES = 4

# 右出口驶入阶段：保守里程计弧线后，由当前帧视觉中线接管。
ENTRY_ARC_SPEED = 0.06
ENTRY_ARC_BLIND_SPEED = 0.06
# 视觉引导尚未建立时只允许保守右转，避免固定追 88 度导致过头。
ENTRY_ARC_BLIND_TARGET_DEG = 68.0
ENTRY_ARC_BLIND_STOP_DEG = 72.0
ENTRY_ARC_MAX_DISTANCE = 0.45
ENTRY_ARC_TIMEOUT = 15.0
ENTRY_ARC_MAX_TURN_DEG = 95.0
DUAL_EDGE_CONFIRM_FRAMES = 8
DUAL_ACCEPT_MIN_TURN_DEG = 65.0

# 双边线对准阶段：只用当前帧真实左右边界，不使用历史中心线判成功。
ALIGN_SPEED = 0.06
ALIGN_MAX_DISTANCE = 0.35
ALIGN_TIMEOUT = 10.0
ALIGN_MIN_DISTANCE = 0.12
ALIGN_HEADING_TOL_DEG = 8.0
ALIGN_CENTER_TOL_PX = 45.0
ALIGN_CONFIRM_FRAMES = 10
ENTRY_GUIDE_MIN_DISTANCE = 0.15
ENTRY_GUIDE_CONFIRM_FRAMES = 8
ROW_GUIDE_MIN_POINTS = 10
ROW_GUIDE_CENTER_TOL_PX = 50.0
ROW_GUIDE_HEADING_TARGET_DEG = 0.0
ROW_GUIDE_HEADING_TOL_DEG = 15.0
ROW_GUIDE_CONFIRM_FRAMES = 6
ROW_GUIDE_ACCEPT_MIN_TURN_DEG = 50.0

# IPM 的目标图宽为 600，标定目标车道宽约 400 px；先使用宽容范围。
IPM_CENTER_X = 300.0
IPM_LANE_WIDTH_MIN = 220.0
IPM_LANE_WIDTH_MAX = 520.0
IPM_LANE_HALF_WIDTH = 200.0  # 400 px ~= 实测 42 cm，半宽约 21 cm

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


def normalize_angle(value):
    return math.atan2(math.sin(value), math.cos(value))


class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.mode = 'DISARMED'
        self.message = '仅感知模式：确认安全后在网页启动'
        self.odom = None
        self.start_pose = None
        self.state_started = time.time()
        self.center_error = None
        self.corners = []
        self.right_edge_frames = 0
        self.right_edge_found = False
        self.distance = 0.0
        self.turn_deg = 0.0
        self.last_image_time = 0.0
        self.hsv_params = dict(DEFAULT_PARAMS)
        self.last_center_points = []
        self.last_center_error = 0.0
        self.left_slope = None
        self.right_slope = None
        self.slope_diff = None
        self.virtual_center_error = None
        self.entry_start_dist = 0.0
        self.phase_start_pose = None
        self.phase_distance = 0.0
        self.dual_lane_frames = 0
        self.align_good_frames = 0
        self.dual_lane_valid = False
        self.lane_center_error = None
        self.lane_heading_error_deg = None
        self.lane_width_ipm = None
        self.entry_guide_valid = False
        self.entry_guide_confirmed = False
        self.entry_guide_center_error = None
        self.entry_guide_heading_error_deg = None
        self.entry_guide_good_frames = 0
        self.row_guide_valid = False
        self.row_guide_center_error = None
        self.row_guide_heading_error_deg = None
        self.row_guide_points = 0
        self.row_guide_good_frames = 0
        self.control_source = 'STOPPED'
        self.command_linear_x = 0.0
        self.command_angular_z = 0.0

    def status(self):
        return {
            'mode': self.mode,
            'message': self.message,
            'center_error_px': self.center_error,
            'virtual_center_error_px': round(self.virtual_center_error, 1) if self.virtual_center_error is not None else None,
            'corners': self.corners,
            'distance_m': round(self.distance, 3),
            'right_turn_deg': round(self.turn_deg, 1),
            'right_edge_confirm': self.right_edge_frames,
            'right_edge_found': self.right_edge_found,
            'ipm_left_slope': round(self.left_slope, 3) if self.left_slope is not None else None,
            'ipm_right_slope': round(self.right_slope, 3) if self.right_slope is not None else None,
            'ipm_slope_diff': round(self.slope_diff, 3) if self.slope_diff is not None else None,
            'phase_distance_m': round(self.phase_distance, 3),
            'dual_lane_confirm': self.dual_lane_frames,
            'dual_lane_valid': self.dual_lane_valid,
            'lane_center_error_ipm': round(self.lane_center_error, 1) if self.lane_center_error is not None else None,
            'lane_heading_error_deg': round(self.lane_heading_error_deg, 1) if self.lane_heading_error_deg is not None else None,
            'lane_width_ipm': round(self.lane_width_ipm, 1) if self.lane_width_ipm is not None else None,
            'align_good_frames': self.align_good_frames,
            'entry_guide_valid': self.entry_guide_valid,
            'entry_guide_confirmed': self.entry_guide_confirmed,
            'entry_guide_center_error_ipm': round(self.entry_guide_center_error, 1) if self.entry_guide_center_error is not None else None,
            'entry_guide_heading_error_deg': round(self.entry_guide_heading_error_deg, 1) if self.entry_guide_heading_error_deg is not None else None,
            'entry_guide_good_frames': self.entry_guide_good_frames,
            'row_guide_valid': self.row_guide_valid,
            'row_guide_center_error_px': round(self.row_guide_center_error, 1) if self.row_guide_center_error is not None else None,
            'row_guide_heading_error_deg': round(self.row_guide_heading_error_deg, 1) if self.row_guide_heading_error_deg is not None else None,
            'row_guide_points': self.row_guide_points,
            'row_guide_good_frames': self.row_guide_good_frames,
            'control_source': self.control_source,
            'command_linear_x': round(self.command_linear_x, 3),
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
                rospy.loginfo("成功载入鸟瞰图 (IPM) 正向及反向矩阵: %s", PERSP_PATH)
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
    else:
        rospy.loginfo("未找到 HSV 配置文件，使用默认白色车道线参数")


def get_contour_slope_ipm(contour):
    """将轮廓转换至鸟瞰图 (IPM) 下拟合斜率 dx/dy，用于平行线判断"""
    if contour is None or len(contour) < 5 or IPM_MATRIX is None:
        return None
    try:
        pts = contour.reshape(-1, 1, 2).astype(np.float32).copy()
        # 由于图像已被 cv2.flip(frame, 1) 翻转，需还原回 IPM 矩阵对应的原始相机坐标系
        pts[:, 0, 0] = 639.0 - pts[:, 0, 0]
        ipm_pts = cv2.perspectiveTransform(pts, IPM_MATRIX).reshape(-1, 2)
        y_coords = ipm_pts[:, 1]
        x_coords = ipm_pts[:, 0]
        if np.max(y_coords) - np.min(y_coords) < 10:
            return None
        # 拟合 x = k * y + b (计算纵向 Y 轴对应的横向 X 斜率)
        slope, _ = np.polyfit(y_coords, x_coords, 1)
        return float(slope)
    except Exception as e:
        rospy.logwarn_throttle(2, f"IPM 斜率计算异常: {e}")
        return None


def get_virtual_extended_midline(right_edge, mask_shape):
    """
    在鸟瞰图 (IPM) 地面物理坐标系下拟合右边界并向下外推延伸至小车脚下 (Y_ipm=600)，
    扣除固定物理车道宽度 (200px中线偏移) 后，利用逆矩阵 IPM_INV_MATRIX
    反向投影回相机 2D 图像视角绘制符合近大远小透视效果的绿色虚线中线。
    """
    if right_edge is None or len(right_edge) < 5 or IPM_MATRIX is None or IPM_INV_MATRIX is None:
        return None, [], []
    
    h, w = mask_shape
    try:
        # 1. 还原翻转前相机原始坐标，转换为鸟瞰图 (IPM) 坐标 (dst_points: width=600, height=600)
        pts_raw = right_edge.reshape(-1, 1, 2).astype(np.float32).copy()
        pts_raw[:, 0, 0] = 639.0 - pts_raw[:, 0, 0]
        pts_ipm = cv2.perspectiveTransform(pts_raw, IPM_MATRIX).reshape(-1, 2)
        
        ys_ipm = pts_ipm[:, 1]
        xs_ipm = pts_ipm[:, 0]
        
        if np.max(ys_ipm) - np.min(ys_ipm) < 15:
            return None, [], []
        
        # 2. 在鸟瞰图物理坐标系中拟合直线 X_ipm = k_ipm * Y_ipm + b_ipm
        k_ipm, b_ipm = np.polyfit(ys_ipm, xs_ipm, 1)
        
        # 鸟瞰图中，车道左右宽度固定为 400px (对应 dst_pts 中 100 到 500)，中线偏移为 -200px
        IPM_MIDLINE_OFFSET_PX = 200.0
        
        y_min_ipm = float(np.min(ys_ipm))
        y_max_ipm = 595.0  # 鸟瞰图最底部 (小车正前方地面)
        
        y_steps = np.linspace(y_min_ipm, y_max_ipm, num=30)
        
        ipm_mid_pts = []
        for y in y_steps:
            x_right_ipm = k_ipm * y + b_ipm
            x_mid_ipm = x_right_ipm - IPM_MIDLINE_OFFSET_PX
            ipm_mid_pts.append([x_mid_ipm, y])
            
        ipm_mid_pts = np.array(ipm_mid_pts, dtype=np.float32).reshape(-1, 1, 2)
        
        # 3. 计算鸟瞰图近处 (Y_ipm 靠近 600) 的像素偏差 (IPM 鸟瞰图中心为 X=300)
        near_ipm_xs = [p[0, 0] for p in ipm_mid_pts if p[0, 1] >= 450.0]
        if len(near_ipm_xs) == 0:
            near_ipm_xs = [p[0, 0] for p in ipm_mid_pts]
        ipm_near_error = float(np.mean(near_ipm_xs) - 300.0)
        
        # 4. 利用逆矩阵 IPM_INV_MATRIX 将鸟瞰图中线反向投影回相机原始视角
        pts_raw_back = cv2.perspectiveTransform(ipm_mid_pts, IPM_INV_MATRIX).reshape(-1, 2)
        
        # 重新镜像 X 坐标以匹配 flipped_frame
        overlay_pts = []
        for p in pts_raw_back:
            ox = int(round(639.0 - p[0]))
            oy = int(round(p[1]))
            # 过滤掉离开图像视角或非常荒谬的点
            if -100 <= ox <= w + 100 and -100 <= oy <= h + 100:
                overlay_pts.append((ox, oy))
                
        # 5. 生成虚线绘制段 (Dashed Line Segments)
        dashed_segments = []
        for i in range(0, len(overlay_pts) - 1, 2):
            dashed_segments.append((overlay_pts[i], overlay_pts[i + 1]))
            
        return ipm_near_error, overlay_pts, dashed_segments

    except Exception as err:
        rospy.logwarn_throttle(2, f"IPM 虚拟延伸计算异常: {err}")
        return None, [], []


def contour_ipm_fit(contour, min_y_span=55.0):
    """Fit x = k*y+b for one current-frame marking in the IPM plane."""
    if contour is None or len(contour) < 5 or IPM_MATRIX is None:
        return None
    pts = contour.reshape(-1, 1, 2).astype(np.float32).copy()
    pts[:, 0, 0] = 639.0 - pts[:, 0, 0]
    ipm = cv2.perspectiveTransform(pts, IPM_MATRIX).reshape(-1, 2)
    ipm = ipm[np.isfinite(ipm).all(axis=1)]
    if len(ipm) < 5:
        return None
    # Keep a generous calibrated ground region while rejecting projection blow-up.
    ipm = ipm[(ipm[:, 0] > -250) & (ipm[:, 0] < 850) &
              (ipm[:, 1] > -100) & (ipm[:, 1] < 700)]
    if len(ipm) < 5 or np.ptp(ipm[:, 1]) < min_y_span:
        return None
    k, b = np.polyfit(ipm[:, 1], ipm[:, 0], 1)
    return {
        'k': float(k), 'b': float(b),
        'y_min': float(np.min(ipm[:, 1])),
        'y_max': float(np.max(ipm[:, 1])),
        'contour': contour,
    }


def build_entry_guide_ipm(contours, mask_shape):
    """Build a full center guide from the long physical-left exit marking.

    The image is mirrored for display, but contour_ipm_fit restores original
    camera coordinates before IPM. Therefore the physical-left marking is the
    fit whose projected x is smaller. A short physical-right marking confirms
    its identity and lane width; the long left fit supplies the stable heading.
    """
    if IPM_MATRIX is None or IPM_INV_MATRIX is None:
        return None

    fits = [fit for fit in
            (contour_ipm_fit(c, min_y_span=18.0) for c in contours[:8]) if fit]
    for fit in fits:
        fit['span'] = fit['y_max'] - fit['y_min']

    best = None
    for left_fit in fits:
        if left_fit['span'] < 90.0:
            continue
        # Evaluate near the visible end, without extrapolating for identity.
        left_ref_y = min(left_fit['y_max'], 560.0)
        left_ref_x = left_fit['k'] * left_ref_y + left_fit['b']
        if left_ref_x > IPM_CENTER_X + 80.0:
            continue

        for right_fit in fits:
            if right_fit is left_fit or right_fit['span'] < 18.0:
                continue
            overlap_min = max(left_fit['y_min'], right_fit['y_min'], 20.0)
            overlap_max = min(left_fit['y_max'], right_fit['y_max'], 590.0)
            if overlap_max - overlap_min < 12.0:
                continue
            y_check = (overlap_min + overlap_max) / 2.0
            left_x = left_fit['k'] * y_check + left_fit['b']
            right_x = right_fit['k'] * y_check + right_fit['b']
            width = right_x - left_x
            slope_diff = abs(left_fit['k'] - right_fit['k'])
            if not (IPM_LANE_WIDTH_MIN <= width <= IPM_LANE_WIDTH_MAX):
                continue
            if slope_diff > 0.55:
                continue
            # Prefer a long left marking, then a plausible 42 cm lane width.
            score = (left_fit['span'] / 500.0 -
                     abs(width - 2.0 * IPM_LANE_HALF_WIDTH) / 500.0 -
                     slope_diff * 0.4)
            if best is None or score > best['score']:
                best = {
                    'score': score,
                    'left_fit': left_fit,
                    'right_fit': right_fit,
                    'measured_width': float(width),
                    'slope_diff': float(slope_diff),
                }

    if best is None:
        return None

    left_fit = best['left_fit']
    # For x=k*y+b, this intercept shift offsets the parallel line by the
    # requested perpendicular distance in the metric-like IPM plane.
    center_k = left_fit['k']
    center_b = (left_fit['b'] +
                IPM_LANE_HALF_WIDTH * math.sqrt(1.0 + center_k * center_k))
    y_near = 585.0
    y_far = max(40.0, min(left_fit['y_min'], best['right_fit']['y_min']))
    center_near = center_k * y_near + center_b
    heading_deg = math.degrees(math.atan2(-center_k, 1.0))

    y_values = np.linspace(y_far, y_near, 36)
    center_ipm = np.float32([
        [center_k * y + center_b, y] for y in y_values
    ]).reshape(-1, 1, 2)
    raw_points = cv2.perspectiveTransform(center_ipm, IPM_INV_MATRIX).reshape(-1, 2)
    h, w = mask_shape
    overlay_points = []
    for x, y in raw_points:
        point = (int(round(639.0 - x)), int(round(y)))
        if -80 <= point[0] < w + 80 and 0 <= point[1] < h:
            overlay_points.append(point)

    best.update({
        'k': float(center_k),
        'b': float(center_b),
        'center_error': float(center_near - IPM_CENTER_X),
        'heading_error_deg': float(heading_deg),
        'overlay_points': overlay_points,
    })
    return best


def detect_dual_lane_ipm(contours, mask_shape):
    """Find two current-frame parallel boundaries and their actual centerline."""
    if IPM_MATRIX is None or IPM_INV_MATRIX is None:
        return None
    fits = [fit for fit in (contour_ipm_fit(c) for c in contours[:6]) if fit]
    best = None
    for i in range(len(fits)):
        for j in range(i + 1, len(fits)):
            first, second = fits[i], fits[j]
            overlap_min = max(first['y_min'], second['y_min'], 80.0)
            overlap_max = min(first['y_max'], second['y_max'], 590.0)
            if overlap_max - overlap_min < 70:
                continue
            y_near = overlap_max
            y_far = max(overlap_min, y_near - 160.0)
            f_near = first['k'] * y_near + first['b']
            s_near = second['k'] * y_near + second['b']
            f_far = first['k'] * y_far + first['b']
            s_far = second['k'] * y_far + second['b']
            width_near = abs(s_near - f_near)
            width_far = abs(s_far - f_far)
            slope_diff = abs(first['k'] - second['k'])
            if not (IPM_LANE_WIDTH_MIN <= width_near <= IPM_LANE_WIDTH_MAX):
                continue
            if not (IPM_LANE_WIDTH_MIN <= width_far <= IPM_LANE_WIDTH_MAX):
                continue
            if slope_diff > 0.30 or abs(width_near - width_far) > 90:
                continue
            score = slope_diff + abs((width_near + width_far) / 2.0 - 400.0) / 400.0
            if best is None or score < best['score']:
                k_center = (first['k'] + second['k']) / 2.0
                b_center = (first['b'] + second['b']) / 2.0
                # Forward travel in the IPM plane is toward decreasing y.
                heading_deg = math.degrees(math.atan2(-k_center, 1.0))
                center_near = k_center * y_near + b_center
                best = {
                    'score': score,
                    'left_fit': first if f_near < s_near else second,
                    'right_fit': second if f_near < s_near else first,
                    'k': k_center, 'b': b_center,
                    'center_error': float(center_near - IPM_CENTER_X),
                    'heading_error_deg': float(heading_deg),
                    'lane_width': float((width_near + width_far) / 2.0),
                    'y_min': y_far, 'y_max': y_near,
                }
    if best is None:
        return None

    y_values = np.linspace(best['y_min'], best['y_max'], 24)
    ipm_center = np.float32([
        [best['k'] * y + best['b'], y] for y in y_values
    ]).reshape(-1, 1, 2)
    raw_center = cv2.perspectiveTransform(ipm_center, IPM_INV_MATRIX).reshape(-1, 2)
    h, w = mask_shape
    overlay_points = []
    for x, y in raw_center:
        point = (int(round(639.0 - x)), int(round(y)))
        if 0 <= point[0] < w and 0 <= point[1] < h:
            overlay_points.append(point)
    best['overlay_points'] = overlay_points
    return best


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


def lane_midline(mask, x_start, x_end, step=6):
    """Sample a thick lane marking as one median-y point per x bin."""
    points = []
    for x in range(x_start, x_end, step):
        ys, xs = np.where(mask[:, x:min(x + step, x_end)] > 0)
        if len(ys) >= 3:
            points.append((int(x + np.median(xs)), int(np.median(ys))))
    return points


def polyline_corner(points, width):
    """Return the strongest internal bend after simplifying a lane midline."""
    if len(points) < 3:
        return None
    curve = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
    approx = cv2.approxPolyDP(curve, 7.0, False)
    pts = [tuple(p[0]) for p in approx]
    if len(pts) < 3:
        return None
    best = None
    for i in range(1, len(pts) - 1):
        if pts[i][0] < 20 or pts[i][0] > width - 21:
            continue
        a = np.array(pts[i - 1], dtype=float) - pts[i]
        b = np.array(pts[i + 1], dtype=float) - pts[i]
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom < 1:
            continue
        angle = math.degrees(math.acos(np.clip(np.dot(a, b) / denom, -1, 1)))
        bend = 180.0 - angle
        if best is None or bend > best[0]:
            best = (bend, pts[i])
    return best[1] if best and best[0] > 18 else None


def current_frame_row_guide(points, image_width):
    """Measure the current two-edge centerline using last year's row scan idea."""
    if len(points) < ROW_GUIDE_MIN_POINTS:
        return None

    near_count = min(5, len(points))
    far_count = min(5, len(points))
    near_x = float(np.mean([p[0] for p in points[:near_count]]))
    far_x = float(np.mean([p[0] for p in points[-far_count:]]))
    near_y = float(np.mean([p[1] for p in points[:near_count]]))
    far_y = float(np.mean([p[1] for p in points[-far_count:]]))
    forward_pixels = near_y - far_y
    if forward_pixels < 45.0:
        return None

    # Positive heading means the lane heads toward display-right. The frame is
    # mirrored, so control below explicitly accounts for this display sign.
    heading_deg = math.degrees(math.atan2(far_x - near_x, forward_pixels))
    return {
        'points': list(points),
        'point_count': len(points),
        'center_error': near_x - image_width / 2.0,
        'heading_error_deg': heading_deg,
    }


def analyze_lanes(mask):
    h, w = mask.shape
    result = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = result[0] if len(result) == 2 else result[1]
    contours = [c for c in contours if cv2.contourArea(c) > 150 and cv2.arcLength(c, False) > 60]
    contours.sort(key=lambda c: cv2.contourArea(c), reverse=True)

    left = [c for c in contours if cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2] / 2 < w / 2]
    right = [c for c in contours if cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2] / 2 >= w / 2]
    left_c = left[0] if left else None
    right_c = right[0] if right else None

    # 构建仅包含主要主车道线轮廓的干净掩膜 clean_mask，防止中线在噪声处产生抖动/凸起
    clean_mask = np.zeros_like(mask)
    if left_c is not None:
        cv2.drawContours(clean_mask, [left_c], -1, 255, -1)
    if right_c is not None:
        cv2.drawContours(clean_mask, [right_c], -1, 255, -1)

    left_midline = lane_midline(clean_mask, 0, w // 2)
    right_midline = lane_midline(clean_mask, w // 2, w)
    corners = [point for point in
               (polyline_corner(left_midline, w), polyline_corner(right_midline, w))
               if point is not None]

    centers = []
    edge_samples = []
    for y in range(h - 1, int(h * 0.45), -8):
        xs = np.flatnonzero(clean_mask[y] > 0)
        if len(xs) == 0:
            continue
        groups = np.split(xs, np.where(np.diff(xs) > 2)[0] + 1)
        means = [int(np.mean(g)) for g in groups if len(g) >= 2]
        left_x = max((x for x in means if x < w / 2), default=None)
        right_x = min((x for x in means if x >= w / 2), default=None)
        if left_x is not None and right_x is not None and 100 < right_x - left_x < w * 0.9:
            cand_x = int((left_x + right_x) / 2)
            # 【借鉴去年逻辑 1：中线跳变约束】
            # 新采样的中线点 X 坐标与上一个中线点的横向偏差不能超过 40 像素
            if len(centers) == 0 or abs(cand_x - centers[-1][0]) < 40:
                centers.append((cand_x, y))
                edge_samples.append(((left_x, y), (right_x, y)))

    # Preserve current-frame geometry before the diagnostic history fallback.
    current_centers = list(centers)
    row_guide = current_frame_row_guide(current_centers, w)

    # 【借鉴去年逻辑 2：历史中线记忆】
    center_error = None
    if len(centers) >= 3:
        near = centers[:min(5, len(centers))]
        center_error = float(np.mean([p[0] for p in near]) - w / 2)
        state.last_center_points = list(centers)
        state.last_center_error = center_error
    elif len(state.last_center_points) >= 3:
        centers = list(state.last_center_points)
        center_error = state.last_center_error

    # 【方案 A：鸟瞰图 (IPM) 平行斜率匹配】
    left_slope = get_contour_slope_ipm(left_c)
    right_slope = get_contour_slope_ipm(right_c)
    slope_diff = None

    right_edge = None
    if right_c is not None:
        rx, ry, rcw, rch = cv2.boundingRect(right_c)
        rlength = cv2.arcLength(right_c, False)
        
        # 默认不认可 (is_parallel = False)，只有通过鸟瞰图 IPM 正向斜率校验才认定为合法右出口车道线
        is_parallel = False
        if left_slope is not None and right_slope is not None:
            slope_diff = abs(left_slope - right_slope)
            # 鸟瞰图中真正的右出口车道线必须指向前方/右侧 (right_slope > 0.05) 且双线平行 (slope_diff < 0.25)
            if slope_diff < 0.25 and right_slope > 0.05:
                is_parallel = True
        elif right_slope is not None:
            # 单线情况：鸟瞰图中真正的右出口车道线必须指向前方/右侧 (right_slope > 0.05)
            # 截图中测出的 V 型路口内侧尖角线斜率为 -0.144 / -0.181 (均 <= 0.05)，将被 100% 拒绝！
            if right_slope > 0.05:
                is_parallel = True
        
        if (rx + rcw / 2 > w * 0.45 and rlength > 80 and is_parallel):
            right_edge = right_c

    entry_guide = build_entry_guide_ipm(contours, mask.shape)
    dual_lane = detect_dual_lane_ipm(contours, mask.shape)

    return (left_c, right_c, left_midline, right_midline, corners, centers,
            edge_samples, center_error, right_edge, left_slope, right_slope, slope_diff,
            row_guide, entry_guide, dual_lane)


def image_cb(msg):
    try:
        frame = bridge.imgmsg_to_cv2(msg, 'passthrough')
        if msg.encoding.lower() == 'rgb8':
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        elif msg.encoding.lower() != 'bgr8':
            frame = bridge.imgmsg_to_cv2(msg, 'bgr8')

        # 进行水平镜像翻转，与去年的 xunxian.py 保持完全一致
        frame = cv2.flip(frame, 1)

        with state.lock:
            hsv_p = dict(state.hsv_params)

        mask, roi = make_mask(frame, hsv_p)
        (left, right, left_midline, right_midline, corners, centers, samples,
         error, right_edge, left_slope, right_slope, slope_diff,
         row_guide, entry_guide, dual_lane) = analyze_lanes(mask)

        overlay = frame.copy()
        cv2.rectangle(overlay, (roi[0], roi[1]), (roi[2] - 1, roi[3] - 1), (255, 120, 0), 2)
        for contour, color in ((left, (0, 140, 255)), (right, (255, 180, 0))):
            if contour is not None:
                cv2.drawContours(overlay, [contour], -1, color, 2)
        for midline, color in ((left_midline, (0, 80, 255)),
                               (right_midline, (255, 80, 0))):
            if len(midline) > 1:
                cv2.polylines(overlay, [np.array(midline, np.int32)], False, color, 2)
        for first, second in samples:
            cv2.circle(overlay, first, 2, (0, 0, 255), -1)
            cv2.circle(overlay, second, 2, (0, 0, 255), -1)
        if len(centers) > 1:
            cv2.polylines(overlay, [np.array(centers, np.int32)], False, (0, 255, 0), 3)
        for point in corners:
            cv2.circle(overlay, point, 9, (255, 0, 255), 3)

        if row_guide is not None:
            cv2.polylines(overlay, [np.array(row_guide['points'], np.int32)],
                          False, (0, 255, 0), 5)
            cv2.putText(overlay, 'CURRENT ROW-SCAN CENTER', (220, 105),
                        cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 2)

        if entry_guide is not None:
            points = entry_guide['overlay_points']
            for index in range(0, len(points) - 1, 2):
                cv2.line(overlay, points[index], points[index + 1], (0, 255, 0), 4)
            cv2.drawContours(overlay, [entry_guide['left_fit']['contour']],
                             -1, (0, 255, 255), 4)
            cv2.drawContours(overlay, [entry_guide['right_fit']['contour']],
                             -1, (255, 0, 255), 3)
            cv2.putText(overlay, 'ENTRY GUIDE (LONG LEFT + SHORT RIGHT)',
                        (175, 80), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 2)
        
        if dual_lane is not None:
            cv2.drawContours(overlay, [dual_lane['left_fit']['contour']], -1, (255, 0, 255), 3)
            cv2.drawContours(overlay, [dual_lane['right_fit']['contour']], -1, (255, 0, 255), 3)
            if len(dual_lane['overlay_points']) > 1:
                cv2.polylines(overlay, [np.array(dual_lane['overlay_points'], np.int32)],
                              False, (255, 255, 0), 4)
            cv2.putText(overlay, 'CURRENT DUAL-LANE CENTER', (230, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 0), 2)

        if right_edge is not None:
            cv2.drawContours(overlay, [right_edge], -1, (0, 255, 255), 3)
            cv2.putText(overlay, 'RIGHT EDGE CANDIDATE', (300, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 255, 255), 2)

        with state.lock:
            state.last_image_time = time.time()
            state.center_error = error
            state.corners = [[int(x), int(y)] for x, y in corners]
            state.left_slope = left_slope
            state.right_slope = right_slope
            state.slope_diff = slope_diff
            state.virtual_center_error = None
            state.dual_lane_valid = dual_lane is not None
            state.lane_center_error = dual_lane['center_error'] if dual_lane else None
            state.lane_heading_error_deg = dual_lane['heading_error_deg'] if dual_lane else None
            state.lane_width_ipm = dual_lane['lane_width'] if dual_lane else None
            state.entry_guide_valid = entry_guide is not None
            state.entry_guide_confirmed = entry_guide is not None
            state.entry_guide_center_error = entry_guide['center_error'] if entry_guide else None
            state.entry_guide_heading_error_deg = entry_guide['heading_error_deg'] if entry_guide else None
            state.row_guide_valid = row_guide is not None
            state.row_guide_center_error = row_guide['center_error'] if row_guide else None
            state.row_guide_heading_error_deg = row_guide['heading_error_deg'] if row_guide else None
            state.row_guide_points = row_guide['point_count'] if row_guide else 0

            if state.mode == 'ENTER_LANE_ARC':
                guide_aligned = (entry_guide is not None and
                                 abs(entry_guide['center_error']) <= ALIGN_CENTER_TOL_PX and
                                 abs(entry_guide['heading_error_deg']) <= ALIGN_HEADING_TOL_DEG)
                state.entry_guide_good_frames = (state.entry_guide_good_frames + 1
                                                 if guide_aligned else 0)
                row_heading_residual = (row_guide['heading_error_deg'] -
                                        ROW_GUIDE_HEADING_TARGET_DEG
                                        if row_guide is not None else None)
                row_aligned = (state.turn_deg >= ROW_GUIDE_ACCEPT_MIN_TURN_DEG and
                               row_guide is not None and
                               abs(row_guide['center_error']) <= ROW_GUIDE_CENTER_TOL_PX and
                               abs(row_heading_residual) <= ROW_GUIDE_HEADING_TOL_DEG)
                state.row_guide_good_frames = (state.row_guide_good_frames + 1
                                               if row_aligned else 0)
            
            if state.mode == 'SEARCH_RIGHT' and state.turn_deg >= SEARCH_START_DEG:
                # 严密判定：既要找到合格且通过 IPM 斜率校验的右出口车道线 (right_edge is not None)，
                # 也要同时提取出稳定的绿色中线 (len(centers) >= 3)
                has_valid_right_exit = (right_edge is not None) and (len(centers) >= 3)
                if has_valid_right_exit:
                    state.right_edge_frames += 1
                else:
                    state.right_edge_frames = max(0, state.right_edge_frames - 1)
                state.right_edge_found = (state.right_edge_frames >= EDGE_CONFIRM_FRAMES)

            if state.mode == 'ENTER_LANE_ARC':
                if (state.turn_deg >= DUAL_ACCEPT_MIN_TURN_DEG and
                        dual_lane is not None):
                    state.dual_lane_frames += 1
                else:
                    state.dual_lane_frames = 0

            if state.mode == 'ALIGN_BETWEEN_LINES':
                if dual_lane is not None:
                    state.dual_lane_frames += 1
                else:
                    state.dual_lane_frames = max(0, state.dual_lane_frames - 1)

            if state.mode == 'ALIGN_BETWEEN_LINES':
                aligned = (dual_lane is not None and
                           abs(dual_lane['center_error']) <= ALIGN_CENTER_TOL_PX and
                           abs(dual_lane['heading_error_deg']) <= ALIGN_HEADING_TOL_DEG)
                state.align_good_frames = state.align_good_frames + 1 if aligned else 0

            status = state.status()

        cv2.putText(overlay, 'STATE: %s' % status['mode'], (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 255, 255), 2)
        cv2.putText(overlay, 'center=%s px  dist=%.3f m  turn=%.1f deg  ipm_diff=%s' %
                    (str(status['center_error_px']) if status['center_error_px'] is not None else 'N/A',
                     status['distance_m'], status['right_turn_deg'],
                     str(status['ipm_slope_diff']) if status['ipm_slope_diff'] is not None else 'N/A'),
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, .50, (0, 255, 255), 2)

        mask_pub.publish(safe_cv2_to_imgmsg(mask, 'mono8'))
        overlay_pub.publish(safe_cv2_to_imgmsg(overlay, 'bgr8'))
    except Exception as exc:
        import traceback
        rospy.logwarn_throttle(2, f'right turn image processing error: {exc}\n{traceback.format_exc()}')


def odom_cb(msg):
    q = msg.pose.pose.orientation
    yaw = euler_from_quaternion((q.x, q.y, q.z, q.w))[2]
    with state.lock:
        state.odom = (msg.pose.pose.position.x, msg.pose.pose.position.y, yaw)
        if state.start_pose:
            dx = state.odom[0] - state.start_pose[0]
            dy = state.odom[1] - state.start_pose[1]
            state.distance = math.hypot(dx, dy)
            state.turn_deg = max(0.0, -math.degrees(normalize_angle(yaw - state.start_pose[2])))
        if state.phase_start_pose:
            phase_dx = state.odom[0] - state.phase_start_pose[0]
            phase_dy = state.odom[1] - state.phase_start_pose[1]
            state.phase_distance = math.hypot(phase_dx, phase_dy)


def publish_stop():
    state.control_source = 'STOPPED'
    state.command_linear_x = 0.0
    state.command_angular_z = 0.0
    if cmd_pub is not None:
        cmd_pub.publish(Twist())


def fail_locked(message):
    state.mode = 'FAULT'
    state.message = message
    publish_stop()


def begin_phase_locked(mode, message, now):
    """Begin a motion phase with an independent odometry distance origin."""
    state.mode = mode
    state.message = message
    state.state_started = now
    state.phase_start_pose = state.odom
    state.phase_distance = 0.0
    publish_stop()


def control_timer(_event):
    with state.lock:
        now = time.time()
        mode = state.mode
        if mode not in ('ADVANCE', 'SEARCH_RIGHT', 'ENTER_LANE_ARC',
                        'ALIGN_BETWEEN_LINES'):
            return
        if state.odom is None or now - state.last_image_time > 0.6:
            fail_locked('里程计不可用或相机超时，已停车')
            return
        cmd = Twist()
        if mode == 'ADVANCE':
            if state.distance >= ADVANCE_DISTANCE:
                state.mode = 'SEARCH_RIGHT'
                state.state_started = now
                state.message = '缓慢右转并搜索右出口边界'
                publish_stop()
                return
            if state.distance > ADVANCE_MAX_DISTANCE or now - state.state_started > ADVANCE_TIMEOUT:
                fail_locked('前进距离或时间超过安全限制')
                return
            cmd.linear.x = ADVANCE_SPEED
            yaw_error = normalize_angle(state.start_pose[2] - state.odom[2])
            cmd.angular.z = max(-0.08, min(0.08, 0.8 * yaw_error))
            state.control_source = 'ADVANCE_ODOM'
        elif mode == 'SEARCH_RIGHT':
            if state.right_edge_found:
                state.dual_lane_frames = 0
                state.align_good_frames = 0
                begin_phase_locked('ENTER_LANE_ARC',
                                   '已锁定右出口，低速弧线继续驶入并寻找真实双边线', now)
                return
            if state.turn_deg >= ROTATE_LIMIT_DEG or now - state.state_started > ROTATE_TIMEOUT:
                fail_locked('达到60度/超时仍未稳定找到右边界')
                return
            cmd.angular.z = -ROTATE_SPEED
            state.control_source = 'SEARCH_RIGHT_ODOM'
        elif mode == 'ENTER_LANE_ARC':
            # Completion checks must run before timeout checks. Otherwise a
            # frame that has just become valid at 12 s would be mislabeled as
            # a timeout even though the vehicle is already in the lane.
            if (state.phase_distance >= ENTRY_GUIDE_MIN_DISTANCE and
                    state.row_guide_good_frames >= ROW_GUIDE_CONFIRM_FRAMES):
                state.mode = 'ENTERED_STOPPED'
                state.message = '当前帧双边中线已稳定确认，已驶入右转车道并停车'
                publish_stop()
                return

            if (state.phase_distance >= ENTRY_GUIDE_MIN_DISTANCE and
                    state.entry_guide_good_frames >= ENTRY_GUIDE_CONFIRM_FRAMES):
                state.mode = 'ENTERED_STOPPED'
                state.message = 'IPM驶入引导已稳定确认，已驶入右转车道并停车'
                publish_stop()
                return

            # Only accept a dual-lane pair after most of the right turn; this
            # prevents the entrance pair from triggering an early handover.
            if (state.turn_deg >= DUAL_ACCEPT_MIN_TURN_DEG and
                    state.dual_lane_frames >= DUAL_EDGE_CONFIRM_FRAMES):
                state.align_good_frames = 0
                begin_phase_locked('ALIGN_BETWEEN_LINES',
                                   '真实双边线已确认，沿当前帧车道中线低速对准', now)
                return
            if (state.phase_distance > ENTRY_ARC_MAX_DISTANCE or
                    state.turn_deg > ENTRY_ARC_MAX_TURN_DEG or
                    now - state.state_started > ENTRY_ARC_TIMEOUT):
                fail_locked('弧线驶入超距/超角/超时，已停车')
                return

            if (state.turn_deg >= ROW_GUIDE_ACCEPT_MIN_TURN_DEG and
                    state.row_guide_valid):
                center_error = state.row_guide_center_error
                heading_raw = state.row_guide_heading_error_deg
                if center_error is None or heading_raw is None:
                    publish_stop()
                    return

                # The perspective row-scan has a measured straight-ahead bias
                # of about +14 deg. Use only the residual; otherwise that bias
                # overwhelms the lateral term and incorrectly commands left.
                heading_error = heading_raw - ROW_GUIDE_HEADING_TARGET_DEG
                rot = (-0.18 * math.radians(heading_error) -
                       0.00045 * center_error)
                cmd.linear.x = ALIGN_SPEED
                cmd.angular.z = max(-0.08, min(0.08, rot))
                state.control_source = 'ROW_GUIDE'

            elif state.entry_guide_valid:
                center_error = state.entry_guide_center_error
                heading_error = state.entry_guide_heading_error_deg
                if center_error is None or heading_error is None:
                    publish_stop()
                    return

                # Positive IPM errors put the guide to the physical right;
                # command negative angular.z (right turn). Both signs are
                # permitted so the vehicle can straighten or correct left.
                rot = (-0.65 * math.radians(heading_error) -
                       0.00075 * center_error)
                cmd.linear.x = ENTRY_ARC_SPEED
                cmd.angular.z = max(-0.14, min(0.12, rot))
                state.control_source = 'IPM_ENTRY_GUIDE'
            else:
                # No trusted long-left + short-right geometry: creep while
                # approaching 68 deg, but never blindly turn beyond 72 deg.
                if state.turn_deg >= ENTRY_ARC_BLIND_STOP_DEG:
                    fail_locked('到达72度仍无可信驶入虚线，已停车防止转过头')
                    return
                yaw_error_deg = ENTRY_ARC_BLIND_TARGET_DEG - state.turn_deg
                cmd.linear.x = ENTRY_ARC_BLIND_SPEED
                cmd.angular.z = max(-0.10, min(0.0, -0.008 * yaw_error_deg))
                state.control_source = 'BLIND_ODOM_RIGHT'

        elif mode == 'ALIGN_BETWEEN_LINES':
            if (state.phase_distance > ALIGN_MAX_DISTANCE or
                    now - state.state_started > ALIGN_TIMEOUT):
                fail_locked('双边线对准超距/超时，未达到驶入成功条件')
                return
            if not state.dual_lane_valid:
                # Never steer using historical or invented lane geometry.
                publish_stop()
                return

            center_error = state.lane_center_error
            heading_error = state.lane_heading_error_deg
            if center_error is None or heading_error is None:
                publish_stop()
                return

            if (state.phase_distance >= ALIGN_MIN_DISTANCE and
                    state.align_good_frames >= ALIGN_CONFIRM_FRAMES):
                state.mode = 'ENTERED_STOPPED'
                state.message = '已完整驶入右转车道并对准，验证任务停车'
                publish_stop()
                return

            # ROS angular.z > 0 turns left. Positive IPM errors mean the
            # current centerline is to the right, so command a right turn.
            heading_rad = math.radians(heading_error)
            rot = -0.75 * heading_rad - 0.0009 * center_error
            cmd.linear.x = ALIGN_SPEED
            cmd.angular.z = max(-0.16, min(0.16, rot))
            state.control_source = 'IPM_DUAL_ALIGN'
        state.command_linear_x = cmd.linear.x
        state.command_angular_z = cmd.angular.z
        cmd_pub.publish(cmd)


PAGE = '''<!doctype html><meta charset="utf-8"><title>右转车道驶入验证</title>
<style>body{font:16px sans-serif;background:#0f172a;color:#f8fafc;margin:20px}main{max-width:1100px;margin:auto}section{background:#1e293b;padding:16px;margin:12px 0;border-radius:10px}img{width:48%;background:#111;margin:1%;border-radius:6px}.start{background:#1677ff;color:white}.stop{background:#d00;color:white}button{padding:12px 22px;border:0;border-radius:6px;margin-right:10px;font-size:16px;cursor:pointer}pre{font-size:15px;background:#0f172a;padding:10px;border-radius:6px;color:#38bdf8}</style>
<main><h2>右转车道驶入验证</h2>
<section><b>默认不运动。确认场地清空并准备实体急停后再启动。</b>
<p>流程：入口前进 → 搜索右出口 → 低速弧线驶入 → 当前帧双边线对准 → 停车。</p>
<p><button class="start" onclick="post('/api/start')">解锁并开始</button><button class="stop" onclick="post('/api/stop')">立即停车</button><button onclick="post('/api/reset')">复位为仅感知</button></p>
<pre id="status">加载状态中...</pre></section>
<section><h3>识别叠加图 (/right_turn/debug/overlay) 与 二值图 (/right_turn/debug/mask)</h3>
<img id="img_overlay"><img id="img_mask">
</section></main>
<script>
let host = window.location.hostname || '192.168.89.176';
document.getElementById('img_overlay').src = 'http://' + host + ':8080/stream?topic=/right_turn/debug/overlay';
document.getElementById('img_mask').src = 'http://' + host + ':8080/stream?topic=/right_turn/debug/mask';

async function post(p){
    try {
        let res = await (await fetch(p,{method:'POST'})).json();
        if(!res.ok && res.error){
            alert("⚠️ 无法启动: " + res.error);
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
                if state.mode not in ('DISARMED', 'FAULT', 'ENTERED_STOPPED'):
                    self.reply({'ok': False, 'error': '任务已经运行'})
                    return
                if state.odom is None:
                    self.reply({'ok': False, 'error': '底盘里程计 (/odom) 未就绪，请先启动底盘驱动 (roslaunch ucar_controller base_driver.launch)'})
                    return
                if time.time() - state.last_image_time > 0.6:
                    self.reply({'ok': False, 'error': '摄像头画面超时'})
                    return
                if state.center_error is None or abs(state.center_error) > CENTER_TOLERANCE_PX:
                    self.reply({'ok': False, 'error': f'未居中 (当前偏离 {state.center_error}px, 限制 {CENTER_TOLERANCE_PX}px)'})
                    return
                state.start_pose = state.odom
                state.distance = state.turn_deg = 0.0
                state.right_edge_frames = 0
                state.right_edge_found = False
                state.phase_start_pose = None
                state.phase_distance = 0.0
                state.dual_lane_frames = 0
                state.align_good_frames = 0
                state.entry_guide_good_frames = 0
                state.row_guide_good_frames = 0
                state.mode = 'ADVANCE'
                state.state_started = time.time()
                state.message = '低速前进25厘米'
                self.reply({'ok': True})
            elif path == '/api/stop':
                state.mode = 'ESTOP'
                state.message = '网页急停'
                publish_stop()
                self.reply({'ok': True})
            elif path == '/api/reset':
                state.mode = 'DISARMED'
                state.message = '仅感知模式'
                state.start_pose = None
                state.distance = state.turn_deg = 0.0
                state.right_edge_frames = 0
                state.right_edge_found = False
                state.phase_start_pose = None
                state.phase_distance = 0.0
                state.dual_lane_frames = 0
                state.align_good_frames = 0
                state.entry_guide_good_frames = 0
                state.row_guide_good_frames = 0
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
                rospy.logwarn(f"检测到 {PORT} 端口被占用，正在清理旧进程...")
                os.system(f"fuser -k {PORT}/tcp 2>/dev/null || pkill -9 -f right_turn_trial.py 2>/dev/null")
                import time
                time.sleep(1)
                super().server_bind()
            else:
                raise


def shutdown():
    for _ in range(5):
        publish_stop()
        time.sleep(0.03)


if __name__ == '__main__':
    rospy.init_node('right_turn_trial', anonymous=False)
    init_ipm()
    load_hsv()
    cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
    mask_pub = rospy.Publisher('/right_turn/debug/mask', Image, queue_size=1)
    overlay_pub = rospy.Publisher('/right_turn/debug/overlay', Image, queue_size=1)
    rospy.Subscriber(IMAGE_TOPIC, Image, image_cb, queue_size=1, buff_size=2 ** 24)
    rospy.Subscriber('/odom', Odometry, odom_cb, queue_size=1)
    rospy.Timer(rospy.Duration(0.05), control_timer)
    rospy.on_shutdown(shutdown)
    server = ReusableHTTPServer(('0.0.0.0', PORT), Handler)
    rospy.loginfo('右转车道驶入节点已启动（默认仅感知）: http://0.0.0.0:%d', PORT)
    server.serve_forever()
