#!/usr/bin/env python3
"""Low-speed right-turn trial node with IPM bird's-eye view slope matching.

The node starts DISARMED and publishes no velocity until /api/start is called.
It advances 0.25 m using odometry, rotates right up to 60 degrees, and stops
when a stable right-edge candidate is visible.
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

DEFAULT_PARAMS = {
    'low_h': 0, 'high_h': 179,
    'low_s': 0, 'high_s': 45,
    'low_v': 170, 'high_v': 255,
    'roi_top': 0.45, 'roi_bottom': 1.0,
    'roi_left': 0.0, 'roi_right': 1.0,
    'blur_ksize': 3, 'erode_iter': 0, 'erode_ksize': 3, 'dilate_iter': 2, 'dilate_ksize': 3
}

IPM_MATRIX = None


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

    virt_near_err, virt_centers, dashed_segs = get_virtual_extended_midline(right_edge, mask.shape)

    return (left_c, right_c, left_midline, right_midline, corners, centers,
            edge_samples, center_error, right_edge, left_slope, right_slope, slope_diff,
            virt_near_err, dashed_segs)


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
         virt_near_err, dashed_segs) = analyze_lanes(mask)

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
        
        # 绘制向小车方向 (画面底部) 延伸的绿色虚拟中线 (虚线)
        if len(dashed_segs) > 0:
            for p1, p2 in dashed_segs:
                cv2.line(overlay, p1, p2, (0, 255, 0), 3)
            cv2.putText(overlay, 'VIRTUAL DASHED MIDLINE', (260, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 255, 0), 2)

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
            state.virtual_center_error = virt_near_err
            
            if state.mode == 'SEARCH_RIGHT' and state.turn_deg >= SEARCH_START_DEG:
                # 严密判定：既要找到合格且通过 IPM 斜率校验的右出口车道线 (right_edge is not None)，
                # 也要同时提取出稳定的绿色中线 (len(centers) >= 3)
                has_valid_right_exit = (right_edge is not None) and (len(centers) >= 3)
                if has_valid_right_exit:
                    state.right_edge_frames += 1
                else:
                    state.right_edge_frames = max(0, state.right_edge_frames - 1)
                state.right_edge_found = (state.right_edge_frames >= EDGE_CONFIRM_FRAMES)

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


def publish_stop():
    if cmd_pub is not None:
        cmd_pub.publish(Twist())


def fail_locked(message):
    state.mode = 'FAULT'
    state.message = message
    publish_stop()


def control_timer(_event):
    with state.lock:
        now = time.time()
        mode = state.mode
        if mode not in ('ADVANCE', 'SEARCH_RIGHT', 'ENTRY_GUIDED'):
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
        elif mode == 'SEARCH_RIGHT':
            if state.right_edge_found:
                state.mode = 'ENTRY_GUIDED'
                state.entry_start_dist = state.distance
                state.state_started = now
                state.message = '延伸虚线引导：驶入右转车道正中央'
                publish_stop()
                return
            if state.turn_deg >= ROTATE_LIMIT_DEG or now - state.state_started > ROTATE_TIMEOUT:
                fail_locked('达到60度/超时仍未稳定找到右边界')
                return
            cmd.angular.z = -ROTATE_SPEED
        elif mode == 'ENTRY_GUIDED':
            entry_dist = state.distance - state.entry_start_dist
            virt_err = state.virtual_center_error
            
            # 若延伸中线像素偏离可用，进行闭环调节；若无，默认微调圆弧
            if virt_err is not None:
                rot = - (0.0035 * virt_err)
                rot = max(-0.45, min(0.10, rot)) - 0.15
            else:
                rot = -0.30
                
            # 切入成功判定：推进约 22~35cm，且偏离像素 <= 35px (说明小车已在右转车道正中央对齐)
            if entry_dist >= 0.22 and (virt_err is not None and abs(virt_err) <= 35.0):
                state.mode = 'ARRIVED_CENTER'
                state.message = '已成功驶入右转车道正中央！'
                publish_stop()
                return
            if entry_dist > 0.35 or now - state.state_started > 6.0:
                state.mode = 'ARRIVED_CENTER'
                state.message = '已到达右转车道中间位置 (距离/时间限制触发)'
                publish_stop()
                return
                
            cmd.linear.x = 0.12
            cmd.angular.z = rot
        cmd_pub.publish(cmd)


PAGE = '''<!doctype html><meta charset="utf-8"><title>右转验证终端 (IPM 鸟瞰图平行校验版)</title>
<style>body{font:16px sans-serif;background:#0f172a;color:#f8fafc;margin:20px}main{max-width:1100px;margin:auto}section{background:#1e293b;padding:16px;margin:12px 0;border-radius:10px}img{width:48%;background:#111;margin:1%;border-radius:6px}.start{background:#1677ff;color:white}.stop{background:#d00;color:white}button{padding:12px 22px;border:0;border-radius:6px;margin-right:10px;font-size:16px;cursor:pointer}pre{font-size:15px;background:#0f172a;padding:10px;border-radius:6px;color:#38bdf8}</style>
<main><h2>右转简单验证控制台 (直行 25cm + IPM 鸟瞰图平行匹配)</h2>
<section><b>默认不运动。确认场地清空并准备实体急停后再启动。</b>
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
                if state.mode not in ('DISARMED', 'FAULT', 'EDGE_FOUND', 'ARRIVED_CENTER'):
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
    rospy.loginfo('右转验证节点已启动 (直行 25cm + IPM 鸟瞰图平行匹配模式): http://0.0.0.0:%d', PORT)
    server.serve_forever()
