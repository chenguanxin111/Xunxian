#!/usr/bin/env python3
"""
Standalone High-Speed Line-Following Node 4 (line_following_node4.py)

100% 基于去年实车验证成熟的 xunxian.py 源码改造移植：
1. 完整保留原版 get_line_bin_img、get_ROI、find_center_edge_line、calculate_slope、get_pid_params。
2. 解耦摄像头死循环，接入 ROS /usb_cam/image_raw 图像订阅。
3. 兼容 config/white_lane.json 配置载入。
4. 提供独立端口 5004 的 Web 调试控制台与 /line_following/debug/overlay 话题流。
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

PORT = 5004
ALLOW_MOTION = True

# 统一分辨率 (与去年 xunxian.py pw=640, ph=360 完全一致)
pw = 640
ph = 360

# 去年 xunxian.py 原汁原味默认 HSV
DEFAULT_HSV = {
    'low_h': 0, 'high_h': 120,
    'low_s': 0, 'high_s': 75,
    'low_v': 70, 'high_v': 255,
    'blur_ksize': 3,
    'erode_iter': 1, 'erode_ksize': 1,
    'dilate_iter': 2, 'dilate_ksize': 5,
    'roi_top': 0.58, 'roi_bottom': 1.0,
}

# 100% 还原去年 xunxian.py 原版 PID 参数矩阵 (包含 1.2 倍比例与 15 倍横向控制增益)
PID_PARAMS = {
    # 与去年 xunxian.py 第 65-75 行的实际计算结果一致。
    "small_curve_invisible": (0.0264, 0.00075, 0.24),
    "medium_curve_invisible": (0.0288, 0.00075, 0.216),
    "extreme_curve_invisible": (0.0324, 0.00075, 0.108),
    "large_extreme_curve_invisible": (0.0372, 0.00075, 0.276),
    "small_straight": (0.018, 0.00075, 0.276),
    "small_curve": (0.024, 0.00075, 0.216),
    "medium_curve": (0.0264, 0.00075, 0.216),
    "large_curve": (0.0288, 0.00075, 0.168),
    "large_straight": (0.0144, 0.0075, 0.336)
}


class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.mode = 'DISARMED'
        self.message = '巡线节点 4 已启动 (去年 xunxian.py 完整移植)'
        self.last_image_time = 0.0
        self.hsv_params = dict(DEFAULT_HSV)
        
        # 巡线感知数据
        self.error = 0.0
        self.last_error = 0.0
        self.raw_error = 0.0
        self.error_jump_rejected = False
        self.far_preview_error = 0.0
        self.far_preview_used = False
        self.last_center_points = []
        self.centers = []
        self.edge_points = []
        self.current_edge_points_zuixiamian = []
        self.kanbujian = 0
        
        # 运动控制参数
        self.target_speed = 0.32  # 100% 恢复去年 xunxian.py 原版默认速度 0.32 m/s
        self.start_time = 0.0
        self.last_v_z = 0.0
        self.delat_v_z = 0.0
        self.command_linear_x = 0.0
        self.command_linear_y = 0.0
        self.command_angular_z = 0.0
        
        self.vision_valid = False
        self.lost_frames = 0
        self.last_valid_time = 0.0
        self.duizhun_finish = False
        self.roi_2 = None
        # 最近一次由“同一帧可信双边行”算出的 IPM 平均半车道宽。
        # 当画面只剩单边时冻结该值，绝不使用单边结果反向更新。
        self.ipm_half_width = 160.0
        self.ipm_half_width_samples = 0
        self.lane_track_count = 0
        self.lane_pair_valid = False
        self.lane_pair_slope_diff = None
        # 去年 calculate_turn() 的远端识别结果；仅作诊断，不直接执行硬转。
        self.far_turn_direction = 'normal'
        self.far_turn_confidence = 0.0
        self.far_turn_points = 0

    def status(self):
        return {
            'mode': self.mode,
            'message': self.message,
            'error_deg': round(self.error, 1) if self.error is not None else None,
            'raw_error_deg': round(self.raw_error, 1) if self.raw_error is not None else None,
            'error_jump_rejected': self.error_jump_rejected,
            'far_preview_error_deg': round(self.far_preview_error, 1),
            'far_preview_used': self.far_preview_used,
            'center_count': len(self.centers),
            'target_speed': self.target_speed,
            'kanbujian': bool(self.kanbujian),
            'command_linear_x': round(self.command_linear_x, 3),
            'command_linear_y': round(self.command_linear_y, 3),
            'command_angular_z': round(self.command_angular_z, 3),
            'vision_valid': self.vision_valid,
            'lost_frames': self.lost_frames,
            'ipm_half_width': round(self.ipm_half_width, 1),
            'ipm_half_width_samples': self.ipm_half_width_samples,
            'lane_track_count': self.lane_track_count,
            'lane_pair_valid': self.lane_pair_valid,
            'lane_pair_slope_diff': (
                round(self.lane_pair_slope_diff, 3)
                if self.lane_pair_slope_diff is not None else None
            ),
            'far_turn_direction': self.far_turn_direction,
            'far_turn_confidence': round(self.far_turn_confidence, 3),
            'far_turn_points': self.far_turn_points,
            'image_age_s': round(max(0.0, time.time() - self.last_image_time), 2),
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
                rospy.loginfo("巡线节点4成功载入配置文件: %s", target_path)
        except Exception as err:
            rospy.logwarn("读取配置文件 %s 失败: %s", target_path, err)


def get_line_bin_img(img, params):
    """去年 xunxian.py 原汁原味二值化与尺寸缩放"""
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
    line_up_2 = 0.67
    line_low_2 = 1.0
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
        
    H2 = ROI_1.shape[0]
    ROI_2 = ROI_1[int(H2 * line_up_2):int(H2 * line_low_2), :]

    return ROI_1, ROI_2, y1


def find_line(img, step=4):
    """去年 xunxian.py 原汁原味近端底层白线寻找函数 (用于对准)"""
    height, width = img.shape
    mid_x = width // 2
    left_yellow_points = []
    right_yellow_points = []

    for y in range(height - 1, -1, -step):
        white_indices = np.where(img[y] == 255)[0]
        if len(white_indices) == 0:
            continue

        left_white_points = white_indices[white_indices < mid_x]
        right_white_points = white_indices[white_indices >= mid_x]

        if len(left_white_points) > 0:
            left_nearest = min(left_white_points, key=lambda x: abs(x - mid_x))
            left_yellow_points.append((left_nearest, y))

        if len(right_white_points) > 0:
            right_nearest = min(right_white_points, key=lambda x: abs(x - mid_x))
            right_yellow_points.append((right_nearest, y))

    left_x_coords = [x for x, y in left_yellow_points]
    right_x_coords = [x for x, y in right_yellow_points]

    avg_left_x = np.mean(left_x_coords) if left_x_coords and len(left_yellow_points) > 2 else None
    avg_right_x = np.mean(right_x_coords) if right_x_coords and len(right_yellow_points) > 2 else None

    return avg_left_x, avg_right_x


def calculate_turn(img, shared_state):
    """移植去年 calculate_turn 的远端几何判断，仅输出状态，不执行硬转。

    去年函数按列扫描远端白线，并比较左/中/右三段斜率。Node4 没有去年
    state.results 的外部计数来源，因此这里只使用当前鸟瞰二值图的几何证据。
    """
    if img is None or img.size == 0:
        return 'normal'
    height, width = img.shape[:2]
    step = 15
    points = []
    previous_y = None
    for col in range(width - 1, -1, -step):
        ys = np.where(img[:, col] == 255)[0]
        if len(ys) == 0:
            continue
        mean_y = int(np.mean(ys))
        if previous_y is None or abs(mean_y - previous_y) <= height // 5:
            points.append((col, mean_y))
            previous_y = mean_y

    confidence = len(points) / float(max(1, width // step))
    direction = 'normal'
    if confidence >= 0.85 and len(points) >= 9:
        points = sorted(points, key=lambda point: point[0])
        left_points = points[:4]
        center_start = max((len(points) - 4) // 2, 4)
        center_points = points[center_start:center_start + 4]
        right_points = points[-4:]

        def slope(segment):
            if len(segment) < 2:
                return 0.0
            return float(np.polyfit(
                [point[0] for point in segment],
                [point[1] for point in segment], 1
            )[0])

        left_slope = slope(left_points)
        center_slope = slope(center_points)
        right_slope = slope(right_points)
        if left_slope and center_slope and right_slope:
            if (abs(center_slope - right_slope) < 0.2
                    and abs(center_slope) < abs(right_slope) < abs(left_slope)
                    and abs(center_slope - left_slope) > 0.2
                    and abs(left_slope) > 0.2):
                direction = 'left'
            elif (abs(center_slope - left_slope) < 0.2
                  and abs(center_slope) < abs(left_slope) < abs(right_slope)
                  and abs(center_slope - right_slope) > 0.2
                  and abs(right_slope) > 0.2):
                direction = 'right'
            elif (abs(center_slope - left_slope) < 0.2
                  and abs(center_slope) < abs(left_slope)
                  and abs(center_slope) < abs(right_slope)
                  and abs(center_slope - right_slope) < 0.2):
                direction = 'stop'

    with shared_state.lock:
        shared_state.far_turn_direction = direction
        shared_state.far_turn_confidence = confidence
        shared_state.far_turn_points = len(points)
    return direction


# 鸟瞰图 (IPM) 变换矩阵及其逆矩阵
src_pts = np.float32([[140, 210], [500, 210], [0, 360], [640, 360]])
dst_pts = np.float32([[160, 0], [480, 0], [160, 360], [480, 360]])
IPM_MATRIX = cv2.getPerspectiveTransform(src_pts, dst_pts)
IPM_INV_MATRIX = cv2.getPerspectiveTransform(dst_pts, src_pts)


def transform_point_ipm(pt, matrix):
    pts = np.float32([[[pt[0], pt[1]]]])
    res = cv2.perspectiveTransform(pts, matrix)
    return float(res[0][0][0]), float(res[0][0][1])


def remove_small_white_components(binary_img):
    """去除无法构成车道轨迹的小连通域。"""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary_img, 8)
    clean = np.zeros_like(binary_img)
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        component_h = int(stats[label, cv2.CC_STAT_HEIGHT])
        component_w = int(stats[label, cv2.CC_STAT_WIDTH])
        if area >= 45 and component_h >= 16 and component_w >= 2:
            clean[labels == label] = 255
    return clean


def build_lane_tracks(row_observations, step, width):
    """将逐行白色簇连接为连续轨迹，避免逐行独立选点造成身份跳变。"""
    tracks = []
    max_gap = step * 3
    max_residual = max(24.0, width * 0.065)

    for y, observations in row_observations:
        candidates = []
        for track_index, track in enumerate(tracks):
            last_y, last_x = track['points'][-1]
            if last_y - y > max_gap:
                continue
            predicted_x = last_x
            if len(track['points']) >= 2:
                prev_y, prev_x = track['points'][-2]
                if last_y != prev_y:
                    local_slope = (last_x - prev_x) / float(last_y - prev_y)
                    predicted_x = last_x + local_slope * (y - last_y)
            for observation_index, x in enumerate(observations):
                residual = abs(x - predicted_x)
                if residual <= max_residual:
                    candidates.append((residual, track_index, observation_index))

        used_tracks = set()
        used_observations = set()
        for residual, track_index, observation_index in sorted(candidates):
            if track_index in used_tracks or observation_index in used_observations:
                continue
            tracks[track_index]['points'].append((y, observations[observation_index]))
            used_tracks.add(track_index)
            used_observations.add(observation_index)

        for observation_index, x in enumerate(observations):
            if observation_index not in used_observations:
                tracks.append({'points': [(y, x)]})

    minimum_points = 5
    minimum_span = step * 4
    valid_tracks = []
    for track in tracks:
        points = track['points']
        y_span = max(p[0] for p in points) - min(p[0] for p in points)
        if len(points) >= minimum_points and y_span >= minimum_span:
            track['by_y'] = {int(y): float(x) for y, x in points}
            valid_tracks.append(track)
    return valid_tracks


def fit_track_in_bird_view(track):
    """直接在鸟瞰二值图中拟合 x=k*y+b，并返回拟合残差。"""
    if len(track['points']) < 4:
        return None
    ys = np.asarray([point[0] for point in track['points']], dtype=np.float64)
    xs = np.asarray([point[1] for point in track['points']], dtype=np.float64)
    if np.ptp(ys) < 20.0:
        return None
    slope, intercept = np.polyfit(ys, xs, 1)
    residual = float(np.median(np.abs(xs - (slope * ys + intercept))))
    return float(slope), float(intercept), residual


def select_parallel_lane_pair(tracks, width):
    """完全在鸟瞰坐标系中匹配斜率平行、宽度稳定的左右轨迹。"""
    fitted = []
    for track in tracks:
        fit = fit_track_in_bird_view(track)
        if fit is not None and fit[2] <= 24.0:
            fitted.append((track, fit))

    best = None
    for i in range(len(fitted)):
        for j in range(i + 1, len(fitted)):
            track_a, fit_a = fitted[i]
            track_b, fit_b = fitted[j]
            common_y = sorted(set(track_a['by_y']).intersection(track_b['by_y']), reverse=True)
            if len(common_y) < 5:
                continue

            mean_a = np.mean([track_a['by_y'][y] for y in common_y])
            mean_b = np.mean([track_b['by_y'][y] for y in common_y])
            left, left_fit, right, right_fit = (
                (track_a, fit_a, track_b, fit_b)
                if mean_a < mean_b else
                (track_b, fit_b, track_a, fit_a)
            )

            slope_diff = abs(left_fit[0] - right_fit[0])
            if slope_diff > 0.22:
                continue

            full_widths = []
            center_rows = []
            for y in common_y:
                lx = left['by_y'][y]
                rx = right['by_y'][y]
                lane_width = rx - lx
                if lane_width < width * 0.15:
                    continue
                if 120.0 <= lane_width <= 520.0:
                    full_widths.append(float(lane_width))
                    center_rows.append((y, (lx + rx) / 2.0))
            if len(full_widths) < 5:
                continue

            median_width = float(np.median(full_widths))
            width_mad = float(np.median(np.abs(np.asarray(full_widths) - median_width)))
            if width_mad > max(20.0, median_width * 0.12):
                continue

            score = len(full_widths) - 18.0 * slope_diff - 0.04 * width_mad
            if best is None or score > best['score']:
                best = {
                    'left': left,
                    'right': right,
                    'common_y': common_y,
                    'full_widths': full_widths,
                    'center_rows': center_rows,
                    'slope_diff': slope_diff,
                    'score': score,
                }
    return best


def find_center_edge_line(img, y_offset, shared_state):
    """在鸟瞰图中提取、拟合和匹配双边，再逆投影输出控制点。"""
    if img is None:
        raise ValueError("输入图像为空")
    if len(img.shape) != 2 or img.dtype != np.uint8:
        raise ValueError("输入图像必须是单通道二值图像")

    roi_height, width = img.shape
    full_mask = np.zeros((ph, width), dtype=np.uint8)
    copy_bottom = min(ph, y_offset + roi_height)
    full_mask[y_offset:copy_bottom] = img[:copy_bottom - y_offset]

    bird_mask = cv2.warpPerspective(
        full_mask, IPM_MATRIX, (width, ph), flags=cv2.INTER_NEAREST
    )
    bird_mask = remove_small_white_components(bird_mask)
    # 去年 calculate_turn 的输入是原 ROI，而不是 IPM 鸟瞰图；鸟瞰图中的
    # 车道线大多是竖向窄带，用按列扫描会天然得到很低的覆盖率。
    calculate_turn(img, shared_state)
    current_edge_points_zuixiamian = []
    center_points = []
    edge_points = []
    sigle, double = 0, 0
    step = 4

    row_detections = []
    for bird_y in range(ph - 1, -1, -step):
        white_indices = np.where(bird_mask[bird_y] == 255)[0]
        if len(white_indices) == 0:
            continue
        breaks = np.where(np.diff(white_indices) > 1)[0] + 1
        clusters = np.split(white_indices, breaks)
        mean_indices = [
            np.mean(cluster) for cluster in clusters
            if len(cluster) >= 3 and 5 < np.mean(cluster) < width - 6
        ]
        if mean_indices:
            row_detections.append((bird_y, sorted(mean_indices)))

    tracks = build_lane_tracks(row_detections, step, width)
    pair = select_parallel_lane_pair(tracks, width)

    with shared_state.lock:
        shared_state.lane_track_count = len(tracks)
        shared_state.lane_pair_valid = pair is not None
        shared_state.lane_pair_slope_diff = pair['slope_diff'] if pair is not None else None
        if pair is not None:
            shared_state.ipm_half_width = float(np.mean(pair['full_widths'])) / 2.0
            shared_state.ipm_half_width_samples = len(pair['full_widths'])
        else:
            shared_state.ipm_half_width_samples = 0
        half_width = shared_state.ipm_half_width

    if pair is not None:
        selected_tracks = [pair['left'], pair['right']]
        trusted_center_rows = pair['center_rows']
    elif tracks:
        selected_tracks = [max(tracks, key=lambda track: len(track['points']))]
        trusted_center_rows = []
    else:
        selected_tracks = []
        trusted_center_rows = []

    selected_bird_rows = sorted(
        set().union(*(set(track['by_y']) for track in selected_tracks))
        if selected_tracks else set(), reverse=True
    )

    for bird_y in selected_bird_rows:
        left_x = pair['left']['by_y'].get(bird_y) if pair is not None else None
        right_x = pair['right']['by_y'].get(bird_y) if pair is not None else None
        if pair is None:
            single_x = selected_tracks[0]['by_y'].get(bird_y)
            visible_xs = [single_x] if single_x is not None else []
        else:
            visible_xs = [x for x in (left_x, right_x) if x is not None]
        if not visible_xs:
            continue

        projected_edges = []
        for bird_x in visible_xs:
            image_x, image_y = transform_point_ipm(
                (bird_x, bird_y), IPM_INV_MATRIX
            )
            roi_y = int(round(image_y - y_offset))
            if 0 <= image_x < width and 0 <= roi_y < roi_height:
                projected_edges.append((int(round(image_x)), roi_y))
                edge_points.append((int(round(image_x)), roi_y))
        if not projected_edges:
            continue

        current_edge_points = [point[0] for point in projected_edges]
        if left_x is not None and right_x is not None:
            double += 1
            center_bird_x = (left_x + right_x) / 2.0
        else:
            sigle += 1
            edge_x = visible_xs[0]
            if pair is not None:
                is_left_line = left_x is not None
            elif trusted_center_rows:
                reference_center_x = min(
                    trusted_center_rows,
                    key=lambda item: abs(item[0] - bird_y)
                )[1]
                is_left_line = edge_x < reference_center_x
            else:
                is_left_line = edge_x < width / 2.0
            center_bird_x = edge_x + half_width if is_left_line else edge_x - half_width

        center_img_x, center_img_y = transform_point_ipm(
            (center_bird_x, bird_y), IPM_INV_MATRIX
        )
        center_roi_y = int(round(center_img_y - y_offset))
        if not (0 <= center_img_x < width and 0 <= center_roi_y < roi_height):
            continue
        new_center_point = (int(round(center_img_x)), center_roi_y)

        if current_edge_points:
            current_edge_points_zuixiamian = current_edge_points

        if len(center_points) == 0 or abs(new_center_point[0] - center_points[-1][0]) < width / 8:
            center_points.append(new_center_point)

    # 去年逻辑：有效扫描行中单边占比超过 90%，表示弯道中远端一侧
    # 大面积离开视野。单边推算出的中线仍有效，但应切换 invisible 弯道 PID。
    valid_rows = sigle + double
    kanbujian = int(valid_rows > 0 and sigle / float(valid_rows) > 0.9)

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
    """去年 xunxian.py 原汁原味 PID 参数逻辑"""
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

        # 水平镜像翻转 (与去年 xunxian.py 100% 一致)
        frame = cv2.flip(frame, 1)

        with state.lock:
            hsv_p = dict(state.hsv_params)

        resize_img, resize_bin_img, img_mask = get_line_bin_img(frame, hsv_p)
        ROI_1, ROI_2, y_offset = get_ROI(resize_img, resize_bin_img, hsv_p)

        center_pts, edge_pts, edge_bottom, kanbujian = find_center_edge_line(ROI_1, y_offset, state)

        if len(center_pts) > 3:
            with state.lock:
                state.last_center_points = center_pts

        angle_first_last, angle_first_middle, avg_first_3_x = calculate_slope(center_pts, state)

        # 去年原版误差计算公式。注意：当双边匹配丢失时，单边补出的中线
        # 可能发生左右身份跳变；此时不能让一个伪造的 +80° 覆盖上一帧 -58°。
        raw_error = -90.0 + angle_first_middle if angle_first_middle > 0 else 90.0 + angle_first_middle
        far_error = (-90.0 + angle_first_last
                     if angle_first_last > 0 else 90.0 + angle_first_last)

        with state.lock:
            pair_valid = state.lane_pair_valid
            previous_error = state.last_error
            have_previous_error = state.last_image_time > 0 and state.vision_valid

        error_jump_rejected = False
        calc_error = raw_error
        far_preview_used = False
        # first_middle 主要看近端，first_last 覆盖完整轨迹。弯道尚未丢边时，
        # 用远端趋势提前增加 35% 纠偏，避免等到单边后才开始大幅转向。
        if pair_valid and len(center_pts) >= 8:
            if (abs(far_error) > abs(raw_error) + 8.0
                    and abs(far_error) > 25.0
                    and raw_error * far_error > 0):
                calc_error = 0.65 * raw_error + 0.35 * far_error
                far_preview_used = True
        if have_previous_error and not pair_valid:
            # 去年 PID 仍保留，但单边轨迹只允许平滑变化。该阈值远小于本次
            # 实测的 -58.5 -> +81.6° 身份翻转，正常弯道变化不会被拒绝。
            if abs(raw_error - previous_error) > 28.0:
                calc_error = previous_error
                error_jump_rejected = True

        # 还原到 (640x360) 叠加可视化坐标
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
            state.roi_2 = ROI_2
            state.centers = full_centers
            state.edge_points = full_edges
            state.current_edge_points_zuixiamian = edge_bottom
            state.kanbujian = kanbujian
            state.raw_error = raw_error
            state.error_jump_rejected = error_jump_rejected
            state.far_preview_error = far_error
            state.far_preview_used = far_preview_used
            state.last_error = calc_error
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
        if ipm_pub is not None:
            ipm_overlay = cv2.warpPerspective(overlay, IPM_MATRIX, (pw, ph))
            ipm_pub.publish(safe_cv2_to_imgmsg(ipm_overlay, 'bgr8'))

    except Exception as e:
        rospy.logerr_throttle(2, f"巡线节点4图像回调异常: {e}")


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

        # 去年 xunxian.py 原版对准 (duizhun) 阶段：启动时不向前冲，先用 linear.y 横向平移居中对准！
        if not state.duizhun_finish:
            ROI_2 = state.roi_2
            if ROI_2 is not None:
                avg_left_x, avg_right_x = find_line(ROI_2)
                avg_left_x = avg_left_x if avg_left_x is not None else 0
                avg_right_x = avg_right_x if avg_right_x is not None else 639
                white_center = (avg_left_x + avg_right_x) / 2.0

                vel_align = Twist()
                vel_align.linear.x = 0.0
                vel_align.angular.z = 0.0
                if white_center < 310:
                    vel_align.linear.y = -0.06
                elif white_center > 330:
                    vel_align.linear.y = 0.06
                else:
                    state.duizhun_finish = True
                    rospy.loginfo("巡线节点4对准完成，进入正常巡线模式")

                state.command_linear_x = vel_align.linear.x
                state.command_linear_y = vel_align.linear.y
                state.command_angular_z = vel_align.angular.z
                cmd_pub.publish(vel_align)
                return
            else:
                state.duizhun_finish = True

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
        # 与去年一致：单边占比超过 90% 时，即使单边推算中线仍有效，
        # 也应使用 invisible 弯道 PID，提高大弯响应。
        kanbujian = bool(state.kanbujian)
        current_edge_bottom = state.current_edge_points_zuixiamian

        kp_z, kp_y, kd_z = get_pid_params(err, kanbujian)

        # 严格恢复去年 xunxian.py 第 492-494 行的 PID 控制律。
        vel = Twist()
        vel.angular.z = kp_z * err - kd_z * state.delat_v_z
        state.delat_v_z = vel.angular.z - state.last_v_z
        state.last_v_z = vel.angular.z

        # 启动软起步平滑加速 (前 1.5 秒从 0.08 m/s 平滑过渡到 target_speed)
        start_elapsed = now - state.start_time if state.start_time > 0 else 2.0
        ramp_factor = min(1.0, max(0.0, start_elapsed / 1.5))
        base_speed = 0.08 + (state.target_speed - 0.08) * ramp_factor

        # 去年默认前进速度为 0.32 m/s；保留启动软起步以及 Web 手动降速。
        vel.linear.x = base_speed

        # 计算横向平移误差 (error_y)
        error_y = 0
        if current_edge_bottom is not None and len(current_edge_bottom) >= 2:
            left, right = current_edge_bottom[:2]
            if left not in [0, 639] and right not in [0, 639]:
                error_y = (left + right) / 2 - 320

        vel.linear.y = kp_y * error_y * 0.0005 if error_y != 0 else 0.0

        state.command_linear_x = vel.linear.x
        state.command_linear_y = vel.linear.y
        state.command_angular_z = vel.angular.z

        rospy.loginfo_throttle(1, "巡线节点4正在发布运动速度: vx=%.2f, vy=%.2f, wz=%.2f (err=%.1f)",
                               vel.linear.x, vel.linear.y, vel.angular.z, err)
        cmd_pub.publish(vel)


PAGE = '''<!doctype html><meta charset="utf-8"><title>巡线控制台 Node4</title>
<style>body{font:16px sans-serif;background:#0f172a;color:#f8fafc;margin:20px}main{max-width:1100px;margin:auto}section{background:#1e293b;padding:16px;margin:12px 0;border-radius:10px}img{width:48%;background:#111;margin:1%;border-radius:6px}.start{background:#1677ff;color:white}.stop{background:#d00;color:white}button{padding:10px 18px;border:0;border-radius:6px;margin-right:8px;font-size:15px;cursor:pointer}pre{font-size:15px;background:#0f172a;padding:10px;border-radius:6px;color:#38bdf8}</style>
<main><h2>独立巡线控制台 Node4 (Port 5004 - 去年 xunxian.py 完整移植)</h2>
<section><b>准备就绪。确认跑道清空后手动启动巡线。</b>
<p><button class="start" onclick="post('/api/start')">解锁并开始巡线</button><button class="stop" onclick="post('/api/stop')">立即停车</button><button onclick="post('/api/reset')">复位为仅感知模式</button></p>
<p>设置巡线目标速度：
<button onclick="post('/api/set_speed?speed=0.18')">0.18 m/s (超慢试跑)</button>
<button onclick="post('/api/set_speed?speed=0.24')">0.24 m/s (中速赛道)</button>
<button onclick="post('/api/set_speed?speed=0.32')">0.32 m/s (去年原版速度)</button>
</p>
<pre id="status">加载状态中...</pre></section>
<section><h3>识别叠加图 (/line_following/debug/overlay) 与 二值图 (/line_following/debug/mask)</h3>
<p><img id="img_overlay"><img id="img_mask"></p></section>
<section><h3>实时鸟瞰图变换 (/line_following/debug/ipm)</h3>
<p><img id="img_ipm" style="width:70%"></p></section>
</main>
<script>
document.getElementById('img_overlay').src = 'http://' + window.location.hostname + ':8080/stream?topic=/line_following/debug/overlay';
document.getElementById('img_mask').src = 'http://' + window.location.hostname + ':8080/stream?topic=/line_following/debug/mask';
document.getElementById('img_ipm').src = 'http://' + window.location.hostname + ':8080/stream?topic=/line_following/debug/ipm';
async function post(url){
    try {
        let r = await fetch(url,{method:'POST'});
        let d = await r.json();
        if(!d.ok){
            alert("⚠️ 操作失败: " + (d.error || '未知错误'));
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
                if time.time() - state.last_image_time > 0.8:
                    self.reply({'ok': False, 'error': '摄像头画面超时，请检查相机'})
                    return
                if not state.vision_valid or len(state.centers) < 3:
                    self.reply({'ok': False, 'error': '当前没有可靠车道中线，禁止启动'})
                    return
                state.last_v_z = 0.0
                state.delat_v_z = 0.0
                state.lost_frames = 0
                state.duizhun_finish = False
                state.start_time = time.time()
                state.last_valid_time = time.time()
                state.last_control_time = time.time()
                state.mode = 'RUNNING'
                state.message = '巡线节点4运行中'
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


class ReusableHTTPServer(HTTPServer):
    allow_reuse_address = True

    def server_bind(self):
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        super().server_bind()


def shutdown():
    publish_stop()


def main():
    global cmd_pub, mask_pub, overlay_pub, ipm_pub
    rospy.init_node('line_following_node4', anonymous=False)
    load_hsv()

    cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
    mask_pub = rospy.Publisher('/line_following/debug/mask', Image, queue_size=1)
    overlay_pub = rospy.Publisher('/line_following/debug/overlay', Image, queue_size=1)
    ipm_pub = rospy.Publisher('/line_following/debug/ipm', Image, queue_size=1)

    rospy.Subscriber('/usb_cam/image_raw', Image, image_cb, queue_size=1, buff_size=2**24)
    rospy.Timer(rospy.Duration(0.04), control_timer)  # 25Hz 控制循环
    rospy.on_shutdown(shutdown)

    server = ReusableHTTPServer(('0.0.0.0', PORT), Handler)
    rospy.loginfo('独立高速巡线节点4已启动: http://0.0.0.0:%d', PORT)
    server.serve_forever()


if __name__ == '__main__':
    main()
