"""视觉感知公共模块（原样搬自 line_following_ss_pure_ipm.py）。

仅做"纯搬运 + 参数化"改动：
- 全局常量 / TUNE 改为入参 tune；
- state 改为显式的 LaneTracker（含 lane_bias_px / poly 滤波等历史状态）；
- resolve_lane 增加可选 trust_right（hybrid 用），默认值 = ss_pure 行为。
"""
import json
import math

import cv2
import numpy as np

# 统一分辨率（IPM 处理用 640x480 原始相机分辨率）
CAM_W = 640
CAM_H = 480
PROC_W = 320
PROC_H = 240
SCALE_X = PROC_W / CAM_W
SCALE_Y = PROC_H / CAM_H

IPM_CENTER_X = 300.0    # IPM 平面横向中线
CANVAS_X0 = 150.0
CANVAS_Y0 = 200.0
CANVAS_W = 300
CANVAS_H = 400
Y_NEAR = 600.0
Y_FAR = 260.0
ALPHA_POLY = 0.45

IPM_MATRIX = None       # src(图像) -> dst(IPM 平面)
IPM_INV_MATRIX = None   # dst -> src
IPM_CANVAS_M = None     # 组合矩阵：平移后的大画布（含 -origin 平移）


def init_ipm(persp_path):
    """载入 IPM 标定并构建放大画布变换矩阵。"""
    global IPM_MATRIX, IPM_INV_MATRIX, IPM_CANVAS_M
    try:
        with open(persp_path, 'r') as f:
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
        return True
    except Exception as exc:
        print('读取 IPM 配置失败 %s: %s' % (persp_path, exc))
        return False


class LaneTracker:
    """车道线身份跟踪 + 中线多项式滤波历史（进入巡线行为前必须 reset）。"""

    def __init__(self, tune):
        self.reset(tune)

    def reset(self, tune=None):
        half = (tune or {}).get('lane_half_width', 84.0)
        self.track_left = None       # {'coeffs','near_x','miss'}
        self.track_right = None
        self.track_half_width = half
        self.track_valid = False
        self.poly_filt = None        # (a,b,c)
        self.poly_filt_mode = None   # 'P'=配对 'S'=单线
        self.poly_filt_y = (0.0, 0.0)
        self.lane_bias_px = 0.0      # hybrid 单线'R'分支偏置，ARC_FOLLOW 保持 0


def get_full_mask(frame, params):
    """在降采样 (PROC_W, PROC_H) 分辨率上做 HSV 二值化（IPM 与停止线共用）。"""
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
    """将 640x480 二值图变换到 IPM 画布，输出 (CANVAS_W, CANVAS_H)。"""
    return cv2.warpPerspective(mask, IPM_CANVAS_M, (CANVAS_W, CANVAS_H),
                               flags=cv2.INTER_NEAREST)


def clean_ipm_mask(warped, tune):
    """保留竖直细长结构（车道线），剔除横向噪声/反射条带。"""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(warped, cv2.MORPH_CLOSE, kernel, iterations=2)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, 8)
    out = np.zeros_like(cleaned)
    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]
        if h >= tune['clean_min_h'] and area >= tune['clean_min_area'] and h >= w * tune['clean_min_ratio']:
            out[labels == label] = 255
    return out


def fit_lane_line(points, tune):
    """对一组 IPM 采样点做直线拟合 x = k*y + b，两轮去离群。"""
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < tune['min_center_pts'] or np.ptp(pts[:, 1]) < tune['min_center_span']:
        return None
    for _ in range(2):
        k, b = np.polyfit(pts[:, 1], pts[:, 0], 1)
        resid = np.abs(pts[:, 0] - (k * pts[:, 1] + b))
        keep = resid <= tune['poly_max_resid']
        if keep.sum() < tune['min_center_pts']:
            return None
        if keep.sum() == len(pts):
            break
        pts = pts[keep]
    if len(pts) < tune['min_center_pts'] or np.ptp(pts[:, 1]) < tune['min_center_span']:
        return None
    k, b = np.polyfit(pts[:, 1], pts[:, 0], 1)
    return {'coeffs': (0.0, float(k), float(b)),
            'y_min': float(np.min(pts[:, 1])),
            'y_max': float(np.max(pts[:, 1]))}


def fit_lane_line_near(points, tune):
    """近端直线段贪心截断拟合（POLYLINE 首个 LINE_FOLLOW 用）。

    车道线弯曲只发生在远端（y 小），近端（y 大）永远是直线。
    因此从近端向远端逐点贪心扩展，一旦整体拟合残差超过 poly_max_resid 就截断，
    只保留近端连续直线段进入拟合，弯折尾部不参与计算，避免中线被偏折带歪。
    """
    pts = sorted(points, key=lambda p: p[1], reverse=True)
    if len(pts) < tune['min_center_pts']:
        return None
    arr = np.asarray(pts, dtype=np.float64)
    best_n = 0
    for n in range(tune['min_center_pts'], len(arr) + 1):
        sub = arr[:n]
        k, b = np.polyfit(sub[:, 1], sub[:, 0], 1)
        resid = np.max(np.abs(sub[:, 0] - (k * sub[:, 1] + b)))
        if resid > tune['poly_max_resid']:
            break
        best_n = n
    if best_n < tune['min_center_pts']:
        return None
    sub = arr[:best_n]
    if np.ptp(sub[:, 1]) < tune['min_center_span']:
        return None
    k, b = np.polyfit(sub[:, 1], sub[:, 0], 1)
    return {'coeffs': (0.0, float(k), float(b)),
            'y_min': float(np.min(sub[:, 1])),
            'y_max': float(np.max(sub[:, 1]))}


def poly_x(coeffs, y):
    a, b, c = coeffs
    return a * y * y + b * y + c


def poly_k(coeffs, y):
    """x(y) 的局部切线斜率 dx/dy。"""
    a, b, _ = coeffs
    return 2.0 * a * y + b


def lanes_matched(left_fit, right_fit, tune):
    y_min = max(left_fit['y_min'], right_fit['y_min'])
    y_max = min(left_fit['y_max'], right_fit['y_max'])
    if y_max - y_min < tune['match_min_overlap']:
        return False
    y_mid = (y_min + y_max) / 2.0
    w = poly_x(right_fit['coeffs'], y_mid) - poly_x(left_fit['coeffs'], y_mid)
    if not (tune['lane_width_min'] - 15.0 <= w <= tune['lane_width_max'] + 15.0):
        return False
    kL = poly_k(left_fit['coeffs'], y_mid)
    kR = poly_k(right_fit['coeffs'], y_mid)
    return abs(kL - kR) <= tune['max_slope_diff']


def _find_sides_from_center(means, center):
    """从画面中心向两侧搜索：分别返回中心左、右侧最靠近中心的一个簇 x。"""
    left = None
    for x in sorted((m for m in means if m < center - 10), reverse=True):
        left = x
        break
    right = None
    for x in sorted(m for m in means if m > center + 10):
        right = x
        break
    return left, right


def extract_raw_lanes(warped, tune, near_straight=False):
    """逐行扫描提取左右车道线采样点并各自直线拟合。返回 (left_fit, right_fit, width_samples)。

    near_straight=True 时用 fit_lane_line_near（近端直线段贪心截断，POLYLINE 首段巡线用）。
    """
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
        means = [int(np.mean(c)) for c in clusters if 2 <= len(c) <= tune['max_cluster_w']]
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
            if tune['lane_width_min'] <= width <= tune['lane_width_max']:
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

    fitter = fit_lane_line_near if near_straight else fit_lane_line
    left_fit = fitter(left_pts, tune)
    right_fit = fitter(right_pts, tune)
    return left_fit, right_fit, width_samples


def _side_of(ls, near_x, tune):
    """结合历史身份给一条线定左右：上一帧是右线，这一帧仍按右线处理。"""
    if ls.track_left is not None and ls.track_right is not None:
        dL = abs(near_x - ls.track_left['near_x'])
        dR = abs(near_x - ls.track_right['near_x'])
        return 'L' if dL <= dR else 'R'
    if ls.track_left is not None:
        if abs(near_x - ls.track_left['near_x']) <= tune['lane_half_width']:
            return 'L'
        return 'L' if near_x < IPM_CENTER_X else 'R'
    if ls.track_right is not None:
        if abs(near_x - ls.track_right['near_x']) <= tune['lane_half_width']:
            return 'R'
        return 'L' if near_x < IPM_CENTER_X else 'R'
    return 'L' if near_x < IPM_CENTER_X else 'R'


def _update_tracker(ls, left_fit, right_fit, half_width, tune):
    """用本帧拟合更新左右身份轨迹（EMA），一侧缺失时保留旧值并计数。"""
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
    if ls.track_left is not None and ls.track_left['miss'] > tune['track_stale_frames']:
        ls.track_left = None
    if ls.track_right is not None and ls.track_right['miss'] > tune['track_stale_frames']:
        ls.track_right = None
    if half_width is not None:
        ls.track_half_width = half_width
    ls.track_valid = (ls.track_left is not None or ls.track_right is not None)


def resolve_lane(ls, left_fit, right_fit, width_samples, tune, trust_right=False):
    """结合历史身份确定左右线、配对或单线，输出中线误差，并更新轨迹。

    trust_right=False 时为 ss_pure 行为；trust_right=True（hybrid 用）时单线强制按右线处理。
    ls.lane_bias_px 在单线判为 'R' 时叠加到半宽上（hybrid 的"跟左线偏右10px"，默认 0）。
    """
    cand = []
    for pos_side, fit in (('L', left_fit), ('R', right_fit)):
        if fit is not None:
            y_ref = min(fit['y_max'], Y_NEAR)
            cand.append({'pos_side': pos_side, 'fit': fit,
                         'near_x': poly_x(fit['coeffs'], y_ref)})

    lane_width = float(np.mean(width_samples)) if width_samples else None
    half_width = ls.track_half_width if ls.track_valid else tune['lane_half_width']
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
        width_ok = tune['lane_width_min'] <= width <= tune['lane_width_max']
        hist_ok = (not ls.track_valid or
                   abs(width - 2.0 * ls.track_half_width) <= tune['max_width_dev'])
        if width_ok and hist_ok and lanes_matched(left_c['fit'], right_c['fit'], tune):
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
            cls = 'R' if trust_right else _side_of(ls, single['near_x'], tune)
            a, b, c = single['fit']['coeffs']
            if cls == 'L':
                coeffs = (a, b, c + half_width)
                left_out, right_out = single['fit'], None
            else:
                coeffs = (a, b, c - (half_width + ls.lane_bias_px))
                left_out, right_out = None, single['fit']
            kanbujian = True
            y_min, y_max = single['fit']['y_min'], single['fit']['y_max']

    if coeffs is None or y_max <= y_min:
        _update_tracker(ls, None, None, None, tune)
        return None

    center_y_range = (y_min, y_max)
    center_error_px = float(poly_x(coeffs, Y_NEAR) - IPM_CENTER_X)
    y_head = min(Y_NEAR - tune['lookahead_px'], y_max)
    kt = poly_k(coeffs, y_head)
    heading_deg = float(math.degrees(math.atan2(-kt, 1.0)))

    _update_tracker(ls, left_out, right_out, half_width if pair_matched else None, tune)

    return {
        'center_error_px': center_error_px,
        'heading_error_deg': heading_deg,
        'lane_width_px': lane_width,
        'kanbujian': kanbujian,
        'pair_matched': pair_matched,
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


def apply_poly_filter(ls, result, tune):
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
    y_head = min(Y_NEAR - tune['lookahead_px'], y1)
    kt = poly_k(coeffs, y_head)
    result['heading_error_deg'] = float(math.degrees(math.atan2(-kt, 1.0)))
    return result


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


def extract_horizontal_bands(bin_img, kernel_w=31):
    if bin_img is None or bin_img.size == 0:
        return bin_img
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 1))
    return cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel)


def remove_horizontal_bands_ipm(warped, tune, kernel_w=41):
    """在 IPM 画布上剔除横向长条（停止线/终点线）。必须放在 clean_ipm_mask 之前。"""
    if warped is None or warped.size == 0:
        return warped
    horiz = extract_horizontal_bands(warped, kernel_w)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(horiz, 8)
    band = np.zeros_like(warped)
    for label in range(1, num_labels):
        x, y, bw, bh, area = stats[label]
        if bw >= tune['lane_width_min'] and bh <= bw * 0.4:
            band[labels == label] = 255
    clean = warped.copy()
    clean[band == 255] = 0
    return clean


def detect_stop_line(bin_img, top_ratio=0.80, width_ratio=0.40, thin_ratio=0.40):
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


# ==================== 路口对面中线检测（移植自 simple_turn_trial.py，原样保持翻转约定）====================
# 注意：simple_turn 先 cv2.flip(frame,1) 再 make_mask，检测内部用 639.0-x 还原原始相机坐标后进 IPM。
# 此处保持完全一致（翻转 + 还原自抵消），保证与实车阶段一行为一致。

OPP_ROI_BOTTOM = 0.85
OPP_ROI_TOP = 0.55
OPP_WIDTH_MIN = 120.0
OPP_WIDTH_MAX = 450.0
OPP_SLOPE_DIFF_MAX = 0.20
OPP_MIN_OVERLAP = 30.0
OPP_MIN_SPAN = 60.0
OPP_FIT_RESID = 30.0


def make_mask640(frame, params):
    """640x480 全分辨率 HSV 二值化 + ROI + 腐蚀膨胀（移植自 simple_turn make_mask）。"""
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


def opp_scan_side_points(mask_center, center_img_x):
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


def opp_detect_side_fit(pts, min_span, ipm_matrix):
    if len(pts) < 4:
        return None
    a = np.float32(pts).reshape(-1, 1, 2).copy()
    a[:, 0, 0] = 639.0 - a[:, 0, 0]  # 还原原始相机坐标（与 simple_turn 翻转约定一致）
    ipm = cv2.perspectiveTransform(a, ipm_matrix).reshape(-1, 2)
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
    """提取路口对面出口中线。mask 需来自 make_mask640(翻转后的帧)。"""
    if IPM_MATRIX is None or IPM_INV_MATRIX is None:
        return None
    mask_center = mask.copy()
    cut_bot = int(mask.shape[0] * OPP_ROI_BOTTOM)
    mask_center[cut_bot:, :] = 0
    cut_top = int(mask.shape[0] * OPP_ROI_TOP)
    mask_center[:cut_top, :] = 0
    h, w = mask_center.shape
    center_x = w // 2

    left_pts, right_pts = opp_scan_side_points(mask_center, center_x)
    left_fit = opp_detect_side_fit(left_pts, OPP_MIN_SPAN, IPM_MATRIX)
    right_fit = opp_detect_side_fit(right_pts, OPP_MIN_SPAN, IPM_MATRIX)
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

    return {
        'score': slope_diff * 10.0 + abs(width - 220.0) / 220.0,
        'k': k_c, 'b': b_c,
        'center_error': float(center_near - IPM_CENTER_X),
        'heading_error_deg': float(heading_deg),
        'width': float(width), 'slope_diff': float(slope_diff),
        'y_min': overlap_min, 'y_max': y_near,
        'left_fit': left_fit, 'right_fit': right_fit,
    }
