"""激光雷达感知公共模块（原样搬自 polyline_following_hybrid.py 雷达感知）。

提供 RadarScan：订阅 /scan 后由 lane serve 线程调用 process() 更新 left_wall/front_wall，
行为(behaviors/polyline)在控制线程里读取快照，双锁隔离。

命名约定（雷达区为"物理几何转角"）：
- seg_feature 返回的 line_angle = PCA 墙线方向（车体系 0=平行车头, 90=正左）—— 物理角
- d_min = 段最近点距离；perp = 到墙垂直距离
- left_wall 按 LEFT_ANG_LO/HI 最近点角扇区选取；front_wall 按 FRONT_ANG_LO/HI。
"""
import math
import threading

import numpy as np


def angle_seg(ranges, a_min, a_inc, max_break=0.25):
    """按角度顺序把有效点切成若干连续段（相邻半径跳变>max_break 或角度间隙>2.5° 断开）。"""
    n = len(ranges)
    angles = a_min + np.arange(n) * a_inc
    valid = np.where(np.isfinite(ranges))[0]
    valid = valid[(ranges[valid] > 0.1) & (ranges[valid] < 10.0)]
    segs, cur = [], []
    for k in range(len(valid)):
        i = valid[k]
        if not cur:
            cur.append(i)
            continue
        prev = valid[k - 1]
        da = abs(angles[i] - angles[prev])
        dr = abs(ranges[i] - ranges[prev])
        if da > math.radians(2.5) or dr > max_break:
            segs.append(cur)
            cur = [i]
        else:
            cur.append(i)
    if cur:
        segs.append(cur)
    return segs, ranges, angles


def seg_feature(idx, ranges, angles):
    """段的最近距离 / 最近点角 / PCA 墙线方向(车体系, 0=平行车头, 90=正左)。"""
    if len(idx) < 6:
        return None
    xs = np.array([ranges[i] * math.cos(angles[i]) for i in idx])
    ys = np.array([ranges[i] * math.sin(angles[i]) for i in idx])
    d = np.hypot(xs, ys)
    i = int(np.argmin(d))
    c_x, c_y = xs.mean(), ys.mean()
    dx, dy = xs - c_x, ys - c_y
    lam, vec = np.linalg.eigh(np.array([[(dx * dx).sum(), (dx * dy).sum()],
                                        [(dx * dy).sum(), (dy * dy).sum()]]))
    line_ang = math.degrees(math.atan2(vec[1, 1], vec[0, 1]))
    line_ang = ((line_ang % 180) + 180) % 180
    dir_x, dir_y = vec[0, 1], vec[1, 1]
    perp = abs(c_x * dir_y - c_y * dir_x)
    return {
        'd_min': float(d[i]),
        'theta': float(math.degrees(math.atan2(ys[i], xs[i]))),
        'line_angle': float(line_ang),
        'perp': float(perp),
        'n_pts': len(idx),
        'span': float(math.sqrt(max(lam[1], 0.0))),
    }


def front_min_dist(ranges, angles, lo, hi):
    """前扇形(-lo~hi)最小测距（贴近墙角时即"雷达到前墙"距离）。"""
    fi = np.where((angles >= math.radians(lo)) & (angles <= math.radians(hi))
                  & np.isfinite(ranges) & (ranges > 0.1))[0]
    return float(ranges[fi].min()) if len(fi) else None


def sector_points(ranges, angles, lo, hi, rmax=5.0, min_pts=8):
    """取车体一侧扇形点（直角坐标），点数不足返回 None。"""
    idx = np.where(np.isfinite(ranges))[0]
    if len(idx) == 0:
        return None
    r = ranges[idx]
    a = angles[idx]
    keep = (r > 0.1) & (r < rmax) & (a >= math.radians(lo)) & (a <= math.radians(hi))
    idx = idx[keep]
    if len(idx) < min_pts:
        return None
    xs = r[keep] * np.cos(a[keep])
    ys = r[keep] * np.sin(a[keep])
    return np.column_stack([xs, ys])


def fit_wall(pts, min_pts=8, out_thresh=0.05, iters=4):
    """扇形点直线拟合（带离群剔除，抗连续墙角污染）。

    返回 dict: ang(墙线方向 vs 车头, 0=平行, +右偏), perp(垂直距离), n, span, dmin。
    """
    x = pts[:, 0]
    y = pts[:, 1]
    if len(pts) < min_pts:
        return None
    mask = np.ones(len(pts), bool)
    v = None
    for it in range(iters):
        xs, ys = x[mask], y[mask]
        if len(xs) < min_pts:
            return None
        cx, cy = xs.mean(), ys.mean()
        dx, dy = xs - cx, ys - cy
        sxx = float((dx * dx).sum())
        syy = float((dy * dy).sum())
        sxy = float((dx * dy).sum())
        lam, vec = np.linalg.eigh(np.array([[sxx, sxy], [sxy, syy]]))
        if lam[1] <= 0:
            return None
        v = vec[:, 1]
        if v[0] < 0:
            v = -v                     # 统一使 x 分量为正(前向)
        if it < iters - 1:
            d = (x - cx) * v[1] - (y - cy) * v[0]
            mask = np.abs(d) <= out_thresh
            if mask.sum() < min_pts:
                return None
    ang = math.degrees(math.atan2(v[1], v[0]))
    s_perp = float(cx * v[1] - cy * v[0])   # 带符号垂直距离
    d = (x - cx) * v[1] - (y - cy) * v[0]
    return {
        'ang': ang,
        'perp': abs(s_perp),
        's_perp': s_perp,
        'n': int(mask.sum()),
        'span': float(np.hypot(x - cx, y - cy).max()),
        'dmin': float(np.hypot(x, y).min()),
    }


class RadarScan:
    """雷达感知状态。激光订阅线程调 process()（或 scan_cb），控制线程读快照。"""

    def __init__(self, cfg):
        self.lock = threading.Lock()
        self.scan_time = 0.0
        self.left_wall = None
        self.front_wall = None
        self.all_min_d = 999.0
        self.ranges = None
        self.angle_min = 0.0
        self.angle_inc = 0.0
        self.cfg = cfg

    def process(self, ranges, a_min, a_inc, now):
        """从 /scan 消息喂入（可在订阅线程调用），更新各类墙特征。"""
        ranges = np.asarray(ranges, dtype=float)
        with self.lock:
            self.scan_time = now
            self.ranges = ranges
            self.angle_min = a_min
            self.angle_inc = a_inc
        finite_idx = np.where(np.isfinite(ranges))[0]
        all_min = 999.0
        if len(finite_idx):
            all_min = float(ranges[finite_idx].min())
        segs, ranges, angles = angle_seg(ranges, a_min, a_inc, self.cfg['break_gap'])
        feats = []
        for s in segs:
            f = seg_feature(s, ranges, angles)
            if f is not None:
                feats.append(f)
        with self.lock:
            self.all_min_d = all_min
        lw, fw = None, None
        for f in feats:
            if self.cfg['left_ang_lo'] <= f['theta'] <= self.cfg['left_ang_hi']:
                if lw is None or f['d_min'] < lw['d_min']:
                    lw = f
            elif self.cfg['front_ang_lo'] <= f['theta'] <= self.cfg['front_ang_hi']:
                if fw is None or f['d_min'] < fw['d_min']:
                    fw = f
        with self.lock:
            self.left_wall = lw
            self.front_wall = fw

    def scan_fresh(self, now):
        with self.lock:
            return self.scan_time > 0.0 and (now - self.scan_time <= self.cfg['scan_timeout'])

    def left_wall_snapshot(self):
        with self.lock:
            return dict(self.left_wall) if self.left_wall else None

    def front_wall_snapshot(self):
        with self.lock:
            return dict(self.front_wall) if self.front_wall else None

    def min_dist(self):
        with self.lock:
            return self.all_min_d

    def has_scan(self):
        with self.lock:
            return self.scan_time > 0.0

    def snapshot(self):
        """返回最新 scan 快照 (ranges, angle_min, angle_inc)；尚无数据返回 None。"""
        with self.lock:
            if self.ranges is None:
                return None
            return np.array(self.ranges), self.angle_min, self.angle_inc