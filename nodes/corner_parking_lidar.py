#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""corner_parking_lidar.py —— 雷达引导墙角停车（独立节点）。

停车目标：小车停进 50cm 方形停车区，最后雷达扫到前墙 ~17cm、
侧墙 ~24cm。**不要求横平竖直**，只要不撞墙、车体在停车位以内。

为什么独立出来：每种巡线路径最后都要靠激光雷达引导停车。视觉巡线
可能没停好（甚至斜着/直直开进停车位），所以本节点从现场激光雷达
数据自行推断"哪一面墙更近"，贴那面墙停车。

策略：
  DETECT -> (PRE_ALIGN) -> DRIVE -> DONE
  DETECT    现场推断目标侧墙（左/右取垂直距离较小且墙线大致平行于车头者）；
            无侧墙且前方开阔则缓慢蠕进尝试入位，否则原地等待。
  PRE_ALIGN 若雷达到侧墙垂直距离偏离 0.25m 超过容差，先用纯侧移(vy)横移到 0.25m
            再入库：离太远(>0.27)朝墙移、靠太近(<0.23)背墙拉开；足够近则跳过本步。
  DRIVE     前进 + 贴墙转向（只用 vx+wz，曲线入库，无原地旋转）：
            vx = KP_F*(前墙距-目标)   前向驶入，接近前墙自动减速、到位刹车；
            wz = KP_ANG*墙线角 + 侧向距离误差项 + 积分项   平滑曲线收敛到
                与侧墙平行(墙线角→0)、垂直距离→24cm；接近前墙时 wz 减弱避免甩尾撞墙。
  DONE    前墙到位 且 侧距/墙线角在放宽区间内（连续 N 帧）停车；
          若已越过前墙目标(前墙距<目标)则无条件立即刹车，不再纠结侧向。

感知（墙角是连通 90° L 形，前墙与侧墙连成一段）：
  * 侧墙：车体一侧扇形点(35°~150°)直线拟合+离群剔除，取垂直距离 perp 与
    墙线方向角 ang(0=平行车头)；排除墙线角>45°的墙(即前墙)防误判。
  * 前墙：前扇形(-30°~30°)最小测距，贴近墙角时即"雷达到前墙"距离。

触发：按 s+回车 解锁；订阅 /parking/start(std_msgs/Bool) true 也可解锁。q 退出。
停车完成信号：/parking/done(std_msgs/Bool, latched) 置 True —— 供语音播报等下游任务触发；
/parking/status(String, latched) 同步发布 "DONE | 原因"。解锁/解除时 /parking/done 置 False。
安全/超时兜底：每阶段都有超时保证不停留——DETECT 超时无墙→ESTOP 安全停；PRE_ALIGN 超时→进 DRIVE；
DRIVE 超时→已近墙角(放宽区间)则 DONE 兜底停车，否则 ESTOP；scan 超时、<0.13m、行程超限均急停。
启动（需 ydlidar 已起；与视觉/混合节点须二选一，勿同时发 /cmd_vel）：
  rosrun corner_parking_lidar.py
"""
import math
import sys
import threading
import time

import numpy as np
import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, String

# ---------------- 停车目标（雷达测距，实车标定后调整） ----------------
FRONT_TARGET = 0.17     # 雷达到前墙的目标距离 m（车头离前墙约 17cm，到位即停）
FRONT_TOL = 0.015       # 前墙到位判定容差 m（|front - FRONT_TARGET| <= 此值视为到位）
SIDE_TARGET = 0.24      # 雷达到侧墙的目标距离 m（DRIVE 阶段转向收敛目标）
SIDE_MIN = 0.16         # 到位判定：侧距下限 m（过小会压侧墙，判定不通过）
SIDE_MAX = 0.34         # 到位判定：侧距上限 m（仍在 50cm 停车位内）
ANG_TOL = 15.0          # 到位判定：墙线角上限 deg（允许不横平竖直，仅防空撞）
DONE_FRAMES = 10        # 到位需连续确认的帧数（控制周期 20Hz => 0.5s）

# ---------------- 入库前平移贴墙（PRE_ALIGN，对称：离太远朝墙移、靠太近背墙拉开） ----------------
SIDE_PRE_DIST = 0.25    # 平移目标：雷达与侧墙垂直距离 m（起始偏离超过容差即对称横移到此值）
PRE_TOL = 0.02          # 平移到位判定容差 m（|perp - SIDE_PRE_DIST| <= 此值进入 DRIVE）
PRE_VEL = 0.15          # 平移(纯侧移 vy)最大速度 m/s
KP_Y = 2.0              # 平移比例增益：vy = KP_Y*(perp - SIDE_PRE_DIST)，误差越大移得越快
PRE_ALIGN_TIMEOUT = 5.0 # 平移最长时长 s，超时即使没到位也进入 DRIVE（防止卡死）

# ---------------- 速度 / 增益（DRIVE 阶段只前进+转向） ----------------
VX_MAX = 0.15           # 前进速度上限 m/s
VX_MIN = 0.06           # 前进速度下限 m/s，越过电机死区保证接近前墙
KP_F = 0.35             # 前进比例增益：vx = KP_F*(front - target)，~0.6m 处被钳位到 VX_MAX
KP_ANG = 0.5            # 墙线角反馈：wz += KP_ANG * ang_rad
KP_SIDE = 1.5           # 侧距反馈：wz += ±KP_SIDE*(perp - SIDE_TARGET)
KI_SIDE = 0.6           # 侧距积分增益，消除稳态侧距偏差
S_INT_MAX = 0.06        # 侧距积分项钳位，防积分过大
WZ_MAX = 0.35           # 转向角速度上限 rad/s
WZ_NEAR_WALL_START = 0.25   # 前墙距低于此值开始按比例减弱转向（避免甩尾蹭墙）m
WZ_NEAR_WALL_END = 0.03     # 前墙距低于此值时转向完全关闭 m

DETECT_CREEP = 0.08     # DETECT 阶段无侧墙时的蠕进速度 m/s（试探前方是否就是墙角）
DETECT_CREEP_MAX_T = 3.0    # 蠕进最长时长 s，超时原地等待，避免开进死胡同

# ---------------- 感知扇区 ----------------
FRONT_ANG_LO = -30.0    # 前墙检测扇区下界 deg（相对车头，右负）
FRONT_ANG_HI = 30.0     # 前墙检测扇区上界 deg（相对车头，左正）
SIDE_ANG_MIN = 35.0     # 左墙检测扇区下界 deg
SIDE_ANG_MAX = 150.0    # 左墙检测扇区上界 deg（右墙对称取负）
SIDE_PARALLEL_TOL = 45.0    # 判定为"真侧墙"的最大墙线角 deg（防把前墙当侧墙）
SIDE_ARM_MAX = 0.60     # 侧墙可用的最大垂直距离 m（再远视为不存在该墙）
SIDE_HOLD_DIST = 0.40   # 前墙距小于此值且无侧墙时必须停下（避免撞墙）m
MIN_PTS = 8             # 扇形直线拟合最少点数，不足则判为该墙不可用

# ---------------- 安全 / 超时（每阶段都有兜底，保证不会卡死） ----------------
MIN_SAFE_DIST = 0.13    # 任意方向最近测距低于此值即急停 m
SCAN_TIMEOUT = 0.6      # 超过此时长未收到 /scan 即判雷达丢失并停车 s
DETECT_TIMEOUT = 10.0   # DETECT 总超时 s：超时仍未推断出侧墙 → 安全停车(ESTOP)，不再空等
DRIVE_TIMEOUT = 15.0    # DRIVE 最长时长 s：超时按"放宽区间"判断已到位则 DONE 兜底停车，否则 ESTOP
DRIVE_TRAVEL_MAX = 1.0  # DRIVE 最大行进距离 m，超限停车（防冲撞）
DRIVE_WIDE_FRONT = 0.35 # DRIVE 超时兜底：前墙距 ≤ 此值才接受"停车完成" m
DRIVE_WIDE_SIDE_TOL = 0.05  # DRIVE 超时兜底：侧距放宽量 m（[SIDE_MIN-tol, SIDE_MAX+tol]）
DRIVE_WIDE_ANG = 30.0   # DRIVE 超时兜底：墙线角放宽上限 deg

# ---------------- 状态 -------------------
state_lock = threading.Lock()
scan_time = 0.0
ranges_global = None
angle_min_g = 0.0
angle_inc_g = 0.0
front_d = None       # 雷达到前墙距离或 None
left_fit = None      # 左扇形墙线拟合 dict
right_fit = None
all_min_d = 999.0
odom = None

mode = 'DETECT'
armed = False
side = None          # 'left' / 'right' / None
int_s = 0.0          # 侧距积分
last_s_t = 0.0
drive_t0 = 0.0
drive_pose0 = None
creep_t0 = 0.0
detect_t0 = 0.0
pre_t0 = 0.0
pre_lost_t = 0.0
done_frames = 0
cmd_pub = None
status_pub = None
done_pub = None


def normalize_angle(a):
    return math.atan2(math.sin(a), math.cos(a))


# ---------------- 感知 ----------------
def sector_points(ranges, angles, lo, hi, rmax=5.0):
    idx = np.where(np.isfinite(ranges))[0]
    if len(idx) == 0:
        return None
    r = ranges[idx]
    a = angles[idx]
    keep = (r > 0.1) & (r < rmax) & (a >= math.radians(lo)) & (a <= math.radians(hi))
    idx = idx[keep]
    if len(idx) < MIN_PTS:
        return None
    xs = r[keep] * np.cos(a[keep])
    ys = r[keep] * np.sin(a[keep])
    return np.column_stack([xs, ys])


def fit_wall(pts, out_thresh=0.05, iters=4):
    """扇形点直线拟合（带离群剔除，抗连续墙角污染）。

    连续 90° 墙角里前墙与侧墙连成一段，侧墙扇形里会混入前墙点（都靠近牛角顶点、垂直于侧墙），
    直接 PCA 会把墙线姿态撇歪。这里用"拟直线-剔除离直线>5cm 的点-再拟"迭代数轮，收敛到多数点
    所在的真实侧墙上。

    返回 dict: ang(墙线方向 vs 车头, 0=平行, +右偏), perp(垂直距离), n, span, dmin。
    """
    x = pts[:, 0]
    y = pts[:, 1]
    if len(pts) < MIN_PTS:
        return None
    mask = np.ones(len(pts), bool)
    v = None
    cx = cy = 0.0
    for it in range(iters):
        xs, ys = x[mask], y[mask]
        if len(xs) < MIN_PTS:
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
            if mask.sum() < MIN_PTS:
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


def scan_cb(msg):
    global scan_time, ranges_global, angle_min_g, angle_inc_g
    global front_d, left_fit, right_fit, all_min_d
    ranges = np.array(msg.ranges)
    a_min, a_inc = msg.angle_min, msg.angle_increment
    angles = a_min + np.arange(len(ranges)) * a_inc
    with state_lock:
        scan_time = time.time()
        ranges_global = ranges
        angle_min_g, angle_inc_g = a_min, a_inc

    fin = np.where(np.isfinite(ranges))[0]
    all_min_d = float(ranges[fin].min()) if len(fin) else 999.0

    fi = np.where((angles >= math.radians(FRONT_ANG_LO))
                  & (angles <= math.radians(FRONT_ANG_HI)) & np.isfinite(ranges)
                  & (ranges > 0.1))[0]
    front_d = float(ranges[fi].min()) if len(fi) else None

    lp = sector_points(ranges, angles, SIDE_ANG_MIN, SIDE_ANG_MAX)
    rp = sector_points(ranges, angles, -SIDE_ANG_MAX, -SIDE_ANG_MIN)
    with state_lock:
        left_fit = fit_wall(lp) if lp is not None else None
        right_fit = fit_wall(rp) if rp is not None else None


def odom_cb(msg):
    global odom
    q = msg.pose.pose.orientation
    yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
    with state_lock:
        odom = (msg.pose.pose.position.x, msg.pose.pose.position.y, yaw)


def publish_stop():
    if cmd_pub is not None:
        cmd_pub.publish(Twist())


def mark_done(src):
    """停车完成：发出明确的完成信号（供语音播报等下游触发）。"""
    global mode
    mode = 'DONE'
    publish_stop()
    if done_pub is not None:
        done_pub.publish(True)
    if status_pub is not None:
        status_pub.publish('DONE | %s' % src)
    rospy.loginfo('=== 停车完成 === %s', src)


def safe_guard():
    if time.time() - scan_time > SCAN_TIMEOUT:
        return 'scan_timeout'
    if all_min_d < MIN_SAFE_DIST:
        return 'too_close_%.3f' % all_min_d
    return None


def pick_side(lw, rw):
    cand = []
    if lw is not None and lw['n'] >= MIN_PTS and abs(lw['ang']) <= SIDE_PARALLEL_TOL and lw['perp'] <= SIDE_ARM_MAX:
        cand.append(('left', lw))
    if rw is not None and rw['n'] >= MIN_PTS and abs(rw['ang']) <= SIDE_PARALLEL_TOL and rw['perp'] <= SIDE_ARM_MAX:
        cand.append(('right', rw))
    if not cand:
        return None
    cand.sort(key=lambda t: t[1]['perp'])
    return cand[0][0]


def side_fit(sd):
    return left_fit if sd == 'left' else right_fit


def control():
    global mode, side, int_s, last_s_t, drive_t0, drive_pose0, creep_t0, detect_t0, pre_t0, pre_lost_t, done_frames
    if not armed:
        publish_stop()
        return
    guard = safe_guard()
    if guard is not None:
        mode = 'ESTOP'
        rospy.logerr('安全触发: %s', guard)
        publish_stop()
        return
    if mode == 'ESTOP':
        publish_stop()
        return

    with state_lock:
        fd = front_d
        lw = dict(left_fit) if left_fit else None
        rw = dict(right_fit) if right_fit else None
        odo = odom
    now = time.time()
    cmd = Twist()
    src = mode

    if mode == 'DETECT':
        if now - detect_t0 > DETECT_TIMEOUT:
            mode = 'ESTOP'
            rospy.logerr('DETECT 超时(%.0fs)未推断出侧墙, 安全停车', DETECT_TIMEOUT)
            publish_stop()
            return
        sd = pick_side(lw, rw)
        if sd is None:
            if fd is not None and fd < SIDE_HOLD_DIST:
                cmd.linear.x = 0.0
                src = 'DETECT_HOLD no-side front=%.3f' % fd
            else:
                if creep_t0 == 0.0:
                    creep_t0 = now
                if now - creep_t0 < DETECT_CREEP_MAX_T:
                    cmd.linear.x = DETECT_CREEP
                    src = 'DETECT_CREEP front=%s' % ('%.3f' % fd if fd is not None else 'inf')
                else:
                    cmd.linear.x = 0.0
                    src = 'DETECT_HOLD creep-timeout front=%s' % ('%.3f' % fd if fd is not None else 'inf')
        else:
            side = sd
            fw = side_fit(side)
            if abs(fw['perp'] - SIDE_PRE_DIST) > PRE_TOL:
                mode = 'PRE_ALIGN'
                pre_t0 = now
                pre_lost_t = 0.0
                src = '->PRE_ALIGN side=%s perp=%.3f (需横移到%.2f)' % (side, fw['perp'], SIDE_PRE_DIST)
            else:
                mode = 'DRIVE'
                int_s = 0.0
                last_s_t = now
                drive_t0 = now
                drive_pose0 = odo
                done_frames = 0
                src = '->DRIVE side=%s perp=%.3f ang=%+.0f' % (side, fw['perp'], fw['ang'])

    elif mode == 'PRE_ALIGN':
        fw = side_fit(side) if side else None
        dt = max(0.0001, now - last_s_t)
        last_s_t = now
        if fw is None:
            pre_lost_t += dt
            cmd.linear.x = 0.0
            src = 'PRE_ALIGN_HOLD wall-lost %.1fs' % pre_lost_t
            if pre_lost_t > 1.0:
                mode = 'DETECT'
                side = None
                creep_t0 = 0.0
                detect_t0 = now
                src = '->DETECT prealign-wall-lost'
        else:
            pre_lost_t = 0.0
            err = fw['perp'] - SIDE_PRE_DIST
            if abs(err) <= PRE_TOL:
                mode = 'DRIVE'
                int_s = 0.0
                drive_t0 = now
                drive_pose0 = odo
                done_frames = 0
                src = '->DRIVE prealign-done perp=%.3f' % fw['perp']
            else:
                ssign = 1.0 if side == 'left' else -1.0
                vy = ssign * max(-PRE_VEL, min(PRE_VEL, KP_Y * err))
                cmd.linear.y = vy
                src = 'PRE_ALIGN side=%s perp=%.3f vy=%.2f' % (side, fw['perp'], vy)
        if now - pre_t0 > PRE_ALIGN_TIMEOUT and mode == 'PRE_ALIGN':
            mode = 'DRIVE'
            int_s = 0.0
            drive_t0 = now
            drive_pose0 = odo
            done_frames = 0
            src = '->DRIVE prealign-timeout perp=%s' % ('%.3f' % fw['perp'] if fw else 'nan')

    elif mode == 'DRIVE':
        fw = side_fit(side) if side else None
        if fw is None:
            mode = 'DETECT'
            side = None
            creep_t0 = 0.0
            detect_t0 = now
            src = '->DETECT drive-wall-lost'
        else:
            ang = fw['ang']
            err_s = fw['perp'] - SIDE_TARGET
            dt = max(0.0001, now - last_s_t)
            last_s_t = now
            int_s += err_s * dt
            int_s = max(-S_INT_MAX, min(S_INT_MAX, int_s))
            ssign = 1.0 if side == 'left' else -1.0

            # 侧向贴墙转向（仅 wz，无侧移）
            steer_ang = KP_ANG * math.radians(ang)
            steer_side = ssign * (KP_SIDE * err_s + KI_SIDE * int_s)
            wz_raw = steer_ang + steer_side

            # 前向速度：接近前墙自动减速，到位刹车
            if fd is None:
                vx = 0.0
                src = 'DRIVE_HOLD no-front front=inf'
            else:
                err_f = fd - FRONT_TARGET
                if err_f > FRONT_TOL:
                    vx = max(VX_MIN, min(VX_MAX, KP_F * err_f))
                else:
                    vx = 0.0
                # 接近前墙减弱转向，避免最后一刻甩尾蹭墙
                wz_scale = max(0.0, min(1.0,
                    (fd - WZ_NEAR_WALL_END) / max(0.01, WZ_NEAR_WALL_START - WZ_NEAR_WALL_END)))
                wz = max(-WZ_MAX, min(WZ_MAX, wz_raw * wz_scale))
                src = 'DRIVE side=%s front=%.3f perp=%.3f ang=%+.1f vx=%.2f wz=%.2f' % (
                    side, fd, fw['perp'], ang, vx, wz)
                cmd.linear.x = vx
                cmd.angular.z = wz

                # 到位判定（放宽区间）
                if fd is not None and abs(fd - FRONT_TARGET) <= FRONT_TOL \
                   and SIDE_MIN <= fw['perp'] <= SIDE_MAX and abs(ang) <= ANG_TOL:
                    done_frames += 1
                else:
                    done_frames = 0

                if fd is not None and fd < FRONT_TARGET - FRONT_TOL:
                    # 已越过前墙目标：无条件刹车，不再纠结侧向
                    mark_done('overshoot front=%.3f' % fd)
                    return
                if done_frames >= DONE_FRAMES:
                    mark_done('front=%.3f perp=%.3f ang=%+.1f' % (fd, fw['perp'], ang))
                    return

            if now - drive_t0 > DRIVE_TIMEOUT:
                wide = (fd is not None and fd <= DRIVE_WIDE_FRONT
                        and SIDE_MIN - DRIVE_WIDE_SIDE_TOL <= fw['perp'] <= SIDE_MAX + DRIVE_WIDE_SIDE_TOL
                        and abs(ang) <= DRIVE_WIDE_ANG)
                if wide:
                    mark_done('drive-timeout wide-accept front=%.3f perp=%.3f ang=%+.1f' % (
                        fd, fw['perp'], ang))
                    return
                mode = 'ESTOP'
                rospy.logerr('DRIVE 超时(%.0fs)且不在车位内, 安全停车', DRIVE_TIMEOUT)
                publish_stop()
                return
            if odo is not None and drive_pose0 is not None:
                trav = math.hypot(odo[0] - drive_pose0[0], odo[1] - drive_pose0[1])
                if trav > DRIVE_TRAVEL_MAX:
                    mode = 'ESTOP'
                    rospy.logerr('DRIVE 行程超限(%.2f m)停车', trav)
                    publish_stop()
                    return

    elif mode == 'DONE':
        publish_stop()
        if status_pub is not None:
            status_pub.publish('DONE')   # 明确停车完成信号（latched 常驻）
        return
    else:
        publish_stop()
        return

    cmd_pub.publish(cmd)
    if status_pub is not None:
        status_pub.publish('%s | %s' % (mode, src))
    if int(now * 5) % 5 == 0:
        rospy.loginfo('[%s] %s', mode, src)


def wait_for_scan(timeout=1.5):
    """等待雷达首帧数据：容忍订阅刚建立时的连接窗口，避免误报"雷达未启动"。

    收到过 /scan 且新鲜度在 SCAN_TIMEOUT 内即返回 True；等满 timeout 仍无
    首帧才返回 False（此时才提示检查 ydlidar）。
    """
    t0 = time.time()
    while time.time() - t0 < timeout:
        if scan_time > 0.0 and time.time() - scan_time <= SCAN_TIMEOUT:
            return True
        time.sleep(0.05)
    return scan_time > 0.0


def arm():
    global mode, armed, side, creep_t0, detect_t0, pre_t0, pre_lost_t, int_s, last_s_t, done_frames
    if not wait_for_scan():
        rospy.logwarn('等 %.1fs 仍未收到 /scan，请确认 ydlidar 已启动且 /scan 在发布', 1.5)
        return False
    with state_lock:
        armed = True
    mode = 'DETECT'
    side = None
    creep_t0 = 0.0
    detect_t0 = time.time()
    pre_t0 = 0.0
    pre_lost_t = 0.0
    int_s = 0.0
    last_s_t = time.time()
    done_frames = 0
    if done_pub is not None:
        done_pub.publish(False)   # 解锁时清掉完成信号
    rospy.loginfo('>>> 已解锁启动，进入 DETECT(推断侧墙)')
    return True


def disarm():
    global armed
    with state_lock:
        armed = False
    publish_stop()
    if done_pub is not None:
        done_pub.publish(False)
    rospy.loginfo('>>> 已停车/解除')


def start_cb(msg):
    if bool(msg.data):
        arm()
    else:
        disarm()


def stdin_thread():
    print('=== corner_parking_lidar ===')
    print('按 s+回车 解锁启动   按 q+回车 退出')
    while True:
        try:
            line = sys.stdin.readline()
        except Exception:
            break
        if not line:
            break
        c = line.strip().lower()
        if c == 's':
            arm()
        elif c == 'q':
            print('>>> 退出')
            rospy.signal_shutdown('user quit')


def main():
    global cmd_pub, status_pub, done_pub
    rospy.init_node('corner_parking_lidar', anonymous=False)
    cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
    status_pub = rospy.Publisher('/parking/status', String, queue_size=1, latch=True)
    done_pub = rospy.Publisher('/parking/done', Bool, queue_size=1, latch=True)
    rospy.Subscriber('/scan', LaserScan, scan_cb, queue_size=1)
    rospy.Subscriber('/odom', Odometry, odom_cb, queue_size=1)
    rospy.Subscriber('/parking/start', Bool, start_cb, queue_size=1)
    rospy.Timer(rospy.Duration(0.05), lambda e: control())
    rospy.on_shutdown(lambda: publish_stop())
    t = threading.Thread(target=stdin_thread, daemon=True)
    t.start()
    rospy.spin()


if __name__ == '__main__':
    main()