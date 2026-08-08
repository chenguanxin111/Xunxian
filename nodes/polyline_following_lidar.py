#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""雷达纯引导测试脚本：斜车道贴墙右转 -> 平行车道直行 -> 前方墙前停车。

不依赖视觉。订阅 /scan 与 /odom，发布 /cmd_vel。
按 bag3 实测三阶段状态机：
  APPROACH  左墙距离 > 0.40m：直行接近 y=5 墙
  TURN      左墙距离 <= 0.40m：右转使墙线趋于平行(90°)，距离收敛到 0.22m
  PARALLEL  墙线平行且距离~0.22m：左墙距离保持 PID，前方 x=8 墙减速刹车

启动方式：rosrun polyline_following_lidar.py  （需雷达 ydlidar 已启动）
安全：scan 超时停车；任意方向 < 0.13m 急停；前方墙 0.30m 减速、0.15m 刹车。
"""
import sys
import math
import time
import threading
import numpy as np

import rospy
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist

# ---------------- 参数 ----------------
KEEP_WALL_DIST = 0.26      # 贴墙安全距离 m（平行段保持目标）
TURN_TRIGGER_DIST = 0.36   # 左墙距离 <= 此值触发转弯（提前转反而会往内侧绕压尖角，保持贴近墙角）
TURN_DONE_DIST = 0.28      # 转弯段距离收敛阈值（略高于 KEEP_WALL_DIST）
PARALLEL_ANGLE = 15.0      # 平行度阈值
FORWARD_SPEED = 0.20       # 前进速度 m/s
TURN_WZ_MAX = -0.40      # 转弯右转最大角速度
TURN_WZ_MIN = -0.18        # 转弯右转最小角速度
KP_TURN_ANG = 0.008        # 转弯段按墙线偏差的比例
TURN_GUARD_DIST = 0.18     # 转弯过深保护：d 低于此值减少右转、提速脱离墙角
TURN_GUARD_WZ = -0.10      # 过深时右转(负)上限：更平缓弧线，避免扎进尖角
KP_PARALLEL = 0.55         # 平行段距离保持 P
KI_PARALLEL = 0.35         # 平行段距离保持 I（消除稳态距离偏差）
PARALLEL_INT_MAX = 0.15    # 积分项钳位（防 windup）
KP_ANG_HEAD = 0.02         # 平行段航向(墙线角)反馈 P
PARALLEL_WZ_CLAMP = 0.30
FRONT_SLOW_DIST = 0.60     # 前方墙距离 m：低于减速
FRONT_STOP_DIST = 0.50     # 前方墙距离 m：刹车
FRONT_SLOW_SPEED = 0.08
MIN_SAFE_DIST = 0.13       # 任意方向最近距离低于此值急停
SAFE_CLAMP_DIST = 0.18     # 左墙距离低于此值才禁止航向项左转(靠墙)
ALIGN_TARGET = 135.0       # 原地对准目标墙线角：135°=车与墙成45°
ALIGN_STOP_ERR = 2.0       # 对准收敛误差(度)：|err|<=2 => 夹角∈[43,47]
KP_ALIGN = 0.06            # 原地对准：角度偏差(度)->角速度(rad/s)
ALIGN_WZ_MAX = 0.35        # 原地对准最大角速度
ALIGN_WZ_MIN = 0.15        # 原地对准最小角速度(越过电机死区，确保转得动)
ALIGN_TIMEOUT = 5.0        # 原地对准超时(s)：卡死则强制进入 APPROACH
LEFT_ANG_LO = 25.0         # 左墙最近点角范围 [25,155]（正角=左）
LEFT_ANG_HI = 155.0
FRONT_ANG_LO = -30.0       # 前墙最近点角范围
FRONT_ANG_HI = 30.0
SCAN_TIMEOUT = 0.6         # scan 超时 s
BREAK_GAP = 0.25           # 聚类相邻点半径跳变阈值 m

# ---------------- 状态 ----------------
state_lock = threading.Lock()
scan_time = 0.0
ranges_global = None
angle_min_g = 0.0
angle_inc_g = 0.0
left_wall = None   # dict: d_min, theta, line_angle, n_pts
front_wall = None
all_min_d = 999.0
odom = None
int_err = 0.0      # 平行段距离保持 I 累加器
last_par_t = 0.0
prealign_t0 = 0.0  # 原地对准进入时刻(超时保护)

mode = 'APPROACH'
cmd_pub = None
armed = False
stdin_ready = threading.Event()


def angle_seg(ranges, a_min, a_inc, max_break=BREAK_GAP):
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
    lam, vec = np.linalg.eigh(np.array([[ (dx * dx).sum(), (dx * dy).sum()],
                                        [ (dx * dy).sum(), (dy * dy).sum()]]))
    line_ang = math.degrees(math.atan2(vec[1, 1], vec[0, 1]))
    line_ang = ((line_ang % 180) + 180) % 180
    return {
        'd_min': float(d[i]),
        'theta': float(math.degrees(math.atan2(ys[i], xs[i]))),
        'line_angle': float(line_ang),
        'n_pts': len(idx),
        'span': float(math.sqrt(max(lam[1], 0.0))),
    }


def scan_cb(msg):
    global scan_time, ranges_global, angle_min_g, angle_inc_g
    global left_wall, front_wall, all_min_d
    ranges = np.array(msg.ranges)
    a_min, a_inc = msg.angle_min, msg.angle_increment
    with state_lock:
        scan_time = time.time()
        ranges_global = ranges
        angle_min_g, angle_inc_g = a_min, a_inc

    finite_idx = np.where(np.isfinite(ranges))[0]
    if len(finite_idx):
        all_min_d = float(ranges[finite_idx].min())
    segs, ranges, angles = angle_seg(ranges, a_min, a_inc)
    feats = []
    for s in segs:
        f = seg_feature(s, ranges, angles)
        if f is not None:
            feats.append(f)
    with state_lock:
        left_wall = None
        front_wall = None
        for f in feats:
            if LEFT_ANG_LO <= f['theta'] <= LEFT_ANG_HI:
                if left_wall is None or f['d_min'] < left_wall['d_min']:
                    left_wall = f
            elif FRONT_ANG_LO <= f['theta'] <= FRONT_ANG_HI:
                if front_wall is None or f['d_min'] < front_wall['d_min']:
                    front_wall = f


def odom_cb(msg):
    global odom
    q = msg.pose.pose.orientation
    yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
    with state_lock:
        odom = (msg.pose.pose.position.x, msg.pose.pose.position.y, yaw)


def publish_stop():
    if cmd_pub is not None:
        cmd_pub.publish(Twist())


def safe_guard():
    """全局安全：scan 超时 / 任意方向过近 → 刹车。返回 True 表示应停车。"""
    if time.time() - scan_time > SCAN_TIMEOUT:
        return 'scan_timeout'
    if all_min_d < MIN_SAFE_DIST:
        return 'too_close_%.3f' % all_min_d
    return None


def control():
    global mode, armed, int_err, last_par_t, prealign_t0
    if not armed:
        publish_stop()
        return
    guard = safe_guard()
    if guard is not None:
        mode = 'ESTOP'
        rospy.logerr('安全触发: %s', guard)
        publish_stop()
        return

    with state_lock:
        lw = dict(left_wall) if left_wall else None
        fw = dict(front_wall) if front_wall else None

    cmd = Twist()
    if mode == 'ESTOP':
        publish_stop()
        return

    if mode == 'PREALIGN':
        if lw is None:
            if time.time() - prealign_t0 > ALIGN_TIMEOUT:
                rospy.logwarn('PREALIGN 超时(%.0fs)且无左墙，强制进入 APPROACH', ALIGN_TIMEOUT)
                mode = 'APPROACH'
                cmd.linear.x = FORWARD_SPEED
                cmd.angular.z = 0.0
                src = '->APPROACH(timeout,no_wall)'
            else:
                cmd.linear.x = 0.0
                cmd.angular.z = 0.0
                src = 'PREALIGN_NO_WALL'
        else:
            L = lw['line_angle']
            err_ang = ALIGN_TARGET - L
            while err_ang > 90.0:
                err_ang -= 180.0
            while err_ang < -90.0:
                err_ang += 180.0
            if abs(err_ang) <= ALIGN_STOP_ERR:
                mode = 'APPROACH'
                cmd.linear.x = FORWARD_SPEED
                cmd.angular.z = 0.0
                src = '->APPROACH'
            elif time.time() - prealign_t0 > ALIGN_TIMEOUT:
                rospy.logwarn('PREALIGN 超时(%.0fs)，夹角偏差仍 %+3.0f°，强制进入 APPROACH', ALIGN_TIMEOUT, err_ang)
                mode = 'APPROACH'
                cmd.linear.x = FORWARD_SPEED
                cmd.angular.z = 0.0
                src = '->APPROACH(timeout)'
            else:
                wz = -KP_ALIGN * err_ang
                if abs(wz) < ALIGN_WZ_MIN:
                    wz = -ALIGN_WZ_MIN if err_ang > 0 else ALIGN_WZ_MIN
                wz = max(-ALIGN_WZ_MAX, min(ALIGN_WZ_MAX, wz))
                cmd.linear.x = 0.0
                cmd.angular.z = wz
                src = 'PREALIGN line=%3.0f err=%+3.0f wz=%.2f' % (L, err_ang, wz)
    elif mode == 'APPROACH':
        if lw is None:
            cmd.linear.x = FORWARD_SPEED
            cmd.angular.z = 0.0
            src = 'APPROACH_NO_WALL'
        elif lw['d_min'] <= TURN_TRIGGER_DIST:
            mode = 'TURN'
            src = '->TURN'
            cmd.linear.x = FORWARD_SPEED
            cmd.angular.z = TURN_WZ_MIN
        else:
            # 直行接近；若墙线明显偏离平行(>120°)，轻微右转提前对准
            cmd.linear.x = FORWARD_SPEED
            ang = lw['line_angle']
            cmd.angular.z = -0.08 if ang > 125.0 else 0.0
            src = 'APPROACH d=%.3f line=%3.0f' % (lw['d_min'], ang)

    elif mode == 'TURN':
        if lw is None:
            cmd.linear.x = FORWARD_SPEED * 0.7
            cmd.angular.z = TURN_WZ_MIN
            src = 'TURN_NO_WALL'
        else:
            d = lw['d_min']
            ang = lw['line_angle']
            par = min(ang, 180.0 - ang)
            done = (d <= TURN_DONE_DIST and par <= PARALLEL_ANGLE)
            if done:
                int_err = 0.0
                last_par_t = time.time()
                mode = 'PARALLEL'
                src = '->PARALLEL'
            else:
                align_wz = -(0.10 + KP_TURN_ANG * max(0.0, par - 20.0))
                dist_wz = KP_PARALLEL * (d - KEEP_WALL_DIST)
                if d < TURN_GUARD_DIST:
                    dist_wz = max(dist_wz, 0.0)   # 过深: 距离项不再增强右转(避免扎尖角)
                    wz = max(align_wz + dist_wz, TURN_GUARD_WZ)
                    speed = FORWARD_SPEED
                    src = 'TURN_DIVE d=%.3f par=%3.0f wz=%.2f' % (d, par, wz)
                else:
                    wz = max(TURN_WZ_MAX, min(-TURN_WZ_MIN, align_wz + dist_wz))
                    speed = FORWARD_SPEED
                    if d < TURN_DONE_DIST:
                        speed = FORWARD_SPEED * 0.5
                    src = 'TURN d=%.3f par=%3.0f wz=%.2f' % (d, par, wz)
                cmd.linear.x = speed
                cmd.angular.z = wz

    elif mode == 'PARALLEL':
        if lw is None:
            cmd.linear.x = FORWARD_SPEED
            cmd.angular.z = 0.0
            src = 'PARALLEL_NO_WALL'
        else:
            now = time.time()
            d = lw['d_min']
            ang = lw['line_angle']
            # 带符号航向偏差: 0/180=平行; line接近180(车头朝墙)为负, line接近0(车头离墙)为正
            dev = ang if ang <= 90.0 else ang - 180.0
            err = d - KEEP_WALL_DIST
            int_err += err * (now - last_par_t)
            last_par_t = now
            int_err = max(-PARALLEL_INT_MAX, min(PARALLEL_INT_MAX, int_err))
            head_wz = KP_ANG_HEAD * dev
            if d < SAFE_CLAMP_DIST:
                head_wz = min(head_wz, 0.0)   # 过近(接近急停)时禁止航向项左转(靠墙)
            i_term = KI_PARALLEL * int_err
            wz = KP_PARALLEL * err + i_term + head_wz
            wz = max(-PARALLEL_WZ_CLAMP, min(PARALLEL_WZ_CLAMP, wz))
            speed = FORWARD_SPEED
            if fw is not None and fw['d_min'] < FRONT_SLOW_DIST:
                if fw['d_min'] < FRONT_STOP_DIST:
                    mode = 'DONE'
                    cmd.linear.x = 0.0
                    cmd.angular.z = 0.0
                    rospy.loginfo('前方墙 %.3f m，停车完成', fw['d_min'])
                    publish_stop()
                    return
                speed = FRONT_SLOW_SPEED
                src = 'PARALLEL_SLOW d=%.3f i=%.2f front=%.3f wz=%.2f' % (d, i_term, fw['d_min'], wz)
            else:
                src = 'PARALLEL d=%.3f line=%3.0f i=%.2f front=%s wz=%.2f' % (
                    d, lw['line_angle'], i_term,
                    ('%.3f' % fw['d_min']) if fw else '---', wz)
            cmd.linear.x = speed
            cmd.angular.z = wz

    elif mode == 'DONE':
        publish_stop()
        return
    else:
        publish_stop()
        return

    cmd_pub.publish(cmd)
    rospy.loginfo('[%s] %s', mode, src)


def stdin_thread():
    global armed
    print('=== 雷达引导测试 ===')
    print('按 s + 回车：解锁启动  按 q + 回车：退出')
    while True:
        try:
            line = sys.stdin.readline()
        except Exception:
            break
        if not line:
            break
        c = line.strip().lower()
        if c == 's':
            with state_lock:
                if scan_time == 0.0:
                    print('警告：尚未收到 /scan，请先启动 ydlidar')
                    continue
                armed = True
            global mode, prealign_t0
            mode = 'PREALIGN'
            prealign_t0 = time.time()
            print('>>> 已解锁启动，原地对准后进入 APPROACH')
        elif c == 'q':
            print('>>> 退出')
            rospy.signal_shutdown('user quit')


def main():
    global cmd_pub, mode
    rospy.init_node('polyline_following_lidar', anonymous=False)
    cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
    rospy.Subscriber('/scan', LaserScan, scan_cb, queue_size=1)
    rospy.Subscriber('/odom', Odometry, odom_cb, queue_size=1)
    rospy.Timer(rospy.Duration(0.05), lambda e: control())
    rospy.on_shutdown(lambda: publish_stop())
    t = threading.Thread(target=stdin_thread, daemon=True)
    t.start()
    rospy.spin()


if __name__ == '__main__':
    main()
