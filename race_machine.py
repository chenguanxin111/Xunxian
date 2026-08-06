#!/usr/bin/env python3
"""
race_machine.py — 巡线状态机（纯终端控制，无 web / 无图像显示流）

流程（方案A 单机统一）：
  BOOT --a--> ALIGN(1 中线对齐) --完成/超时--> DIRECTION(2 方向识别)
  DIRECTION: 终端输入 0=左转 1=折线 2=右转
    0/2 -> TURN(3.1.1 路口转弯) -> ARC_FOLLOW(3.1.2 弧线巡线) -> PARK(4 停车)
    1   -> POLYLINE(3.2 折线巡线) -> PARK(4 停车)

当前实现：ALIGN + DIRECTION(终端0/1/2) + TURN(3.1.1) + ARC_FOLLOW(3.1.2) + POLYLINE(3.2，
含视觉段 ADVANCE→LINE_FOLLOW→TURN_RIGHT→SEARCH_RIGHT 与雷达段接管/兜底，最终停车)；PARK(4) 为占位。
调试：z 键可跳过对准直接进入 ARC_FOLLOW。

终端控制（单键即时，无需回车）：
  a            从 BOOT 开始（进入 ALIGN 中线对齐）
  0 / 1 / 2    在 DIRECTION 中选方向
  z            调试：直接进入 ARC_FOLLOW
  s            急停(ESTOP)
  r            复位到 BOOT
  v [速度]     设置目标速度，如 v 0.36（回车生效）
  q            退出
"""
import os

os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["OPENCV_UI_BACKEND"] = "HEADLESS"
if "DISPLAY" in os.environ:
    del os.environ["DISPLAY"]

import math
import select
import sys
import termios
import threading
import time
import tty

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, LaserScan
from tf.transformations import euler_from_quaternion

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common.config as cfg
import common.lidar as lidar
import common.vision as vision
from behaviors.align import AlignBehavior
from behaviors.arc_follow import ArcFollowBehavior
from behaviors.modes import (MODE_ALIGN, MODE_ARC_DONE, MODE_ARC_FOLLOW,
                             MODE_BOOT, MODE_DIRECTION, MODE_ESTOP, MODE_FAULT,
                             MODE_PARK, MODE_POLYLINE, MODE_TURN)
from behaviors.park import ParkBehavior
from behaviors.polyline import PolylineBehavior
from behaviors.turn import TurnBehavior

IMAGE_TOPIC = '/usb_cam/image_raw'
PERSISTENT_HZ = 20.0

# ---- 全局参数（来自 config/*.json）----
HSV = cfg.load_hsv()
LANE_TUNE = cfg.load_lane_tune()
STEER = cfg.load_steering()
STOP = cfg.load_stopline()
ALIGN_CFG = cfg.load_align()
TURN_CFG = cfg.load_turn()
POLY_CFG = cfg.load_polyline()
PARK_CFG = cfg.load_park()

CAMERA_TIMEOUT = STOP['camera_timeout']
CREEP_SPEED = STOP['creep_speed']
CREEP_DISTANCE = STOP['creep_distance']
STOP_LINE_ROI_TOP_RATIO = STOP['stop_line_roi_top_ratio']
STOP_LINE_WIDTH_RATIO = STOP['stop_line_width_ratio']
STOP_LINE_THIN_RATIO = STOP['stop_line_thin_ratio']

GENTLE_START_DURATION = 2.0
DEFAULT_SPEED = 0.36


class _RawStdout:
    """raw 模式下 OPOST 被关闭，'\n' 不会回到行首；显式写 '\r\n' 修复换行错乱。"""

    def __init__(self, stream):
        self._s = stream

    def write(self, data):
        self._s.write(data.replace('\n', '\r\n'))
        self._s.flush()

    def flush(self):
        self._s.flush()

    def isatty(self):
        return self._s.isatty()


class Ctx:
    def __init__(self):
        self.lock = threading.Lock()
        self.machine_time = 0.0
        self.odom = None                # (x, y, yaw)
        self.last_image_time = 0.0
        self.vision = {
            'heading_error_deg': 0.0,
            'center_error_px': 0.0,
            'lane_valid': False,
            'lane_width_px': None,
            'kanbujian': False,
            'stop_line_detected': False,
            'stop_y': -1,
            'opp_valid': False,
            'opp_center_error': None,
            'opp_heading_error_deg': None,
            'pair_matched': False,
            'phys_left_found': False,
            'phys_right_found': False,
            'right_fit_ok': False,
            'right_near_x': None,
            'right_good_frames': 0,
        }
        self.poly_roi = {'use_wide': False, 'bias': 0.0}   # POLYLINE 每帧由行为写入
        self.poly_stop = {'enabled': False, 'enable_time': 0.0}
        self._right_good_frames = 0      # POLYLINE 可信右边界连续帧计数
        self._right_last_near_x = None
        self.command = {'x': 0.0, 'y': 0.0, 'z': 0.0}
        self.status = {'mode': MODE_BOOT, 'message': '等待解锁', 'direction': '-'}

    def snapshot_vision(self):
        with self.lock:
            return dict(self.vision)

    def make_twist(self):
        return Twist()


ctx = Ctx()
bridge = CvBridge()
cmd_pub = None

tracker = vision.LaneTracker(LANE_TUNE)
align = AlignBehavior(ALIGN_CFG)
arc = ArcFollowBehavior(LANE_TUNE, STEER, STOP, DEFAULT_SPEED, GENTLE_START_DURATION, tracker)
turn = TurnBehavior(TURN_CFG, tracker)
radar = lidar.RadarScan(POLY_CFG)
poly = PolylineBehavior(POLY_CFG, LANE_TUNE, STEER, tracker, radar, POLY_CFG['target_speed'])
park = ParkBehavior(PARK_CFG, radar)
mode = MODE_BOOT
direction = '-'          # '0'=左转  '1'=折线  '2'=右转


def publish_stop():
    ctx.command['x'] = 0.0
    ctx.command['y'] = 0.0
    ctx.command['z'] = 0.0
    if cmd_pub is not None:
        cmd_pub.publish(Twist())


def publish(cmd):
    ctx.command['x'] = cmd.linear.x
    ctx.command['y'] = cmd.linear.y
    ctx.command['z'] = cmd.angular.z
    if cmd_pub is not None:
        cmd_pub.publish(cmd)


def image_cb(msg):
    try:
        frame = bridge.imgmsg_to_cv2(msg, 'passthrough')
        if msg.encoding.lower() == 'rgb8':
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        elif msg.encoding.lower() != 'bgr8':
            frame = bridge.imgmsg_to_cv2(msg, 'bgr8')

        full_mask = vision.get_full_mask(frame, HSV)
        stop_detected, stop_y = vision.detect_stop_line(
            full_mask, STOP_LINE_ROI_TOP_RATIO, STOP_LINE_WIDTH_RATIO, STOP_LINE_THIN_RATIO)

        # POLYLINE：ROI 宽紧 + lane_bias + 停止线 enable/delay 门控（由 PolylineBehavior 每帧写入）
        if mode == MODE_POLYLINE:
            with ctx.lock:
                proi = dict(ctx.poly_roi)
                pstop = dict(ctx.poly_stop)
            roi_ratio = (POLY_CFG['roi_wide_bottom_ratio'] if proi['use_wide']
                         else POLY_CFG['roi_tight_bottom_ratio'])
            tracker.lane_bias_px = proi['bias']
            stop_eff = stop_detected and pstop['enabled'] and (
                time.time() - pstop['enable_time'] >= POLY_CFG['stop_line_enable_delay_sec'])
            if not stop_eff:
                stop_detected, stop_y = False, -1
        else:
            roi_ratio = LANE_TUNE['roi_bottom_ratio']
            tracker.lane_bias_px = 0.0

        # 路口对面中线（仅 ALIGN 阶段计算，保持 simple_turn 的翻转约定）
        opp = None
        if mode == MODE_ALIGN:
            opp_params = dict(ALIGN_CFG['hsv'])
            for k in ('roi_top', 'roi_bottom', 'roi_left', 'roi_right',
                      'blur_ksize', 'erode_iter', 'erode_ksize',
                      'dilate_iter', 'dilate_ksize'):
                if k in ALIGN_CFG:
                    opp_params[k] = ALIGN_CFG[k]
            opp_frame = cv2.flip(frame, 1)
            opp_mask, _ = vision.make_mask640(opp_frame, opp_params)
            opp = vision.detect_opposite_centerline(opp_mask)

        mask_roi = full_mask.copy()
        y_cut = int(vision.PROC_H * (1.0 - roi_ratio))
        mask_roi[:y_cut, :] = 0

        warped = vision.warp_to_ipm(mask_roi)
        warped = vision.remove_horizontal_bands_ipm(warped, LANE_TUNE)
        cleaned = vision.clean_ipm_mask(warped, LANE_TUNE)
        left_fit, right_fit, width_samples = vision.extract_raw_lanes(cleaned, LANE_TUNE)
        result = vision.resolve_lane(tracker, left_fit, right_fit, width_samples, LANE_TUNE)

        if result is not None:
            result = vision.apply_poly_filter(tracker, result, LANE_TUNE)
            # 摄像头原始画面为镜像（IPM x 与物理左右相反），取反得到物理符号
            result['center_error_px'] = -result['center_error_px']
            result['heading_error_deg'] = -result['heading_error_deg']

        with ctx.lock:
            if result is not None:
                ctx.vision['heading_error_deg'] = result['heading_error_deg']
                ctx.vision['center_error_px'] = result['center_error_px']
                ctx.vision['lane_width_px'] = result['lane_width_px']
                ctx.vision['kanbujian'] = result['kanbujian']
            ctx.vision['lane_valid'] = result is not None
            ctx.vision['stop_line_detected'] = stop_detected
            ctx.vision['stop_y'] = stop_y

            # POLYLINE 专属感知量（仅折线阶段计算）
            if mode == MODE_POLYLINE:
                ctx.vision['pair_matched'] = bool(result['pair_matched']) if result else False
                # IPM 大 x = 物理左线(紫)，小 x = 物理右线(红)（历史命名残留，勿改）
                ctx.vision['phys_left_found'] = right_fit is not None
                ctx.vision['phys_right_found'] = left_fit is not None
                r_nx = None
                if right_fit is not None:
                    span = right_fit['y_max'] - right_fit['y_min']
                    y_ref = min(right_fit['y_max'], vision.Y_NEAR)
                    r_nx = vision.poly_x(right_fit['coeffs'], y_ref)
                    jitter_ok = (ctx._right_last_near_x is None or
                                 abs(r_nx - ctx._right_last_near_x) <= POLY_CFG['right_trust_jitter_px'])
                    if (span >= POLY_CFG['right_trust_span_min'] and
                            r_nx >= POLY_CFG['right_trust_nx_min'] and jitter_ok):
                        ctx._right_good_frames += 1
                    else:
                        ctx._right_good_frames = 0
                else:
                    ctx._right_good_frames = 0
                ctx._right_last_near_x = r_nx
                ctx.vision['right_near_x'] = r_nx
                ctx.vision['right_good_frames'] = ctx._right_good_frames
                ctx.vision['right_fit_ok'] = ctx._right_good_frames >= POLY_CFG['right_trust_frames']
            if opp is not None:
                ctx.vision['opp_valid'] = True
                ctx.vision['opp_center_error'] = opp['center_error']
                ctx.vision['opp_heading_error_deg'] = opp['heading_error_deg']
            else:
                ctx.vision['opp_valid'] = False
                ctx.vision['opp_center_error'] = None
                ctx.vision['opp_heading_error_deg'] = None
            ctx.last_image_time = time.time()
    except Exception as exc:
        import traceback
        rospy.logwarn_throttle(2, 'image_cb 异常: %s\n%s', exc, traceback.format_exc())


def odom_cb(msg):
    q = msg.pose.pose.orientation
    yaw = euler_from_quaternion((q.x, q.y, q.z, q.w))[2]
    with ctx.lock:
        ctx.odom = (msg.pose.pose.position.x, msg.pose.pose.position.y, yaw)


def control_timer(_event):
    global mode, direction
    now = time.time()
    ctx.machine_time = now
    cmd = Twist()

    if mode == MODE_ALIGN:
        if ctx.odom is None:
            publish_stop()
        elif ctx.last_image_time == 0 or now - ctx.last_image_time > CAMERA_TIMEOUT:
            mode = MODE_FAULT
            ctx.status['message'] = '摄像头画面超时，紧急停车'
            publish_stop()
            print('>> FAULT: %s' % ctx.status['message'], flush=True)
        else:
            c, nxt = align.step(ctx, now)
            if nxt is None:
                publish(c)
            else:
                publish_stop()
                if nxt == MODE_DIRECTION:
                    mode = MODE_DIRECTION
                    direction = '-'
                    ctx.status['message'] = 'DIRECTION: 终端输入 0=左转 1=折线 2=右转'
                    print('>> 对准完成/超时 → 方向识别，请输入 0/1/2', flush=True)
                elif nxt == MODE_FAULT:
                    mode = MODE_FAULT
                    print('>> FAULT: %s' % ctx.status['message'], flush=True)
    elif mode == MODE_DIRECTION:
        publish_stop()
    elif mode == MODE_TURN:
        if ctx.odom is None or ctx.last_image_time == 0 or now - ctx.last_image_time > CAMERA_TIMEOUT:
            mode = MODE_FAULT
            ctx.status['message'] = 'TURN: 里程计不可用或相机超时，紧急停车'
            publish_stop()
            print('>> FAULT: %s' % ctx.status['message'], flush=True)
        else:
            c, nxt = turn.step(ctx, now)
            if nxt is None:
                publish(c)
            else:
                publish_stop()
                if nxt == MODE_ARC_FOLLOW:
                    arc.enter(ctx)
                    mode = MODE_ARC_FOLLOW
                    print('>> 转弯找到弧线中线 → ARC_FOLLOW', flush=True)
                elif nxt == MODE_FAULT:
                    mode = MODE_FAULT
                    print('>> FAULT: %s' % ctx.status['message'], flush=True)
    elif mode == MODE_POLYLINE:
        if ctx.odom is None or ctx.last_image_time == 0 or now - ctx.last_image_time > POLY_CFG['camera_timeout']:
            mode = MODE_FAULT
            ctx.status['message'] = 'POLYLINE: 里程计不可用或相机超时，紧急停车'
            publish_stop()
            print('>> FAULT: %s' % ctx.status['message'], flush=True)
        else:
            # 全局安全：任意方向过近急停（雷达有数据时）
            if radar.has_scan() and radar.min_dist() < POLY_CFG['min_safe_dist']:
                mode = MODE_ESTOP
                ctx.status['message'] = '雷达安全触发: too_close %.3f' % radar.min_dist()
                publish_stop()
                print('>> ESTOP: %s' % ctx.status['message'], flush=True)
            else:
                c, nxt = poly.step(ctx, now)
                if nxt is None:
                    publish(c)
                else:
                    publish_stop()
                    if nxt == MODE_PARK:
                        park.enter(ctx)
                        mode = MODE_PARK
                        ctx.status['message'] = '折线巡线段结束，进入雷达墙角停车'
                        print('>> POLYLINE_DONE → PARK: %s' % ctx.status['message'], flush=True)
                    elif nxt == MODE_ESTOP:
                        mode = MODE_ESTOP
                        print('>> ESTOP: %s' % ctx.status['message'], flush=True)
                    elif nxt == MODE_FAULT:
                        mode = MODE_FAULT
                        print('>> FAULT: %s' % ctx.status['message'], flush=True)
    elif mode == MODE_PARK:
        if ctx.odom is None:
            publish_stop()
        else:
            if radar.has_scan() and radar.min_dist() < PARK_CFG['min_safe_dist']:
                mode = MODE_ESTOP
                ctx.status['message'] = '雷达安全触发: too_close %.3f' % radar.min_dist()
                publish_stop()
                print('>> ESTOP: %s' % ctx.status['message'], flush=True)
            else:
                c, nxt = park.step(ctx, now)
                if nxt == MODE_ESTOP:
                    mode = MODE_ESTOP
                    publish_stop()
                    print('>> ESTOP: %s' % ctx.status['message'], flush=True)
                else:
                    publish(c)
                    if nxt == MODE_PARK and not park.done_reported:
                        park.done_reported = True
                        print('>> PARK_DONE: %s' % ctx.status['message'], flush=True)
    elif mode == MODE_ARC_FOLLOW:
        if ctx.odom is None:
            publish_stop()
        elif ctx.last_image_time == 0 or now - ctx.last_image_time > CAMERA_TIMEOUT:
            mode = MODE_FAULT
            ctx.status['message'] = '摄像头画面超时，紧急停车'
            publish_stop()
            print('>> FAULT: %s' % ctx.status['message'], flush=True)
        else:
            c, nxt = arc.step(ctx, now)
            if nxt is None:
                publish(c)
            else:
                publish_stop()
                if nxt == MODE_ARC_DONE:
                    mode = MODE_ARC_DONE
                    ctx.status['message'] = '弧线巡线段结束（下一步接 PARK）'
                    print('>> ARC_DONE: %s' % ctx.status['message'], flush=True)
                elif nxt == MODE_FAULT:
                    mode = MODE_FAULT
                    print('>> FAULT: %s' % ctx.status['message'], flush=True)
    elif mode == MODE_FAULT:
        publish_stop()
    elif mode == MODE_ESTOP:
        publish_stop()
    elif mode == MODE_ARC_DONE:
        if ctx.odom is None:
            publish_stop()
        else:
            park.enter(ctx)
            mode = MODE_PARK
            ctx.status['message'] = '弧线段结束，进入雷达墙角停车'
            print('>> ARC_DONE → PARK: %s' % ctx.status['message'], flush=True)
    else:  # BOOT
        publish_stop()

    if now - getattr(control_timer, 'last_print', 0.0) >= 1.0:
        control_timer.last_print = now
        with ctx.lock:
            h = ctx.vision['heading_error_deg']
            c = ctx.vision['center_error_px']
            lv = ctx.vision['lane_valid']
            stop = ctx.vision['stop_line_detected']
            opp = ctx.vision['opp_valid']
            odom = ctx.odom
        o = 'odom:---' if odom is None else 'yaw=%.1f' % math.degrees(odom[2])
        print('[%s] dir=%s h=%+.1fdeg c=%+.1fpx lane=%s stop=%s opp=%s %s | vx=%.2f vz=%.2f | %s'
              % (mode, direction, h, c, lv, stop, opp, o,
                 ctx.command['x'], ctx.command['z'], ctx.status['message']))


def handle_line(line):
    """处理一行终端命令（单键即时：a/0/1/2/z/s/r/q 按下即生效）。"""
    global mode, direction
    line = line.strip()
    if not line:
        return
    c = line.split()
    key = c[0].lower()
    try:
        if key == 'a':
            if mode == MODE_BOOT:
                if ctx.odom is None:
                    print('>> odom 未就绪，稍后再试', flush=True)
                else:
                    align.enter(ctx)
                    mode = MODE_ALIGN
                    direction = '-'
                    print('>> 已开始，进入 ALIGN 中线对齐', flush=True)
            else:
                print('>> 当前状态 %s，不能直接开始' % mode, flush=True)
        elif key in ('0', '1', '2'):
            if mode in (MODE_DIRECTION, MODE_TURN, MODE_POLYLINE):
                direction = key
                if key in ('0', '2'):
                    if ctx.odom is None:
                        print('>> odom 未就绪，稍后再试', flush=True)
                    else:
                        turn.direction = key
                        turn.enter(ctx)
                        mode = MODE_TURN
                        ctx.status['message'] = 'TURN(3.1.1): 路口转弯'
                        print('>> 方向 %s=转弯，进入 TURN' % key, flush=True)
                else:
                    if ctx.odom is None:
                        print('>> odom 未就绪，稍后再试', flush=True)
                    else:
                        poly.enter(ctx)
                        mode = MODE_POLYLINE
                        ctx.status['message'] = 'POLYLINE(3.2): 折线巡线'
                        print('>> 方向 1=折线，进入 POLYLINE', flush=True)
            else:
                print('>> 当前不在 DIRECTION/TURN/POLYLINE，无法选择方向', flush=True)
        elif key == 'z':
            if mode in (MODE_BOOT, MODE_DIRECTION, MODE_TURN, MODE_POLYLINE, MODE_ARC_DONE):
                if ctx.odom is None:
                    print('>> odom 未就绪，稍后再试', flush=True)
                else:
                    arc.enter(ctx)
                    mode = MODE_ARC_FOLLOW
                    print('>> 调试：直接进入 ARC_FOLLOW', flush=True)
            else:
                print('>> 当前状态 %s 不支持 z 跳转' % mode, flush=True)
        elif key == 's':
            mode = MODE_ESTOP
            ctx.status['message'] = '终端急停'
            print('>> ESTOP', flush=True)
        elif key == 'r':
            mode = MODE_BOOT
            direction = '-'
            ctx.status['message'] = '已复位'
            print('>> 复位到 BOOT', flush=True)
        elif key == 'v' and len(c) >= 2:
            try:
                arc.pid.target_speed = float(c[1])
                print('>> 目标速度 %.2f' % arc.pid.target_speed, flush=True)
            except Exception:
                print('>> 速度格式错误', flush=True)
        elif key == 'q':
            print('>> 退出', flush=True)
            rospy.signal_shutdown('user quit')
        else:
            print(">> 未知命令 '%s'（a / 0/1/2 / z / s / r / v 速度 / q）" % line, flush=True)
    except Exception as exc:
        import traceback
        print('>> 命令异常: %s\n%s' % (exc, traceback.format_exc()), flush=True)


def stdin_thread():
    """单键即时读取：a/0/1/2/z/s/r/q 按下即生效；v 后跟数字回车设置速度。"""
    print('=== race_machine [ALIGN→DIRECTION→TURN→ARC_FOLLOW] ===', flush=True)
    print('命令(单键): a=开始 0/1/2=选方向 z=直接ARC s=急停 r=复位 q=退出 | v 0.36=调速', flush=True)
    old = None
    try:
        old = termios.tcgetattr(sys.stdin.fileno())
        tty.setraw(sys.stdin.fileno())
    except Exception:
        old = None  # 非 tty（如日志重定向），退化为逐行读取
    try:
        buf = ''
        while not rospy.is_shutdown():
            if old is not None:
                r, _, _ = select.select([sys.stdin], [], [], 0.1)
                if not r:
                    continue
                ch = sys.stdin.read(1)
                if ch in ('\r', '\n'):
                    handle_line(buf)
                    buf = ''
                    continue
                if ch in ('\x03', '\x04'):  # ctrl-c / ctrl-d
                    rospy.signal_shutdown('user quit')
                    break
                if ch in ('\x7f', '\b'):    # 退格
                    buf = buf[:-1]
                    continue
                buf += ch
                b = buf.strip().lower()
                if b in ('a', '0', '1', '2', 'z', 's', 'r', 'q'):
                    handle_line(buf)
                    buf = ''
            else:
                line = sys.stdin.readline()
                if not line:
                    time.sleep(0.2)
                    continue
                handle_line(line)
    finally:
        if old is not None:
            try:
                termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, old)
            except Exception:
                pass


def shutdown():
    for _ in range(5):
        publish_stop()
        time.sleep(0.03)


def main():
    global cmd_pub
    sys.stdout = _RawStdout(sys.stdout)
    rospy.init_node('race_machine', anonymous=False)
    if not vision.init_ipm(cfg.persp_path()):
        rospy.logwarn('IPM 标定未加载，弧线检测将不可用')
    cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
    rospy.Subscriber(IMAGE_TOPIC, Image, image_cb, queue_size=1, buff_size=2 ** 24)
    rospy.Subscriber('/odom', Odometry, odom_cb, queue_size=1)
    rospy.Subscriber('/scan', LaserScan, lambda msg: radar.process(
        msg.ranges, msg.angle_min, msg.angle_increment, time.time()), queue_size=1)
    threading.Thread(target=stdin_thread, daemon=True).start()
    rospy.Timer(rospy.Duration(1.0 / PERSISTENT_HZ), control_timer)
    rospy.on_shutdown(shutdown)
    rospy.loginfo('race_machine 已启动，终端输入 a 开始')
    rospy.spin()


if __name__ == '__main__':
    main()
