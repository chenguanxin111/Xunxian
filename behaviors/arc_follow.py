"""3.1.2 弧线车道巡线 — 移植自 race_machine（源自 line_following_ss_pure_ipm 的 RUNNING）。

逻辑与之前完全一致：
- 停止线检测→蠕动走满 creep_distance→ARC_DONE；
- 视觉丢失 2s→FAULT；相机超时→FAULT；
- 软启动 + 弯道降速。
"""
import math

from behaviors.base import Behavior
from behaviors.modes import MODE_ARC_DONE, MODE_FAULT


class ArcFollowBehavior(Behavior):
    name = 'ARC_FOLLOW'

    def __init__(self, tune, steer, stop_cfg, default_speed, gentle_duration, tracker):
        import common.control as control
        self.tracker = tracker
        self.pid = control.PidState()
        self.tune = tune
        self.steer = steer
        self.stop = stop_cfg
        self.default_speed = default_speed
        self.gentle_duration = gentle_duration
        self.creep_speed = stop_cfg['creep_speed']
        self.creep_distance = stop_cfg['creep_distance']
        self.camera_timeout = stop_cfg['camera_timeout']
        self.stop_line_detected = False
        self.creep_started = False
        self.creep_start_x = 0.0
        self.creep_start_y = 0.0
        self.creep_start_yaw = 0.0
        self.creep_angular_z = 0.0
        self.wz_history = []
        self.lost_creep_start = 0.0
        self.start_time = 0.0

    def enter(self, ctx):
        import common.control as control
        self.tracker.reset(self.tune)
        self.pid.reset()
        self.pid.target_speed = self.default_speed
        self.stop_line_detected = False
        self.creep_started = False
        self.creep_start_x = 0.0
        self.creep_start_y = 0.0
        self.creep_start_yaw = 0.0
        self.creep_angular_z = 0.0
        self.wz_history = []
        self.lost_creep_start = 0.0
        self.start_time = ctx.machine_time
        ctx.status['message'] = 'ARC_FOLLOW: 弧线车道巡线'

    def step(self, ctx, now):
        import common.control as control
        v = ctx.snapshot_vision()
        odom = ctx.odom
        cmd = ctx.make_twist()

        if odom is None:
            ctx.status['message'] = 'ARC_FOLLOW: 无里程计，紧急停车'
            return cmd, MODE_FAULT

        if now - ctx.last_image_time > self.camera_timeout:
            ctx.status['message'] = '摄像头画面超时，紧急停车'
            return cmd, MODE_FAULT

        # 停止线蠕动（走满 CREEP_DISTANCE 即结束弧线段）
        if self.creep_started or v['stop_line_detected']:
            if not self.creep_started:
                self.creep_started = True
                self.creep_start_x = odom[0]
                self.creep_start_y = odom[1]
                self.creep_start_yaw = odom[2]
                recent = self.wz_history[-10:]
                avg_wz = float(sum(recent) / len(recent)) if recent else 0.0
                self.creep_angular_z = max(-0.15, min(0.15, avg_wz))
            traveled = math.hypot(odom[0] - self.creep_start_x,
                                  odom[1] - self.creep_start_y)
            if traveled >= self.creep_distance:
                ctx.status['message'] = '检测到停止线，蠕动到位，弧线段结束'
                return cmd, MODE_ARC_DONE
            cmd.linear.x = self.creep_speed
            cmd.linear.y = 0.0
            d_yaw = odom[2] - self.creep_start_yaw
            d_yaw = (d_yaw + math.pi) % (2 * math.pi) - math.pi
            if abs(d_yaw) >= math.radians(5.0):
                cmd.angular.z = 0.0
            else:
                cmd.angular.z = max(-0.15, min(0.15, self.creep_angular_z))
            return cmd, None

        # 视觉丢失降级蠕动
        if not v['lane_valid']:
            if self.lost_creep_start == 0.0:
                self.lost_creep_start = now
            if now - self.lost_creep_start > 2.0:
                ctx.status['message'] = '视觉持续丢失，已停车'
                return cmd, MODE_FAULT
            recent = self.wz_history[-10:]
            avg_wz = float(sum(recent) / len(recent)) if recent else 0.0
            cmd.linear.x = self.creep_speed
            cmd.linear.y = 0.0
            cmd.angular.z = max(-0.20, min(0.20, avg_wz))
            return cmd, None

        vel, h, c = control.compute_pid_ipm(
            v['heading_error_deg'], v['center_error_px'], self.pid, self.steer)

        # 软启动（只限速不覆盖弯道降速结果）
        is_gentle = (now - self.start_time) < self.gentle_duration
        if is_gentle:
            vel.linear.x = 0.15
            vel.angular.z *= 0.5
        else:
            elapsed = now - self.start_time - self.gentle_duration
            ramp = 0.15 + elapsed * 0.15
            vel.linear.x = min(vel.linear.x, max(0.15, ramp))

        self.lost_creep_start = 0.0
        self.wz_history.append(float(vel.angular.z))
        if len(self.wz_history) > 10:
            self.wz_history = self.wz_history[-10:]
        return vel, None
