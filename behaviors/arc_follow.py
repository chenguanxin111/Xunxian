"""3.1.2 弧线车道巡线 — 移植自 race_machine（源自 line_following_ss_pure_ipm 的 RUNNING）。

逻辑与之前完全一致：
- 停止线检测→蠕动走满 creep_distance→ARC_DONE；
- 视觉丢失先降级蠕动 lost_creep_timeout，仍丢失则原地双向搜索找回中线；
  SEARCH 双向都找不到→FAULT；相机超时→FAULT；
- 软启动 + 弯道降速。
"""
import math

from behaviors.base import Behavior
from behaviors.modes import MODE_ARC_DONE, MODE_FAULT


def _norm(a):
    return math.atan2(math.sin(a), math.cos(a))


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
        self.creep_timeout = stop_cfg.get('creep_timeout', 10.0)
        self.camera_timeout = stop_cfg['camera_timeout']
        self.lost_creep_timeout = stop_cfg.get('lost_creep_timeout', 2.0)
        self.search_rotate_wz = stop_cfg.get('search_rotate_wz', 0.35)
        self.search_confirm_frames = stop_cfg.get('search_confirm_frames', 8)
        self.search_first_sec = stop_cfg.get('search_first_sec', 5.0)
        self.search_first_deg = stop_cfg.get('search_first_deg', 75.0)
        self.search_second_sec = stop_cfg.get('search_second_sec', 6.0)
        self.search_second_deg = stop_cfg.get('search_second_deg', 90.0)
        self.stop_line_detected = False
        self.creep_started = False
        self.creep_start_t = 0.0
        self.creep_start_x = 0.0
        self.creep_start_y = 0.0
        self.creep_start_yaw = 0.0
        self.creep_angular_z = 0.0
        self.wz_history = []
        self.lost_creep_start = 0.0
        self.start_time = 0.0
        self.search_active = False
        self.search_dir = 1
        self.search_flipped = False
        self.search_t0 = 0.0
        self.search_start_yaw = 0.0
        self.search_accum_deg = 0.0
        self.search_good_frames = 0

    def enter(self, ctx):
        import common.control as control
        self.tracker.reset(self.tune)
        self.pid.reset()
        self.pid.target_speed = self.default_speed
        self.stop_line_detected = False
        self.creep_started = False
        self.creep_start_t = 0.0
        self.creep_start_x = 0.0
        self.creep_start_y = 0.0
        self.creep_start_yaw = 0.0
        self.creep_angular_z = 0.0
        self.wz_history = []
        self.lost_creep_start = 0.0
        self.start_time = ctx.machine_time
        self.search_active = False
        self.search_dir = 1
        self.search_flipped = False
        self.search_t0 = 0.0
        self.search_start_yaw = 0.0
        self.search_accum_deg = 0.0
        self.search_good_frames = 0
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
                self.creep_start_t = now
                self.creep_start_x = odom[0]
                self.creep_start_y = odom[1]
                self.creep_start_yaw = odom[2]
                recent = self.wz_history[-10:]
                avg_wz = float(sum(recent) / len(recent)) if recent else 0.0
                self.creep_angular_z = max(-0.15, min(0.15, avg_wz))
            traveled = math.hypot(odom[0] - self.creep_start_x,
                                  odom[1] - self.creep_start_y)
            if now - self.creep_start_t > self.creep_timeout:
                ctx.status['message'] = '检测到停止线，蠕动超时(%.0fs)未走完，已停车' % self.creep_timeout
                return cmd, MODE_FAULT
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

        # 恢复搜索（主动找回中线）
        if self.search_active:
            return self._step_search(ctx, now, v, odom, cmd)

        # 视觉丢失降级蠕动 → 超时进 SEARCH 原地搜索找回中线
        if not v['lane_valid']:
            if self.lost_creep_start == 0.0:
                self.lost_creep_start = now
            if now - self.lost_creep_start > self.lost_creep_timeout:
                self._enter_search(ctx, now, odom)
                return self._step_search(ctx, now, v, odom, cmd)
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

    def _enter_search(self, ctx, now, odom):
        recent = self.wz_history[-10:]
        avg_wz = float(sum(recent) / len(recent)) if recent else 0.0
        self.search_active = True
        self.search_dir = 1 if avg_wz >= 0.0 else -1
        self.search_flipped = False
        self.search_t0 = now
        self.search_start_yaw = odom[2]
        self.search_accum_deg = 0.0
        self.search_good_frames = 0
        ctx.status['message'] = 'ARC_FOLLOW: 视觉持续丢失，原地搜索找回中线'

    def _step_search(self, ctx, now, v, odom, cmd):
        if v['lane_valid']:
            self.search_good_frames += 1
            if self.search_good_frames >= self.search_confirm_frames:
                ctx.status['message'] = 'ARC_FOLLOW: SEARCH 找回中线，恢复巡线'
                self.search_active = False
                self.lost_creep_start = 0.0
                return cmd, None
        else:
            self.search_good_frames = 0

        d_yaw = _norm(odom[2] - self.search_start_yaw)
        self.search_accum_deg = abs(math.degrees(d_yaw))

        if not self.search_flipped:
            if now - self.search_t0 > self.search_first_sec or self.search_accum_deg > self.search_first_deg:
                self.search_flipped = True
                self.search_dir = -self.search_dir
                self.search_t0 = now
                self.search_start_yaw = odom[2]
                self.search_accum_deg = 0.0
                ctx.status['message'] = 'ARC_FOLLOW: SEARCH 未找到，反向搜索'
        else:
            if now - self.search_t0 > self.search_second_sec or self.search_accum_deg > self.search_second_deg:
                ctx.status['message'] = 'ARC_FOLLOW: SEARCH 双向未找到中线，已停车'
                return cmd, MODE_FAULT

        cmd.linear.x = 0.0
        cmd.linear.y = 0.0
        cmd.angular.z = self.search_dir * self.search_rotate_wz
        side = 'L' if self.search_dir > 0 else 'R'
        ctx.status['message'] = 'ARC_FOLLOW: SEARCH 原地%s转 %.0f°' % (side, self.search_accum_deg)
        return cmd, None
