"""3.1.1 路口转弯 — 移植自 simple_turn_trial.py（ADVANCE→ROTATE→SEARCH_ARC）。

机器内 ALIGN 已单独成行为，此处承接 F_YAW → ADVANCE → ROTATE → SEARCH_ARC：
- F_YAW:    原地转回对准起始 yaw（消除平移带偏的车头），收敛或超时 → ADVANCE
- ADVANCE:  直行 advance_distance，锁定航向角；到位或超时 → ROTATE
- ROTATE:   同方向旋转 rotate_target_deg；到位或超时 → SEARCH_ARC
- SEARCH_ARC: 同向慢速续转，读到弧线中线连续 confirm 帧 → 返回 ARC_FOLLOW
超时/超距一律 FAULT 停车；SEARCH 清空 tracker 历史后开始。
"""
import math

from behaviors.base import Behavior
from behaviors.modes import MODE_ARC_FOLLOW, MODE_FAULT


def _norm(a):
    return math.atan2(math.sin(a), math.cos(a))


class TurnBehavior(Behavior):
    name = 'TURN'

    def __init__(self, cfg, tracker):
        self.cfg = cfg
        self.tracker = tracker
        self.direction = '0'
        self.phase = 'F_YAW'
        self.phase_started = 0.0
        self.f_align_start_yaw = 0.0
        self.f_yaw_started = 0.0
        self.advance_start_pose = None
        self.advance_dist = 0.0
        self.rotate_start_yaw = None
        self.turn_deg = 0.0
        self.search_start_yaw = None
        self.search_start_time = 0.0
        self.search_good_frames = 0

    def _sign(self):
        return 1.0 if self.direction in ('0', 'LEFT') else -1.0

    def _dir_str(self):
        return '左转' if self.direction in ('0', 'LEFT') else '右转'

    def enter(self, ctx):
        self.tracker.reset()
        self.phase = 'F_YAW'
        self.phase_started = ctx.machine_time
        self.f_align_start_yaw = ctx.odom[2]
        self.f_yaw_started = ctx.machine_time
        self.advance_start_pose = None
        self.advance_dist = 0.0
        self.rotate_start_yaw = None
        self.turn_deg = 0.0
        self.search_start_yaw = None
        self.search_start_time = 0.0
        self.search_good_frames = 0
        ctx.status['message'] = 'TURN: 车头微调（%s）' % self._dir_str()

    def step(self, ctx, now):
        cfg = self.cfg
        cmd = ctx.make_twist()
        odom = ctx.odom

        if odom is None:
            ctx.status['message'] = 'TURN: 无里程计，紧急停车'
            return cmd, MODE_FAULT

        if self.phase == 'F_YAW':
            elapsed = now - self.f_yaw_started
            yaw_error = _norm(self.f_align_start_yaw - odom[2])
            if abs(math.degrees(yaw_error)) <= cfg['f_yaw_tol_deg'] or elapsed > cfg['f_yaw_max_time']:
                self.advance_start_pose = odom
                self.advance_dist = 0.0
                self.phase = 'ADVANCE'
                self.phase_started = now
                ctx.status['message'] = 'TURN: 车头微调完成，直行 %dcm' % int(cfg['advance_distance'] * 100)
                return cmd, None
            cmd.linear.x = 0.0
            cmd.linear.y = 0.0
            cmd.angular.z = max(-cfg['f_yaw_wz'], min(cfg['f_yaw_wz'], cfg['f_yaw_kp'] * yaw_error))
            ctx.status['message'] = 'TURN: 车头微调'
            return cmd, None

        if self.phase == 'ADVANCE':
            if now - self.phase_started > cfg['advance_timeout']:
                ctx.status['message'] = 'TURN: 直行超时，已停车'
                return cmd, MODE_FAULT
            self.advance_dist = math.hypot(odom[0] - self.advance_start_pose[0],
                                           odom[1] - self.advance_start_pose[1])
            if self.advance_dist >= cfg['advance_distance']:
                self.rotate_start_yaw = odom[2]
                self.turn_deg = 0.0
                self.phase = 'ROTATE'
                self.phase_started = now
                ctx.status['message'] = 'TURN: 直行完成，开始%s %d 度' % (
                    self._dir_str(), cfg['rotate_target_deg'])
                return cmd, None
            cmd.linear.x = cfg['advance_speed']
            cmd.linear.y = 0.0
            yaw_error = _norm(self.advance_start_pose[2] - odom[2])
            cmd.angular.z = max(-0.08, min(0.08, 0.8 * yaw_error))
            return cmd, None

        if self.phase == 'ROTATE':
            if now - self.phase_started > cfg['rotate_timeout']:
                ctx.status['message'] = 'TURN: 旋转超时，已停车'
                return cmd, MODE_FAULT
            self.turn_deg = math.degrees(abs(_norm(odom[2] - self.rotate_start_yaw)))
            if self.turn_deg >= cfg['rotate_target_deg']:
                self.tracker.reset()  # 清历史（转弯结束、开始找弧线中线）
                self.search_start_yaw = odom[2]
                self.search_start_time = now
                self.search_good_frames = 0
                self.phase = 'SEARCH_ARC'
                ctx.status['message'] = 'TURN: 转弯完成，同向慢速搜索弧线中线'
                return cmd, None
            cmd.linear.x = 0.0
            cmd.linear.y = 0.0
            cmd.angular.z = self._sign() * cfg['rotate_speed']
            return cmd, None

        if self.phase == 'SEARCH_ARC':
            if now - self.search_start_time > cfg['search_timeout']:
                ctx.status['message'] = 'TURN: 弧线中线搜索超时，未找到车道线，已停车'
                return cmd, MODE_FAULT
            accum = math.degrees(abs(_norm(odom[2] - self.search_start_yaw)))
            if accum > cfg['search_accum_limit_deg']:
                ctx.status['message'] = 'TURN: 弧线中线搜索超限(>%d°)，未找到车道线，已停车' % cfg['search_accum_limit_deg']
                return cmd, MODE_FAULT
            v = ctx.snapshot_vision()
            if v.get('lane_valid', False):
                self.search_good_frames += 1
                if self.search_good_frames >= cfg['search_confirm_frames']:
                    ctx.status['message'] = 'TURN: 已找到弧线车道中线，进入弧线巡线'
                    return cmd, MODE_ARC_FOLLOW
            else:
                self.search_good_frames = 0
            cmd.linear.x = 0.0
            cmd.linear.y = 0.0
            cmd.angular.z = self._sign() * cfg['search_rotate_wz']
            return cmd, None

        return cmd, MODE_FAULT
