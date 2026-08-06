"""1. 中线对齐（路口对面中线平移对准）— 移植自 simple_turn_trial.py 阶段一 F_ALIGN。

规则（原样）：
- 禁止前进，linear.y 平移 + 角速度耦合纠偏；
- 短暂掉帧(opp 无效但 last_error 存在)沿用上次误差继续纠偏；
- 超时(ALIGN.timeout s)或累计位移 >= ALIGN.max_distance → 直接进下一阶段；
- 连续 confirm_frames 帧 |center|<=tol && |heading|<=tol → 对准完成，进下一阶段。
"""
import math

from behaviors.base import Behavior
from behaviors.modes import MODE_DIRECTION, MODE_FAULT


class AlignBehavior(Behavior):
    name = 'ALIGN'

    def __init__(self, cfg):
        self.cfg = cfg
        self.good_frames = 0
        self.last_error = None
        self.start_pose = None
        self.start_time = 0.0

    def enter(self, ctx):
        self.good_frames = 0
        self.last_error = None
        self.start_pose = ctx.odom[:2] if ctx.odom is not None else None
        self.start_time = ctx.machine_time
        ctx.status['message'] = 'ALIGN: 路口对面中线平移对准'

    def step(self, ctx, now):
        cfg = self.cfg
        cmd = ctx.make_twist()
        odom = ctx.odom

        if odom is None:
            ctx.status['message'] = 'ALIGN: 无里程计，无法对准'
            return cmd, MODE_FAULT

        if self.start_pose is None:
            self.start_pose = odom[:2]
        traveled = math.hypot(odom[0] - self.start_pose[0],
                              odom[1] - self.start_pose[1])

        # 超时 / 超距兜底：直接进下一阶段
        if traveled >= cfg['max_distance'] or now - self.start_time > cfg['timeout']:
            ctx.status['message'] = 'ALIGN: 对准超时/超距，进方向识别'
            return cmd, MODE_DIRECTION

        v = ctx.snapshot_vision()
        opp_valid = v.get('opp_valid', False)
        center_error = v.get('opp_center_error')
        heading_error = v.get('opp_heading_error_deg')

        if not opp_valid or center_error is None:
            self.good_frames = 0
            if self.last_error is not None:
                # 短暂掉帧：沿用上次有效误差继续纠偏
                e = self.last_error
                cmd.linear.x = 0.0
                cmd.linear.y = cfg['kp_y'] * e
                cmd.angular.z = max(-cfg['wz_clamp'],
                                    min(cfg['wz_clamp'], cfg['kp_z'] * e))
                ctx.status['message'] = 'ALIGN: HOLD_LAST 沿用上次误差纠偏'
            else:
                ctx.status['message'] = 'ALIGN: 等待检测到对面中线'
            return cmd, None

        self.last_error = center_error
        heading_error = heading_error if heading_error is not None else 0.0

        if (abs(center_error) <= cfg['center_tol_px']
                and abs(heading_error) <= cfg['heading_tol_deg']):
            self.good_frames += 1
            if self.good_frames >= cfg['confirm_frames']:
                ctx.status['message'] = 'ALIGN: 对准完成，进方向识别'
                return cmd, MODE_DIRECTION
        else:
            self.good_frames = 0

        cmd.linear.x = 0.0
        cmd.linear.y = cfg['kp_y'] * center_error
        cmd.angular.z = max(-cfg['wz_clamp'],
                            min(cfg['wz_clamp'], cfg['kp_z'] * center_error))
        ctx.status['message'] = 'ALIGN: 平移纠偏'
        return cmd, None
