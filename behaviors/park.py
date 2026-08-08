"""4 停车 — 移植自 corner_parking_lidar.py（雷达引导墙角停车）。

DETECT -> (PRE_ALIGN) -> DRIVE -> DONE：
- DETECT    现场推断目标侧墙（左/右取垂直距离较小且墙线大致平行于车头者）；
             无侧墙且前方开阔则缓慢蠕进尝试入位，否则原地等待；超时安全停。
- PRE_ALIGN 若雷达到侧墙垂直距离偏离 side_pre_dist(0.25m) 超容差，纯侧移(vy)横移到位；
             足够近则跳过本步。
- DRIVE     前进 + 贴墙转向（只用 vx+wz，曲线入库）：vx 按前墙距减速、wz 收敛到与侧墙
             平行且侧距 24cm；前墙到位且侧距/墙线角在区间内连续 N 帧 -> DONE。
- DONE      MODE_PARK 终点，保持停车。
安全/超时：scan 超时、<min_safe、DETECT/DRIVE 超时、行程超限均 ESTOP 安全停车。
"""
import math

import numpy as np

from behaviors.base import Behavior
from behaviors.modes import MODE_ESTOP, MODE_PARK


class ParkBehavior(Behavior):
    name = 'PARK'

    def __init__(self, cfg, radar):
        self.cfg = cfg
        self.radar = radar
        self.phase = 'WAIT_SCAN'
        self.side = None
        self.int_s = 0.0
        self.last_s_t = 0.0
        self.drive_t0 = 0.0
        self.drive_pose0 = None
        self.creep_t0 = 0.0
        self.detect_t0 = 0.0
        self.pre_t0 = 0.0
        self.pre_lost_t = 0.0
        self.pre_start_pose = None
        self.done_frames = 0
        self.scan_wait_t0 = 0.0
        self.done_reported = False

    def enter(self, ctx):
        self.phase = 'WAIT_SCAN' if not self.radar.has_scan() else 'DETECT'
        self.side = None
        self.int_s = 0.0
        self.last_s_t = ctx.machine_time
        self.drive_t0 = 0.0
        self.drive_pose0 = ctx.odom
        self.creep_t0 = 0.0
        self.detect_t0 = ctx.machine_time if self.phase == 'DETECT' else 0.0
        self.pre_t0 = 0.0
        self.pre_lost_t = 0.0
        self.pre_start_pose = None
        self.done_frames = 0
        self.scan_wait_t0 = ctx.machine_time
        self.done_reported = False
        ctx.status['message'] = 'PARK: 雷达墙角停车'

    def _guard(self, now):
        if not self.radar.scan_fresh(now):
            return 'scan_timeout'
        if self.radar.min_dist() < self.cfg['min_safe_dist']:
            return 'too_close_%.3f' % self.radar.min_dist()
        return None

    def _perceive(self):
        """从最新 scan 快照算前墙距与左/右扇形墙拟合。返回 (fd, lw, rw)。"""
        snap = self.radar.snapshot()
        if snap is None:
            return None, None, None
        import common.lidar as lidar
        ranges, a_min, a_inc = snap
        angles = a_min + np.arange(len(ranges)) * a_inc
        fd = lidar.front_min_dist(ranges, angles, self.cfg['front_ang_lo'], self.cfg['front_ang_hi'])
        lp = lidar.sector_points(ranges, angles, self.cfg['side_ang_min'], self.cfg['side_ang_max'],
                                 self.cfg['rmax'], self.cfg['min_pts'])
        rp = lidar.sector_points(ranges, angles, -self.cfg['side_ang_max'], -self.cfg['side_ang_min'],
                                 self.cfg['rmax'], self.cfg['min_pts'])
        lw = lidar.fit_wall(lp, self.cfg['min_pts']) if lp is not None else None
        rw = lidar.fit_wall(rp, self.cfg['min_pts']) if rp is not None else None
        return fd, lw, rw

    def _pick_side(self, lw, rw):
        cfg = self.cfg
        cand = []
        if lw is not None and lw['n'] >= cfg['min_pts'] and abs(lw['ang']) <= cfg['side_parallel_tol'] \
                and lw['perp'] <= cfg['side_arm_max']:
            cand.append(('left', lw))
        if rw is not None and rw['n'] >= cfg['min_pts'] and abs(rw['ang']) <= cfg['side_parallel_tol'] \
                and rw['perp'] <= cfg['side_arm_max']:
            cand.append(('right', rw))
        if not cand:
            return None
        cand.sort(key=lambda t: t[1]['perp'])
        return cand[0][0]

    def _side_fit(self, lw, rw):
        return lw if self.side == 'left' else rw

    def _start_drive(self, ctx, msg):
        self.phase = 'DRIVE'
        self.int_s = 0.0
        self.last_s_t = ctx.machine_time
        self.drive_t0 = ctx.machine_time
        self.drive_pose0 = ctx.odom
        self.done_frames = 0
        ctx.status['message'] = msg

    def _pre_shift(self, ctx):
        """PRE_ALIGN 开始以来的横向位移（odom 投影到起始车头法向）。"""
        if ctx.odom is None or self.pre_start_pose is None:
            return 0.0
        dx = ctx.odom[0] - self.pre_start_pose[0]
        dy = ctx.odom[1] - self.pre_start_pose[1]
        yaw0 = self.pre_start_pose[2]
        return abs(-dx * math.sin(yaw0) + dy * math.cos(yaw0))

    def step(self, ctx, now):
        cfg = self.cfg
        cmd = ctx.make_twist()

        guard = self._guard(now)
        if guard is not None and self.phase != 'WAIT_SCAN':
            ctx.status['message'] = 'PARK 安全停: %s' % guard
            return cmd, MODE_ESTOP

        if self.phase == 'DONE':
            ctx.status['message'] = 'PARK: 停车完成'
            return cmd, MODE_PARK

        if self.phase == 'WAIT_SCAN':
            if self.radar.scan_fresh(now):
                self.phase = 'DETECT'
                self.detect_t0 = now
                ctx.status['message'] = 'PARK: 雷达就绪，推断侧墙'
                return cmd, None
            if now - self.scan_wait_t0 > cfg['scan_wait']:
                ctx.status['message'] = 'PARK: 等待雷达超时'
                return cmd, MODE_ESTOP
            ctx.status['message'] = 'PARK: 等待雷达...'
            return cmd, None

        fd, lw, rw = self._perceive()

        if self.phase == 'DETECT':
            if now - self.detect_t0 > cfg['detect_timeout']:
                ctx.status['message'] = 'PARK: DETECT 超时未推断出侧墙'
                return cmd, MODE_ESTOP
            sd = self._pick_side(lw, rw)
            if sd is None:
                if fd is not None and fd < cfg['side_hold_dist']:
                    cmd.linear.x = 0.0
                    ctx.status['message'] = 'PARK_DETECT_HOLD front=%.3f' % fd
                else:
                    if self.creep_t0 == 0.0:
                        self.creep_t0 = now
                    if now - self.creep_t0 < cfg['detect_creep_max_t']:
                        cmd.linear.x = cfg['detect_creep']
                        ctx.status['message'] = 'PARK_DETECT_CREEP'
                    else:
                        cmd.linear.x = 0.0
                        ctx.status['message'] = 'PARK_DETECT_HOLD creep-timeout'
            else:
                self.side = sd
                fw = self._side_fit(lw, rw)
                if abs(fw['perp'] - cfg['side_pre_dist']) > cfg['pre_tol']:
                    self.phase = 'PRE_ALIGN'
                    self.pre_t0 = now
                    self.pre_lost_t = 0.0
                    self.pre_start_pose = ctx.odom
                    ctx.status['message'] = 'PARK->PRE_ALIGN side=%s perp=%.3f' % (sd, fw['perp'])
                else:
                    self._start_drive(ctx, 'PARK->DRIVE side=%s perp=%.3f' % (sd, fw['perp']))
            return cmd, None

        if self.phase == 'PRE_ALIGN':
            fw = self._side_fit(lw, rw)
            dt = max(0.0001, now - self.last_s_t)
            self.last_s_t = now
            if self._pre_shift(ctx) >= cfg['pre_max_shift']:
                self._start_drive(ctx, 'PARK->DRIVE prealign-shift-cap(%.0fmm)' % (cfg['pre_max_shift'] * 1000))
                return cmd, None
            if fw is None:
                self.pre_lost_t += dt
                cmd.linear.x = 0.0
                ctx.status['message'] = 'PARK_PRE_ALIGN_HOLD wall-lost %.1fs' % self.pre_lost_t
                if self.pre_lost_t > 1.0:
                    self.phase = 'DETECT'
                    self.side = None
                    self.creep_t0 = 0.0
                    self.detect_t0 = now
                    ctx.status['message'] = 'PARK->DETECT prealign-wall-lost'
            else:
                self.pre_lost_t = 0.0
                err = fw['perp'] - cfg['side_pre_dist']
                if abs(err) <= cfg['pre_tol']:
                    self._start_drive(ctx, 'PARK->DRIVE prealign-done perp=%.3f' % fw['perp'])
                else:
                    ssign = 1.0 if self.side == 'left' else -1.0
                    cmd.linear.y = ssign * max(-cfg['pre_vel'], min(cfg['pre_vel'], cfg['kp_y'] * err))
                    cmd.linear.x = 0.0
                    ctx.status['message'] = 'PARK_PRE_ALIGN perp=%.3f vy=%.2f' % (fw['perp'], cmd.linear.y)
            if now - self.pre_t0 > cfg['pre_align_timeout'] and self.phase == 'PRE_ALIGN':
                self._start_drive(ctx, 'PARK->DRIVE prealign-timeout perp=%s' % (
                    '%.3f' % fw['perp'] if fw else 'nan'))
            return cmd, None

        if self.phase == 'DRIVE':
            fw = self._side_fit(lw, rw)
            if fw is None:
                self.phase = 'DETECT'
                self.side = None
                self.creep_t0 = 0.0
                self.detect_t0 = now
                ctx.status['message'] = 'PARK->DETECT drive-wall-lost'
                return cmd, None
            ang = fw['ang']
            err_s = fw['perp'] - cfg['side_target']
            dt = max(0.0001, now - self.last_s_t)
            self.last_s_t = now
            self.int_s += err_s * dt
            self.int_s = max(-cfg['s_int_max'], min(cfg['s_int_max'], self.int_s))
            ssign = 1.0 if self.side == 'left' else -1.0

            steer_ang = cfg['kp_ang'] * math.radians(ang)
            steer_side = ssign * (cfg['kp_side'] * err_s + cfg['ki_side'] * self.int_s)
            wz_raw = steer_ang + steer_side

            if fd is None:
                cmd.linear.x = 0.0
                cmd.angular.z = 0.0
                ctx.status['message'] = 'PARK_DRIVE_HOLD no-front'
                return cmd, None

            err_f = fd - cfg['front_target']
            if err_f > cfg['front_tol']:
                vx = max(cfg['vx_min'], min(cfg['vx_max'], cfg['kp_f'] * err_f))
            else:
                vx = 0.0
            wz_scale = max(0.0, min(1.0, (fd - cfg['wz_near_wall_end']) /
                                    max(0.01, cfg['wz_near_wall_start'] - cfg['wz_near_wall_end'])))
            wz = max(-cfg['wz_max'], min(cfg['wz_max'], wz_raw * wz_scale))
            cmd.linear.x = vx
            cmd.angular.z = wz
            ctx.status['message'] = 'PARK_DRIVE side=%s front=%.3f perp=%.3f ang=%+.1f vx=%.2f wz=%.2f' % (
                self.side, fd, fw['perp'], ang, vx, wz)

            if abs(fd - cfg['front_target']) <= cfg['front_tol'] \
                    and cfg['side_min'] <= fw['perp'] <= cfg['side_max'] and abs(ang) <= cfg['ang_tol_deg']:
                self.done_frames += 1
            else:
                self.done_frames = 0

            if fd < cfg['front_target'] - cfg['front_tol']:
                self.phase = 'DONE'
                ctx.status['message'] = 'PARK_DONE overshoot front=%.3f' % fd
                return cmd, MODE_PARK
            if self.done_frames >= cfg['done_frames']:
                self.phase = 'DONE'
                ctx.status['message'] = 'PARK_DONE front=%.3f perp=%.3f ang=%+.1f' % (fd, fw['perp'], ang)
                return cmd, MODE_PARK

            if now - self.drive_t0 > cfg['drive_timeout']:
                wide = (fd <= cfg['drive_wide_front']
                        and cfg['side_min'] - cfg['drive_wide_side_tol'] <= fw['perp'] <= cfg['side_max'] + cfg['drive_wide_side_tol']
                        and abs(ang) <= cfg['drive_wide_ang'])
                if wide:
                    self.phase = 'DONE'
                    ctx.status['message'] = 'PARK_DONE drive-timeout wide-accept'
                    return cmd, MODE_PARK
                ctx.status['message'] = 'PARK: DRIVE 超时且不在车位内'
                return cmd, MODE_ESTOP
            if ctx.odom is not None and self.drive_pose0 is not None:
                trav = math.hypot(ctx.odom[0] - self.drive_pose0[0], ctx.odom[1] - self.drive_pose0[1])
                if trav > cfg['drive_travel_max']:
                    ctx.status['message'] = 'PARK: DRIVE 行程超限(%.2f m)' % trav
                    return cmd, MODE_ESTOP
            return cmd, None

        return cmd, MODE_ESTOP