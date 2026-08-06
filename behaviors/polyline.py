"""3.2 折线巡线 — 移植自 polyline_following_hybrid.py（视觉段 + 雷达段混合）。

机器内 ALIGN 已独立成行为，此处承接 ADVANCE → LINE_FOLLOW → TURN_RIGHT → SEARCH_RIGHT
之后雷达接管（RADAR_PREALIGN→APPROACH→TURN→PARALLEL），最终 RADAR_DONE 停车。

符号约定（历史命名残留，勿改数值）：
- resolve_lane 的 left/right 按 **IPM x 大小侧** 分（左=小x、右=大x）；
- 物理上 IPM 大 x = 物理左线(紫)，小 x = 物理右线(红)；
- 因此本文件里的 rank/turn 方向一律以"物理行为"描述：
  TURN_RIGHT = 向右转（angular.z 负），与 simple_turn 相反；phys_left_found 来自 right_fit。
"""
import math

from behaviors.base import Behavior
from behaviors.modes import MODE_FAULT, MODE_PARK


def _norm(a):
    return math.atan2(math.sin(a), math.cos(a))


class PolylineBehavior(Behavior):
    name = 'POLYLINE'

    def __init__(self, cfg, tune, steer, tracker, radar, default_speed):
        import common.control as control
        self.cfg = cfg
        self.tune = tune
        self.steer = steer
        self.tracker = tracker
        self.radar = radar
        self.pid = control.PidState()
        self.pid.target_speed = default_speed if default_speed > 0 else cfg['target_speed']
        self.phase = 'ADVANCE'
        self.phase_started = 0.0
        self.advance_start_pose = None
        self.advance_dist = 0.0
        self.line_follow_start_yaw = 0.0
        self.roi_wide_remaining = 0
        self.turn_forced = False
        self.turn_start_yaw = 0.0
        self.turn_accum_deg = 0.0
        self.lost_creep_start = 0.0
        self.stop_line_enabled = False
        self.stop_line_enable_time = 0.0
        self.creep_started = False
        self.creep_start_pose = None
        self.creep_angular_z = 0.0
        self.y_turn_done = False
        self.radar_ready_frames = 0
        self.radar_wall_lost_frames = 0
        self.radar_fallback_time = 0.0
        self.int_err = 0.0
        self.last_par_t = 0.0
        self.lane_good_frames = 0

    # ---------- 状态复位 ----------
    def _reset_vision(self, ctx):
        self.tracker.reset(self.tune)
        self.pid.reset()

    def enter(self, ctx):
        self.last_wz = 0.0
        self.pid.target_speed = self.cfg['target_speed'] or 0.2
        self._reset_vision(ctx)
        self.phase = 'ADVANCE'
        self.phase_started = ctx.machine_time
        self.advance_start_pose = ctx.odom
        self.advance_dist = 0.0
        self.line_follow_start_yaw = ctx.odom[2] if ctx.odom else 0.0
        self.roi_wide_remaining = self.cfg['roi_wide_frames_after_follow']
        self.turn_forced = False
        self.turn_start_yaw = 0.0
        self.turn_accum_deg = 0.0
        self.lost_creep_start = 0.0
        self.stop_line_enabled = False
        self.stop_line_enable_time = 0.0
        self.creep_started = False
        self.creep_start_pose = None
        self.creep_angular_z = 0.0
        self.y_turn_done = False
        self.radar_ready_frames = 0
        self.radar_wall_lost_frames = 0
        self.radar_fallback_time = 0.0
        self.int_err = 0.0
        self.last_par_t = ctx.machine_time
        self._ctrl_good_frames = 0
        self._set_poly_callbacks(ctx, use_wide=False, bias=0.0, stop_enabled=False)
        with ctx.lock:
            ctx._right_good_frames = 0
            ctx._right_last_near_x = None
        ctx.status['message'] = 'POLYLINE: 前进 %dcm 尝试巡线' % int(self.cfg['advance_distance'] * 100)

    def _set_poly_callbacks(self, ctx, use_wide, bias, stop_enabled):
        with ctx.lock:
            ctx.poly_roi['use_wide'] = bool(use_wide)
            ctx.poly_roi['bias'] = float(bias)
            ctx.poly_stop['enabled'] = bool(stop_enabled)
            ctx.poly_stop['enable_time'] = ctx.machine_time if stop_enabled else ctx.poly_stop['enable_time']

    # ---------- 巡线段工具 ----------
    def _enter_line_follow_from_turn(self, ctx, message):
        self.phase = 'LINE_FOLLOW'
        self.line_follow_start_yaw = ctx.odom[2] if ctx.odom else 0.0
        self.y_turn_done = True
        self.roi_wide_remaining = 0
        self.lost_creep_start = 0.0
        self.stop_line_enabled = True
        self.stop_line_enable_time = ctx.machine_time
        self._set_poly_callbacks(ctx, use_wide=False, bias=0.0, stop_enabled=True)
        ctx.status['message'] = message
        return None, False

    def _enter_turn_right(self, ctx, forced):
        self.turn_forced = forced
        self.turn_start_yaw = ctx.odom[2] if ctx.odom else 0.0
        self.turn_accum_deg = 0.0
        self.phase = 'TURN_RIGHT'
        self.phase_started = ctx.machine_time
        self.lost_creep_start = 0.0
        self._reset_vision(ctx)
        ctx.status['message'] = 'POLYLINE: 视觉丢失/左偏，前进右转%d°' % int(self.cfg['turn_yaw_deg'])
        return None, False

    # ---------- 雷达接管 / 兜底 ----------
    def _try_radar_handoff(self, now):
        if now - self.radar_fallback_time < self.cfg['radar_retry_cooldown']:
            return False
        scan_ok = self.radar.scan_fresh(now)
        lw = self.radar.left_wall_snapshot()
        gate = scan_ok and (lw is not None) and (
            self.cfg['radar_handoff_perp_min'] <= lw['perp'] <= self.cfg['radar_handoff_perp_max'])
        if gate:
            self.radar_ready_frames += 1
        else:
            self.radar_ready_frames = 0
        return self.radar_ready_frames >= self.cfg['radar_handoff_frames']

    def _radar_fallback(self, ctx, reason):
        """雷达异常 -> 复位视觉感知状态 -> 切回 LINE_FOLLOW。"""
        self._reset_vision(ctx)
        self.phase = 'LINE_FOLLOW'
        self.line_follow_start_yaw = ctx.odom[2] if ctx.odom else 0.0
        self.y_turn_done = True
        self.roi_wide_remaining = 0
        self.lost_creep_start = 0.0
        self.stop_line_enabled = True
        self.stop_line_enable_time = ctx.machine_time
        self._set_poly_callbacks(ctx, use_wide=False, bias=0.0, stop_enabled=True)
        self.radar_ready_frames = 0
        self.radar_wall_lost_frames = 0
        self.radar_fallback_time = ctx.machine_time
        ctx.status['message'] = '雷达异常回退: %s' % reason

    # ---------- 主步进 ----------
    def step(self, ctx, now):
        cfg = self.cfg
        cmd = ctx.make_twist()
        v = ctx.snapshot_vision()
        odom = ctx.odom

        if odom is None:
            ctx.status['message'] = 'POLYLINE: 无里程计，紧急停车'
            return cmd, MODE_FAULT
        if now - ctx.last_image_time > cfg['camera_timeout']:
            ctx.status['message'] = 'POLYLINE: 摄像头画面超时，紧急停车'
            return cmd, MODE_FAULT

        if self.phase == 'LINE_FOLLOW' and self.y_turn_done and self._try_radar_handoff(now):
            self.phase = 'RADAR_PREALIGN'
            self.phase_started = now
            self.radar_wall_lost_frames = 0
            self._set_poly_callbacks(ctx, use_wide=False, bias=0.0, stop_enabled=False)
            ctx.status['message'] = '雷达接管：贴墙引导'
            return cmd, None

        if self.phase == 'ADVANCE':
            return self._step_advance(ctx, now, v, odom, cmd)
        if self.phase == 'LINE_FOLLOW':
            return self._step_line_follow(ctx, now, v, odom, cmd)
        if self.phase == 'TURN_RIGHT':
            return self._step_turn_right(ctx, now, v, odom, cmd)
        if self.phase == 'SEARCH_RIGHT':
            return self._step_search_right(ctx, now, v, odom, cmd)
        if self.phase in ('RADAR_PREALIGN', 'RADAR_APPROACH', 'RADAR_TURN', 'RADAR_PARALLEL'):
            return self._step_radar(ctx, now, cmd)
        if self.phase == 'RADAR_DONE':
            ctx.status['message'] = '前方墙停车完成，折线段结束'
            return cmd, MODE_PARK

        return cmd, MODE_FAULT

    def _step_advance(self, ctx, now, v, odom, cmd):
        cfg = self.cfg
        self.advance_dist = math.hypot(odom[0] - self.advance_start_pose[0],
                                       odom[1] - self.advance_start_pose[1])
        if self.advance_dist >= cfg['advance_distance']:
            self.phase = 'LINE_FOLLOW'
            self.line_follow_start_yaw = odom[2]
            self.roi_wide_remaining = cfg['roi_wide_frames_after_follow']
            self._set_poly_callbacks(ctx, use_wide=True, bias=cfg['lane_bias_right_px'], stop_enabled=False)
            ctx.status['message'] = 'POLYLINE: 前进到位，巡线直行'
            return None, False
        if now - self.phase_started > cfg['advance_timeout']:
            self.phase = 'LINE_FOLLOW'
            self.line_follow_start_yaw = odom[2]
            self.roi_wide_remaining = cfg['roi_wide_frames_after_follow']
            self._set_poly_callbacks(ctx, use_wide=True, bias=cfg['lane_bias_right_px'], stop_enabled=False)
            ctx.status['message'] = 'POLYLINE: 前进超时，巡线直行'
            return None, False
        if v.get('pair_matched', False):
            self._ctrl_good_frames += 1
            if self._ctrl_good_frames >= cfg['lane_stable_frames']:
                self.phase = 'LINE_FOLLOW'
                self.line_follow_start_yaw = odom[2]
                self.roi_wide_remaining = cfg['roi_wide_frames_after_follow']
                self._set_poly_callbacks(ctx, use_wide=True, bias=cfg['lane_bias_right_px'], stop_enabled=False)
                ctx.status['message'] = 'POLYLINE: 巡线双线稳定，进入巡线PID'
                return None, False
        else:
            self._ctrl_good_frames = 0
        cmd.linear.x = cfg['advance_speed']
        cmd.linear.y = 0.0
        yaw_error = _norm(self.advance_start_pose[2] - odom[2])
        cmd.angular.z = max(-cfg['advance_wz_clamp'], min(cfg['advance_wz_clamp'], cfg['advance_kp_yaw'] * yaw_error))
        use_wide = self.advance_dist >= cfg['advance_roi_switch_dist']
        bias = cfg['lane_bias_right_px'] if use_wide else 0.0
        self._set_poly_callbacks(ctx, use_wide=use_wide, bias=bias, stop_enabled=False)
        ctx.status['message'] = 'POLYLINE: 前进 %.2fm' % self.advance_dist
        return cmd, None

    def _step_line_follow(self, ctx, now, v, odom, cmd):
        cfg = self.cfg
        if self.creep_started or v['stop_line_detected']:
            if not self.creep_started:
                self.creep_started = True
                self.creep_start_pose = odom
                self.creep_angular_z = max(-0.15, min(0.15, self.last_wz))
            traveled = 0.0
            if self.creep_start_pose is not None:
                traveled = math.hypot(odom[0] - self.creep_start_pose[0],
                                      odom[1] - self.creep_start_pose[1])
            if traveled >= cfg['creep_distance']:
                ctx.status['message'] = 'POLYLINE: 检测到停止线，蠕动到位，停车'
                return cmd, MODE_PARK
            cmd.linear.x = cfg['creep_speed']
            cmd.linear.y = 0.0
            cmd.angular.z = self.creep_angular_z
            return cmd, None

        if v['lane_valid']:
            self.lost_creep_start = 0.0
            if ctx.odom is not None:
                delta_yaw = math.degrees(_norm(odom[2] - self.line_follow_start_yaw))
                if delta_yaw > cfg['left_yaw_limit_deg']:
                    self._enter_turn_right(ctx, forced=True)
                    return None, False
            import common.control as control
            vel, h, c = control.compute_pid_ipm(
                v['heading_error_deg'], v['center_error_px'], self.pid, self.steer)
            self.last_wz = vel.angular.z
            self._set_poly_callbacks(ctx,
                                     use_wide=self.roi_wide_remaining > 0,
                                     bias=cfg['lane_bias_right_px'] if self.roi_wide_remaining > 0 else 0.0,
                                     stop_enabled=self.stop_line_enabled)
            self.roi_wide_remaining = max(0, self.roi_wide_remaining - 1)
            return vel, None
        else:
            if self.lost_creep_start == 0.0:
                self.lost_creep_start = now
            if now - self.lost_creep_start > cfg['lost_creep_timeout']:
                self._enter_turn_right(ctx, forced=False)
                return None, False
            cmd.linear.x = cfg['creep_speed']
            cmd.linear.y = 0.0
            cmd.angular.z = 0.0
            return cmd, None

    def _step_turn_right(self, ctx, now, v, odom, cmd):
        cfg = self.cfg
        if not self.turn_forced and v['right_fit_ok']:
            return self._enter_line_follow_from_turn(ctx, '右转中已找到可信右边界，切入巡线')
        self.turn_accum_deg = math.degrees(max(0.0, _norm(self.turn_start_yaw - odom[2])))
        if self.turn_accum_deg >= cfg['turn_yaw_deg']:
            self.turn_forced = False
            if v['phys_left_found']:
                return self._enter_line_follow_from_turn(ctx, '右转%d°完成且已找到左边界，恢复巡线' % int(cfg['turn_yaw_deg']))
            self.phase = 'SEARCH_RIGHT'
            self.phase_started = now
            ctx.status['message'] = '右转完成但未找到左边界，原地右转搜索入口'
            return None, False
        if now - self.phase_started > cfg['turn_timeout']:
            ctx.status['message'] = 'POLYLINE: 右转超时，停车'
            return cmd, MODE_FAULT
        cmd.linear.x = cfg['turn_advance_speed']
        cmd.linear.y = 0.0
        cmd.angular.z = cfg['turn_drive_wz']
        self._set_poly_callbacks(ctx, use_wide=False, bias=0.0, stop_enabled=False)
        return cmd, None

    def _step_search_right(self, ctx, now, v, odom, cmd):
        cfg = self.cfg
        if v['pair_matched']:
            return self._enter_line_follow_from_turn(ctx, '原地搜索已找到双线，切入巡线')
        if v['phys_left_found']:
            return self._enter_line_follow_from_turn(ctx, '原地搜索已找到物理左边界，切入巡线')
        self.turn_accum_deg = math.degrees(max(0.0, _norm(self.turn_start_yaw - odom[2])))
        if self.turn_accum_deg >= cfg['search_angle_deg']:
            return self._enter_line_follow_from_turn(ctx, '原地搜索达阈值仍未找到入口，硬切巡线前进')
        if now - self.phase_started > cfg['search_timeout']:
            ctx.status['message'] = 'POLYLINE: 原地搜索超时，停车'
            return cmd, MODE_FAULT
        cmd.linear.x = 0.0
        cmd.linear.y = 0.0
        cmd.angular.z = cfg['search_rotate_wz']
        self._set_poly_callbacks(ctx, use_wide=False, bias=0.0, stop_enabled=False)
        ctx.status['message'] = 'POLYLINE: 原地缓慢右转搜索入口'
        return cmd, None

    def _step_radar(self, ctx, now, cmd):
        cfg = self.cfg
        if not self.radar.scan_fresh(now):
            self._radar_fallback(ctx, 'scan超时，回退视觉巡线')
            return None, False
        if self.phase in ('RADAR_APPROACH', 'RADAR_TURN', 'RADAR_PARALLEL'):
            lw = self.radar.left_wall_snapshot()
            if lw is None:
                self.radar_wall_lost_frames += 1
            else:
                self.radar_wall_lost_frames = 0
            if self.radar_wall_lost_frames >= cfg['radar_wall_lost_frames']:
                self._radar_fallback(ctx, '左墙连续丢失，回退视觉巡线')
                return None, False

        lw = self.radar.left_wall_snapshot()
        fw = self.radar.front_wall_snapshot()

        if self.phase == 'RADAR_PREALIGN':
            return self._radar_prealign(ctx, now, lw, cmd)
        if self.phase == 'RADAR_APPROACH':
            return self._radar_approach(ctx, lw, cmd)
        if self.phase == 'RADAR_TURN':
            return self._radar_turn(ctx, now, lw, cmd)
        if self.phase == 'RADAR_PARALLEL':
            return self._radar_parallel(ctx, now, lw, fw, cmd)
        return cmd, MODE_FAULT

    def _radar_prealign(self, ctx, now, lw, cmd):
        cfg = self.cfg
        if cfg['skip_prealign']:
            self.phase = 'RADAR_APPROACH'
            cmd.linear.x = cfg['forward_speed']
            cmd.angular.z = 0.0
            ctx.status['message'] = 'RADAR->APPROACH(跳过原地对准)'
            return cmd, None
        if lw is None:
            if now - self.phase_started > cfg['align_timeout']:
                self._radar_fallback(ctx, 'PREALIGN超时且无左墙，回退视觉')
                return None, False
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            ctx.status['message'] = 'RADAR_PREALIGN_NO_WALL'
            return cmd, None
        err_ang = cfg['align_target_deg'] - lw['line_angle']
        while err_ang > 90.0:
            err_ang -= 180.0
        while err_ang < -90.0:
            err_ang += 180.0
        if abs(err_ang) <= cfg['align_stop_err_deg']:
            self.phase = 'RADAR_APPROACH'
            cmd.linear.x = cfg['forward_speed']
            cmd.angular.z = 0.0
            ctx.status['message'] = 'RADAR->APPROACH'
            return cmd, None
        if now - self.phase_started > cfg['align_timeout']:
            self.phase = 'RADAR_APPROACH'
            cmd.linear.x = cfg['forward_speed']
            cmd.angular.z = 0.0
            ctx.status['message'] = 'RADAR->APPROACH(超时)'
            return cmd, None
        wz = -cfg['kp_align'] * err_ang
        if abs(wz) < cfg['align_wz_min']:
            wz = -cfg['align_wz_min'] if err_ang > 0 else cfg['align_wz_min']
        wz = max(-cfg['align_wz_max'], min(cfg['align_wz_max'], wz))
        cmd.linear.x = 0.0
        cmd.angular.z = wz
        ctx.status['message'] = 'PREALIGN line=%.0f err=%+.0f' % (lw['line_angle'], err_ang)
        return cmd, None

    def _radar_approach(self, ctx, lw, cmd):
        cfg = self.cfg
        if lw is None:
            cmd.linear.x = cfg['forward_speed']
            cmd.angular.z = 0.0
            ctx.status['message'] = 'RADAR_APPROACH_NO_WALL'
            return cmd, None
        if lw['d_min'] <= cfg['turn_trigger_dist']:
            self.phase = 'RADAR_TURN'
            cmd.linear.x = cfg['forward_speed']
            cmd.angular.z = cfg['turn_wz_min']
            ctx.status['message'] = 'RADAR->TURN'
            return cmd, None
        ang = lw['line_angle']
        cmd.linear.x = cfg['forward_speed']
        cmd.angular.z = -0.08 if ang > 125.0 else 0.0
        ctx.status['message'] = 'RADAR_APPROACH d=%.3f line=%.0f' % (lw['d_min'], ang)
        return cmd, None

    def _radar_turn(self, ctx, now, lw, cmd):
        cfg = self.cfg
        if lw is None:
            return cmd, MODE_FAULT
        d = lw['d_min']
        ang = lw['line_angle']
        par = min(ang, 180.0 - ang)
        done = (d <= cfg['turn_done_dist'] and par <= cfg['parallel_angle_deg'])
        if done:
            self.int_err = 0.0
            self.last_par_t = now
            self.phase = 'RADAR_PARALLEL'
            ctx.status['message'] = 'RADAR->PARALLEL'
            return None, False
        align_wz = -(0.10 + cfg['kp_turn_ang'] * max(0.0, par - 20.0))
        dist_wz = cfg['kp_parallel'] * (d - cfg['keep_wall_dist'])
        if d < cfg['turn_guard_dist']:
            dist_wz = max(dist_wz, 0.0)
            wz = max(align_wz + dist_wz, cfg['turn_guard_wz'])
            speed = cfg['forward_speed']
        else:
            wz = max(cfg['turn_wz_max'], min(cfg['turn_wz_min'], align_wz + dist_wz))
            speed = cfg['forward_speed']
            if d < cfg['turn_done_dist']:
                speed = cfg['forward_speed'] * 0.5
        cmd.linear.x = speed
        cmd.angular.z = wz
        ctx.status['message'] = 'RADAR_TURN d=%.3f par=%.0f wz=%.2f' % (d, par, wz)
        return cmd, None

    def _radar_parallel(self, ctx, now, lw, fw, cmd):
        cfg = self.cfg
        if lw is None:
            return cmd, MODE_FAULT
        d = lw['d_min']
        ang = lw['line_angle']
        dev = ang if ang <= 90.0 else ang - 180.0
        err = d - cfg['keep_wall_dist']
        self.int_err += err * (now - self.last_par_t)
        self.last_par_t = now
        self.int_err = max(-cfg['parallel_int_max'], min(cfg['parallel_int_max'], self.int_err))
        head_wz = cfg['kp_ang_head'] * dev
        if d < cfg['safe_clamp_dist']:
            head_wz = min(head_wz, 0.0)
        i_term = cfg['ki_parallel'] * self.int_err
        wz = cfg['kp_parallel'] * err + i_term + head_wz
        wz = max(-cfg['parallel_wz_clamp'], min(cfg['parallel_wz_clamp'], wz))
        speed = cfg['forward_speed']
        if fw is not None and fw['d_min'] < cfg['front_slow_dist']:
            if fw['d_min'] < cfg['front_stop_dist']:
                self.phase = 'RADAR_DONE'
                ctx.status['message'] = '前方墙 %.3f m，折线停车完成' % fw['d_min']
                return None, False
            speed = cfg['front_slow_speed']
        cmd.linear.x = speed
        cmd.angular.z = wz
        ctx.status['message'] = 'RADAR_PARALLEL d=%.3f i=%.2f front=%s wz=%.2f' % (
            d, i_term, ('%.3f' % fw['d_min']) if fw else '---', wz)
        return cmd, None