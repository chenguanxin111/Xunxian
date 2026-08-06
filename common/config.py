"""统一从 config/*.json 加载参数，带默认值兜底。"""
import json
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(REPO_ROOT, 'config')


def _load(name, fallback):
    path = os.path.join(CONFIG_DIR, name)
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as exc:
        print('读取配置失败 %s: %s' % (path, exc))
        return dict(fallback)


def persp_path():
    return os.path.join(CONFIG_DIR, 'perspective_params.json')


def load_hsv():
    return _load('white_lane.json', {
        'low_h': 42, 'high_h': 179, 'low_s': 5, 'high_s': 71,
        'low_v': 116, 'high_v': 255, 'blur_ksize': 4,
    })


def load_lane_tune():
    return _load('lane_detect.json', {
        'roi_bottom_ratio': 0.30, 'min_center_pts': 3, 'min_center_span': 25.0,
        'poly_max_resid': 20.0, 'lane_width_min': 110.0, 'lane_width_max': 230.0,
        'lane_half_width': 84.0, 'max_slope_diff': 0.55, 'clean_min_h': 12,
        'clean_min_area': 40, 'clean_min_ratio': 0.35, 'lookahead_px': 120.0,
        'track_stale_frames': 10, 'max_width_dev': 35.0, 'max_cluster_w': 48,
        'match_min_overlap': 18.0,
    })


def load_steering():
    return _load('steering.json', {
        'kp_heading_lo': 0.8, 'kp_heading_mid': 0.8, 'kp_heading_hi': 0.75,
        'hk1_deg': 5.0, 'hk2_deg': 10.0, 'hk3_deg': 18.0, 'kp_center': 0.018,
        'kd_heading': 0.10, 'kd_center': 0.005, 'heading_spike_max_deg': 5.0,
        'ema_h': 0.35, 'ema_c': 0.55, 'wz_slew': 3.0, 'wz_max': 0.55,
        'speed_h_deg': 12.0, 'speed_k_deg': 0.004, 'kp_lat': 0.0,
        'heading_bias_deg': 0.0, 'deadband_center_px': 3.0,
        'deadband_heading_deg': 1.2,
    })


def load_stopline():
    return _load('stopline.json', {
        'stop_line_roi_top_ratio': 0.80, 'stop_line_width_ratio': 0.40,
        'stop_line_thin_ratio': 0.40, 'creep_speed': 0.10,
        'creep_distance': 0.10, 'camera_timeout': 0.8,
    })


def load_align():
    return _load('align.json', {
        'speed': 0.04, 'center_tol_px': 18.0, 'heading_tol_deg': 12.0,
        'confirm_frames': 10, 'kp_y': 0.0022, 'kp_z': 0.0015,
        'wz_clamp': 0.10, 'max_distance': 0.40, 'timeout': 20.0,
        'hsv': {
            'low_h': 0, 'high_h': 179, 'low_s': 0, 'high_s': 45,
            'low_v': 170, 'high_v': 255,
        },
        'roi_top': 0.45, 'roi_bottom': 1.0, 'roi_left': 0.0, 'roi_right': 1.0,
        'blur_ksize': 3, 'erode_iter': 0, 'erode_ksize': 3,
        'dilate_iter': 2, 'dilate_ksize': 3,
    })


def load_turn():
    return _load('turn.json', {
        'f_yaw_max_time': 0.6, 'f_yaw_tol_deg': 1.5, 'f_yaw_kp': 1.2,
        'f_yaw_wz': 0.15, 'advance_speed': 0.15, 'advance_distance': 0.30,
        'advance_timeout': 10.0, 'rotate_speed': 0.35,
        'rotate_target_deg': 65.0, 'rotate_timeout': 12.0,
        'search_rotate_wz': 0.15, 'search_accum_limit_deg': 85.0,
        'search_timeout': 10.0, 'search_confirm_frames': 8,
    })


def load_polyline():
    return _load('polyline.json', {
        'target_speed': 0.2, 'advance_speed': 0.15,
        'advance_distance': 0.50, 'advance_timeout': 8.0,
        'advance_kp_yaw': 0.8, 'advance_wz_clamp': 0.08,
        'lane_stable_frames': 10, 'roi_wide_bottom_ratio': 0.48,
        'roi_tight_bottom_ratio': 0.30, 'roi_wide_frames_after_follow': 15,
        'advance_roi_switch_dist': 0.30, 'lane_bias_right_px': 10.0,
        'camera_timeout': 0.8, 'creep_speed': 0.10,
        'creep_distance': 0.10, 'lost_creep_timeout': 2.0,
        'stop_line_enable_delay_sec': 1.0, 'left_yaw_limit_deg': 20.0,
        'turn_advance_speed': 0.131, 'turn_drive_wz': -0.29,
        'turn_yaw_deg': 47.0, 'turn_timeout': 25.0,
        'right_trust_nx_min': 340.0, 'right_trust_span_min': 60.0,
        'right_trust_frames': 5, 'right_trust_jitter_px': 25.0,
        'search_rotate_wz': -0.25, 'search_angle_deg': 70.0,
        'search_timeout': 15.0, 'keep_wall_dist': 0.26,
        'turn_trigger_dist': 0.36, 'turn_done_dist': 0.28,
        'parallel_angle_deg': 15.0, 'forward_speed': 0.20,
        'turn_wz_max': -0.40, 'turn_wz_min': -0.18,
        'kp_turn_ang': 0.008, 'turn_guard_dist': 0.18,
        'turn_guard_wz': -0.10, 'kp_parallel': 0.55,
        'ki_parallel': 0.35, 'parallel_int_max': 0.15,
        'kp_ang_head': 0.02, 'parallel_wz_clamp': 0.30,
        'front_slow_dist': 0.60, 'front_stop_dist': 0.50,
        'front_slow_speed': 0.08, 'min_safe_dist': 0.13,
        'safe_clamp_dist': 0.18, 'align_target_deg': 135.0,
        'align_stop_err_deg': 2.0, 'kp_align': 0.06,
        'align_wz_max': 0.35, 'align_wz_min': 0.15,
        'align_timeout': 5.0, 'skip_prealign': True,
        'left_ang_lo': 25.0, 'left_ang_hi': 155.0,
        'front_ang_lo': -30.0, 'front_ang_hi': 30.0,
        'scan_timeout': 0.6, 'break_gap': 0.25,
        'radar_handoff_perp_min': 0.25, 'radar_handoff_perp_max': 0.40,
        'radar_handoff_frames': 5, 'radar_wall_lost_frames': 12,
        'radar_retry_cooldown': 3.0,
    })


def load_park():
    return _load('park.json', {
        'front_target': 0.17, 'front_tol': 0.015, 'side_target': 0.24,
        'side_min': 0.16, 'side_max': 0.34, 'ang_tol_deg': 15.0,
        'done_frames': 10, 'side_pre_dist': 0.25, 'pre_tol': 0.02,
        'pre_vel': 0.15, 'kp_y': 2.0, 'pre_align_timeout': 5.0,
        'vx_max': 0.15, 'vx_min': 0.06, 'kp_f': 0.35, 'kp_ang': 0.5,
        'kp_side': 1.5, 'ki_side': 0.6, 's_int_max': 0.06,
        'wz_max': 0.35, 'wz_near_wall_start': 0.25,
        'wz_near_wall_end': 0.03, 'detect_creep': 0.08,
        'detect_creep_max_t': 3.0, 'front_ang_lo': -30.0,
        'front_ang_hi': 30.0, 'side_ang_min': 35.0,
        'side_ang_max': 150.0, 'side_parallel_tol': 45.0,
        'side_arm_max': 0.60, 'side_hold_dist': 0.40,
        'min_pts': 8, 'rmax': 5.0, 'min_safe_dist': 0.13,
        'scan_timeout': 0.6, 'scan_wait': 2.0, 'detect_timeout': 10.0,
        'drive_timeout': 15.0, 'drive_travel_max': 1.0,
        'drive_wide_front': 0.35, 'drive_wide_side_tol': 0.05,
        'drive_wide_ang': 30.0,
    })
