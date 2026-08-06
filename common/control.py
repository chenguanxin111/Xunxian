"""控制公共模块（compute_pid_ipm 原样搬自 line_following_ss_pure_ipm.py）。

PidState 存放滤波/微分/限幅等控制历史，compute_pid_ipm(heading, center, st, tune) 与 ss_pure 行为一致。
"""
import collections
import math
import time

from geometry_msgs.msg import Twist


def normalize_angle(value):
    return math.atan2(math.sin(value), math.cos(value))


class PidState:
    def __init__(self):
        self.target_speed = 0.36
        self.heading_filt = 0.0
        self.center_filt = 0.0
        self.heading_prev = None
        self.heading_prev_t = 0.0
        self.last_wz = 0.0
        self.ctl = {}
        self.heading_samples = collections.deque(maxlen=32)
        self.center_samples = collections.deque(maxlen=32)

    def reset(self):
        self.heading_filt = 0.0
        self.center_filt = 0.0
        self.heading_prev = None
        self.heading_prev_t = 0.0
        self.last_wz = 0.0
        self.ctl = {}
        self.heading_samples.clear()
        self.center_samples.clear()


def compute_pid_ipm(heading_deg, center_px, st, tune):
    """前瞻方位角 + 分段增益控制（去年：直线低增益防摆，弯道高增益强打）。"""
    heading_deg -= tune['heading_bias_deg']

    if st.heading_prev is not None and tune['heading_spike_max_deg'] > 0.0:
        if abs(heading_deg - st.heading_filt) > tune['heading_spike_max_deg']:
            heading_deg = st.heading_filt

    eh = tune['ema_h']
    ec = tune['ema_c']
    h = eh * heading_deg + (1 - eh) * st.heading_filt
    c = ec * center_px + (1 - ec) * st.center_filt
    st.heading_filt = h
    st.center_filt = c

    now = time.time()

    deriv_h = 0.0
    st.heading_samples.append((now, h))
    while len(st.heading_samples) >= 2 and now - st.heading_samples[0][0] > 0.25:
        st.heading_samples.popleft()
    if len(st.heading_samples) >= 2:
        t0, h0 = st.heading_samples[0]
        dt = now - t0
        if dt > 0.05:
            deriv_h = (h - h0) / dt

    deriv_c = 0.0
    st.center_samples.append((now, c))
    while len(st.center_samples) >= 2 and now - st.center_samples[0][0] > 0.25:
        st.center_samples.popleft()
    if len(st.center_samples) >= 2:
        t0, c0 = st.center_samples[0]
        dt = now - t0
        if dt > 0.05:
            deriv_c = (c - c0) / dt
    st.heading_prev = h
    st.heading_prev_t = now

    ah = abs(h)
    if ah <= tune['hk1_deg']:
        kp = tune['kp_heading_lo']
    elif ah >= tune['hk2_deg']:
        t = (ah - tune['hk2_deg']) / max(1e-6, tune['hk3_deg'] - tune['hk2_deg'])
        kp = tune['kp_heading_mid'] + (tune['kp_heading_hi'] - tune['kp_heading_mid']) * min(1.0, t)
    else:
        t = (ah - tune['hk1_deg']) / max(1e-6, tune['hk2_deg'] - tune['hk1_deg'])
        kp = tune['kp_heading_lo'] + (tune['kp_heading_mid'] - tune['kp_heading_lo']) * t

    if abs(c) < tune['deadband_center_px'] and abs(h) < tune['deadband_heading_deg']:
        wz = 0.0
    else:
        wz = -(kp * math.radians(h) + tune['kp_center'] * c) \
             - tune['kd_heading'] * math.radians(deriv_h) - tune['kd_center'] * deriv_c
    wz = max(-tune['wz_max'], min(tune['wz_max'], wz))
    wz_pre_slew = wz

    if tune['wz_slew'] > 0.0:
        slew_tick = tune['wz_slew'] * 0.05
        st.last_wz = max(st.last_wz - slew_tick, min(st.last_wz + slew_tick, wz))
        wz = st.last_wz

    st.ctl = {
        'raw_h': heading_deg, 'raw_c': center_px,
        'h': h, 'c': c, 'deriv_h': deriv_h, 'deriv_c': deriv_c,
        'kp': kp, 'wz_pre': wz_pre_slew, 'wz_post': wz,
    }

    speed = st.target_speed
    if ah > tune['speed_h_deg']:
        speed = max(0.18, st.target_speed - tune['speed_k_deg'] * (ah - tune['speed_h_deg']))

    vel = Twist()
    vel.angular.z = wz
    vel.linear.y = -tune['kp_lat'] * c
    vel.linear.x = speed
    return vel, h, c
