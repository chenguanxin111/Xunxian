#!/usr/bin/env python3
"""安全巡线节点 5：沿用去年逐行聚类思路，默认仅感知，Web 端手动解锁。"""
import json
import math
import os
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(PROJECT_DIR, "config", "white_line.json")
PORT = 5005
NODE_NAME = "line_following_node5"
IMAGE_TOPIC = "/usb_cam/image_raw"
CMD_TOPIC = "/cmd_vel"
MASK_TOPIC = "/line_following_node5/debug/mask"
OVERLAY_TOPIC = "/line_following_node5/debug/overlay"

REQUIRED_CONFIG = (
    "low_h", "high_h", "low_s", "high_s", "low_v", "high_v",
    "roi_top", "roi_bottom", "roi_left", "roi_right",
    "blur_ksize", "erode_iter", "erode_ksize", "dilate_iter", "dilate_ksize",
)

PID_TABLE = {
    "straight": (0.0144, 0.28),
    "small": (0.0180, 0.24),
    "medium": (0.0240, 0.20),
    "large": (0.0288, 0.168),
}


class State:
    def __init__(self):
        self.lock = threading.RLock()
        self.mode = "DISARMED"
        self.message = "节点启动，默认仅感知"
        self.config = None
        self.config_ok = False
        self.config_error = "配置尚未加载"
        self.last_image_time = 0.0
        self.last_valid_time = 0.0
        self.valid_streak = 0
        self.lost_frames = 0
        self.vision_valid = False
        self.confidence = 0.0
        self.centers = []
        self.edges = []
        self.bottom_edges = []
        self.last_centers = []
        self.angle_error = 0.0
        self.center_error = 0.0
        self.single_edge_ratio = 0.0
        self.target_speed = 0.12
        self.last_error = 0.0
        self.last_control_time = 0.0
        self.last_angular = 0.0
        self.command_x = 0.0
        self.command_y = 0.0
        self.command_z = 0.0

    def status(self):
        return {
            "mode": self.mode,
            "message": self.message,
            "config_ok": self.config_ok,
            "config_path": CONFIG_PATH,
            "config_error": self.config_error,
            "vision_valid": self.vision_valid,
            "confidence": round(self.confidence, 3),
            "valid_streak": self.valid_streak,
            "lost_frames": self.lost_frames,
            "center_count": len(self.centers),
            "single_edge_ratio": round(self.single_edge_ratio, 3),
            "angle_error_deg": round(self.angle_error, 2),
            "center_error_px": round(self.center_error, 2),
            "target_speed": self.target_speed,
            "command_linear_x": round(self.command_x, 3),
            "command_linear_y": round(self.command_y, 3),
            "command_angular_z": round(self.command_z, 3),
            "image_age_s": None if not self.last_image_time else round(time.time() - self.last_image_time, 2),
        }


state = State()
bridge = CvBridge()
cmd_pub = None
mask_pub = None
overlay_pub = None


def validate_config(data):
    missing = [key for key in REQUIRED_CONFIG if key not in data]
    if missing:
        raise ValueError("缺少字段: " + ", ".join(missing))
    for lo, hi, maximum in (("low_h", "high_h", 179), ("low_s", "high_s", 255), ("low_v", "high_v", 255)):
        if not (0 <= int(data[lo]) <= int(data[hi]) <= maximum):
            raise ValueError("HSV 范围非法: %s/%s" % (lo, hi))
    for key in ("roi_top", "roi_bottom", "roi_left", "roi_right"):
        if not 0.0 <= float(data[key]) <= 1.0:
            raise ValueError("ROI 字段越界: " + key)
    if float(data["roi_top"]) >= float(data["roi_bottom"]):
        raise ValueError("roi_top 必须小于 roi_bottom")
    if float(data["roi_left"]) >= float(data["roi_right"]):
        raise ValueError("roi_left 必须小于 roi_right")
    result = dict(data)
    for key in ("low_h", "high_h", "low_s", "high_s", "low_v", "high_v",
                "blur_ksize", "erode_iter", "erode_ksize", "dilate_iter", "dilate_ksize"):
        result[key] = int(result[key])
    for key in ("roi_top", "roi_bottom", "roi_left", "roi_right"):
        result[key] = float(result[key])
    return result


def load_config():
    with state.lock:
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as stream:
                state.config = validate_config(json.load(stream))
            state.config_ok = True
            state.config_error = ""
            state.message = "配置加载成功，等待有效车道线"
            rospy.loginfo("node5 已加载配置: %s", CONFIG_PATH)
        except Exception as exc:
            state.config = None
            state.config_ok = False
            state.config_error = str(exc)
            state.mode = "FAULT"
            state.message = "配置错误，禁止运行"
            rospy.logerr("node5 配置加载失败: %s", exc)


def make_mask(frame, cfg):
    blur = cfg["blur_ksize"]
    if blur >= 3:
        blur += 1 if blur % 2 == 0 else 0
        frame = cv2.GaussianBlur(frame, (blur, blur), 0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv,
                       np.array([cfg["low_h"], cfg["low_s"], cfg["low_v"]], np.uint8),
                       np.array([cfg["high_h"], cfg["high_s"], cfg["high_v"]], np.uint8))
    h, w = mask.shape
    x1, x2 = int(w * cfg["roi_left"]), int(w * cfg["roi_right"])
    y1, y2 = int(h * cfg["roi_top"]), int(h * cfg["roi_bottom"])
    roi = np.zeros_like(mask)
    roi[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
    for name in ("erode", "dilate"):
        count = max(0, cfg[name + "_iter"])
        size = max(1, cfg[name + "_ksize"])
        size += 1 if size % 2 == 0 else 0
        if count:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
            roi = getattr(cv2, name)(roi, kernel, iterations=count)
    return roi, (x1, y1, x2, y2)


def row_clusters(row, min_width=2):
    xs = np.flatnonzero(row == 255)
    if not len(xs):
        return []
    groups = np.split(xs, np.flatnonzero(np.diff(xs) > 1) + 1)
    return [float(np.mean(group)) for group in groups if len(group) >= min_width]


def extract_centerline(mask, roi):
    """去年逐行聚类算法的安全化版本；点顺序为近端（底部）到远端（顶部）。"""
    h, w = mask.shape
    x1, y1, x2, y2 = roi
    centers, edges = [], []
    bottom_edges = []
    single_rows = valid_rows = 0
    previous_center = None
    max_jump = max(20.0, w / 8.0)

    for y in range(y2 - 1, y1 - 1, -4):
        clusters = [x for x in row_clusters(mask[y]) if x1 <= x < x2]
        if not clusters:
            continue
        valid_rows += 1
        chosen = []
        if len(clusters) == 1:
            single_rows += 1
            edge = clusters[0]
            # 与去年代码一致，用图像边界构造不可见边；历史趋势只用于消除左右抖动。
            if len(centers) >= 2:
                virtual = 0.0 if centers[-1][0] < centers[-2][0] else float(w - 1)
            else:
                virtual = float(w - 1) if edge < w / 2.0 else 0.0
            chosen = [edge, virtual]
        else:
            if previous_center is None:
                left = [x for x in clusters if x < w / 2.0]
                right = [x for x in clusters if x >= w / 2.0]
                chosen = [max(left) if left else 0.0, min(right) if right else float(w - 1)]
            else:
                pairs = [(a, b) for i, a in enumerate(clusters) for b in clusters[i + 1:]
                         if (b - a) >= w * 0.20]
                if pairs:
                    chosen = list(min(pairs, key=lambda p: abs((p[0] + p[1]) / 2.0 - previous_center)))
                else:
                    edge = min(clusters, key=lambda x: abs(x - previous_center))
                    chosen = [edge, float(w - 1) if edge < previous_center else 0.0]
                    single_rows += 1

        center = float(sum(chosen) / 2.0)
        if previous_center is not None and abs(center - previous_center) > max_jump:
            continue
        point = (int(round(center)), y)
        centers.append(point)
        previous_center = center
        for edge in chosen:
            edges.append((int(round(edge)), y))
        if not bottom_edges:
            bottom_edges = [int(round(x)) for x in chosen]

    single_ratio = float(single_rows) / valid_rows if valid_rows else 1.0
    roi_rows = max(1, int((y2 - y1) / 4))
    coverage = min(1.0, len(centers) / float(max(8, roi_rows)))
    continuity = 0.0
    if len(centers) >= 3:
        jumps = np.abs(np.diff([p[0] for p in centers]))
        continuity = max(0.0, 1.0 - float(np.mean(jumps)) / max_jump)
    confidence = coverage * (0.55 + 0.45 * continuity) * (0.75 if single_ratio > 0.9 else 1.0)
    return centers, edges, bottom_edges, single_ratio, confidence


def calculate_errors(centers, width):
    if len(centers) < 3:
        raise ValueError("中心点不足")
    near_count = min(4, len(centers))
    center_error = float(np.mean([p[0] for p in centers[:near_count]]) - width / 2.0)
    look_index = min(len(centers) - 1, max(2, int(round(len(centers) / 3.5))))
    near = centers[0]
    look = centers[look_index]
    forward = max(1.0, float(near[1] - look[1]))
    lateral = float(look[0] - near[0])
    # 镜像图中向右偏对应小车左转，因此保持去年误差符号。
    angle_error = math.degrees(math.atan2(lateral, forward))
    return angle_error, center_error, look


def safe_imgmsg(image, encoding):
    try:
        return bridge.cv2_to_imgmsg(image, encoding=encoding)
    except Exception:
        msg = Image()
        msg.height, msg.width = image.shape[:2]
        msg.encoding = encoding
        msg.is_bigendian = 0
        msg.step = msg.width if image.ndim == 2 else msg.width * image.shape[2]
        msg.data = image.tobytes()
        return msg


def image_cb(msg):
    try:
        frame = bridge.imgmsg_to_cv2(msg, "bgr8")
        frame = cv2.flip(frame, 1)
        with state.lock:
            cfg = dict(state.config) if state.config_ok else None
        if cfg is None:
            return
        mask, roi = make_mask(frame, cfg)
        centers, edges, bottom, single_ratio, confidence = extract_centerline(mask, roi)
        valid = len(centers) >= 6 and confidence >= 0.28
        look = None
        if valid:
            angle, offset, look = calculate_errors(centers, frame.shape[1])
        else:
            angle, offset = 0.0, 0.0

        now = time.time()
        with state.lock:
            state.last_image_time = now
            state.centers = centers
            state.edges = edges
            state.bottom_edges = bottom
            state.single_edge_ratio = single_ratio
            state.confidence = confidence
            state.vision_valid = valid
            if valid:
                state.angle_error = angle
                state.center_error = offset
                state.last_centers = centers
                state.last_valid_time = now
                state.valid_streak += 1
                state.lost_frames = 0
            else:
                state.valid_streak = 0
                state.lost_frames += 1
            mode = state.mode

        overlay = frame.copy()
        cv2.rectangle(overlay, (roi[0], roi[1]), (roi[2] - 1, roi[3] - 1), (255, 150, 0), 2)
        for point in edges:
            cv2.circle(overlay, point, 2, (0, 0, 255), -1)
        if centers:
            cv2.polylines(overlay, [np.asarray(centers, np.int32)], False, (0, 255, 0), 3)
        if look:
            cv2.circle(overlay, look, 7, (255, 0, 255), -1)
        cv2.putText(overlay, "%s valid=%s conf=%.2f" % (mode, valid, confidence), (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 255, 255), 2)
        cv2.putText(overlay, "angle=%.1f center=%.1f single=%.2f" % (angle, offset, single_ratio), (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
        mask_pub.publish(safe_imgmsg(mask, "mono8"))
        overlay_pub.publish(safe_imgmsg(overlay, "bgr8"))
    except Exception as exc:
        rospy.logerr_throttle(2.0, "node5 图像回调异常: %s", exc)


def reset_controller():
    state.last_error = 0.0
    state.last_control_time = 0.0
    state.last_angular = 0.0


def publish_stop():
    state.command_x = state.command_y = state.command_z = 0.0
    reset_controller()
    if cmd_pub is not None:
        cmd_pub.publish(Twist())


def fail(message):
    state.mode = "FAULT"
    state.message = message
    publish_stop()


def control_timer(_event):
    with state.lock:
        if state.mode != "RUNNING":
            return
        now = time.time()
        if now - state.last_image_time > 0.6:
            fail("摄像头超时，已停车")
            return
        if not state.vision_valid:
            # 极短丢线期低速保持上次方向；超过 0.3 秒或 7 帧立即停车。
            if now - state.last_valid_time <= 0.30 and state.lost_frames <= 7:
                cmd = Twist()
                cmd.linear.x = 0.05
                cmd.angular.z = max(-0.25, min(0.25, state.last_angular))
                state.command_x, state.command_y, state.command_z = cmd.linear.x, 0.0, cmd.angular.z
                cmd_pub.publish(cmd)
                return
            fail("持续丢失有效车道线，已停车")
            return

        error = state.angle_error
        dt = now - state.last_control_time if state.last_control_time else 0.04
        dt = max(0.02, min(0.15, dt))
        derivative = (error - state.last_error) / dt if state.last_control_time else 0.0
        ae = abs(error)
        kp, kd = PID_TABLE["straight" if ae < 15 else "small" if ae < 30 else "medium" if ae < 50 else "large"]
        angular = kp * error + 0.0010 * state.center_error + kd * derivative * 0.04
        angular = max(-0.75, min(0.75, angular))
        # 同时按转角、横向偏差、置信度降速。
        speed_scale = max(0.45, 1.0 - ae / 95.0 - min(abs(state.center_error) / 500.0, 0.25))
        speed_scale *= max(0.65, min(1.0, state.confidence / 0.55))
        speed = max(0.055, state.target_speed * speed_scale)
        lateral = 0.0
        if state.single_edge_ratio < 0.75 and len(state.bottom_edges) >= 2:
            lateral = max(-0.025, min(0.025, -0.00012 * state.center_error))

        cmd = Twist()
        cmd.linear.x, cmd.linear.y, cmd.angular.z = speed, lateral, angular
        state.last_error = error
        state.last_control_time = now
        state.last_angular = angular
        state.command_x, state.command_y, state.command_z = speed, lateral, angular
        cmd_pub.publish(cmd)


PAGE = """<!doctype html><meta charset='utf-8'><title>巡线 Node5</title>
<style>body{font:16px sans-serif;background:#101827;color:#eee;max-width:1100px;margin:20px auto}section{background:#1f2937;padding:16px;margin:12px;border-radius:10px}button{padding:11px 18px;margin:5px;border:0;border-radius:6px}img{width:48%}pre{color:#67e8f9}</style>
<h2>巡线 Node5（安全逐行中线算法）</h2><section><b>默认不运动；必须连续识别有效中线后才能启动。</b><p>
<button onclick="post('/api/start')">开始</button><button onclick="post('/api/stop')">停车</button>
<button onclick="post('/api/reset')">复位</button><button onclick="post('/api/reload')">重载配置</button></p><pre id='s'></pre></section>
<section><img id='o'><img id='m'></section><script>
o.src='http://'+location.hostname+':8080/stream?topic=/line_following_node5/debug/overlay';
m.src='http://'+location.hostname+':8080/stream?topic=/line_following_node5/debug/mask';
async function post(p){let d=await(await fetch(p,{method:'POST'})).json();if(!d.ok)alert(d.error)}
setInterval(async()=>s.textContent=JSON.stringify(await(await fetch('/api/status')).json(),null,2),400);</script>"""


class Handler(BaseHTTPRequestHandler):
    def reply(self, obj, code=200):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if urlparse(self.path).path == "/":
            data = PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        elif urlparse(self.path).path == "/api/status":
            with state.lock:
                self.reply(state.status())
        else:
            self.send_error(404)

    def do_POST(self):
        path = urlparse(self.path).path
        with state.lock:
            if path == "/api/start":
                if not state.config_ok:
                    self.reply({"ok": False, "error": "配置无效: " + state.config_error})
                elif time.time() - state.last_image_time > 0.6:
                    self.reply({"ok": False, "error": "摄像头画面超时"})
                elif not state.vision_valid or state.valid_streak < 5:
                    self.reply({"ok": False, "error": "尚未连续识别到有效车道中线"})
                else:
                    reset_controller()
                    state.mode = "RUNNING"
                    state.message = "低速巡线运行中"
                    self.reply({"ok": True})
            elif path == "/api/stop":
                state.mode, state.message = "STOPPED", "用户停车"
                publish_stop()
                self.reply({"ok": True})
            elif path == "/api/reset":
                state.mode, state.message = "DISARMED", "已复位为仅感知"
                publish_stop()
                self.reply({"ok": True})
            elif path == "/api/reload":
                publish_stop()
                state.mode = "DISARMED"
                load_config()
                self.reply({"ok": state.config_ok, "error": state.config_error})
            else:
                self.send_error(404)

    def log_message(self, _format, *_args):
        return


class Server(ThreadingHTTPServer):
    allow_reuse_address = True

    def server_bind(self):
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        super().server_bind()


def shutdown():
    with state.lock:
        publish_stop()


def main():
    global cmd_pub, mask_pub, overlay_pub
    rospy.init_node(NODE_NAME, anonymous=False)
    cmd_pub = rospy.Publisher(CMD_TOPIC, Twist, queue_size=1)
    mask_pub = rospy.Publisher(MASK_TOPIC, Image, queue_size=1)
    overlay_pub = rospy.Publisher(OVERLAY_TOPIC, Image, queue_size=1)
    load_config()
    rospy.Subscriber(IMAGE_TOPIC, Image, image_cb, queue_size=1, buff_size=2 ** 24)
    rospy.Timer(rospy.Duration(0.04), control_timer)
    rospy.on_shutdown(shutdown)
    rospy.loginfo("巡线 node5 已启动（仅感知）: http://0.0.0.0:%d", PORT)
    Server(("0.0.0.0", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()