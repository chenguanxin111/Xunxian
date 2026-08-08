#!/usr/bin/env python3
"""独立标定页面：在网页上点 4 个角，自动计算单应矩阵并写回 perspective_params.json。

用法（车上）：
    python3 calib_page.py
浏览器打开 http://192.168.89.176:5009

流程：
  1. 页面显示原始相机画面（2× 放大，滚轮缩放，右键拖拽平移）
  2. 按顺序点击近左、近右、远左、远右 4 个角
  3. 填入板子参数（近边距离 d、缩放 S）
  4. 点击"计算并写入"→ 自动写 json + 重启主节点
"""

import json
import os
import subprocess
import time
import numpy as np
import cv2
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse
import threading

PORT = 5009
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config', 'perspective_params.json')
CAMERA_TOPIC = '/usb_cam/image_raw'
RESTART_CMD = [
    'bash', '-c',
    'pkill -9 -f straight_intersection_pass.py; sleep 1; '
    'cd /home/ucar/Xunxian_standalone/nodes && '
    'nohup python3 straight_intersection_pass.py > /home/ucar/Xunxian_standalone/nodes/straight_ipm.log 2>&1 &'
]

bridge = CvBridge()
latest_raw = None
lock = threading.Lock()


def image_cb(msg):
    global latest_raw
    try:
        cv2_img = bridge.imgmsg_to_cv2(msg, 'bgr8')
        with lock:
            latest_raw = cv2_img
    except Exception:
        pass


PAGE = r'''<!doctype html><meta charset="utf-8"><title>IPM 标定</title>
<style>
body{font:16px sans-serif;background:#0f172a;color:#f8fafc;margin:20px}
main{max-width:1200px;margin:auto}
section{background:#1e293b;padding:16px;margin:12px 0;border-radius:10px}
#calib-canvas{border:2px solid #334155;border-radius:8px;cursor:crosshair;display:block;margin:10px 0;background:#111}
.controls{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin:10px 0}
.controls input{width:90px;padding:6px;background:#0f172a;color:#f8fafc;border:1px solid #475569;border-radius:4px;font-size:14px}
.controls label{font-size:13px;white-space:nowrap}
button{padding:10px 20px;background:#1677ff;color:white;border:0;border-radius:6px;cursor:pointer;font-size:14px}
button:disabled{background:#475569;cursor:not-allowed}
button.danger{background:#dc2626}
.status{padding:10px;background:#0f172a;border-radius:6px;margin:10px 0;font-size:13px;min-height:40px}
.points{display:flex;gap:8px;flex-wrap:wrap;margin:8px 0}
.point-badge{background:#334155;padding:4px 10px;border-radius:4px;font-size:13px;display:flex;align-items:center;gap:6px}
.point-badge .dot{width:10px;height:10px;border-radius:50%;display:inline-block}
.zoom-info{font-size:12px;color:#94a3b8;margin-top:4px}
</style>
<main><h2>IPM 标定 — 点 4 个角</h2>
<section>
  <b>步骤</b>
  <ol>
    <li>把 0.3×1.0m 板子平放在车前地面，近边正对车，横向居中</li>
    <li>量近边到相机正下方地面的距离 d（米）</li>
    <li>在放大图上依次点击 4 个角：<b>近左 → 近右 → 远左 → 远右</b></li>
    <li>填入 d 和缩放 S，点击"计算并写入"</li>
  </ol>
  <div class="controls">
    <label>近边距离 d (m): <input id="inp_d" type="number" step="0.05" value="0.5"></label>
    <label>板宽 (m): <input id="inp_w" type="number" step="0.01" value="0.3" disabled></label>
    <label>板长 (m): <input id="inp_l" type="number" step="0.01" value="1.0" disabled></label>
    <label>缩放 S (px/m): <input id="inp_s" type="number" step="10" value="400"></label>
    <button id="btn-calc" disabled>计算并写入</button>
    <button id="btn-reset" class="danger">清除点</button>
  </div>
  <div class="status" id="status">等待点击…</div>
  <div class="points" id="points-list"></div>
  <div class="zoom-info">滚轮缩放（朝鼠标） · 右键拖拽平移 · 双击重置视图</div>
</section>
<section>
  <h3>原始相机（已放大，点击取点）</h3>
  <canvas id="calib-canvas" width="1280" height="720"></canvas>
</section>
</main>
<script>
const canvas = document.getElementById('calib-canvas');
const ctx = canvas.getContext('2d');
const statusEl = document.getElementById('status');
const pointsListEl = document.getElementById('points-list');
const btnCalc = document.getElementById('btn-calc');
const btnReset = document.getElementById('btn-reset');
const inpD = document.getElementById('inp_d');
const inpS = document.getElementById('inp_s');

let zoom = 2.0;
let panX = 0, panY = 0;
let dragging = false, dragStartX = 0, dragStartY = 0, dragPanX = 0, dragPanY = 0;
let img = new Image();
let imgNaturalW = 0, imgNaturalH = 0;
const pts = [];
const MAX_PTS = 4;
const LABELS = ['近左', '近右', '远左', '远右'];
const COLORS = ['#22d3ee','#a78bfa','#fbbf24','#f87171'];

function resizeCanvas() {
    canvas.width = Math.min(window.innerWidth - 40, 1280);
    canvas.height = Math.min(window.innerHeight - 200, 720);
    draw();
}
window.addEventListener('resize', resizeCanvas);
resizeCanvas();

function screenToImg(sx, sy) {
    return [(sx - panX) / zoom, (sy - panY) / zoom];
}

function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = '#111';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    if (imgNaturalW === 0) return;
    ctx.save();
    ctx.setTransform(zoom, 0, 0, zoom, panX, panY);
    ctx.drawImage(img, 0, 0);
    // 画点标记（image coords → screen via transform）
    for (let i = 0; i < pts.length; i++) {
        const [px, py] = pts[i];
        ctx.beginPath();
        ctx.arc(px, py, 8, 0, Math.PI * 2);
        ctx.fillStyle = COLORS[i];
        ctx.fill();
        ctx.strokeStyle = 'white';
        ctx.lineWidth = 2;
        ctx.stroke();
        ctx.fillStyle = 'white';
        ctx.font = 'bold 12px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(LABELS[i], px, py - 14);
    }
    // 画连线
    if (pts.length >= 2) {
        ctx.beginPath();
        ctx.moveTo(pts[0][0], pts[0][1]);
        for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
        ctx.strokeStyle = 'rgba(255,255,255,0.3)';
        ctx.lineWidth = 1;
        ctx.stroke();
    }
    ctx.restore();
}

// 定期拉取单帧 JPEG
async function fetchFrame() {
    try {
        const res = await fetch('/api/raw');
        const blob = await res.blob();
        const url = URL.createObjectURL(blob);
        img.onload = () => {
            imgNaturalW = img.naturalWidth;
            imgNaturalH = img.naturalHeight;
            URL.revokeObjectURL(url);
            draw();
        };
        img.src = url;
    } catch(e) {}
    setTimeout(fetchFrame, 100);
}
fetchFrame();

// 鼠标点击取点
canvas.addEventListener('mousedown', e => {
    if (e.button === 2) { dragging = true; dragStartX = e.clientX; dragStartY = e.clientY; dragPanX = panX; dragPanY = panY; e.preventDefault(); return; }
    if (e.button !== 0) return;
    const rect = canvas.getBoundingClientRect();
    const sx = e.clientX - rect.left;
    const sy = e.clientY - rect.top;
    const [ix, iy] = screenToImg(sx, sy);
    if (ix < 0 || iy < 0 || ix > imgNaturalW || iy > imgNaturalH) return;
    if (pts.length < MAX_PTS) {
        pts.push([ix, iy]);
        statusEl.textContent = '已选 ' + (pts.length) + '/' + MAX_PTS + ': ' + LABELS[pts.length - 1] + ' (' + Math.round(ix) + ', ' + Math.round(iy) + ')';
        if (pts.length === MAX_PTS) btnCalc.disabled = false;
        draw();
    }
});
canvas.addEventListener('mousemove', e => {
    if (dragging) {
        panX = dragPanX + (e.clientX - dragStartX);
        panY = dragPanY + (e.clientY - dragStartY);
        draw();
    }
});
canvas.addEventListener('mouseup', () => { dragging = false; });
canvas.addEventListener('contextmenu', e => e.preventDefault());
canvas.addEventListener('wheel', e => {
    e.preventDefault();
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    const delta = e.deltaY > 0 ? 0.9 : 1.1;
    const newZoom = Math.max(0.5, Math.min(8, zoom * delta));
    panX = mx - (mx - panX) * (newZoom / zoom);
    panY = my - (my - panY) * (newZoom / zoom);
    zoom = newZoom;
    draw();
}, {passive: false});
canvas.addEventListener('dblclick', () => { zoom = 2; panX = 0; panY = 0; draw(); });

// 触摸
canvas.addEventListener('touchstart', e => {
    if (e.touches.length === 1) {
        const t = e.touches[0];
        const rect = canvas.getBoundingClientRect();
        const sx = t.clientX - rect.left;
        const sy = t.clientY - rect.top;
        const [ix, iy] = screenToImg(sx, sy);
        if (ix >= 0 && iy >= 0 && ix <= imgNaturalW && iy <= imgNaturalH && pts.length < MAX_PTS) {
            pts.push([ix, iy]);
            statusEl.textContent = '已选 ' + (pts.length) + '/' + MAX_PTS + ': ' + LABELS[pts.length - 1] + ' (' + Math.round(ix) + ', ' + Math.round(iy) + ')';
            if (pts.length === MAX_PTS) btnCalc.disabled = false;
            draw();
        }
    } else if (e.touches.length === 2) {
        dragging = true;
        dragStartX = (e.touches[0].clientX + e.touches[1].clientX) / 2;
        dragStartY = (e.touches[0].clientY + e.touches[1].clientY) / 2;
        dragPanX = panX; dragPanY = panY;
    }
    e.preventDefault();
}, {passive: false});
canvas.addEventListener('touchmove', e => {
    if (dragging && e.touches.length === 2) {
        const mx = (e.touches[0].clientX + e.touches[1].clientX) / 2;
        const my = (e.touches[0].clientY + e.touches[1].clientY) / 2;
        panX = dragPanX + (mx - dragStartX);
        panY = dragPanY + (my - dragStartY);
        draw();
    }
    e.preventDefault();
}, {passive: false});
canvas.addEventListener('touchend', () => { dragging = false; });

function renderPoints() {
    pointsListEl.innerHTML = pts.map((p, i) =>
        '<span class="point-badge"><span class="dot" style="background:' + COLORS[i] + '"></span>' + LABELS[i] + ' (' + Math.round(p[0]) + ', ' + Math.round(p[1]) + ')</span>'
    ).join('');
}

btnReset.addEventListener('click', () => {
    pts.length = 0;
    btnCalc.disabled = true;
    statusEl.textContent = '已清除，重新点击 4 个角';
    renderPoints();
    draw();
});

btnCalc.addEventListener('click', async () => {
    if (pts.length !== MAX_PTS) return;
    const d = parseFloat(inpD.value);
    const s = parseFloat(inpS.value);
    const w = 0.3;
    const l = 1.0;
    if (isNaN(d) || isNaN(s) || d <= 0 || s <= 0) {
        statusEl.textContent = '请输入有效的 d 和 S';
        return;
    }
    const dst = [
        [300 - 0.15 * s, 600],
        [300 + 0.15 * s, 600],
        [300 - 0.15 * s, 600 - s],
        [300 + 0.15 * s, 600 - s],
    ];
    const src = pts.map(p => [p[0], p[1]]);
    statusEl.textContent = '计算中…';
    try {
        const resp = await fetch('/api/calib', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({src, dst, d, w, l, s, img_w: imgNaturalW, img_h: imgNaturalH}),
        });
        const data = await resp.json();
        if (data.ok) {
            statusEl.textContent = '标定完成！H 已写入 ' + data.path + '，正在重启节点…';
            btnCalc.disabled = true;
            try { await fetch('/api/restart', {method: 'POST'}); statusEl.textContent += ' 重启指令已发送。'; } catch(e) {}
        } else {
            statusEl.textContent = '错误: ' + (data.error || '未知');
        }
    } catch(e) {
        statusEl.textContent = '请求失败: ' + e;
    }
    renderPoints();
    draw();
});
</script>'''


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def reply_json(self, obj):
        data = json.dumps(obj, ensure_ascii=False).encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == '/':
            data = PAGE.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        elif path == '/api/raw':
            with lock:
                img = latest_raw.copy() if latest_raw is not None else None
            if img is not None:
                ret, jpeg = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ret:
                    self.send_response(200)
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', str(len(jpeg)))
                    self.end_headers()
                    self.wfile.write(jpeg.tobytes())
                    return
            self.send_error(404)
        else:
            self.send_error(404)

    def do_POST(self):
        path = urlparse(self.path).path
        length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(length) if length > 0 else b'{}'
        if path == '/api/calib':
            try:
                data = json.loads(body)
                src = np.float32(data['src'])
                dst = np.float32(data['dst'])
                if src.shape != (4, 2) or dst.shape != (4, 2):
                    self.reply_json({'ok': False, 'error': '需要 4 个点'})
                    return
                h, _ = cv2.findHomography(src, dst)
                if h is None:
                    self.reply_json({'ok': False, 'error': '单应矩阵计算失败'})
                    return
                # 图像尺寸取真实帧，避免被页面的板宽 w 覆盖
                cam_w = cam_h = 0
                with lock:
                    if latest_raw is not None:
                        cam_h, cam_w = latest_raw.shape[:2]
                if cam_w <= 0 or cam_h <= 0:
                    cam_w, cam_h = 640, 480
                # 配置文件允许不存在：从零生成，保证始终写入 config 目录
                if os.path.exists(CONFIG_PATH):
                    with open(CONFIG_PATH, 'r') as f:
                        cfg = json.load(f)
                else:
                    cfg = {}
                cfg['src_points'] = [[float(x) for x in p] for p in src.tolist()]
                cfg['dst_points'] = [[float(x) for x in p] for p in dst.tolist()]
                cfg['image_width'] = int(data.get('img_w', cam_w))
                cfg['image_height'] = int(data.get('img_h', cam_h))
                with open(CONFIG_PATH, 'w') as f:
                    json.dump(cfg, f, indent=2)
                self.reply_json({'ok': True, 'path': CONFIG_PATH, 'h': h.tolist()})
            except Exception as e:
                self.reply_json({'ok': False, 'error': str(e)})
        elif path == '/api/restart':
            try:
                subprocess.Popen(RESTART_CMD, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                self.reply_json({'ok': True, 'msg': '重启指令已发送'})
            except Exception as e:
                self.reply_json({'ok': False, 'error': str(e)})
        else:
            self.send_error(404)


def main():
    rospy.init_node('calib_page', anonymous=True)
    rospy.Subscriber(CAMERA_TOPIC, Image, image_cb, queue_size=1, buff_size=2**24)
    rospy.loginfo('标定页面已启动: http://0.0.0.0:%d', PORT)
    server = HTTPServer(('0.0.0.0', PORT), Handler)
    server.serve_forever()


if __name__ == '__main__':
    main()