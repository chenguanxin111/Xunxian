#!/usr/bin/env python3
"""Universal Visual Tuner Node for HSV, ROI, and Morphology.
Supports presets (white lane, yellow line, traffic lights) and custom JSON files.
Serves HTTP GUI on port 5000 and publishes debug mask & overlay ROS Image topics.
"""
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

BASE_DIR = '/home/ucar/Xunxian_standalone'
CONFIG_DIR = os.path.join(BASE_DIR, 'config')

DEFAULT_PARAMS = {
    'low_h': 0, 'high_h': 179,
    'low_s': 0, 'high_s': 255,
    'low_v': 70, 'high_v': 255,
    'roi_top': 0.45, 'roi_bottom': 1.0,
    'roi_left': 0.0, 'roi_right': 1.0,
    'blur_ksize': 3,
    'erode_iter': 1, 'erode_ksize': 3,
    'dilate_iter': 2, 'dilate_ksize': 3
}

PRESET_PROFILES = {
    'white_lane': {
        'low_h': 42, 'high_h': 179,
        'low_s': 5, 'high_s': 71,
        'low_v': 116, 'high_v': 255,
        'roi_top': 0.45, 'roi_bottom': 1.0,
        'roi_left': 0.0, 'roi_right': 1.0,
        'blur_ksize': 4, 'erode_iter': 0, 'erode_ksize': 3, 'dilate_iter': 2, 'dilate_ksize': 3
    },
    'yellow_line': {
        'low_h': 0, 'high_h': 35,
        'low_s': 0, 'high_s': 255,
        'low_v': 70, 'high_v': 255,
        'roi_top': 0.50, 'roi_bottom': 1.0,
        'roi_left': 0.0, 'roi_right': 1.0,
        'blur_ksize': 3, 'erode_iter': 1, 'erode_ksize': 3, 'dilate_iter': 2, 'dilate_ksize': 3
    },
    'traffic_light_red': {
        'low_h': 0, 'high_h': 10,
        'low_s': 100, 'high_s': 255,
        'low_v': 100, 'high_v': 255,
        'roi_top': 0.10, 'roi_bottom': 0.60,
        'roi_left': 0.20, 'roi_right': 0.80,
        'blur_ksize': 3, 'erode_iter': 1, 'erode_ksize': 3, 'dilate_iter': 1, 'dilate_ksize': 3
    },
    'traffic_light_green': {
        'low_h': 35, 'high_h': 85,
        'low_s': 100, 'high_s': 255,
        'low_v': 100, 'high_v': 255,
        'roi_top': 0.10, 'roi_bottom': 0.60,
        'roi_left': 0.20, 'roi_right': 0.80,
        'blur_ksize': 3, 'erode_iter': 1, 'erode_ksize': 3, 'dilate_iter': 1, 'dilate_ksize': 3
    }
}


class State:
    def __init__(self):
        self.lock = threading.Lock()
        self.params = dict(DEFAULT_PARAMS)
        self.info = {
            'status': '初始化完成',
            'mask_nonzero': 0,
            'image_width': 0,
            'image_height': 0
        }


state = State()
bridge = CvBridge()
mask_pub = None
overlay_pub = None


def image_cb(msg):
    try:
        frame = bridge.imgmsg_to_cv2(msg, 'passthrough')
        if msg.encoding.lower() == 'rgb8':
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        elif msg.encoding.lower() != 'bgr8':
            frame = bridge.imgmsg_to_cv2(msg, 'bgr8')

        # 水平镜像翻转
        frame = cv2.flip(frame, 1)

        with state.lock:
            p = dict(state.params)

        h, w = frame.shape[:2]

        # 1. 高斯滤波 (Gaussian Blur)
        blur_k = int(p.get('blur_ksize', 0))
        if blur_k >= 3:
            if blur_k % 2 == 0:
                blur_k += 1
            frame_blur = cv2.GaussianBlur(frame, (blur_k, blur_k), 0)
        else:
            frame_blur = frame

        # 2. HSV 颜色阈值分割
        hsv = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(
            hsv,
            np.array([p['low_h'], p['low_s'], p['low_v']]),
            np.array([p['high_h'], p['high_s'], p['high_v']])
        )

        # 3. 4 向 ROI 区域切边
        y1 = int(h * min(p.get('roi_top', 0.0), p.get('roi_bottom', 1.0)))
        y2 = int(h * max(p.get('roi_top', 0.0), p.get('roi_bottom', 1.0)))
        x1 = int(w * min(p.get('roi_left', 0.0), p.get('roi_right', 1.0)))
        x2 = int(w * max(p.get('roi_left', 0.0), p.get('roi_right', 1.0)))

        y1 = max(0, min(h - 1, y1))
        y2 = max(y1 + 1, min(h, y2))
        x1 = max(0, min(w - 1, x1))
        x2 = max(x1 + 1, min(w, x2))

        roi_mask = np.zeros_like(mask)
        roi_mask[y1:y2, x1:x2] = mask[y1:y2, x1:x2]

        # 4. 腐蚀操作 (Erode)
        erode_iter = int(p.get('erode_iter', 0))
        erode_k = int(p.get('erode_ksize', 3))
        if erode_iter > 0 and erode_k > 0:
            if erode_k % 2 == 0:
                erode_k += 1
            e_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (erode_k, erode_k))
            roi_mask = cv2.erode(roi_mask, e_kernel, iterations=erode_iter)

        # 5. 膨胀操作 (Dilate)
        dilate_iter = int(p.get('dilate_iter', 0))
        dilate_k = int(p.get('dilate_ksize', 3))
        if dilate_iter > 0 and dilate_k > 0:
            if dilate_k % 2 == 0:
                dilate_k += 1
            d_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dilate_k, dilate_k))
            roi_mask = cv2.dilate(roi_mask, d_kernel, iterations=dilate_iter)

        # 6. 轮廓提取与 Overlay 画框
        res = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = res[0] if len(res) == 2 else res[1]
        overlay = frame.copy()

        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 120, 0), 2)
        cv2.putText(overlay, 'ROI Area', (x1 + 5, max(20, y1 + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 120, 0), 2)

        if contours:
            c = max(contours, key=cv2.contourArea)
            if cv2.contourArea(c) > 20:
                cv2.drawContours(overlay, [c], -1, (0, 0, 255), 2)
                cx, cy, cw, ch = cv2.boundingRect(c)
                cv2.rectangle(overlay, (cx, cy), (cx + cw, cy + ch), (0, 255, 0), 2)
                cv2.putText(overlay, 'Target Found', (cx, max(20, cy - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 0), 2)

        cv2.putText(overlay, 'HSV [%d,%d,%d] - [%d,%d,%d]' %
                    (p['low_h'], p['low_s'], p['low_v'], p['high_h'], p['high_s'], p['high_v']),
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 255), 2)

        mask_pub.publish(bridge.cv2_to_imgmsg(roi_mask, 'mono8'))
        overlay_pub.publish(bridge.cv2_to_imgmsg(overlay, 'bgr8'))

        with state.lock:
            state.info = {
                'status': '运行正常',
                'mask_nonzero': int(cv2.countNonZero(roi_mask)),
                'image_width': w,
                'image_height': h,
                'contour_count': len(contours)
            }
    except Exception as e:
        rospy.logwarn_throttle(2, f"image processing error: {e}")


PAGE = '''<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>通用视觉在线调参控制台</title>
<style>
body { font-family: 'Segoe UI', system-ui, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 20px; }
main { max-width: 1200px; margin: 0 auto; }
h2 { color: #38bdf8; border-bottom: 2px solid #334155; padding-bottom: 10px; }
section { background: #1e293b; border-radius: 12px; padding: 20px; margin-bottom: 20px; box-shadow: 0 4px 15px rgba(0,0,0,0.3); }
h3 { margin-top: 0; color: #f1f5f9; font-size: 16px; border-left: 4px solid #38bdf8; padding-left: 8px; }
.grid-ctrl { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 15px; }
.ctrl-item { background: #0f172a; padding: 12px; border-radius: 8px; border: 1px solid #334155; }
.ctrl-item label { display: block; font-size: 13px; color: #94a3b8; margin-bottom: 5px; font-weight: 600; }
.ctrl-item input[type=range] { width: 70%; accent-color: #38bdf8; cursor: pointer; }
.ctrl-item b { float: right; color: #38bdf8; font-size: 15px; }
.btn-group { display: flex; gap: 10px; flex-wrap: wrap; align-items: center; }
button { padding: 8px 16px; border: none; border-radius: 6px; font-weight: 600; cursor: pointer; transition: 0.2s; font-size: 13px; }
button.btn-preset { background: #334155; color: #38bdf8; border: 1px solid #0284c7; }
button.btn-preset:hover { background: #0284c7; color: white; }
button.btn-save { background: #16a34a; color: white; }
button.btn-save:hover { background: #15803d; }
button.btn-load { background: #2563eb; color: white; }
button.btn-load:hover { background: #1d4ed8; }
button.btn-reset { background: #dc2626; color: white; }
button.btn-reset:hover { background: #b91c1c; }
input.txt-file { padding: 8px 12px; border-radius: 6px; border: 1px solid #334155; background: #0f172a; color: white; font-size: 14px; width: 180px; }
select.sel-file { padding: 8px 12px; border-radius: 6px; border: 1px solid #334155; background: #0f172a; color: white; font-size: 14px; }
#status { font-weight: bold; color: #4ade80; margin-left: 10px; }
.img-container { display: flex; flex-wrap: wrap; gap: 15px; justify-content: space-between; }
.img-card { flex: 1 1 30%; min-width: 300px; background: #0f172a; padding: 10px; border-radius: 8px; box-shadow: inset 0 0 10px rgba(0,0,0,0.5); }
.img-card h4 { margin: 0 0 8px 0; color: #cbd5e1; font-size: 14px; text-align: center; }
.img-card img { width: 100%; height: auto; display: block; border-radius: 4px; background: #000; min-height: 200px; }
pre { background: #0f172a; padding: 12px; border-radius: 6px; color: #38bdf8; font-size: 13px; overflow-x: auto; }
</style>
</head>
<body>
<main>
<h2>通用视觉 HSV 阈值 / ROI 选区 / 形态学在线调参控制台</h2>

<section>
<h3>预设套件与自定义参数文件管理 (Presets & File Manager)</h3>
<div class="btn-group">
  <span style="color:#cbd5e1;font-weight:600;">一键套件预设:</span>
  <button class="btn-preset" onclick="loadPreset('white_lane')">白色车道线</button>
  <button class="btn-preset" onclick="loadPreset('yellow_line')">黄色停止线</button>
  <button class="btn-preset" onclick="loadPreset('traffic_light_red')">红灯识别</button>
  <button class="btn-preset" onclick="loadPreset('traffic_light_green')">绿灯识别</button>
</div>
<div class="btn-group" style="margin-top: 15px;">
  <span style="color:#cbd5e1;font-weight:600;">已保存文件:</span>
  <select id="file_list" class="sel-file"></select>
  <button class="btn-load" onclick="loadSelectedFile()">加载选中文件</button>
  <span style="color:#cbd5e1;font-weight:600;margin-left:15px;">自定义保存文件名:</span>
  <input id="save_filename" type="text" class="txt-file" value="white_lane.json" placeholder="例如: white_lane.json">
  <button class="btn-save" onclick="saveCustomFile()">保存到自定义文件</button>
  <button class="btn-reset" onclick="reset()">重置默认</button>
  <span id="status"></span>
</div>
</section>

<section>
<h3>1. HSV 颜色阈值调节 (HSV Color Thresholds)</h3>
<div id="hsv_controls" class="grid-ctrl"></div>
</section>

<section>
<h3>2. ROI 兴趣区域框定 (上下左右 4 向切边)</h3>
<div id="roi_controls" class="grid-ctrl"></div>
</section>

<section>
<h3>3. 高斯滤波 & 腐蚀膨胀 (形态学预处理)</h3>
<div id="morph_controls" class="grid-ctrl"></div>
</section>

<section>
<h3>实时视频流与检测状态</h3>
<div class="img-container">
  <div class="img-card"><h4>Camera Raw (/usb_cam/image_raw)</h4><img id="img_raw"></div>
  <div class="img-card"><h4>Debug Overlay (/xunxian/debug/overlay)</h4><img id="img_overlay"></div>
  <div class="img-card"><h4>HSV Mask (/xunxian/debug/hsv_mask)</h4><img id="img_mask"></div>
</div>
<h4 style="margin-top:15px;color:#94a3b8;">检测统计信息 (Live Info)</h4>
<pre id="info">加载中...</pre>
</section>

</main>

<script>
let host = window.location.hostname || '192.168.89.176';
document.getElementById('img_raw').src = 'http://' + host + ':8080/stream?topic=/usb_cam/image_raw';
document.getElementById('img_overlay').src = 'http://' + host + ':8080/stream?topic=/xunxian/debug/overlay';
document.getElementById('img_mask').src = 'http://' + host + ':8080/stream?topic=/xunxian/debug/hsv_mask';

let hsv_names = ['low_h','high_h','low_s','high_s','low_v','high_v'];
let roi_names = ['roi_top','roi_bottom','roi_left','roi_right'];
let morph_names = ['blur_ksize','erode_iter','erode_ksize','dilate_iter','dilate_ksize'];

let ranges = {
  low_h:[0,179,1], high_h:[0,179,1],
  low_s:[0,255,1], high_s:[0,255,1],
  low_v:[0,255,1], high_v:[0,255,1],
  roi_top:[0,1,0.01], roi_bottom:[0,1,0.01],
  roi_left:[0,1,0.01], roi_right:[0,1,0.01],
  blur_ksize:[0,15,2],
  erode_iter:[0,10,1], erode_ksize:[1,15,2],
  dilate_iter:[0,10,1], dilate_ksize:[1,15,2]
};

let labels = {
  low_h: 'Low H (色调下限)', high_h: 'High H (色调上限)',
  low_s: 'Low S (饱和度下限)', high_s: 'High S (饱和度上限)',
  low_v: 'Low V (明度下限)', high_v: 'High V (明度上限)',
  roi_top: 'ROI Top (上边界)', roi_bottom: 'ROI Bottom (下边界)',
  roi_left: 'ROI Left (左边界)', roi_right: 'ROI Right (右边界)',
  blur_ksize: '高斯滤波核大小',
  erode_iter: '腐蚀迭代次数', erode_ksize: '腐蚀核大小',
  dilate_iter: '膨胀迭代次数', dilate_ksize: '膨胀核大小'
};

async function updateFileList(){
  try {
    let res = await (await fetch('/api/files')).json();
    let sel = document.querySelector('#file_list');
    sel.innerHTML = res.files.map(f => `<option value="${f}">${f}</option>`).join('');
  } catch(e){}
}

async function load(){
  try {
    let p = await (await fetch('/api/params')).json();
    
    let hc = document.querySelector('#hsv_controls');
    hc.innerHTML = hsv_names.map(n => `
      <div class="ctrl-item">
        <label>${labels[n]}</label>
        <input id="${n}" type="range" min="${ranges[n][0]}" max="${ranges[n][1]}" step="${ranges[n][2]}" value="${p[n]}" oninput="change('${n}',this.value)">
        <b id="v_${n}">${p[n]}</b>
      </div>
    `).join('');

    let rc = document.querySelector('#roi_controls');
    rc.innerHTML = roi_names.map(n => `
      <div class="ctrl-item">
        <label>${labels[n]}</label>
        <input id="${n}" type="range" min="${ranges[n][0]}" max="${ranges[n][1]}" step="${ranges[n][2]}" value="${p[n] !== undefined ? p[n] : ranges[n][0]}" oninput="change('${n}',this.value)">
        <b id="v_${n}">${p[n] !== undefined ? p[n] : ranges[n][0]}</b>
      </div>
    `).join('');

    let mc = document.querySelector('#morph_controls');
    mc.innerHTML = morph_names.map(n => `
      <div class="ctrl-item">
        <label>${labels[n]}</label>
        <input id="${n}" type="range" min="${ranges[n][0]}" max="${ranges[n][1]}" step="${ranges[n][2]}" value="${p[n] !== undefined ? p[n] : ranges[n][0]}" oninput="change('${n}',this.value)">
        <b id="v_${n}">${p[n] !== undefined ? p[n] : ranges[n][0]}</b>
      </div>
    `).join('');

    await updateFileList();
  } catch(e) { console.error(e); }
}

async function change(n,v){
  document.querySelector('#v_'+n).textContent = v;
  await fetch('/api/params', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({[n]:Number(v)})
  });
}

async function loadPreset(key){
  let res = await (await fetch('/api/load', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({preset:key})
  })).json();
  load();
  showStatus('✓ 已载入预设套件: ' + key);
}

async function loadSelectedFile(){
  let fn = document.querySelector('#file_list').value;
  if(!fn) return;
  await fetch('/api/load', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({filename:fn})
  });
  load();
  document.querySelector('#save_filename').value = fn;
  showStatus('✓ 已载入配置文件: ' + fn);
}

async function saveCustomFile(){
  let fn = document.querySelector('#save_filename').value.trim();
  if(!fn) { alert('请输入有效的保存文件名!'); return; }
  if(!fn.endsWith('.json')) fn += '.json';
  await fetch('/api/save', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({filename:fn})
  });
  await updateFileList();
  document.querySelector('#file_list').value = fn;
  showStatus('✓ 参数已保存至: ' + fn);
}

async function reset(){
  await fetch('/api/reset', {method:'POST'});
  load();
  showStatus('✓ 参数已恢复默认');
}

function showStatus(msg){
  let st = document.querySelector('#status');
  st.textContent = msg;
  setTimeout(() => st.textContent = '', 3500);
}

setInterval(async () => {
  try {
    let info = await (await fetch('/api/info')).json();
    document.querySelector('#info').textContent = JSON.stringify(info, null, 2);
  } catch(e){}
}, 1000);

load();
</script>
</body>
</html>
'''


class Handler(BaseHTTPRequestHandler):
    def reply(self, obj):
        data = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == '/':
            data = PAGE.encode()
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        elif path == '/api/params':
            with state.lock:
                self.reply(state.params)
        elif path == '/api/info':
            with state.lock:
                self.reply(state.info)
        elif path == '/api/files':
            os.makedirs(CONFIG_DIR, exist_ok=True)
            files = [f for f in os.listdir(CONFIG_DIR) if f.endswith('.json')]
            files.sort()
            self.reply({'files': files, 'presets': list(PRESET_PROFILES.keys())})
        else:
            self.send_error(404)

    def do_POST(self):
        path = urlparse(self.path).path
        if path == '/api/params':
            n = int(self.headers.get('Content-Length', 0))
            data = json.loads(self.rfile.read(n))
            with state.lock:
                for k, v in data.items():
                    if k in state.params or k in DEFAULT_PARAMS:
                        if k in ('roi_top', 'roi_bottom', 'roi_left', 'roi_right'):
                            state.params[k] = max(0.0, min(1.0, float(v)))
                        elif k in ('low_h', 'high_h'):
                            state.params[k] = max(0, min(179, int(v)))
                        elif k in ('low_s', 'high_s', 'low_v', 'high_v'):
                            state.params[k] = max(0, min(255, int(v)))
                        else:
                            state.params[k] = max(0, min(15, int(v)))
            self.reply(state.params)

        elif path == '/api/save':
            n = int(self.headers.get('Content-Length', 0))
            data = json.loads(self.rfile.read(n)) if n > 0 else {}
            filename = data.get('filename', 'white_lane_right.json')
            if not filename.endswith('.json'):
                filename += '.json'
            os.makedirs(CONFIG_DIR, exist_ok=True)
            target_path = os.path.join(CONFIG_DIR, os.path.basename(filename))
            with state.lock:
                with open(target_path, 'w') as f:
                    json.dump(state.params, f, indent=2)
            rospy.loginfo(f"参数已存入: {target_path}")
            self.reply({'ok': True, 'file': filename})

        elif path == '/api/load':
            n = int(self.headers.get('Content-Length', 0))
            data = json.loads(self.rfile.read(n)) if n > 0 else {}
            if 'preset' in data and data['preset'] in PRESET_PROFILES:
                with state.lock:
                    state.params.update(PRESET_PROFILES[data['preset']])
                rospy.loginfo(f"已加载预设套件: {data['preset']}")
                self.reply({'ok': True, 'params': state.params})
            elif 'filename' in data:
                filename = os.path.basename(data['filename'])
                target_path = os.path.join(CONFIG_DIR, filename)
                if os.path.exists(target_path):
                    with state.lock:
                        with open(target_path, 'r') as f:
                            saved = json.load(f)
                            state.params.update(saved)
                    rospy.loginfo(f"已从文件读取参数: {target_path}")
                    self.reply({'ok': True, 'params': state.params})
                else:
                    self.reply({'ok': False, 'error': 'File not found'})
            else:
                self.reply({'ok': False, 'error': 'Invalid request'})

        elif path == '/api/reset':
            with state.lock:
                state.params.update(DEFAULT_PARAMS)
            self.reply(state.params)
        else:
            self.send_error(404)

    def log_message(self, *args):
        pass


class ReusableHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True

    def server_bind(self):
        try:
            super().server_bind()
        except OSError as e:
            if getattr(e, 'errno', None) == 98:
                rospy.logwarn("检测到 5000 端口被占用，正在清理旧进程...")
                os.system("fuser -k 5000/tcp 2>/dev/null || pkill -9 -f tuner_node.py 2>/dev/null")
                import time
                time.sleep(1)
                super().server_bind()
            else:
                raise


if __name__ == '__main__':
    rospy.init_node('xunxian_browser_tuner', anonymous=False)
    mask_pub = rospy.Publisher('/xunxian/debug/hsv_mask', Image, queue_size=1)
    overlay_pub = rospy.Publisher('/xunxian/debug/overlay', Image, queue_size=1)
    rospy.Subscriber('/usb_cam/image_raw', Image, image_cb, queue_size=1, buff_size=2**24)
    rospy.loginfo("通用视觉调参 HTTP 服务器已启动，访问端口: 5000")
    ReusableHTTPServer(('0.0.0.0', 5000), Handler).serve_forever()
