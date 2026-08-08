#!/usr/bin/env python3
"""Node6：去年 xunxian.py 的巡线部分 standalone 化。

保留去年算法的：HSV 二值化、ROI、逐行白色簇、单边虚拟边、中心点连续性、
calculate_slope 和 get_pid_params；移除里程计、雷达、任务终点、音频和直接
打开摄像头，只接入 ROS 图像，并增加配置校验、丢线停车和 Web 解锁。
"""
import json, math, os, socket, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse
import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG = os.path.join(ROOT, "config", "white_line.json")
PORT = 5006

PID_PARAMS = {
    "small_curve_invisible": (0.0264, 0.00075, 0.24),
    "medium_curve_invisible": (0.0288, 0.00075, 0.216),
    "extreme_curve_invisible": (0.0324, 0.00075, 0.108),
    "large_extreme_curve_invisible": (0.0372, 0.00075, 0.276),
    "small_straight": (0.018, 0.00075, 0.276),
    "small_curve": (0.024, 0.00075, 0.216),
    "medium_curve": (0.0264, 0.00075, 0.216),
    "large_curve": (0.0288, 0.00075, 0.168),
    "large_straight": (0.0144, 0.0075, 0.336),
}


class State:
    def __init__(self):
        self.lock = threading.RLock(); self.mode = "DISARMED"
        self.message = "Node6 启动，默认仅感知"; self.cfg = None; self.config_ok = False
        self.config_error = "未加载"; self.last_image = 0.; self.last_valid = 0.
        self.valid_streak = 0; self.lost_frames = 0; self.vision_valid = False
        self.centers = []; self.edges = []; self.bottom_edges = []; self.last_centers = []
        self.kanbujian = 0; self.error = 0.; self.last_error = 0.; self.last_vz = 0.; self.delta_vz = 0.
        self.last_control = 0.; self.speed = .12; self.image_width = 640; self.cmd = Twist()

    def status(self):
        return {"mode": self.mode, "message": self.message, "config_ok": self.config_ok,
                "config_error": self.config_error, "vision_valid": self.vision_valid,
                "valid_streak": self.valid_streak, "lost_frames": self.lost_frames,
                "center_count": len(self.centers), "kanbujian": bool(self.kanbujian),
                "error_deg": round(self.error, 2), "linear_x": round(self.cmd.linear.x, 3),
                "linear_y": round(self.cmd.linear.y, 3), "angular_z": round(self.cmd.angular.z, 3),
                "image_age_s": None if not self.last_image else round(time.time()-self.last_image, 2)}


state = State(); bridge = CvBridge(); cmd_pub = mask_pub = overlay_pub = None


def load_config():
    required = ("low_h", "high_h", "low_s", "high_s", "low_v", "high_v",
                "roi_top", "roi_bottom", "roi_left", "roi_right", "blur_ksize",
                "erode_iter", "erode_ksize", "dilate_iter", "dilate_ksize")
    try:
        with open(CONFIG, encoding="utf-8") as f: data = json.load(f)
        missing = [x for x in required if x not in data]
        if missing: raise ValueError("缺少字段: " + ",".join(missing))
        for lo, hi, top in (("low_h", "high_h", 179), ("low_s", "high_s", 255), ("low_v", "high_v", 255)):
            if not 0 <= int(data[lo]) <= int(data[hi]) <= top: raise ValueError("HSV范围非法")
        if not 0 <= float(data["roi_top"]) < float(data["roi_bottom"]) <= 1: raise ValueError("ROI纵向范围非法")
        if not 0 <= float(data["roi_left"]) < float(data["roi_right"]) <= 1: raise ValueError("ROI横向范围非法")
        with state.lock:
            state.cfg = data; state.config_ok = True; state.config_error = ""
            state.message = "配置加载成功，等待有效中线"
    except Exception as e:
        with state.lock:
            state.cfg = None; state.config_ok = False; state.config_error = str(e)
            state.mode = "FAULT"; state.message = "配置错误，禁止运行"
        rospy.logerr("Node6 配置失败: %s", e)


def binary_and_roi(frame, p):
    k = int(p.get("blur_ksize", 3)); k += k % 2 == 0 if k >= 3 else 0
    if k >= 3: frame_hsv = cv2.GaussianBlur(frame, (k, k), 0)
    else: frame_hsv = frame
    hsv = cv2.cvtColor(frame_hsv, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([p["low_h"],p["low_s"],p["low_v"]], np.uint8),
                       np.array([p["high_h"],p["high_s"],p["high_v"]], np.uint8))
    h, w = mask.shape; x1=int(w*p.get("roi_left",0)); x2=int(w*p.get("roi_right",1))
    y1=int(h*p["roi_top"]); y2=int(h*p["roi_bottom"]); roi=np.zeros_like(mask); roi[y1:y2,x1:x2]=mask[y1:y2,x1:x2]
    for name in ("erode", "dilate"):
        n=int(p.get(name+"_iter", 0)); size=max(1,int(p.get(name+"_ksize", 1))); size += size % 2 == 0
        if n: roi=getattr(cv2,name)(roi,cv2.getStructuringElement(cv2.MORPH_RECT,(size,size)),iterations=n)
    return roi, (x1,y1,x2,y2)


def find_center_edge_line(img):
    """去年 xunxian.py 的逐行中心点算法，去掉窗口显示。"""
    h,w=img.shape; centers=[]; edges=[]; bottom=[]; single=double=0; previous=None
    for y in range(h-1,-1,-4):
        xs=np.where(img[y]==255)[0]
        if not len(xs): continue
        groups=np.split(xs,np.where(np.diff(xs)>1)[0]+1); means=[np.mean(g) for g in groups]
        selected=[]
        if len(means)==1:
            single+=1; edge=int(means[0]); virtual=(0 if centers[-1][0]<centers[-2][0] else w-1) if len(centers)>1 else (w-1 if edge<w//2 else 0); selected=[edge,virtual]
        else:
            double+=1
            if len(means)==2:
                selected=[int(x) for x in means]
                if abs(selected[0]-selected[1])<w/3:
                    keep=min(selected,key=lambda x:abs(x-w/2)); selected=[keep,w-1 if keep<w/2 else 0]; single+=1; double-=1
            else:
                left=[x for x in means if x<w/2]; right=[x for x in means if x>=w/2]
                selected=[int(max(left)) if left else 0,int(min(right)) if right else w-1]
        center=int(sum(selected)/2)
        if previous is not None and abs(center-previous)>=w/8: continue
        previous=center; centers.append((center,y)); edges.extend((int(x),y) for x in selected)
        if not bottom: bottom=list(map(int,selected))
    ratio=single/float(single+double) if single+double else 1.
    return centers,edges,bottom,int(ratio>.9),ratio


def calculate_slope(points, fallback):
    if len(points)<3: points=fallback if len(fallback)>=3 else [(320,150),(320,140),(320,130)]
    first,last=points[0],points[-1]; middle=points[min(len(points)-1,int(round(len(points)/3.5)))]
    angle=np.degrees(np.arctan2(middle[1]-first[1],middle[0]-first[0]))
    error=-90+angle if angle>0 else 90+angle
    return float(error)


def get_pid_params(error, invisible):
    a=abs(error)
    if invisible:
        key="small_curve_invisible" if 33.5<a<=51 else "medium_curve_invisible" if 51<a<=62 else "extreme_curve_invisible" if 62<a<=64 else "large_extreme_curve_invisible" if a>64 else "large_straight"
    else:
        key="small_straight" if 30<a<=34 else "small_curve" if 34<a<=55 else "medium_curve" if 55<a<=60 else "large_curve" if a>60 else "large_straight"
    return PID_PARAMS[key]


def image_cb(msg):
    try:
        frame=cv2.flip(bridge.imgmsg_to_cv2(msg,"bgr8"),1)
        with state.lock: p=dict(state.cfg) if state.config_ok else None
        if p is None: return
        mask,roi=binary_and_roi(frame,p); points,edges,bottom,invisible,ratio=find_center_edge_line(mask[roi[1]:roi[3],roi[0]:roi[2]])
        points=[(x+roi[0],y+roi[1]) for x,y in points]; edges=[(x+roi[0],y+roi[1]) for x,y in edges]; bottom=[x+roi[0] for x in bottom]
        valid=len(points)>=6; now=time.time()
        with state.lock:
            state.last_image=now; state.image_width=frame.shape[1]; state.centers=points; state.edges=edges; state.bottom_edges=bottom; state.kanbujian=invisible; state.vision_valid=valid
            if valid: state.error=calculate_slope(points,state.last_centers); state.last_centers=points; state.last_valid=now; state.valid_streak+=1; state.lost_frames=0
            else: state.valid_streak=0; state.lost_frames+=1
            mode=state.mode
        overlay=frame.copy()
        for x,y in edges: cv2.circle(overlay,(x,y),2,(0,0,255),-1)
        if points: cv2.polylines(overlay,[np.asarray(points,np.int32)],False,(0,255,0),3)
        cv2.putText(overlay,"%s valid=%s err=%.1f single=%.2f"%(mode,valid,state.error,ratio),(10,25),0,.55,(0,255,255),2)
        mask_pub.publish(bridge.cv2_to_imgmsg(mask,"mono8")); overlay_pub.publish(bridge.cv2_to_imgmsg(overlay,"bgr8"))
    except Exception as e: rospy.logerr_throttle(2,"Node6 图像异常: %s",e)


def stop():
    state.cmd=Twist(); state.last_error=state.last_vz=state.delta_vz=0.; state.last_control=0.
    if cmd_pub: cmd_pub.publish(Twist())


def control(_event):
    with state.lock:
        if state.mode!="RUNNING": return
        now=time.time()
        if now-state.last_image>.6: state.mode="FAULT"; state.message="图像超时停车"; stop(); return
        if not state.vision_valid:
            if now-state.last_valid<.25 and state.lost_frames<=5:
                c=Twist(); c.linear.x=.04; c.angular.z=max(-.25,min(.25,state.last_vz)); state.cmd=c; cmd_pub.publish(c); return
            state.mode="FAULT"; state.message="丢失中线停车"; stop(); return
        kp,ky,kd=get_pid_params(state.error,state.kanbujian)
        # 保留去年控制律：角速度比例项减去上一周期角速度增量。
        z=kp*state.error-kd*state.delta_vz; z=max(-.75,min(.75,z))
        c=Twist(); c.linear.x=max(.055,state.speed*(1-min(abs(state.error)/100.,.55))); c.angular.z=z
        if not state.kanbujian and len(state.bottom_edges)>=2: c.linear.y=max(-.025,min(.025,-ky*((sum(state.bottom_edges[:2])/2)-state.image_width/2)*.0005))
        state.delta_vz=z-state.last_vz; state.cmd=c; state.last_error=state.error; state.last_vz=z; state.last_control=now; cmd_pub.publish(c)


PAGE="""<meta charset='utf-8'><h2>巡线 Node6（去年算法）</h2><button onclick=go('/start')>开始</button><button onclick=go('/stop')>停车</button><button onclick=go('/reset')>复位</button><pre id=s></pre><script>async function go(x){let d=await(await fetch('/api'+x,{method:'POST'})).json();if(!d.ok)alert(d.error)}setInterval(async()=>s.textContent=JSON.stringify(await(await fetch('/api/status')).json(),null,2),400)</script>"""
class Handler(BaseHTTPRequestHandler):
    def reply(self,x):
        b=json.dumps(x,ensure_ascii=False).encode(); self.send_response(200); self.send_header("Content-Type","application/json"); self.send_header("Content-Length",str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        if urlparse(self.path).path=="/": self.send_response(200); self.end_headers(); self.wfile.write(PAGE.encode())
        elif urlparse(self.path).path=="/api/status":
            with state.lock: self.reply(state.status())
        else: self.send_error(404)
    def do_POST(self):
        with state.lock:
            p=urlparse(self.path).path
            if p=="/api/start":
                ok=state.config_ok and state.vision_valid and state.valid_streak>=5 and time.time()-state.last_image<.6
                if ok: state.mode="RUNNING"; state.message="去年算法巡线运行中"; state.last_error=state.last_vz=0.; self.reply({"ok":True})
                else: self.reply({"ok":False,"error":"配置/图像/有效中线未就绪"})
            elif p=="/api/stop": state.mode="STOPPED"; state.message="用户停车"; stop(); self.reply({"ok":True})
            elif p=="/api/reset": state.mode="DISARMED"; state.message="仅感知"; stop(); self.reply({"ok":True})
            else: self.send_error(404)
    def log_message(self,*_): pass


class ReusableHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True

    def server_bind(self):
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        super().server_bind()

def shutdown():
    with state.lock: stop()
def main():
    global cmd_pub,mask_pub,overlay_pub
    rospy.init_node("line_following_node6",anonymous=False); cmd_pub=rospy.Publisher("/cmd_vel",Twist,queue_size=1); mask_pub=rospy.Publisher("/line_following_node6/debug/mask",Image,queue_size=1); overlay_pub=rospy.Publisher("/line_following_node6/debug/overlay",Image,queue_size=1); load_config(); rospy.Subscriber("/usb_cam/image_raw",Image,image_cb,queue_size=1,buff_size=2**24); rospy.Timer(rospy.Duration(.04),control); rospy.on_shutdown(shutdown)
    try:
        server=ReusableHTTPServer(("0.0.0.0",PORT),Handler)
    except OSError as exc:
        stop()
        rospy.logfatal("Node6 Web 端口 %d 无法绑定（可能已有 Node6 在运行）: %s",PORT,exc)
        raise SystemExit(2)
    rospy.loginfo("Node6 已启动，默认仅感知: http://0.0.0.0:%d",PORT)
    try: server.serve_forever()
    finally: server.server_close(); stop()
if __name__=="__main__": main()