import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

bridge = CvBridge()

def cb(msg):
    print("Received frame: encoding=", msg.encoding, "size=", msg.width, msg.height)
    frame = bridge.imgmsg_to_cv2(msg, "passthrough")
    print("Frame shape:", frame.shape, "dtype:", frame.dtype)
    if msg.encoding.lower() == "rgb8":
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    p = {"low_h": 0, "high_h": 120, "low_s": 0, "high_s": 75, "low_v": 70, "high_v": 255, "roi_top": 0.58, "roi_bottom": 1.0}
    h, w = frame.shape[:2]
    y1, y2 = int(h * p["roi_top"]), int(h * p["roi_bottom"])
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([p["low_h"], p["low_s"], p["low_v"]]), np.array([p["high_h"], p["high_s"], p["high_v"]]))
    print("Mask non-zero:", np.count_nonzero(mask))
    rospy.signal_shutdown("done")

rospy.init_node("diag_node")
rospy.Subscriber("/usb_cam/image_raw", Image, cb)
rospy.spin()
