#!/usr/bin/env python3
"""用当前 Node4 二值话题离线验证 /tmp 中候选算法，不启动候选节点。"""
import importlib.util

import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image


rospy.init_node("node4_candidate_diag", anonymous=True, disable_signals=True)
message = rospy.wait_for_message("/line_following/debug/mask", Image, timeout=3)
roi_mask = CvBridge().imgmsg_to_cv2(message, "mono8")
spec = importlib.util.spec_from_file_location(
    "node4_candidate", "/tmp/line_following_node4_candidate.py"
)
node4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(node4)

state = node4.SharedState()
centers, edges, bottom, invisible = node4.find_center_edge_line(
    roi_mask, 162, state
)
jumps = np.abs(np.diff([point[0] for point in centers])) if len(centers) > 1 else []
print("RESULT", {
    "roi_shape": roi_mask.shape,
    "tracks": state.lane_track_count,
    "pair_valid": state.lane_pair_valid,
    "slope_diff_bird": state.lane_pair_slope_diff,
    "half_width_bird": state.ipm_half_width,
    "width_samples": state.ipm_half_width_samples,
    "centers": len(centers),
    "edges": len(edges),
    "bottom": bottom,
    "invisible": invisible,
    "max_center_jump": float(np.max(jumps)) if len(jumps) else None,
    "p95_center_jump": float(np.percentile(jumps, 95)) if len(jumps) else None,
})