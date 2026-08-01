#!/usr/bin/env python3
import importlib.util

import cv2
import numpy as np


spec = importlib.util.spec_from_file_location(
    "node4_candidate", "/tmp/line_following_node4_candidate.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

height, width = 198, 640
y_offset = 162
mask = np.zeros((height, width), np.uint8)
# 先在 IPM 中生成平行、定宽车道，再逆投影到相机图。
for lane_x in (185.0, 455.0):
    camera_points = []
    for ipm_y in np.linspace(0, 360, 361):
        camera_x, camera_y = module.transform_point_ipm(
            (lane_x + 0.00035 * (ipm_y - 180.0) ** 2, ipm_y),
            module.IPM_INV_MATRIX,
        )
        roi_y = int(round(camera_y - y_offset))
        if 0 <= roi_y < height and 0 <= camera_x < width:
            camera_points.append((int(round(camera_x)), roi_y))
    cv2.polylines(mask, [np.asarray(camera_points, np.int32)], False, 255, 7)

for x, y in [(310, 180), (90, 140), (550, 110), (260, 70), (390, 40), (35, 25)]:
    cv2.circle(mask, (x, y), 2, 255, -1)

state = module.SharedState()
clean = module.remove_small_white_components(mask)
rows = []
for y in range(height - 1, -1, -4):
    xs = np.where(clean[y] == 255)[0]
    if len(xs):
        groups = np.split(xs, np.where(np.diff(xs) > 1)[0] + 1)
        means = [np.mean(group) for group in groups if len(group) >= 3]
        if means:
            rows.append((y, sorted(means)))
debug_tracks = module.build_lane_tracks(rows, 4, width)
print("TRACKS", [len(track["points"]) for track in debug_tracks])
print("FITS", [module.fit_track_in_ipm(track, y_offset) for track in debug_tracks])
if len(debug_tracks) >= 2:
    common = sorted(set(debug_tracks[0]["by_y"]) & set(debug_tracks[1]["by_y"]))
    widths = []
    for y in common:
        a = module.transform_point_ipm((debug_tracks[0]["by_y"][y], y + y_offset), module.IPM_MATRIX)
        b = module.transform_point_ipm((debug_tracks[1]["by_y"][y], y + y_offset), module.IPM_MATRIX)
        widths.append(abs(b[0] - a[0]))
    print("WIDTHS", len(widths), round(float(np.median(widths)), 2), round(float(np.median(np.abs(np.asarray(widths)-np.median(widths)))), 2))
centers, edges, _, _ = module.find_center_edge_line(mask, y_offset, state)
print(
    "DOUBLE",
    len(centers), len(edges), state.lane_track_count, state.lane_pair_valid,
    round(state.lane_pair_slope_diff or 0.0, 3),
    round(state.ipm_half_width, 2), state.ipm_half_width_samples,
)
assert state.lane_pair_valid
assert len(centers) > 30
assert len(edges) <= 2 * len(centers) + 4

mixed_mask = mask.copy()
mixed_mask[:100, 320:] = 0
mixed_state = module.SharedState()
mixed_centers, _, _, _ = module.find_center_edge_line(
    mixed_mask, y_offset, mixed_state
)
jumps = np.abs(np.diff([point[0] for point in mixed_centers]))
print(
    "MIXED",
    len(mixed_centers), mixed_state.lane_pair_valid,
    round(float(np.max(jumps)), 2),
    round(float(np.percentile(jumps, 95)), 2),
)
assert len(mixed_centers) > 20
assert float(np.max(jumps)) < 80.0
print("SYNTHETIC_TEST_OK")