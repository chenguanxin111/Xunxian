# 直接转弯，不绕环岛,停车转弯，避障只执行一次防止终点时触发避障，加入终点检测
# 加入了软启动逻辑
# 避障不用odom,直接计时

#直线视野加大
#回归用雷达

import sys
# 假设 voice_manager.py 在这个路径下，如果不在，请修改为你的实际路径
sys.path.append('/home/ucar/ucar_ws/vision/src')
from voice_manager import VoiceManager
import cv2
import numpy as np
from PIL import Image as PImage
import math
import rospy
from geometry_msgs.msg import Twist
import os
from std_msgs.msg import String
import signal
from sklearn.linear_model import LinearRegression
from collections import deque
import time
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from nav_msgs.msg import Path, Odometry
from sensor_msgs.msg import LaserScan

# 全局变量定义
global slope
global state
global line_count_near, left_near, right_near, line_count_last, left_near_las
global bu
ph = 360  # 图片高度
pw = 640  # 图片宽度
RESULT_ROW = 480
RESULT_COL = 640
USED_ROW = 480
USED_COL = 640

X_qujibian = np.zeros((RESULT_ROW, RESULT_COL), dtype=np.uint16)
Y_qujibian = np.zeros((RESULT_ROW, RESULT_COL), dtype=np.uint16)

def thread_job():
    rospy.spin()

def get_pid_params(error, kanbujian):
    """ 根据误差和是否看到单边线，生成用于显示的PID状态文本 """
    abs_error = abs(error)
    if kanbujian:
        if 33.5 < abs_error <= 51: return "看不见：小大弯！"
        if 51 < abs_error <= 62: return "看不见：中弯！"
        if 62 < abs_error <= 64: return "看不见：极弯"
        if abs_error > 64: return "看不见：大极弯！"
        return "看不见：直线！"
    else:
        if 30 < abs_error <= 34: return "小直线！"
        if 34 < abs_error <= 55: return "小弯，一般情况！"
        if 55 < abs_error <= 60: return "中弯！"
        if abs_error > 60: return "大弯！"
        return "大直线！"

def display_and_log_debug_info(debug_data, enable_gui=False, origin_img=None, vis_img=None, roi_1=None, roi_2=None):
    """
    【再次升级】增加了对路口调试信息的打印。
    """
    # 1. 在终端打印一个详细、美观的多行日志块
    log_separator = f"--- [ {time.time():.2f} ] "
    print(log_separator.ljust(70, '-'))
    
    print(f"  {'State':<10}: {debug_data.get('main_state', 'N/A')}")
    print(f"  {'Turn':<10}: {debug_data.get('turn_decision', 'N/A'):<15} (L/R Flags: {debug_data.get('zuozhuan_flag', 0)}/{debug_data.get('youzhuan_flag', 0)})")
    print(f"  {'PID':<10}: Error= {debug_data.get('pid_error', 0.0):<6.2f} | Message= {debug_data.get('pid_message', 'N/A')}")
    print(f"  {'Motion':<10}: Vel(X/Z)= {debug_data.get('vel_x', 0.0):.2f}/{debug_data.get('vel_z', 0.0):.3f} | Dist(F/R)= {debug_data.get('dist_front', 999):.2f}/{debug_data.get('dist_right', 999):.2f}m")
    print(f"  {'System':<10}: FPS= {debug_data.get('fps', 0.0):.1f}")
    
    # --- 新增的调试信息打印 ---
    intersection_debug = debug_data.get('intersection_debug')
    if intersection_debug:
        int_line = (
            f"  {'Intersect':<10}: "
            f"L_px(avg/cur)={intersection_debug.get('avg_lp', 0)}/{intersection_debug.get('clp', 0)} | "
            f"R_px(avg/cur)={intersection_debug.get('avg_rp', 0)}/{intersection_debug.get('crp', 0)} | "
            f"Confirm(L/R)={intersection_debug.get('lcc', 0)}/{intersection_debug.get('rcc', 0)} | "
            f"H-Line(L/R)={intersection_debug.get('l_h_max', 0)}/{intersection_debug.get('r_h_max', 0)}"
        )
        print(int_line)

    print("") # 输出一个空行，增加可读性

    # 2. 如果启用了GUI，才执行所有与cv2显示相关的代码
    if enable_gui:
        # 检查图像是否存在，防止程序崩溃
        if vis_img is None or origin_img is None or roi_1 is None or roi_2 is None:
            print("警告：请求显示GUI，但未提供必需的图像帧。")
            return

        # --- 以下是之前版本中的图形绘制代码，原封不动地搬过来 ---
        display_panel = vis_img.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        color = (0, 255, 255)
        thickness = 1
        line_type = cv2.LINE_AA

        overlay = display_panel.copy()
        cv2.rectangle(overlay, (5, 5), (320, 190), (0, 0, 0), -1)
        alpha = 0.6
        cv2.addWeighted(overlay, alpha, display_panel, 1 - alpha, 0, display_panel)

        lines = [
            f"State: {debug_data.get('main_state', 'N/A')}",
            f"Turn Decision: {debug_data.get('turn_decision', 'N/A')}",
            f"L/R Turn Flags: {debug_data.get('zuozhuan_flag', 0)}/{debug_data.get('youzhuan_flag', 0)}",
            "------------------------------------",
            f"PID Error: {debug_data.get('pid_error', 0.0):.2f}",
            f"PID Message: {debug_data.get('pid_message', 'N/A')}",
            "------------------------------------",
            f"Velocity (X/Z): {debug_data.get('vel_x', 0.0):.2f} / {debug_data.get('vel_z', 0.0):.3f}",
            f"Dist (Front/Right): {debug_data.get('dist_front', 999):.2f} / {debug_data.get('dist_right', 999):.2f}m",
            f"FPS: {debug_data.get('fps', 0.0):.1f}",
        ]

        y0, dy = 25, 18
        for i, line in enumerate(lines):
            y = y0 + i * dy
            cv2.putText(display_panel, line, (10, y), font, font_scale, color, thickness, line_type)

        cv2.imshow("Original Image", origin_img)
        cv2.imshow("ROI 1 (Path Finding)", roi_1)
        cv2.imshow("Lane Visualization & Info Panel", display_panel)
        cv2.waitKey(1)


def image_perspective_init( ):
    cameraMatrix = np.array([[417.27599, 0.0, 322.66696],
                             [0.0, 416.23499, 229.45303],
                             [0.0, 0.0, 1.0]])
    distCoeffs = np.array([-0.317235, 0.095383, 0.001505, 0.000861, 0.0])
    move_xy = [0, 0]

    fx = cameraMatrix[0][0]
    fy = cameraMatrix[1][1]
    ux = cameraMatrix[0][2]
    uy = cameraMatrix[1][2]
    k1 = distCoeffs[0]
    k2 = distCoeffs[1]
    k3 = distCoeffs[4]
    p1 = distCoeffs[2]
    p2 = distCoeffs[3]

    move_x = move_xy[0]
    move_y = move_xy[1]

    for i in range(RESULT_ROW-1, 0, -1):
        for j in range(-move_x, RESULT_COL - move_x):
            xCorrected = (j - ux) / fx
            yCorrected = (i - uy) / fy
            r2 = xCorrected * xCorrected + yCorrected * yCorrected
            deltaRa = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
            deltaRb = 1.0 / 1.0
            deltaTx = 2.0 * p1 * xCorrected * yCorrected + p2 * (r2 + 2.0 * xCorrected * xCorrected)
            deltaTy = p1 * (r2 + 2.0 * yCorrected * yCorrected) + 2.0 * p2 * xCorrected * yCorrected
            xDistortion = xCorrected * deltaRa * deltaRb + deltaTx
            yDistortion = yCorrected * deltaRa * deltaRb + deltaTy
            xDistortion = xDistortion * fx + ux
            yDistortion = yDistortion * fy + uy
            X_qujibian[i][j] = int(xDistortion)
            Y_qujibian[i][j] = int(yDistortion)
    finish_flag = 1
    return finish_flag

def signal_handler(sig, frame):
    print('\n您按下了 Ctrl+C! 程序正在关闭...')
    cv2.destroyAllWindows()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

count = 0
last_hei = 0
count_1 = 0
list_hei = []
i = 0
low = [0, 0, 180]# 阈值
high = [255, 65, 255]
slope = 0
angle = 0
xunxian_flag  = 0
global crossroads
crossroads = False

def xunxian_flag_callback(data):
    global xunxian_flag
    xunxian_flag = int(data.data)

def Normalize(data, scale):
    mx = max(data)
    mn = min(data)
    m = (mx - mn)/2 + mn
    data_return = []
    for i in data:
        if ((mx - mn) != 0):
            data_return.append(((float(i) - m) / (mx - mn) + 0.5) * scale)
        else:
            data_return.append(0)
    return data_return

def Get_Hist_For_Dotted(oir_img):
    img = oir_img
    img = np.sum(img, axis=0)
    img_sum = img
    img.reshape(img.size, 1)
    hist = np.int32(np.around(Normalize(img, 255)))
    list_pixels = []
    for num in img:
        list_pixels.append(num/255)
    # print("每列白色像素数：list=",list_pixels) # 已被日志系统替代
    img = np.zeros((len(list_pixels), len(list_pixels), 3))
    bins = np.arange(len(list_pixels)).reshape(len(list_pixels), 1)
    pts = np.int64(np.column_stack((bins, list_pixels)))
    cv2.polylines(img, [pts], False, (0, 255, 0))
    img = np.flipud(img)
    return img_sum, hist

def new_get_yellow_lane_bin_img(frame, low_rh, high_rh, low_gs, high_gs, low_bv, high_bv):
    lower_array = np.array([low_rh, low_gs, low_bv])
    upper_array = np.array([high_rh, high_gs, high_bv])
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lowerb=lower_array, upperb=upper_array)
    _img = mask
    small_img = cv2.resize(_img, (pw, ph), cv2.INTER_AREA)
    small_img = small_img.astype(np.uint8)
    retval, bin_img = cv2.threshold(small_img, 125, 255, cv2.THRESH_BINARY)
    small_img = small_img.astype(np.float32)
    img_3chanel = cv2.cvtColor(small_img, cv2.COLOR_GRAY2BGR)
    origin_img = cv2.resize(frame, (pw, ph), cv2.INTER_AREA)
    return origin_img, bin_img, mask

def new_get_results(yuan_image, line_up_ratio=0.69): # 增加 line_up_ratio 参数
    global line_count_near, last_green_points
    # line_up = 0.69 # 删除固定的 line_up
    line_low = 1
    line_up_2 = 0.69 #0.55
    line_low_2= 1
    origin_img, bin_img, mask = new_get_yellow_lane_bin_img(yuan_image, 0, 120, 0, 65, 80, 255)
    H = origin_img.shape[0]
    W = origin_img.shape[1]
    # 使用传入的 line_up_ratio 动态设置ROI
    bin_img_rectangle_ROI = bin_img[int(H * line_up_ratio):int(H * line_low), :]
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    bin_img_rectangle_ROI = cv2.erode(bin_img_rectangle_ROI, kernel_erode, iterations=1)
    bin_img_rectangle_ROI = cv2.dilate(bin_img_rectangle_ROI, kernel_dilate, iterations=2)
    H2 = bin_img_rectangle_ROI.shape[0]
    bin_img_rectangle_ROI_2 = bin_img_rectangle_ROI[int(H2 * line_up_2):int(H2 * line_low_2), :]
    green_points, red_points, current_red_points_zuixiamian, vis, kanbujian = find_white_pixel_indices(bin_img_rectangle_ROI_2)
    if len(green_points) > 3:
        last_green_points = green_points
    return green_points, red_points, last_green_points, bin_img_rectangle_ROI, bin_img_rectangle_ROI_2, current_red_points_zuixiamian, vis, kanbujian

def calculate_metrics(green_points,last_green_points,bin_img_rectangle_ROI):
    # print(f"Green points count: {len(green_points)}") # 已被日志系统替代
    if len(green_points) < 3:
        if len(last_green_points)<2:
            last_green_points=[(244, 62), (246, 57), (247, 52), (249, 47), (252, 42), (253, 37)]
        green_points = last_green_points
    first_point = green_points[0]
    last_point =  green_points[-1]
    middle_point = green_points[len(green_points) // 2]
    if first_point[0] != last_point[0]:
        slope_first_last = (last_point[1] - first_point[1]) / (last_point[0] - first_point[0])
        angle_first_last = np.degrees(np.arctan(slope_first_last))
    else:
        angle_first_last = 90.0 if (last_point[1] - first_point[1]) > 0 else -90.0
    middle_point = green_points[round(len(green_points) /3.5)]
    if first_point[0] != middle_point[0]:
        slope_first_middle = (middle_point[1] - first_point[1]) / (middle_point[0] - first_point[0])
        angle_first_middle = np.degrees(np.arctan(slope_first_middle))
    else:
        angle_first_middle = 90.0 if (middle_point[1] - first_point[1]) > 0 else -90.0
    last_3_x_coords = [point[0] for point in green_points[:3]]
    avg_last_3_x = np.mean(last_3_x_coords)
    return angle_first_last, angle_first_middle, avg_last_3_x

def has_horizontal_line(img_roi, threshold_width):
    """
    检查ROI中是否存在足够长的水平线，并返回找到的最大宽度。
    """
    height, width = img_roi.shape
    max_horizontal_pixels = 0
    
    for y in range(height):
        horizontal_pixels = np.sum(img_roi[y, :] == 255)
        if horizontal_pixels > max_horizontal_pixels:
            max_horizontal_pixels = horizontal_pixels
            
    found_line = max_horizontal_pixels > threshold_width
    return found_line, int(max_horizontal_pixels)

def calculate_blue_slopes(img, is_turning_left, is_turning_right, front_dist):
    """
    【V3版 - 仅作触发器】
    这个函数现在只被用作视觉触发器。
    它返回"left"或"right"仅代表“可能”是一个路口，不再作为最终决策。
    """
    # 安全检查：如果前方距离太近，直接忽略所有路口检测
    if front_dist < 0.7:
        return "normal", {}

    global left_pixels_history, right_pixels_history
    global left_turn_confirmation_counter, right_turn_confirmation_counter

    intersection_debug = {} # 调试信息可以简化
    if is_turning_left or is_turning_right:
        return "normal", intersection_debug

    turn_direction = "normal"
    height, width = img.shape
    mid_point = width // 2
    left_zone = img[:, :mid_point]
    right_zone = img[:, mid_point:]
    current_left_pixels = np.sum(left_zone == 255)
    current_right_pixels = np.sum(right_zone == 255)

    baseline_threshold = 1800
    left_spike_threshold = 1550
    right_spike_threshold = 1550
    horizontal_line_threshold = 125 

    is_right_spike_detected = False
    if len(right_pixels_history) == right_pixels_history.maxlen:
        avg_right_pixels = np.mean(right_pixels_history)
        if avg_right_pixels > baseline_threshold and current_right_pixels > avg_right_pixels + right_spike_threshold:
            is_right_spike_detected = True

    if is_right_spike_detected:
        right_turn_confirmation_counter += 1
    else:
        right_turn_confirmation_counter = 0

    is_left_spike_detected = False
    if len(left_pixels_history) == left_pixels_history.maxlen:
        avg_left_pixels = np.mean(left_pixels_history)
        if avg_left_pixels > baseline_threshold and current_left_pixels > avg_left_pixels + left_spike_threshold:
            is_left_spike_detected = True
    
    if is_left_spike_detected:
        left_turn_confirmation_counter += 1
    else:
        left_turn_confirmation_counter = 0
            
    if right_turn_confirmation_counter >= CONFIRMATION_FRAMES_REQUIRED: # CONFIRMATION_FRAMES_REQUIRED 通常为 1
        found_line, _ = has_horizontal_line(right_zone, horizontal_line_threshold)
        if found_line:
            turn_direction = "left" 
        else:
            right_turn_confirmation_counter = 0

    if left_turn_confirmation_counter >= CONFIRMATION_FRAMES_REQUIRED:
        found_line, _ = has_horizontal_line(left_zone, horizontal_line_threshold)
        if found_line:
            turn_direction = "right"
        else:
            left_turn_confirmation_counter = 0

    left_pixels_history.append(current_left_pixels)
    right_pixels_history.append(current_right_pixels)

    if turn_direction != "normal":
        left_pixels_history.clear()
        right_pixels_history.clear()
        left_turn_confirmation_counter = 0
        right_turn_confirmation_counter = 0
    
    # 我们只关心它是否返回 "normal"，方向不重要
    return turn_direction, intersection_debug
def make_hybrid_junction_decision(img_roi, error_history, is_turning_left, is_turning_right, front_dist):
    """
    【V4版 - 带安全延迟的混合决策】
    在函数最开始增加一个计时器，在程序启动初期忽略所有路口信号。
    """
    # 【新增】引入全局的启动时间变量
    global program_start_time

    # --- 1. 安全延迟检查 ---
    elapsed_time = time.time() - program_start_time
    if elapsed_time < JUNCTION_DETECTION_DELAY:
        # 如果还在安全延迟期内，直接返回"normal"，不进行任何路口检测
        return "normal"

    final_decision = "normal"

    # --- 2. 调用“侦察兵” (视觉触发器) ---
    trigger_decision, _ = calculate_blue_slopes(img_roi, is_turning_left, is_turning_right, front_dist)

    if trigger_decision != "normal":
        # --- 3. “侦察兵”发现情况，呼叫“指挥官”（轨迹分析） ---
        rospy.loginfo(f"--- [Hybrid Logic] Visual Trigger FIRED! (Signal: {trigger_decision}) ---")
        
        if len(error_history) == PID_ERROR_HISTORY_LENGTH:
            average_error = np.mean(error_history)
            rospy.loginfo(f"--- [Hybrid Logic] Analyzing history... Avg PID Error: {average_error:.2f} ---")

            # --- 4. “指挥官”根据历史轨迹做出最终决策 ---
            if average_error < -JUNCTION_PID_ERROR_THRESHOLD:
                rospy.loginfo("--- [Hybrid Logic] FINAL DECISION: Clockwise detected -> LEFT ---")
                final_decision = "left"
            elif average_error > JUNCTION_PID_ERROR_THRESHOLD:
                rospy.loginfo("--- [Hybrid Logic] FINAL DECISION: Counter-Clockwise detected -> RIGHT ---")
                final_decision = "right"
            
            if final_decision != "normal":
                error_history.clear()
        else:
            rospy.logwarn("[Hybrid Logic] Trigger fired, but history buffer is not full. Ignoring.")

    return final_decision
def has_horizontal_line(img_roi, threshold_width):
    """
    【修改后】检查ROI中是否存在足够长的水平线，并返回找到的最大宽度。
    :param img_roi: 要检查的二值图像区域。
    :param threshold_width: 水平线被认为是“足够长”的像素宽度阈值。
    :return: (bool: 是否找到符合阈值的横线, int: 找到的最大横线宽度)
    """
    height, width = img_roi.shape
    max_horizontal_pixels = 0
    
    # 遍历所有行，找到最长的那条横线
    for y in range(height):
        horizontal_pixels = np.sum(img_roi[y, :] == 255)
        if horizontal_pixels > max_horizontal_pixels:
            max_horizontal_pixels = horizontal_pixels
            
    # 比较找到的最大值和阈值，得出最终判断
    found_line = max_horizontal_pixels > threshold_width
    
    return found_line, int(max_horizontal_pixels)

def find_white_pixel_indices(img):
    current_red_points_zuixiamian =[]
    height, width = img.shape
    global kanbujian , sigle , double
    sigle , double = 0,0
    green_points = []
    red_points = []
    for y in range(height - 1, -1, -4):
        white_indices = np.where(img[y] == 255)[0]
        if len(white_indices) == 0:
            continue
        diff = np.diff(white_indices)
        breaks = np.where(diff > 1)[0] + 1
        clusters = np.split(white_indices, breaks)
        mean_indices = [np.mean(cluster) for cluster in clusters]
        current_red_points = []
        new_green_point = None
        if len(mean_indices) == 1:
            sigle += 1
            red_x = int(mean_indices[0])
            current_red_points.append(red_x)
            if len(green_points) > 1:
                last_green_x = green_points[-1][0]
                second_last_green_x = green_points[-2][0]
                virtual_red_x = 0 if last_green_x < second_last_green_x else width - 1
            else:
                virtual_red_x = width - 1 if red_x < width // 2 else 0
            current_red_points.append(virtual_red_x)
            avg_index = np.mean(current_red_points)
            new_green_point = (int(avg_index), y)
        elif len(mean_indices) > 1:
            double += 1
            for idx in mean_indices:
                current_red_points.append(int(idx))
            if len(current_red_points) == 2:
                if abs(current_red_points[0] - current_red_points[1]) < width / 3:
                    if abs(current_red_points[0] - width // 2) > abs(current_red_points[1] - width // 2):
                        current_red_points = [current_red_points[1], width - 1 if current_red_points[1] < width // 2 else 0]
                    else:
                        current_red_points = [current_red_points[0], width - 1 if current_red_points[0] < width // 2 else 0]
                avg_index = np.mean(current_red_points)
                new_green_point = (int(avg_index), y)
            else:
                mid_x = width // 2
                left_red_points = [pt for pt in current_red_points if pt < mid_x]
                right_red_points = [pt for pt in current_red_points if pt >= mid_x]
                left_nearest = min(left_red_points, key=lambda x: abs(x - mid_x)) if left_red_points else 0
                right_nearest = min(right_red_points, key=lambda x: abs(x - mid_x)) if right_red_points else width - 1
                current_red_points = [left_nearest, right_nearest]
                avg_index = np.mean(current_red_points)
                new_green_point = (int(avg_index), y)
        if current_red_points:
            current_red_points_zuixiamian = current_red_points
        for rp in current_red_points:
            red_points.append((rp, y))
        if sigle + double > 0 and sigle / (sigle + double) > 0.9:
            kanbujian = 1
        else:
            kanbujian = 0
        if new_green_point and (len(green_points) == 0 or abs(new_green_point[0] - green_points[-1][0]) < width / 8):
            green_points.append(new_green_point)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for x, y in green_points:
        cv2.circle(vis, (x, y), 3, (0, 255, 0), -1)
    for x, y in red_points:
        cv2.circle(vis, (x, y), 3, (0, 0, 255), -1)
    return green_points, red_points, current_red_points_zuixiamian, vis, kanbujian

kanbujian = 0
zhuanxiang_time = 0

def xunxian_2(angle_first_last, angle_first_middle, avg_last_3_x, zhuanxiang, current_red_points_zuixiamian):
    """
    【V2 - 视觉闭环转弯版】
    - 将转弯分为两个阶段：
      1. 粗略转向：使用里程计转动约80%的角度。
      2. 视觉精调：使用PID控制器根据视觉误差进行精确对准。
    """
    vel = Twist()
    # 引入所有需要的全局变量
    global last_v_z, delat_v_z, chongci
    global results, kp_z, kp_y, kd_z
    global odom_yaw, goal_yaw
    global xunxian_input, zhuanxiang_time
    global zuozhuan_flag, youzhuan_flag
    global turn_start_time, TURN_TIMEOUT, turn_align_counter # <--- 修改：引入新变量
    global intersection_passed_flag
    global GENTLE_START_DURATION, program_start_time

    # 初始化局部变量
    kp_z, kp_y, kd_z, chongci = 0, 0, 0, 0

    # 计算PID误差 (这个误差在巡线和精调阶段都会用到)
    if angle_first_middle > 0:
        error = -90 + angle_first_middle
    else:
        error = 90 + angle_first_middle

    # --- 1. 左转弯逻辑 (采用两阶段法) ---
    if zhuanxiang == "left" or zuozhuan_flag > 0: # <--- 修改：条件从 ==1 改为 >0，以处理多个转弯阶段
        # --- 阶段 0: 初始化转弯 ---
        if zuozhuan_flag == 0:
            rospy.loginfo("--- [转弯] 开始执行左转 (阶段1: 粗略转向)... ---")
            goal_yaw = odom_yaw + math.pi*(80/180)
            if goal_yaw > math.pi:
                goal_yaw -= 2 * math.pi
            zuozhuan_flag = 1  # 进入阶段1
            turn_start_time = time.time()
            turn_align_counter = 0 # 重置计数器

        # --- 超时检查 (对所有转弯阶段都有效) ---
        if time.time() - turn_start_time > TURN_TIMEOUT:
            rospy.logwarn("!!! 左转超时，强制复位标志位 !!!")
            zuozhuan_flag = 0
            last_v_z, delat_v_z = 0, 0
            return Twist()

        # --- 阶段 1: 粗略转向 (依赖里程计) ---
        if zuozhuan_flag == 1:
            vel.linear.x = 0.04
            vel.angular.z = 0.6 # 恒定角速度旋转

            # 使用更严格的阈值判断是否完成粗略转向
            if abs(odom_yaw - goal_yaw) < 0.25: # <--- 修改：更严格的阈值 (约14度)
                rospy.loginfo("--- [转弯] 粗略转向完成，进入 (阶段2: 视觉精调)... ---")
                zuozhuan_flag = 2 # 进入阶段2
            else:
                # 这部分平移逻辑可以保留，以实现更圆滑的转弯
                delat_xuanzuhan = odom_yaw - goal_yaw
                if abs(delat_xuanzuhan) > math.pi:
                    delat_xuanzuhan += math.pi * 2 if delat_xuanzuhan < 0 else -math.pi * 2
                vel.linear.y = -0.035 if abs(delat_xuanzuhan) > 1.4 else 0

        # --- 阶段 2: 视觉精调 (依赖PID控制器) ---
        elif zuozhuan_flag == 2:
            vel.linear.x = 0.05 # <--- 新增：在精调时极慢速前进
            
            # 使用巡线PID进行精细对准
            kp_z, kd_z = 0.022, 0.2 # 使用一组温和的PID参数
            vel.angular.z = kp_z * error - kd_z * delat_v_z
            delat_v_z = vel.angular.z - last_v_z

            # 检查对齐是否稳定
            if abs(error) < TURN_ALIGN_ERROR_THRESHOLD and kanbujian == 1:
                turn_align_counter += 1
                rospy.loginfo(f"--- [转弯] 视觉对齐中... Error: {error:.2f}, Stable Frames: {turn_align_counter}/{TURN_ALIGN_FRAMES}")
            else:
                turn_align_counter = 0 # 如果误差太大，重置计数器

            # 如果连续多帧都对齐，则认为转弯完成
            if turn_align_counter >= TURN_ALIGN_FRAMES:
                rospy.loginfo("--- [转弯] 视觉精调完成，左转结束 ---")
                if not intersection_passed_flag:
                    rospy.loginfo("--- 避障功能已解锁 ---")
                    intersection_passed_flag = True
                zuozhuan_flag = 0
                last_v_z, delat_v_z = 0, 0
                return Twist() # 返回零速，平滑过渡

    # --- 2. 右转弯逻辑 (与左转同理) ---
    elif zhuanxiang == "right" or youzhuan_flag > 0: # <--- 修改：条件从 ==1 改为 >0
        # 阶段 0: 初始化
        if youzhuan_flag == 0:
            rospy.loginfo("--- [转弯] 开始执行右转 (阶段1: 粗略转向)... ---")
            goal_yaw = odom_yaw - math.pi*(80/180)
            if goal_yaw < -math.pi:
                goal_yaw += 2 * math.pi
            youzhuan_flag = 1
            turn_start_time = time.time()
            turn_align_counter = 0

        # 超时检查
        if time.time() - turn_start_time > TURN_TIMEOUT:
            rospy.logwarn("!!! 右转超时，强制复位标志位 !!!")
            youzhuan_flag = 0
            last_v_z, delat_v_z = 0, 0
            return Twist()

        # 阶段 1: 粗略转向
        if youzhuan_flag == 1:
            vel.linear.x = 0.04
            vel.angular.z = -0.6
            if abs(odom_yaw - goal_yaw) < 0.25: # <--- 修改：更严格的阈值
                rospy.loginfo("--- [转弯] 粗略转向完成，进入 (阶段2: 视觉精调)... ---")
                youzhuan_flag = 2
            else:
                delat_xuanzuhan = odom_yaw - goal_yaw
                if abs(delat_xuanzuhan) > math.pi:
                    delat_xuanzuhan += math.pi * 2 if delat_xuanzuhan < 0 else -math.pi * 2
                vel.linear.y = 0.035 if abs(delat_xuanzuhan) > 1.4 else 0

        # 阶段 2: 视觉精调
        elif youzhuan_flag == 2:
            vel.linear.x = 0.05
            kp_z, kd_z = 0.022, 0.2
            vel.angular.z = kp_z * error - kd_z * delat_v_z
            delat_v_z = vel.angular.z - last_v_z

            if abs(error) < TURN_ALIGN_ERROR_THRESHOLD and kanbujian == 1:
                turn_align_counter += 1
                rospy.loginfo(f"--- [转弯] 视觉对齐中... Error: {error:.2f}, Stable Frames: {turn_align_counter}/{TURN_ALIGN_FRAMES}")
            else:
                turn_align_counter = 0

            if turn_align_counter >= TURN_ALIGN_FRAMES:
                rospy.loginfo("--- [转弯] 视觉精调完成，右转结束 ---")
                if not intersection_passed_flag:
                    rospy.loginfo("--- 避障功能已解锁 ---")
                    intersection_passed_flag = True
                youzhuan_flag = 0
                last_v_z, delat_v_z = 0, 0
                return Twist()

    elif zhuanxiang == "stop" :
        vel = Twist() 
            
    else: # 正常巡线PID控制 (这部分保持不变)
        pid_message = get_pid_params(error, kanbujian)
        if "看不见" in pid_message:
            if "小大弯" in pid_message: kp_z, kp_y, kd_z = 0.024, 0.00005, 0.22
            elif "中弯" in pid_message: kp_z, kp_y, kd_z = 0.026, 0.00005, 0.2
            elif "极弯" in pid_message: kp_z, kp_y, kd_z = 0.029, 0.00005, 0.1
            elif "大极弯" in pid_message: kp_z, kp_y, kd_z = 0.033, 0.00005, 0.25
        else:
            if "小直线" in pid_message: kp_z, kp_y, kd_z = 0.016, 0.00005, 0.25
            elif "小弯" in pid_message: kp_z, kp_y, kd_z = 0.022, 0.00005, 0.2
            elif "中弯" in pid_message: kp_z, kp_y, kd_z = 0.024, 0.00005, 0.2
            elif "大弯" in pid_message: kp_z, kp_y, kd_z = 0.0265, 0.00005, 0.15
            else: # Big Straight
                chongci = 0.0
                kp_z, kp_y, kd_z = 0.013, 0.0005, 0.3

        is_gentle_start = (time.time() - program_start_time) < GENTLE_START_DURATION
        if is_gentle_start:
            rospy.loginfo_once("--- Gentle Start Mode ON ---")
            vel.linear.x = 0.15
            chongci = 0
            kp_z *= 0.45
            kd_z *= 0.5
        else:
            rospy.loginfo_once("--- Gentle Start Mode OFF ---")
            vel.linear.x = 0.36 + chongci
            kp_z *= 1.22
            kd_z *= 1.2
            
        kp_y *= 15

        vel.angular.z = kp_z * error - kd_z * delat_v_z
        delat_v_z = vel.angular.z - last_v_z 

        if results["left"] or results["right"] > 5:
             pass

        error_y = 0
        if current_red_points_zuixiamian and all(p not in [0, 639] for p in current_red_points_zuixiamian):
            error_y = np.mean(current_red_points_zuixiamian)

        if error_y != 0:
            error_y_contorl = error_y - 320
            vel.linear.y = kp_y * error_y_contorl * 0.0005
        else:
            vel.linear.y = 0
            
    return vel
def check_for_end_line(img_roi, width_threshold_ratio=0.5):
    """
    检查给定的ROI中是否存在一条横跨大部分宽度的线（终点线）。

    :param img_roi: 要检查的二值图像区域。
    :param width_threshold_ratio: 宽度阈值比例，横线像素数超过该比例*总宽度，则认为是终点线。
    :return: bool: 如果检测到终点线，则返回 True。
    """
    height, width = img_roi.shape
    # 计算需要达到的像素宽度阈值
    width_threshold = int(width * width_threshold_ratio)
    
    # 遍历ROI的每一行
    for y in range(height):
        # 计算当前行的白色像素数量
        horizontal_pixels = np.sum(img_roi[y, :] == 255)
        # 如果某一行的白色像素数量超过了阈值，就认为是终点线
        if horizontal_pixels > width_threshold:
            rospy.loginfo(f"检测到潜在终点线！行: {y}, 像素数: {horizontal_pixels}, 阈值: {width_threshold}")
            return True
            
    return False

def callback_read_current_position(data):
    global odom_yaw, odom_x, odom_y
    odom_x = data.pose.pose.position.x
    odom_y = data.pose.pose.position.y
    qx, qy, qz, qw = data.pose.pose.orientation.x, data.pose.pose.orientation.y, data.pose.pose.orientation.z, data.pose.pose.orientation.w
    euler = euler_from_quaternion((qx, qy, qz, qw))
    odom_yaw = euler[2]

def scan_callback(scan):
    """
    【修复后】处理激光雷达数据，使用正确的方式获取扫描点数。
    """
    global distance_front, distance_right
    
    min_front_dist = 999.0 
    min_right_dist = 999.0
    
    # --- 核心修复 ---
    # 错误的方式：count = int(scan.scan_time / scan.time_increment)
    # 正确、健壮的方式是直接获取 ranges 数组的长度
    count = len(scan.ranges)
    
    for i in range(count):
        degree = math.degrees(scan.angle_min + scan.angle_increment * i)
        
        # 直接访问 scan.ranges[i] 现在是安全的
        dist = scan.ranges[i]
        
        # 忽略无效的读数 (无穷大或0)
        if not math.isinf(dist) and dist > 0.1:
            # 检测正前方障碍物
            if -15 < degree < 15 and dist < min_front_dist:
                min_front_dist = dist
            # 检测正右方障碍物
            if -92 < degree < -88 and dist < min_right_dist:
                min_right_dist = dist

    # 更新全局变量
    distance_front = min_front_dist
    distance_right = min_right_dist

ENABLE_GUI = False    
# ROS 初始化
rospy.init_node('detector_refactored', anonymous=True)
vel_puber = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
xunxian_flag_sub = rospy.Subscriber('/xunxian_flag', String, xunxian_flag_callback, queue_size=1)
leida_sub = rospy.Subscriber("/scan", LaserScan, scan_callback, queue_size=1)
pose_sub = rospy.Subscriber('/odom', Odometry, callback_read_current_position, queue_size=1) #after_final_and_amcl_odom
voice_pub = rospy.Publisher('/voice/announce_cmd', String, queue_size=1)
# 全局变量和状态初始化
bu = 0
zhuanxiang = 0
odom_x = 0
odom_y = 0
odom_yaw = 0
goal_yaw = 0
last_v_z = 0
delat_v_z = 0
zuozhuan_flag = 0
youzhuan_flag = 0
results = {"left": 0, "right": 0}
state = "else"
last_green_points = []
xunxian_input = [] 
# 新增：用于转弯超时的变量
turn_start_time = 0
TURN_TIMEOUT = 8.0 # <--- 修改：稍微增加超时时间，因为新流程可能耗时更长
turn_align_counter = 0 # <--- 新增：用于视觉对齐的稳定帧计数器
TURN_ALIGN_FRAMES = 1 # <--- 新增：需要连续10帧对齐才算成功
TURN_ALIGN_ERROR_THRESHOLD = 28.0 # <--- 新增：视觉对齐的PID误差阈值

# 【新增】软启动相关变量
GENTLE_START_DURATION = 2.0 # 软启动持续时间（秒）
LARGE_FOV_DURATION_AFTER_START = 2.5 # 【新增】软启动后，使用远视野持续的时间
program_start_time = 0.0      # 程序启动时间

# 【新增】路口轨迹分析法相关变量
PID_ERROR_HISTORY_LENGTH = 30              # 记录最近30帧的PID误差
pid_error_history = deque(maxlen=PID_ERROR_HISTORY_LENGTH) # 用于存储误差历史的双端队列
JUNCTION_PID_ERROR_THRESHOLD = 5.0         # 平均误差超过该阈值，才认为是有效转向

# 【新增】路口检测安全延迟
JUNCTION_DETECTION_DELAY = 5.5 # 程序启动后8秒内，忽略所有路口触发信号

# 转弯检测相关
left_turn_confirmation_counter = 0
right_turn_confirmation_counter = 0
CONFIRMATION_FRAMES_REQUIRED = 1 
left_pixels_history = deque(maxlen=5)
right_pixels_history = deque(maxlen=5)

# 避障状态机相关
distance_front = 999.0
distance_right = 999.0
current_state = 'LINE_FOLLOWING' 
intersection_passed_flag = False             # 是否通过了路口
obstacle_avoidance_executed_flag = False     # 是否已执行了避障
final_stop_triggered = False                 # 是否已触发终点停车

OBSTACLE_DETECT_DISTANCE = 0.45
AVOIDANCE_STRAFE_DISTANCE = 0.50  # 准备向侧方平移多远 (米)
AVOIDANCE_FORWARD_DISTANCE = 0.60  # 平移后，向前开多远以越过障碍物 (米)
AVOIDANCE_SPEED_X = 0.4
AVOIDANCE_SPEED_Y = 0.4
STRAFE_KP = 1.5
STRAFE_TARGET_TOLERANCE = 0.02
# strafe_target_dist = 0.0
# original_strafe_ref_dist = 0.0

# 【新增】基于时间的避障控制变量
AVOIDANCE_FORWARD_DURATION = 1.45      # 向前开的时间 (秒)
AVOIDANCE_STRAFE_DURATION = 1.65      # 【向左】平移避开障碍物的时间 (秒)
  
AVOIDANCE_SPEED_X = 0.4               # 避障时前进的速度
AVOIDANCE_RETURN_SPEED_Y = 0.135      # 闭环回归时，向右平移的较慢速度
AVOIDANCE_CENTER_TOLERANCE_PX = 100   # 居中判断的容差（单位：像素）
AVOIDANCE_STRAFE_SPEED_Y = 0.3 
AVOIDANCE_RETURN_KP = 0.002          # 【新增】回归时P控制器的比例增益
# --- 【新增】为实现完美停车引入的状态和变量 ---
FINISH_LINE_CROSSING_DURATION = 0.42  # 检测到终点线后，继续向前开的时间(秒)
finish_line_crossing_start_time = 0.0 # 记录开始穿越终点线的时间

# 【新增】语音播报相关
# voice_manager = None
purchase_summary_announced = False  # 防止重复播报的标志位


# --- 主循环 ---

if __name__ == '__main__':
    while 1:    
        if xunxian_flag == 1:
            capture = cv2.VideoCapture("/dev/video0")
            if not capture.isOpened():
                rospy.logerr("无法打开摄像头 /dev/video0，程序退出。")
                sys.exit(1)
            
            rospy.loginfo("摄像头已打开，开始执行任务...")
            program_start_time = time.time()
            
            

            rate = rospy.Rate(20)

            yuan_image, vis, roi_1, roi_2 = None, None, None, None

            while not rospy.is_shutdown():
                time1 = time.time()
                debug_data = {}
                vel = Twist()

                # 无论处于何种状态，每一轮循环都先读取一次摄像头
                ret, yuan_image = capture.read()
                if not ret:
                    rospy.logwarn("警告：无法从摄像头读取图像，跳过此循环。")
                    rate.sleep()
                    continue

                current_time_from_start = time.time() - program_start_time
                is_soft_start = current_time_from_start < GENTLE_START_DURATION
                is_post_start_straight = GENTLE_START_DURATION <= current_time_from_start < (GENTLE_START_DURATION + LARGE_FOV_DURATION_AFTER_START)
                is_finding_lines_for_avoidance = (current_state == 'STRAFING_BACK_VISION')

                if is_soft_start or is_finding_lines_for_avoidance:
                    # 状态1: 软启动或避障返回时，使用广角视野看得更近更宽
                    vision_roi_up_ratio = 0.40
                    log_msg = "软启动/避障广角视野 (看得近)"
                elif is_post_start_straight:
                    # 状态2: 软启动刚结束的直线阶段，使用远视野看得更远，提高稳定性
                    vision_roi_up_ratio = 0.50 
                    log_msg = f"直线远视野 (看得远，剩余 {((GENTLE_START_DURATION + LARGE_FOV_DURATION_AFTER_START) - current_time_from_start):.1f}s)"
                else:
                    # 状态3: 其他所有情况（如进入弯道），使用此文件原始的标准视野
                    vision_roi_up_ratio = 0.69 
                    log_msg = "标准视野 (常规)"

                rospy.loginfo_throttle(1.0, f"当前视野模式: {log_msg} | Ratio: {vision_roi_up_ratio}")
                
                # 【新增】如果已触发最终停车，则直接发布零速并跳过所有逻辑
                if final_stop_triggered:
                    vel_puber.publish(Twist())
                    rate.sleep()
                    continue

                # =================== 状态机核心逻辑 ===================
                if current_state == 'LINE_FOLLOWING':
                    # --- 【修改】1. 避障触发判断 (使用新逻辑) ---
                    if distance_front < OBSTACLE_DETECT_DISTANCE and intersection_passed_flag and not obstacle_avoidance_executed_flag:
                        rospy.loginfo("!!! [状态切换] 检测到障碍物，先短暂停车再避障 !!!")
                        vel_puber.publish(Twist())
                        rospy.sleep(0.2)
                        
                        # 【【【核心新增】】】 在这里捕获初始右侧距离！
                        original_strafe_ref_dist = distance_right
                        rospy.loginfo(f"--- 捕获到初始右侧距离为: {original_strafe_ref_dist:.2f} m ---")

                        avoid_forward_start_time = time.time() 
                        current_state = 'STRAFING_AWAY'
                        continue

                    # --- 2. 视觉处理与巡线 (这部分逻辑在软启动步骤中已修改好) ---
                    green_points, red_points, last_green_points, roi_1, roi_2, current_red_points_zuixiamian, vis, kanbujian = new_get_results(yuan_image, line_up_ratio=vision_roi_up_ratio)
                    
                    # --- 3. 终点线判断 (保持不变) ---
                    if obstacle_avoidance_executed_flag:
                        # check_for_end_line 的阈值可以根据实际情况调整，0.8 是比较严格的
                        if check_for_end_line(roi_1, width_threshold_ratio=0.80): 
                            rospy.loginfo("!!! [状态切换] 检测到终点线，开始穿越 !!!")
                            # 记录穿越开始时间
                            finish_line_crossing_start_time = time.time()
                            # 进入新的穿越状态
                            current_state = 'CROSSING_FINISH_LINE'
                            continue

                    # --- 4. 正常巡线与路口判断 (【核心修改】) ---
                    
                    # a. 计算当前帧的PID误差 (不变)
                    angle_first_last, angle_first_middle, avg_last_3_x = calculate_metrics(green_points, last_green_points, roi_1)
                    if angle_first_middle > 0: error = -90 + angle_first_middle
                    else: error = 90 + angle_first_middle
                    
                    # b. 将当前误差存入“记忆” (不变)
                    if zuozhuan_flag == 0 and youzhuan_flag == 0:
                        pid_error_history.append(error)

                    # c. 【修改】调用全新的混合决策函数
                    zhuanxiang = make_hybrid_junction_decision(roi_1, pid_error_history, zuozhuan_flag, youzhuan_flag, distance_front)
                    
                    # d. 更新调试信息并获取速度指令 (不变)
                    pid_message = get_pid_params(error, kanbujian)
                    vel = xunxian_2(angle_first_last, angle_first_middle, avg_last_3_x, zhuanxiang, current_red_points_zuixiamian)
                    
                    # e. 更新日志数据
                    debug_data.update({
                        'main_state': 'LINE_FOLLOWING', 'turn_decision': zhuanxiang,
                        'pid_error': error, 'pid_message': pid_message,
                        'zuozhuan_flag': zuozhuan_flag, 'youzhuan_flag': youzhuan_flag,
                        # 可以选择性地在日志中也显示平均误差，方便调试
                        'intersection_debug': {'avg_error': np.mean(pid_error_history) if len(pid_error_history) > 0 else 0}
                    })

                elif current_state == 'STRAFING_AWAY':
                    rospy.loginfo_throttle(0.5, "避障阶段1: 开环向左平移...")
                    vel.linear.y = AVOIDANCE_STRAFE_SPEED_Y  # 以固定速度向左
                    vel.linear.x = 0.1 # 平移时稍微给点前进速度，更流畅
                    
                    # 检查向左平移的时间是否足够
                    if time.time() - avoid_forward_start_time >= AVOIDANCE_STRAFE_DURATION:
                        rospy.loginfo("--- 左平移完成，准备前进越障 ---")
                        # 重置计时器，用于下一步
                        avoid_forward_start_time = time.time()
                        current_state = 'AVOIDING_FORWARD'
                    debug_data['main_state'] = 'STRAFING_AWAY'

                elif current_state == 'AVOIDING_FORWARD':
                    rospy.loginfo_throttle(0.5, "避障阶段2: 【开环】向前直行...")
                    vel.linear.x = AVOIDANCE_SPEED_X
                    vel.linear.y = 0
                    
                    # 检查向前直行的时间是否足够
                    if time.time() - avoid_forward_start_time >= AVOIDANCE_FORWARD_DURATION:
                        rospy.loginfo("--- 前进越障完成，开始实时决策回归方式 ---")
                        
                        # 【【【全新的决策逻辑：直接使用当前实时距离】】】
                        # 判断当前右侧2米内是否有参考物
                        if distance_right < 2.5:
                            rospy.loginfo(f"决策结果：当前右侧有参考物 (实时距离 {distance_right:.2f}m)，进入雷达回归状态。")
                            current_state = 'STRAFING_BACK_LIDAR'
                        else:
                            rospy.logwarn(f"决策结果：当前右侧无参考物 (实时距离 {distance_right:.2f}m)，降级为视觉回归状态。")
                            current_state = 'STRAFING_BACK_VISION'

                        
                    debug_data['main_state'] = 'AVOIDING_FORWARD'

                elif current_state == 'STRAFING_BACK_LIDAR':
                    rospy.loginfo_throttle(0.5, "避障阶段3: 【闭环P控-雷达】向右回归...")
                    
                    # 【【【核心修改：目标不再是旧数据，而是一个固定值】】】
                    strafe_target_dist = 1.1  # 设定一个固定的理想右侧距离为x米
                    
                    # 计算与目标距离的误差
                    error = strafe_target_dist - distance_right
                    
                    # 判断是否已回归到目标位置
                    if abs(error) < STRAFE_TARGET_TOLERANCE:
                        rospy.loginfo("--- [雷达] 成功回归赛道，恢复巡线！ ---")
                        obstacle_avoidance_executed_flag = True
                        rospy.loginfo("--- 终点线检测功能已解锁 ---")
                        
                        # 【注意】我们不再需要 original_strafe_ref_dist，可以把重置它的代码删掉
                        current_state = 'LINE_FOLLOWING'
                        continue 

                    # P控制器计算回归速度
                    speed = STRAFE_KP * error
                    vel.linear.y = np.clip(speed, -AVOIDANCE_STRAFE_SPEED_Y, AVOIDANCE_STRAFE_SPEED_Y)
                    debug_data['main_state'] = 'STRAFING_BACK_LIDAR'

                # --- 【全新】的视觉回归状态 ---
                elif current_state == 'STRAFING_BACK_VISION':
                    rospy.loginfo_throttle(0.5, "避障阶段3: 【闭环P控】视觉搜索并对准赛道中心...")
                    
                    # 1. 使用更大的视野进行视觉处理
                    green_points, _, _, _, _, current_red_points_zuixiamian, _, _ = new_get_results(yuan_image, line_up_ratio=0.30)

                    # 2. 检查是否看到了两条真实的线
                    found_two_real_lines = (len(current_red_points_zuixiamian) == 2 and 
                                            0 not in current_red_points_zuixiamian and 
                                            (pw - 1) not in current_red_points_zuixiamian)

                    # 3. 如果真的看到了两条线，才执行P控制器进行对准
                    if found_two_real_lines:
                        if len(green_points) > 5:
                            lane_center_x = np.mean([p[0] for p in green_points])
                            center_error = lane_center_x - (pw / 2)

                            # 【核心修改】应用P控制器
                            # 平移速度 = -Kp * 误差。误差为正(偏右)时向左移(y为正)，误差为负(偏左)时向右移(y为负)
                            # 我们是在赛道左边，所以误差初始为负，-Kp*负误差 = 正的速度，这是不对的，应该向右移。
                            # 我们的场景是车在左侧，线在右侧，lane_center_x > 320, error为正。我们需要向右平移（y为负）。
                            # 所以公式应为 vel.linear.y = -Kp * error
                            vel.linear.y = -AVOIDANCE_RETURN_KP * center_error
                            vel.linear.x = 0.05 # 保持一点前进动力

                            rospy.loginfo_throttle(0.2, f"  [P控回归] 中心误差: {center_error:.1f}px, 平移速度: {vel.linear.y:.3f}")

                            # 判断是否已经对准 (速度足够小，也意味着误差足够小)
                            if abs(center_error) < AVOIDANCE_CENTER_TOLERANCE_PX:
                                rospy.loginfo("--- 【闭环】成功回归赛道中心，恢复巡线！ ---")
                                obstacle_avoidance_executed_flag = True
                                rospy.loginfo("--- 终点线检测功能已解锁 ---")
                                current_state = 'LINE_FOLLOWING'
                                vel = Twist() # 速度清零
                            
                        else:
                            # 看到两条线但green_points不够，先保持一个慢速向右移动
                            vel.linear.y = -0.15 
                            vel.linear.x = 0.05
                    
                    # 4. 如果还没看到两条真实的线，就用一个恒定慢速向右平移搜索
                    else:
                        rospy.logwarn_throttle(0.5, "  [P控回归] 未找到两条真实赛道线，继续向右平移搜索...")
                        vel.linear.y = -0.20 # 搜索时速度可以稍快
                        vel.linear.x = 0.05
                    
                    debug_data['main_state'] = 'STRAFING_BACK_VISION'

                # --- 【新增】穿越终点线状态 ---
                elif current_state == 'CROSSING_FINISH_LINE':
                    rospy.loginfo_throttle(0.5, "正在穿越终点线...")
                    # 以一个较慢的速度向前冲线
                    vel.linear.x = 0.2
                    vel.angular.z = 0.0 # 确保不转弯
                    
                    # 检查是否已达到指定的穿越时间
                    if time.time() - finish_line_crossing_start_time >= FINISH_LINE_CROSSING_DURATION:
                        rospy.loginfo("!!! 穿越终点线完成，车辆最终停止 !!!")
                        # 穿越完成，进入最终停止状态
                        current_state = 'FINAL_STOP'
                        vel = Twist() # 速度清零，平滑过渡

                # 【新增】终点停车状态
                elif current_state == 'FINAL_STOP':
                    # 只有在第一次进入此状态时才播报
                    if not purchase_summary_announced:
                        rospy.loginfo("开始播报采购总结...")
                        voice_pub.publish('5')  # 直接发布指令
                        purchase_summary_announced = True
                        rospy.loginfo("采购总结播报完成")
                    
                    rospy.loginfo_once("已到达终点，车辆停止。")
                    vel = Twist() # 确保速度为0
                    final_stop_triggered = True # 触发永久停车
                
                vel_puber.publish(vel)

                time2 = time.time()
                if (time2 - time1) > 0:
                    debug_data['fps'] = 1 / (time2 - time1)
                
                debug_data.update({
                    'dist_front': distance_front, 'dist_right': distance_right,
                    'vel_x': vel.linear.x, 'vel_z': vel.angular.z
                })

                display_and_log_debug_info(
                    debug_data, enable_gui=ENABLE_GUI,
                    origin_img=yuan_image, vis_img=vis, 
                    roi_1=roi_1, roi_2=roi_2
                )
                
                rate.sleep()

            # 循环结束后的清理工作
            rospy.loginfo("循环结束，释放摄像头资源。")
            capture.release()
            if ENABLE_GUI:
                cv2.destroyAllWindows()
            rospy.signal_shutdown("任务完成或被中断")