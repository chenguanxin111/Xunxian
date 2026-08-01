import cv2
import numpy as np
import sys
import math
import rospy
from geometry_msgs.msg import Twist
import os
from std_msgs.msg import String
from collections import deque
import time
from tf.transformations import euler_from_quaternion
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from playsound import playsound
import argparse  # 添加命令行参数解析
import signal    # 添加信号处理

# 全局常量
ph = 360  # 图片高度
pw = 640  # 图片宽度



class RobotState:
    def __init__(self):
        # 初始化所有状态变量
        self.duizhun_finisah = 1       # 对准完成标志
        self.xunxian_flag = 1          # 巡线标志
        self.odom_x = 0.0              # 里程计 x 坐标
        self.odom_y = 0.0              # 里程计 y 坐标
        self.odom_yaw = 0.0            # 里程计偏航角
        self.distance_90 = 0.0         # 90度方向的距离
        self.distance_70 = 0.0         # 70度方向的距离
        self.distance_110 = 0.0        # 110度方向的距离
        self.sound_finish = 0           # 声音播放完成标志
        self.results = {"left": 0, "right": 0}  # 左右结果
        self.last_v_z = 0.0            # 上次的角速度
        self.zhuanxiang_time = 0       # 转向时间
        self.last_error = 0.0          # 上次的误差
        self.error = 0.0
        self.last_center_points = []
        self.delat_v_z = 0
        self.last_v_z = 0
        self.custom_speed = 0.32       # 自定义速度，默认值
        self.is_running = True         # 程序运行状态
        # 根据需要添加其他状态变量

    def stop(self, vel_pub):
        """停止机器人并关闭程序

        Args:
            vel_pub: 速度发布者
        """
        self.is_running = False
        vel = Twist()
        vel_pub.publish(vel)
        print("机器人已停止")
        try:
            rospy.signal_shutdown("用户请求停止")
        except:
            pass
        return

# 定义PID参数常量字典
PID_PARAMS = {
    # 调整PID参数，适应新的摄像头位置
    "small_curve_invisible": (0.022*1.2, 0.00005*15, 0.20*1.2),   # 小大弯 (原来是0.024, 0.00005, 0.22)
    "medium_curve_invisible": (0.024*1.2, 0.00005*15, 0.18*1.2),   # 中弯 (原来是0.026, 0.00005, 0.2)
    "extreme_curve_invisible": (0.027*1.2, 0.00005*15, 0.09*1.2),  # 极弯 (原来是0.029, 0.00005, 0.1)
    "large_extreme_curve_invisible": (0.031*1.2, 0.00005*15, 0.23*1.2),   # 大极弯 (原来是0.033, 0.00005, 0.25)
    "small_straight": (0.015*1.2, 0.00005*15, 0.23*1.2),  # 小直线 (原来是0.016, 0.00005, 0.25)
    "small_curve": (0.020*1.2, 0.00005*15, 0.18*1.2),  # 小弯 (原来是0.022, 0.00005, 0.2)
    "medium_curve": (0.022*1.2, 0.00005*15, 0.18*1.2),  # 中弯 (原来是0.024, 0.00005, 0.2)
    "large_curve": (0.024*1.2, 0.00005*15, 0.14*1.2),  # 大弯 (原来是0.0265, 0.00005, 0.15)
    "large_straight": (0.012*1.2, 0.0005*15, 0.28*1.2)   # 大直线 (原来是0.013, 0.0005, 0.3)
}

def get_pid_params(error, kanbujian):
    """根据误差和可见性选择PID参数。

    Args:
        error: 当前角度误差。
        kanbujian: 路径是否不可见（1为不可见，0为可见）。

    Returns:
        tuple: (kp_z, kp_y, kd_z) PID参数。
    """
    abs_error = abs(error)
    if kanbujian:
        if 33.5 < abs_error <= 51:
            print("看不见：小大弯！"* 8)
            return PID_PARAMS["small_curve_invisible"]
        elif 51 < abs_error <= 62:
            print("看不见：中弯！"* 8)
            return PID_PARAMS["medium_curve_invisible"]
        elif 62 < abs_error <= 64:
            print("看不见：极弯！"* 8)
            return PID_PARAMS["extreme_curve_invisible"]
        elif abs_error > 64:
            print("看不见：大极弯！"* 8)
            return PID_PARAMS["large_extreme_curve_invisible"]
        else:
            print("看不见：默认大直线！")
            print("看不见：默认大直线！")
            print("看不见：默认大直线！")
            return PID_PARAMS["large_straight"]  # 添加默认返回值
    else:
        if 30 < abs_error <= 34:
            print("小直线！")
            print("小直线！")
            print("小直线！")
            return PID_PARAMS["small_straight"]
        elif 34 < abs_error <= 55:
            print("小弯，一般情况！")
            print("小弯，一般情况！")
            print("小弯，一般情况！")
            return PID_PARAMS["small_curve"]
        elif 55 < abs_error <= 60:
            print("中弯！")
            print("中弯！")
            print("中弯！")
            return PID_PARAMS["medium_curve"]
        elif abs_error > 60:
            print("大弯！")
            print("大弯！")
            print("大弯！")
            return PID_PARAMS["large_curve"]
        print("大直线！")
        print("大直线！")
        print("大直线！")
        return PID_PARAMS["large_straight"]


def do_turn(vel_pub, direction, odom_yaw, zhuanxiang_time, xunxian_input):
    """执行转向并发布速度命令。

    Args:
        vel_pub: 速度发布者。
        direction: 转向方向（TurnDirection枚举）。
        odom_yaw: 当前偏航角。
        zhuanxiang_time: 转向计时。
        xunxian_input: 转向指令序列。

    Returns:
        int: 更新后的转向计时。
    """
    vel = Twist()
    if direction == "stop":
        vel.linear.x = 0
        vel.angular.z = 0
        vel_pub.publish(vel)
        return zhuanxiang_time

    angular_z = 0.6 if direction == "left" else -0.6
    yaw_adjust = math.pi / 2 if direction == "left" else -math.pi / 2
    goal_yaw = odom_yaw + yaw_adjust
    goal_yaw = (goal_yaw + math.pi) % (2 * math.pi) - math.pi  # 归一化到[-pi, pi]

    vel.angular.z = angular_z
    delta_yaw = odom_yaw - goal_yaw
    delta_yaw = (delta_yaw + math.pi) % (2 * math.pi) - math.pi  # 归一化到[-pi, pi]

    vel.linear.x = 0.04
    vel.linear.y = 0.035 * (1 if direction == "left" else -1) if abs(delta_yaw) > 1.4 else 0

    vel_pub.publish(vel)
    if abs(odom_yaw - goal_yaw) < 0.4:
        vel.linear.x = 0
        vel.angular.z = 0
        vel_pub.publish(vel)
        return zhuanxiang_time + 1
    return zhuanxiang_time


def get_line_bin_img(img, low_rh, high_rh, low_gs, high_gs, low_bv, high_bv):
    if img is None:
        raise ValueError("输入图像为空")
    lower_array = np.array([low_rh, low_gs, low_bv])
    upper_array = np.array([high_rh, high_gs, high_bv])
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_mask = cv2.inRange(hsv, lowerb=lower_array, upperb=upper_array)
    resize_bin_img = cv2.resize(img_mask, (pw, ph), interpolation=cv2.INTER_AREA)
    resize_img = cv2.resize(img, (pw, ph), interpolation=cv2.INTER_AREA)
    return resize_img, resize_bin_img, img_mask

def find_center_edge_line(img):
    if img is None:
        raise ValueError("输入图像为空")
    # 检查输入图像是否为二值图像
    if len(img.shape) != 2 or img.dtype != np.uint8:
        raise ValueError("输入图像必须是单通道的二值图像 (值为 0 或 255)")

    height, width = img.shape
    current_edge_points_zuixiamian = []
    center_points = []
    edge_points = []
    sigle, double = 0, 0
    kanbujian = 0

    # 修改步长，适应新的摄像头位置
    step = 4  # 原来是4，现在改为5，因为摄像头位置变低，图像中的线条可能更密集
    for y in range(height - 1, -1, -step):
        white_indices = np.where(img[y] == 255)[0]
        if len(white_indices) == 0:
            continue

        diff = np.diff(white_indices)
        breaks = np.where(diff > 1)[0] + 1
        clusters = np.split(white_indices, breaks)
        mean_indices = [np.mean(cluster) for cluster in clusters]

        current_edge_points = []
        if len(mean_indices) == 1:
            sigle += 1
            edge_x = int(mean_indices[0])
            current_edge_points.append(edge_x)

            if len(center_points) > 1:
                last_center_x = center_points[-1][0]
                second_last_center_x = center_points[-2][0]
                virtual_edge_x = 0 if last_center_x < second_last_center_x else width - 1
            else:
                virtual_edge_x = width - 1 if edge_x < width // 2 else 0

            current_edge_points.append(virtual_edge_x)
            avg_index = np.mean(current_edge_points)
            new_center_point = (int(avg_index), y)
        elif len(mean_indices) > 1:
            double += 1
            for idx in mean_indices:
                current_edge_points.append(int(idx))

            if len(current_edge_points) == 2:
                if abs(current_edge_points[0] - current_edge_points[1]) < width / 3:
                    if abs(current_edge_points[0] - width // 2) > abs(current_edge_points[1] - width // 2):
                        current_edge_points = [current_edge_points[1], width - 1 if current_edge_points[1] < width // 2 else 0]
                    else:
                        current_edge_points = [current_edge_points[0], width - 1 if current_edge_points[0] < width // 2 else 0]
                avg_index = np.mean(current_edge_points)
                new_center_point = (int(avg_index), y)
            else:
                mid_x = width // 2
                left_edge_points = [pt for pt in current_edge_points if pt < mid_x]
                right_edge_points = [pt for pt in current_edge_points if pt >= mid_x]
                left_nearest = min(left_edge_points, key=lambda x: abs(x - mid_x)) if left_edge_points else 0
                right_nearest = min(right_edge_points, key=lambda x: abs(x - mid_x)) if right_edge_points else width - 1
                current_edge_points = [left_nearest, right_nearest]
                avg_index = np.mean(current_edge_points)
                new_center_point = (int(avg_index), y)

        if current_edge_points:
            current_edge_points_zuixiamian = current_edge_points
        for rp in current_edge_points:
            edge_points.append((rp, y))

        if sigle + double > 0 and sigle / (sigle + double) > 0.9:
            kanbujian = 1

        if len(center_points) == 0 or abs(new_center_point[0] - center_points[-1][0]) < width / 8:
            center_points.append(new_center_point)

    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for x, y in center_points:
        cv2.circle(vis, (x, y), 3, (0, 255, 0), -1)
    for x, y in edge_points:
        cv2.circle(vis, (x, y), 3, (0, 0, 255), -1)

    cv2.imshow("White Pixel Clusters", vis)
    cv2.waitKey(10)


    return center_points, edge_points, current_edge_points_zuixiamian, kanbujian

def get_ROI(resize_img, resize_bin_img, img_mask):
    # 摄像头位置变低，调整ROI区域，降低ROI区域的位置
    line_up, line_low = 0.58, 1  # 原来是0.6, 1
    line_up_2, line_low_2 = 0.67, 1  # 原来是0.71, 1
    H, W = resize_img.shape[:2]
    ROI_1 = resize_bin_img[int(H * line_up):int(H * line_low), :]
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    ROI_1 = cv2.erode(ROI_1, kernel_erode, iterations=1)
    ROI_1 = cv2.dilate(ROI_1, kernel_dilate, iterations=2)
    H2 = ROI_1.shape[0]
    ROI_2 = ROI_1[int(H2 * line_up_2):int(H2 * line_low_2), :]
    #cv2.imshow("ROI_1", ROI_1)
    #cv2.imshow("ROI_2", ROI_2)
    #cv2.imshow("resize_bin_img", resize_bin_img)
    #cv2.waitKey(10)
    return  ROI_1, ROI_2

def find_line(img, step=4):
    height, width = img.shape
    mid_x = width // 2
    left_yellow_points = []
    right_yellow_points = []

    for y in range(height - 1, -1, -step):
        white_indices = np.where(img[y] == 255)[0]
        if len(white_indices) == 0:
            continue

        left_white_points = white_indices[white_indices < mid_x]
        right_white_points = white_indices[white_indices >= mid_x]

        if len(left_white_points) > 0:
            left_nearest = min(left_white_points, key=lambda x: abs(x - mid_x))
            left_yellow_points.append((left_nearest, y))

        if len(right_white_points) > 0:
            right_nearest = min(right_white_points, key=lambda x: abs(x - mid_x))
            right_yellow_points.append((right_nearest, y))

    left_x_coords = [x for x, y in left_yellow_points]
    right_x_coords = [x for x, y in right_yellow_points]

    avg_left_x = np.mean(left_x_coords) if left_x_coords and len(left_yellow_points) > 2 else None
    avg_right_x = np.mean(right_x_coords) if right_x_coords and len(right_yellow_points) > 2 else None

    return avg_left_x, avg_right_x

def calculate_turn(img, state):
    turn_direction = "normal"
    if img is None or img.size == 0:
        raise ValueError("输入的图像为空")

    height, width = img.shape[0], img.shape[1]
    step = 15
    all_blue_points = []
    prev_mean_idx = None

    for col in range(width - 1, -1, -step):
        white_indices = np.where(img[:, col] == 255)[0]
        if len(white_indices) == 0:
            continue
        mean_idx = int(np.mean(white_indices)) if len(white_indices) > 1 else white_indices[0]
        if prev_mean_idx is None or abs(mean_idx - prev_mean_idx) <= height // 5:
            all_blue_points.append((col, mean_idx))
            prev_mean_idx = mean_idx

    if len(all_blue_points) >= (width // step) * 0.85:
        if state.results["left"] > state.results["right"]:
            turn_direction = "left"
            print("计算左转")
            print("计算左转")
            print("计算左转")
        elif state.results["left"] < state.results["right"]:
            turn_direction = "right"
            print("计算右转")
            print("计算右转")
            print("计算右转")
        else:
            all_blue_points.sort(key=lambda point: point[0])
            total_points = len(all_blue_points)
            left_points = all_blue_points[:4]
            center_start_idx = max((total_points - 4) // 2, 4)
            center_points = all_blue_points[center_start_idx:center_start_idx + 4] if total_points > 12 else []
            right_points = all_blue_points[-4:]

            # # 可视化蓝色点和局部极大值
            # vis3 = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            # for x, y in all_blue_points:
            #     cv2.circle(vis3, (x, y), 3, (255, 0, 0), -1)
            #cv2.imshow("All Blue Points Visualization", vis3)


            def calculate_average_slope(points):
                if len(points) < 2:
                    return 0
                x_coords, y_coords = zip(*points)
                slope, _ = np.polyfit(x_coords, y_coords, 1)
                return slope

            left_slope = calculate_average_slope(left_points)
            center_slope = calculate_average_slope(center_points) if center_points else 0
            right_slope = calculate_average_slope(right_points)

            abs_left_slope, abs_center_slope, abs_right_slope = abs(left_slope), abs(center_slope), abs(right_slope)
            print(f"left_slope: {left_slope}, center_slope: {center_slope}, right_slope: {right_slope}")

            if left_slope != 0 and center_slope != 0 and right_slope != 0:
                if (abs(center_slope - right_slope) < 0.2 and abs_center_slope < abs_right_slope < abs_left_slope
                        and abs(center_slope - left_slope) > 0.2 and abs_left_slope > 0.2):
                    turn_direction = "left"
                    print("计算左转")
                    print("计算左转")
                    print("计算左转")
                elif (abs(center_slope - left_slope) < 0.2 and abs_center_slope < abs_left_slope < abs_right_slope
                        and abs(center_slope - right_slope) > 0.2 and abs_right_slope > 0.2):
                    turn_direction = "right"
                    print("计算右转")
                    print("计算右转")
                    print("计算右转")
                elif (abs(center_slope - left_slope) < 0.2 and abs_center_slope < abs_left_slope
                        and abs_center_slope < abs_right_slope and abs(center_slope - right_slope) < 0.2):
                    turn_direction = "stop"
                    print("计算停停停")
                    print("计算停停停")
                    print("计算停停停")
            elif len(all_blue_points) >= (width // step) * 0.95:
                turn_direction = "left"
                print("瞎****左转")


    return turn_direction

def calculate_slope(center_points, state):
    if not center_points or len(center_points) < 3:
        if not state.last_center_points or len(state.last_center_points) < 2:
            # 修改默认中心点坐标，适应新的摄像头位置
            state.last_center_points = [(320, 150), (320, 140), (320, 130), (320, 120), (320, 110), (320, 100)]
        center_points = state.last_center_points

    if not center_points:  # 额外保护
        # 修改默认中心点坐标，适应新的摄像头位置
        center_points = [(320, 150), (320, 140), (320, 130)]

    first_point = center_points[0]
    last_point = center_points[-1]
    middle_idx = min(len(center_points) - 1, int(round(len(center_points) / 3.5)))
    middle_point = center_points[middle_idx]
    angle_first_last = np.degrees(np.arctan2(last_point[1] - first_point[1], last_point[0] - first_point[0]))
    angle_first_middle = np.degrees(np.arctan2(middle_point[1] - first_point[1], middle_point[0] - first_point[0]))
    avg_first_3_x = np.mean([point[0] for point in center_points[:3]])
    #print(f"Returning from calculate_slope: {angle_first_last}, {angle_first_middle}, {avg_first_3_x}")
    return angle_first_last, angle_first_middle, avg_first_3_x

def xunxian(angle_first_last, angle_first_middle, avg_last_3_x, turn_direction,
              current_edge_points_zuixiamian, vel_pub, kanbujian, xunxian_input,state):
    """
    控制机器人运动的主函数。

    Args:
        angle_first_last: 首尾点角度（未使用）。
        angle_first_middle: 用于误差计算的角度。
        avg_last_3_x: 最后3个点平均x（未使用）。
        turn_direction: 当前转向指令。
        current_edge_points_zuixiamian: 边缘点列表。
        vel_pub: 速度发布者。
        kanbujian: 路径是否不可见。
        xunxian_input: 转向指令序列。
    """

    vel = Twist()
    zhuanxiang_time = 0

    # 计算角度误差  右上（↗）：负角度。
    #左下（↙）：负角度。 直上（↑）：-90°, "右歪 -90 左歪 90，摆头找直
    state.error = -90 + angle_first_middle if angle_first_middle > 0 else 90 + angle_first_middle
    print(f"误差: {state.error}, 转向: {turn_direction}, 偏航角: {state.odom_yaw}, 转向计时: {zhuanxiang_time}")

    # 更新转向指令
    if turn_direction == "left" or turn_direction == "right" or turn_direction == "stop":
        if zhuanxiang_time <= len(xunxian_input) - 2:
            turn_direction = xunxian_input[zhuanxiang_time]
        elif zhuanxiang_time > len(xunxian_input) - 2:
            zhuanxiang_time = len(xunxian_input)
            turn_direction = "stop"
    # 处理转向或停止
    if turn_direction == "right":
        zhuanxiang_time = do_turn(vel_pub, "right", state.odom_yaw, zhuanxiang_time, xunxian_input)
        print("执行右转")
        print("执行右转")
        print("执行右转")
        print("执行右转")
    elif turn_direction == "left":
        zhuanxiang_time = do_turn(vel_pub, "left", state.odom_yaw, zhuanxiang_time, xunxian_input)
        print("执行左转")
        print("执行左转")
        print("执行左转")
        print("执行左转")
    elif turn_direction == "stop":
        vel_pub.publish(vel)
        if not state.sound_finish:
            playsound('完成人质解救工作.mp3')
            state.sound_finish = 1
        print("执行停车 ")
        print("执行停车 ")
        print("执行停车 ")
        print("执行停车 ")
    else:
        # PID控制直线运动
        kp_z, kp_y, kd_z = get_pid_params(state.error, kanbujian)
        # 调整PID参数，适应新的摄像头位置
        kp_z *= 1  # 原来是1.2，降低一点比例系数
        kp_y *= 1   # 原来是15，降低一点横向控制增益
        kd_z *= 1
        # kp_z = 0.6  # 原来是1.2，降低一点比例系数
        # kp_y = 1   # 原来是15，降低一点横向控制增益
        # kd_z = 1.45  # 原来是1.2，降低一点微分系数

        vel.angular.z = kp_z * state.error - kd_z * state.delat_v_z
        state.delat_v_z = vel.angular.z - state.last_v_z
        state.last_v_z = vel.angular.z

        # 使用自定义速度或默认速度
        if hasattr(state, 'custom_speed'):
            vel.linear.x = state.custom_speed
        else:
            # 调整前进速度，适应新的摄像头位置
            vel.linear.x = 0.32  # 原来是0.36，降低一点速度以增加稳定性

        if state.results is None:
            print("警告: state.results 为 None，重新初始化")
            state.results = {"left": 0, "right": 0}
        if state.results.get("left", 0) > 5 or state.results.get("right", 0) > 5:
            vel.linear.x *= 1

        # 计算横向误差
        error_y = 0
        if current_edge_points_zuixiamian is not None and len(current_edge_points_zuixiamian) >= 2:
            left, right = current_edge_points_zuixiamian[:2]
            if left not in [0, 639] and right not in [0, 639]:
                error_y = (left + right) / 2 - 320
        else:
            print("current_edge_points_zuixiamian 无效或不足2个点，跳过横向误差计算")
        vel.linear.y = kp_y * error_y * 0.0005 if error_y != 0 else 0
        print(f"横向误差: {error_y}")

    vel_pub.publish(vel)

def xunxian_flag_callback(data,state):
    global xunxian_flag
    state.xunxian_flag = int(data.data)

def scan_callback(scan,state):
    count = int(scan.scan_time / scan.time_increment)
    for i in range(count):
        degree = math.degrees(scan.angle_min + scan.angle_increment * i)
        if 88 < degree < 92:
            temp = 0
            while (i + temp < count) and (math.isinf(scan.ranges[i + temp]) or scan.ranges[i + temp] == 0):
                temp += 1
            if i + temp < count:
                state.distance_90 = float(scan.ranges[i + temp])
        elif 108 < degree < 112:
            temp = 0
            while (i + temp < count) and (math.isinf(scan.ranges[i + temp]) or scan.ranges[i + temp] == 0):
                temp += 1
            if i + temp < count:
                state.distance_110 = float(scan.ranges[i + temp])
        elif 68 < degree < 72:
            temp = 0
            while (i + temp < count) and (math.isinf(scan.ranges[i + temp]) or scan.ranges[i + temp] == 0):
                temp += 1
            if i + temp < count:
                state.distance_70 = float(scan.ranges[i + temp])

def callback_read_current_position(data,state):
    state.odom_x = data.pose.pose.position.x
    state.odom_y = data.pose.pose.position.y
    quaternion = (data.pose.pose.orientation.x, data.pose.pose.orientation.y,
                  data.pose.pose.orientation.z, data.pose.pose.orientation.w)
    euler = euler_from_quaternion(quaternion)
    state.odom_yaw = euler[2]

def main():
    # 添加命令行参数解析
    parser = argparse.ArgumentParser(description='巡线程序')
    parser.add_argument('--start', action='store_true', help='启动巡线程序')
    parser.add_argument('--stop', action='store_true', help='停止巡线程序')
    parser.add_argument('--debug', action='store_true', help='启用调试模式，显示更多图像')
    parser.add_argument('--speed', type=float, default=0.32, help='设置机器人前进速度 (默认: 0.32)')
    args = parser.parse_args()

    try:
        state = RobotState()
        rospy.init_node('detector', anonymous=True)
        vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        rospy.Subscriber('/xunxian_flag', String, lambda data: xunxian_flag_callback(data, state), queue_size=1)
        rospy.Subscriber("/scan", LaserScan, lambda scan: scan_callback(scan, state), queue_size=1)
        rospy.Subscriber('/after_final_and_amcl_odom', Odometry, lambda data: callback_read_current_position(data, state), queue_size=1)

        # 添加信号处理器，处理Ctrl+C
        def signal_handler(sig, frame):
            print("\n捕获到Ctrl+C，正在停止机器人...")
            state.stop(vel_pub)
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        print("按Ctrl+C可以随时停止程序")

        # 如果指定了--stop参数，则立即停止机器人并退出
        if args.stop:
            print("停止机器人")
            state.stop(vel_pub)
            return

        # 如果未指定--start参数，提示用户如何使用并退出
        if not args.start:
            print("使用方法:")
            print("  启动巡线: python xunxian.py --start [--debug] [--speed 0.32]")
            print("  停止机器人: python xunxian.py --stop")
            return

        xunxian_input = ["stop"]
        goal_x, goal_y = 1.8, 0.15
        kongzheng_flag = 0

        # 使用命令行参数中的速度
        robot_speed = args.speed
        print(f"设置机器人速度为: {robot_speed}")

        # 是否启用调试模式
        debug_mode = args.debug
        if debug_mode:
            print("调试模式已启用，将显示更多图像")

        while not rospy.is_shutdown():
            if state.xunxian_flag == 1 and not state.sound_finish:
                capture = cv2.VideoCapture("/dev/video0")
                if not capture.isOpened():
                    raise RuntimeError("无法打开摄像头 /dev/video0")
                else:
                   while True:
                        time_start = time.time()
                        if not kongzheng_flag:
                            for _ in range(3):
                                ret, _ = capture.read()
                                if not ret:
                                    raise RuntimeError("初始化摄像头时读取帧失败")
                            kongzheng_flag = 1

                        ret, origin_image = capture.read()
                        if not ret:
                            print("无法捕获帧")
                            break
                        origin_image = cv2.flip(origin_image, 1)
                        height, width, channels = origin_image.shape  # 检查原始图像

                        # 修改HSV阈值，适应新的摄像头位置
                        resize_img, resize_bin_img, img_mask = get_line_bin_img(origin_image, 0, 120, 0, 75, 70, 255)
                        height, width = resize_bin_img.shape
                        print(f"resize_bin_img 的 Height: {height}, Width: {width}, Channels: {channels}")

                        # 在调试模式下显示更多图像
                        if debug_mode:
                            cv2.imshow("Original Image", origin_image)
                            cv2.imshow("Binary Image", resize_bin_img)

                        ROI_1, ROI_2 = get_ROI(resize_img, resize_bin_img, img_mask)
                        print("获取兴趣区成功")

                        # 在调试模式下显示ROI区域
                        if debug_mode:
                            cv2.imshow("ROI_1", ROI_1)
                            cv2.imshow("ROI_2", ROI_2)

                        center_points, edge_points, current_edge_points_zuixiamian, kanbujian = find_center_edge_line(ROI_1)
                        print("获取车道线中心点和边缘成功")
                        if len(center_points) > 3:
                            state.last_center_points = center_points
                        print(f"center_points: {center_points}")
                        #print(f"current_edge_points: {current_edge_points_zuixiamian}")

                        if not state.duizhun_finisah:
                            print("没有对准")
                            avg_left_x, avg_right_x = find_line(ROI_2)
                            avg_left_x = avg_left_x if avg_left_x is not None else 0
                            avg_right_x = avg_right_x if avg_right_x is not None else 639

                            white_center = (avg_left_x + avg_right_x) / 2
                            print(f"white_center: {white_center}")
                            vel = Twist()
                            vel.angular.z = 0
                            vel.linear.x = 0
                            if white_center < 310:
                                print("1")
                                vel.linear.y = -0.06
                            elif white_center > 330:
                                print("2")
                                vel.linear.y = 0.06
                            else:
                                print("3")
                                state.duizhun_finisah = 1
                            vel_pub.publish(vel)
                            print("4")

                        else:
                            print("对准成功")
                            zhuanxiang = calculate_turn(ROI_1,state)
                            angle_first_last, angle_first_middle, avg_first_3_x = calculate_slope(center_points, state)
                            print("计算斜率成功")

                            # 使用命令行参数中的速度
                            state.custom_speed = robot_speed

                            xunxian(angle_first_last, angle_first_middle, avg_first_3_x, zhuanxiang, current_edge_points_zuixiamian,
                                    vel_pub, kanbujian, xunxian_input, state)

                        time2 = time.time()
                        FPS = 1 / (time2 - time_start)
                        print(f"FPS: {FPS}")

                        xunxian_ting = (state.distance_90 >= 0.5 and state.distance_110 >= 0.5 and state.distance_70 >= 0.5
                                        and (state.distance_90 + state.distance_110 + state.distance_70) < 4)
                        print(f"distance_70 :{state.distance_70}, distance_90 : {state.distance_90}, distance_110 : {state.distance_110}")

                        if abs(state.odom_x - goal_x) < 0.4 and -0.05 < state.odom_y - goal_y < 0.3:
                            print('00000000000000')
                        if abs(state.odom_x - goal_x) < 0.4 and -0.05 < state.odom_y - goal_y < 1 and xunxian_ting:
                            vel = Twist()
                            vel_pub.publish(vel)
                            if not state.sound_finish:
                                playsound('完成人质解救工作.mp3')
                                state.sound_finish = 1
                           # print(f"{state.odom_x}, {state.odom_y}, {xunxian_ting}")
                            capture.release()
                            break

                        # 检查键盘输入，按ESC键或q键退出
                        key = cv2.waitKey(1) & 0xFF
                        if key == 27 or key == ord('q'):  # ESC键或q键
                            print("用户终止程序")
                            # 停止机器人
                            state.stop(vel_pub)
                            capture.release()
                            cv2.destroyAllWindows()
                            rospy.signal_shutdown("用户终止程序")
                            return

                        rospy.sleep(0.05)

    except Exception as e:
        print(f"运行时错误: {e}")
        rospy.signal_shutdown("程序异常退出")
    finally:
        # 确保在任何情况下都停止机器人
        try:
            state.stop(vel_pub)
            cv2.destroyAllWindows()
        except:
            pass


def start_xunxian(cap,global_vars):
    try:
        state = RobotState()
        vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        rospy.Subscriber("/scan", LaserScan, lambda scan: scan_callback(scan, state), queue_size=1)
        rospy.Subscriber('/after_final_and_amcl_odom', Odometry, lambda data: callback_read_current_position(data, state), queue_size=1)

        xunxian_input = ["stop"]
        goal_x, goal_y = 1.8, 0.15
        kongzheng_flag = 0

        while not rospy.is_shutdown():
            if global_vars.xunxian_flag == 1 and not state.sound_finish:


                while True:
                    time_start = time.time()
                    if not kongzheng_flag:
                        for _ in range(3):
                            ret, _ = cap.read()
                            if not ret:
                                raise RuntimeError("初始化摄像头时读取帧失败")
                        kongzheng_flag = 1

                    ret, origin_image = cap.read()
                    if not ret:
                        print("无法捕获帧")
                        break
                    origin_image = cv2.flip(origin_image, 1)
                    height, width, channels = origin_image.shape  # 检查原始图像

                    # 修改HSV阈值，适应新的摄像头位置
                    resize_img, resize_bin_img, img_mask = get_line_bin_img(origin_image, 0, 120, 0, 75, 70, 255)
                    height, width = resize_bin_img.shape
                    print(f"resize_bin_img 的 Height: {height}, Width: {width}, Channels: {channels}")
                    # origin_img, bin_img, mask = new_get_yellow_lane_bin_img(yuan_image, 0, 105, 0, 80, 140, 255)
                    #origin_img, bin_img, mask = new_get_yellow_lane_bin_img(yuan_image, 0, 255, 0, 90, 80, 255)
                    # cv2.imshow("resize_bin_img.jpg", resize_bin_img)
                    ROI_1, ROI_2 = get_ROI(resize_img, resize_bin_img, img_mask)
                    print("获取兴趣区成功")
                    center_points, edge_points, current_edge_points_zuixiamian, kanbujian = find_center_edge_line(ROI_1)
                    print("获取车道线中心点和边缘成功")
                    if len(center_points) > 3:
                        state.last_center_points = center_points
                    print(f"center_points: {center_points}")
                    #print(f"current_edge_points: {current_edge_points_zuixiamian}")

                    if not state.duizhun_finisah:
                        print("没有对准")
                        avg_left_x, avg_right_x = find_line(ROI_2)
                        avg_left_x = avg_left_x if avg_left_x is not None else 0
                        avg_right_x = avg_right_x if avg_right_x is not None else 639

                        white_center = (avg_left_x + avg_right_x) / 2
                        print(f"white_center: {white_center}")
                        vel = Twist()
                        vel.angular.z = 0
                        vel.linear.x = 0
                        if white_center < 310:
                            print("1")
                            vel.linear.y = -0.06
                        elif white_center > 330:
                            print("2")
                            vel.linear.y = 0.06
                        else:
                            print("3")
                            state.duizhun_finisah = 1
                        vel_pub.publish(vel)
                        print("4")

                    else:
                        print("对准成功")
                        zhuanxiang = calculate_turn(ROI_1,state)
                        angle_first_last, angle_first_middle, avg_first_3_x = calculate_slope(center_points, state)
                        print("计算斜率成功")
                        xunxian(angle_first_last, angle_first_middle, avg_first_3_x, zhuanxiang, current_edge_points_zuixiamian,
                                vel_pub, kanbujian, xunxian_input,state)

                    time2 = time.time()
                    FPS = 1 / (time2 - time_start)
                    print(f"FPS: {FPS}")

                    xunxian_ting = (state.distance_90 >= 0.5 and state.distance_110 >= 0.5 and state.distance_70 >= 0.5
                                    and (state.distance_90 + state.distance_110 + state.distance_70) < 4)
                    print(f"distance_70 :{state.distance_70}, distance_90 : {state.distance_90}, distance_110 : {state.distance_110}")

                    if abs(state.odom_x - goal_x) < 0.4 and -0.05 < state.odom_y - goal_y < 0.3:
                        print('00000000000000')
                    if abs(state.odom_x - goal_x) < 0.4 and -0.05 < state.odom_y - goal_y < 1 and xunxian_ting:
                        vel = Twist()
                        vel_pub.publish(vel)
                        if not state.sound_finish:
                            playsound('完成人质解救工作.mp3')
                            state.sound_finish = 1
                        # print(f"{state.odom_x}, {state.odom_y}, {xunxian_ting}")
                        cap.release()
                        break

                    # 检查键盘输入，按ESC键或q键退出
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27 or key == ord('q'):  # ESC键或q键
                        print("用户终止程序")
                        # 停止机器人
                        state.stop(vel_pub)
                        cv2.destroyAllWindows()
                        rospy.signal_shutdown("用户终止程序")
                        return

                    rospy.sleep(0.1)

    except Exception as e:
        print(f"运行时错误: {e}")
        rospy.signal_shutdown("程序异常退出")
    finally:
        # 确保在任何情况下都停止机器人
        try:
            state.stop(vel_pub)
            cv2.destroyAllWindows()
        except:
            pass

if __name__ == '__main__':