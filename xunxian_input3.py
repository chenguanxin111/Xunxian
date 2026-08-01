import cv2
import numpy as np
from PIL import Image as PImage
import sys
import math
import rospy
from geometry_msgs.msg import Twist
import os
from std_msgs.msg import String
import signal
from sklearn.linear_model import LinearRegression
from collections import deque
import time
global slope
global state
global line_count_near, left_near, right_near, line_count_last, left_near_las
global bu
ph = 360  #图片高度
pw = 640  #图片宽度
RESULT_ROW = 480
RESULT_COL = 640
USED_ROW = 480
USED_COL = 640

X_qujibian = np.zeros((RESULT_ROW, RESULT_COL), dtype=np.uint16)
Y_qujibian = np.zeros((RESULT_ROW, RESULT_COL), dtype=np.uint16)

# 没用到， 不会产生新的线程本身，只是在当前线程里循环等待并调用回调
def thread_job():
    rospy.spin()

# 没用到，相机畸变矫正函数
def image_perspective_init( ):
    cameraMatrix = np.array([[417.27599, 0.0, 322.66696],
                             [0.0, 416.23499, 229.45303],
                             [0.0, 0.0, 1.0]])
    distCoeffs = np.array([-0.317235, 0.095383, 0.001505, 0.000861, 0.0])
    move_xy = [0, 0]

    fx = cameraMatrix[0][0] # x方向焦距
    fy = cameraMatrix[1][1] # y方向焦距
    ux = cameraMatrix[0][2] # x方向主点坐标（图像中心坐标）
    uy = cameraMatrix[1][2] # y方向主点坐标
    k1 = distCoeffs[0]  # 径向畸变系数1
    k2 = distCoeffs[1]  # 径向畸变系数2
    k3 = distCoeffs[4]  # 径向畸变系数3
    p1 = distCoeffs[2]  # 切向畸变系数1
    p2 = distCoeffs[3]  # 切向畸变系数2

    move_x = move_xy[0]
    move_y = move_xy[1]

    for i in range(RESULT_ROW-1, 0, -1):
        for j in range(-move_x, RESULT_COL - move_x):
            # 1. 把像素坐标(j, i)转换为相机坐标系下的归一化坐标(xCorrected, yCorrected)
            xCorrected = (j - ux) / fx
            yCorrected = (i - uy) / fy
            # 2. 计算径向畸变
            r2 = xCorrected * xCorrected + yCorrected * yCorrected
            deltaRa = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
            deltaRb = 1.0 / 1.0
            # 3. 计算切向畸变
            deltaTx = 2.0 * p1 * xCorrected * yCorrected + p2 * (r2 + 2.0 * xCorrected * xCorrected)
            deltaTy = p1 * (r2 + 2.0 * yCorrected * yCorrected) + 2.0 * p2 * xCorrected * yCorrected
            # 4. 计算畸变后的像素坐标(xDistortion, yDistortion)
            xDistortion = xCorrected * deltaRa * deltaRb + deltaTx
            yDistortion = yCorrected * deltaRa * deltaRb + deltaTy
            xDistortion = xDistortion * fx + ux
            yDistortion = yDistortion * fy + uy
            # 5. 将畸变后的像素坐标(xDistortion, yDistortion)存储在X_qujibian和Y_qujibian数组中
            X_qujibian[i][j] = int(xDistortion)
            Y_qujibian[i][j] = int(yDistortion)
    finish_flag = 1
    return finish_flag

# 这个函数将在接收到SIGINT时被调用
def signal_handler(sig, frame):
    print('您按下了 Ctrl+C!')
    # 在这里执行任何清理工作
    # 例如，释放资源、关闭文件等
    sys.exit(0)

# 在程序开始时设置信号处理器
signal.signal(signal.SIGINT, signal_handler)

count = 0   #没用到
last_hei = 0    #没用到
count_1 = 0  #没用到
list_hei = []   #没用到
i = 0   #没用到

# hsv阈值
low = [0, 0, 180]# 阈值
high = [255, 65, 255]

slope = 0   #没用到
angle = 0   #没用到

xunxian_flag  = 1   #是否循线
global crossroads
# vel = Twist()
crossroads = False

# ？
def xunxian_flag_callback(data):
    global xunxian_flag
    xunxian_flag = int(data.data)

def Normalize(data, scale):
    """
    归一化-->[0.0, 1.0]
    :param data: 一组数据
    :param scale: 缩放倍率
    :return: [-0.5, 0.5]*scale
    """
    mx = max(data)  # 获取数据中的最大值
    mn = min(data)  # 获取数据中的最小值
    m = (mx - mn)/2 + mn  # 计算数据的中间值
    data_return = []
    # 对每个数据进行归一化处理，并乘以指定的缩放因子
    for i in data:
        if ((mx - mn) != 0):
            data_return.append(((float(i) - m) / (mx - mn) + 0.5) * scale)
        else:
            data_return.append(0)
    return data_return

# 没用到的图像处理函数，包含了图像的高斯模糊、颜色空间转换和颜色过滤等步骤
def detection(img):
    #cv2.imshow('frame', img)  # 在窗口中显示原始图像
    frameBGR = cv2.GaussianBlur(img, (7, 7), 0)  # 对图像进行高斯模糊，平滑图像以降低噪声
    hsv = cv2.cvtColor(frameBGR, cv2.COLOR_BGR2HSV)  # 将BGR颜色空间转换为HSV颜色空间

    colorLow = np.array(low)  # 从变量low中获取颜色的下界
    colorHigh = np.array(high)  # 从变量high中获取颜色的上界
    mask = cv2.inRange(hsv, colorLow, colorHigh)  # 使用颜色的上下界创建一个二值掩模，HSV抠图

    mask = cv2.resize(mask, (320, 240))# 缩小

    #cv2.imwrite("result.jpg",mask)  # 如果需要，将抠图结果保存为图像文件
    #cv2.imshow("result1", mask)  # 在窗口中显示抠图结果


def Get_Hist(img):
    """
        计算直方图数组
    """
    img = np.sum(img, axis=0)  # 沿着垂直方向对图像进行求和，得到一维数组
    img.reshape(img.size, 1)  # 将数组形状修改为 (img.size, 1)，但此处没有保存修改后的结果
    hist = np.int32(np.around(Normalize(img, 255)))  # 将一维数组归一化到 [0, 255] 范围，并将元素转换为整
    list = []
    for num in img:
        list.append(num/255)
    img = np.zeros((hist.size, hist.size, 3))  # 创建一个全零图像，用于绘制直方图
    bins = np.arange(hist.size).reshape(hist.size, 1)  # 直方图中各bin的顶点位置，构建二维数组

    pts = np.column_stack((bins, hist))  # 将 bins 和 hist 合并为一个二维数组
    cv2.polylines(img, [pts], False, (0, 255, 0))  # 使用多边形线段连接直方图的顶点，绘制直方图曲线
    img = np.flipud(img)  # 上下翻转图

    return img, hist

def Get_Hist_For_Dotted(oir_img):

    """
        计算直方图数组
    """
    img = oir_img
    img = np.sum(img, axis=0)
    img_sum = img
    img.reshape(img.size, 1)
    hist = np.int32(np.around(Normalize(img, 255)))  # 对数组进行归一化，得到直方图
    list = []
    for num in img:
        list.append(num/255)
    print("每列白色像素数：list=",list)
    img = np.zeros((len(list), len(list), 3))  # 创建用于绘制直方图的全0图像
    bins = np.arange(len(list)).reshape(len(list), 1)  # 直方图中各bin的顶点位置
    pts = np.int64(np.column_stack((bins, list)))
    cv2.polylines(img, [pts], False, (0, 255, 0))
    img = np.flipud(img)
    #cv2.imshow("try3",img)
    #cv2.waitKey(1)
    return img_sum, hist #这个img为白色像素含量折线图


def create_video_front(frame):  # 保存图片用
    current_dir = os.path.dirname(os.path.abspath(__file__))
    loca=time.strftime('%Y-%m-%d-%H-%M)')
    new_name = str(loca)
    new_folder_path = os.path.join(current_dir, new_name)
    if not os.path.exists(new_folder_path):
        os.mkdir(new_folder_path)
    num=len(os.listdir(new_folder_path))+1
    imgname=str(num)+".jpg"
    turefilename=os.path.join(new_folder_path,imgname)
    cv2.imwrite(turefilename, frame)

# 此函数重定义
def new_get_yellow_lane_bin_img(frame, low_rh, high_rh, low_gs, high_gs, low_bv, high_bv):


    lower_array = np.array([low_rh, low_gs, low_bv])  # 第二个参数调节亮度
    upper_array = np.array([high_rh, high_gs, high_bv])
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # 使用HSV进行车道线识别
    mask = cv2.inRange(hsv, lowerb=lower_array, upperb=upper_array)


    _img = mask

    small_img = np.zeros((pw, ph))  # 创建全0图像
    small_img = cv2.resize(_img, (pw, ph), small_img, cv2.INTER_AREA)

    small_img = small_img.astype(np.uint8)
    retval, bin_img = cv2.threshold(small_img, 125, 255, cv2.THRESH_BINARY)

    small_img = small_img.astype(np.float32)
    img_3chanel = cv2.cvtColor(small_img, cv2.COLOR_GRAY2BGR)
    origin_img = np.zeros((pw, ph, 3))  # 创建全0图像
    origin_img = cv2.resize(frame, (pw, ph), origin_img, cv2.INTER_AREA)
    # cv2.waitKey(10)
    return origin_img, bin_img,mask

# 没用到
def new_fliter_left_right_bias(hist_array):
    global line_up
    left = right = 0
    len_right = len_left = 0
    mn_width = 25.0
    mn_height = 55.0
    mn_area = 2000.0
    mn_distance = 40.0
    """
        滤波器函数，过滤直方图中的干扰
        :param hist_array:        直方图数组
        :return:脉冲数量，最左边的脉冲峰顶的索引位置，最右边脉冲峰顶的索引位置，过滤后的图像波形
    """

    left_distance = right_distance = 0
    shape = list(hist_array)
    # 计算脉冲个数
    pulse_flag = 0  # 波形连续标志
    pulse_height = []
    pulse_index = []
    # 1 对直方图进行波形切割，记录了波峰的高度和索引
    for i in range(len(shape)):
        if pulse_flag and (shape[i] > 25 and i < len(shape) - 1):
            _pulse_height.append(shape[i])
            _pulse_index.append(i)
        elif pulse_flag and ((shape[i] <= 25) or (i == (len(shape) - 1))):
            pulse_flag = 0  # 波形分割
            pulse_height.append(_pulse_height)
            pulse_index.append(_pulse_index)
        elif shape[i] > 0:
            pulse_flag = 1
            _pulse_height = []
            _pulse_index = []
            _pulse_height.append(shape[i])
            _pulse_index.append(i)

    pulse_point_preprocess = []
    pulse_height_preprocess = []
    # 2 根据脉冲高度和面积，过滤掉干扰脉冲
    for i, points in enumerate(pulse_height):
        height = max(points)
        area = sum(points)
        width = len(points)
        if (width >= mn_width and (area >= mn_area or height >= mn_height)):
            pulse_height_preprocess.append(points)
            pulse_point_preprocess.append(pulse_index[i])
    # 对过滤后的脉冲波形进行绘图
    __array = np.array([0] * len(shape))
    for points in pulse_point_preprocess:
        for ind in points:
            __array[ind] = shape[ind]
    __array.reshape((1, len(shape)))
    img = np.zeros((len(shape), len(shape), 3))  # 创建用于绘制直方图的全0图像
    bins = np.arange(len(shape)).reshape(len(shape), 1)  # 作为索引，0~len()
    pts = np.column_stack((bins, __array))
    cv2.polylines(img, [pts], False, (0, 255, 0))
    hist_img = np.flipud(img)  # 生成滤波后的波形图

    #print(pulse_point_preprocess)
    for sublist in pulse_point_preprocess:
        first = sublist[0]  # 获取子列表的第一个元素
        last = sublist[-1]  # 获取子列表的最后一个元素
        if first < 240:
            left = first
        if last > 240:
            right = pw - last
    if 0 < left < right:
        right = 0
    elif 0 < right < left:
        left = 0

    #print(left, right)

    # 3 对于过滤后的脉冲进行左右双峰的定位
    if len(pulse_point_preprocess) == 1:  # 单峰
        points = pulse_point_preprocess[0]
        if points[0] < 240:
            len_left = len(points)
        elif points[0] > 240:
            len_right = len(points)
        left_distance = right_distance = pulse_point_preprocess[0][int(len(points) / 2)]

    elif len(pulse_point_preprocess) == 2:  # 双峰
        # 方法2：取峰顶中心点
        points = pulse_point_preprocess[0]
        left_distance = pulse_point_preprocess[0][int(len(points) / 2)]
        len_left = len(points)
        points = pulse_point_preprocess[-1]
        right_distance = pulse_point_preprocess[-1][int(len(points) / 2)]
        len_right = len(points)

    #print(len_left, len_right)
    elif len(pulse_point_preprocess) > 2:  # 多峰，从左向右，从右向左取最高的2个
        left_ind = 0
        right_ind = len(pulse_point_preprocess) - 1
        left_mx = []
        left_mx_ind = []
        right_mx = []
        right_mx_ind = []
        while left_ind < right_ind:
            points = pulse_height_preprocess[left_ind]
            if sum(points) > sum(left_mx):
                left_mx = points
                left_mx_ind = pulse_point_preprocess[left_ind]
            points = pulse_height_preprocess[right_ind]
            if sum(points) > sum(right_mx):
                right_mx = points
                right_mx_ind = pulse_point_preprocess[right_ind]
            left_ind += 1
            right_ind -= 1
            pass
        if left_ind == right_ind:
            points = pulse_height_preprocess[right_ind]
            if sum(points) > sum(right_mx):
                right_mx = points
                right_mx_ind = pulse_point_preprocess[right_ind]

        # 多峰变双峰
        # 方法2 求左侧最高峰的中心点，右侧最高峰的中心点
        left_distance = int(sum(left_mx_ind) / len(left_mx_ind))
        right_distance = int(sum(right_mx_ind) / len(right_mx_ind))

    return len(pulse_point_preprocess), left_distance, right_distance, hist_img, left, right, len_left, len_right


def new_get_results(yuan_image):
    global line_count_near, line_up,last_green_points
    line_up = 0.6
    line_low = 1
    line_up_2 = 0.71
    line_low_2= 1
    # origin_img, bin_img, mask = new_get_yellow_lane_bin_img(yuan_image, 0, 105, 0, 80, 140, 255)  # 获取缩放后的原图与二值图
    origin_img, bin_img, mask = new_get_yellow_lane_bin_img(yuan_image, 0, 120, 0, 65, 80, 255)
    #origin_img, bin_img, mask = new_get_yellow_lane_bin_img(yuan_image, 0, 255, 0, 90, 80, 255) # 获取缩放后的原图与二值图
    # cv2.imshow("bin_img.jpg", bin_img)

    H = origin_img.shape[0]
    W = origin_img.shape[1]
    bin_img_rectangle_ROI = bin_img[int(H * line_up):int(H * line_low), :]  # 获取二值图的ROI区 原图下边0.85到0.99
    #cv2.imshow("roi_image", bin_img_rectangle_ROI)
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    bin_img_rectangle_ROI = cv2.erode(bin_img_rectangle_ROI, kernel_erode, iterations=1)
    bin_img_rectangle_ROI = cv2.dilate(bin_img_rectangle_ROI, kernel_dilate, iterations=2)
    H2 = bin_img_rectangle_ROI.shape[0]
    bin_img_rectangle_ROI_2 = bin_img_rectangle_ROI[int(H2 * line_up_2):int(H2 * line_low_2), :] 
    #cv2.imshow("bin_img_rectangle_ROI", bin_img_rectangle_ROI)
    #cv2.imshow("bin_img_rectangle_ROI_2", bin_img_rectangle_ROI_2)
    #cv2.imshow("bin_img", bin_img)
    green_points, red_points, current_red_points_zuixiamian = find_white_pixel_indices(bin_img_rectangle_ROI_2)
    if len(green_points) > 3:
        last_green_points = green_points
    else:
        last_green_points = last_green_points
    #cv2.waitKey(10)
    return green_points, red_points,last_green_points,bin_img_rectangle_ROI,bin_img_rectangle_ROI_2,current_red_points_zuixiamian
def calculate_metrics(green_points,last_green_points,bin_img_rectangle_ROI):
    print(len(green_points))
    if len(green_points) < 3:
        if len(last_green_points)<2:
            last_green_points=[(244, 62), (246, 57), (247, 52), (249, 47), (252, 42), (253, 37)]
        green_points = last_green_points
        #raise ValueError("Not enough points to calculate slopes and averages")


    # 获取第一个、最后一个和中间的绿色点
    first_point = green_points[0]
    last_point =  green_points[-1]
    middle_point = green_points[len(green_points) // 2]

    # 计算第一个和最后一个绿色点的斜率并转换为角度
    if first_point[0] != last_point[0]:
        slope_first_last = (last_point[1] - first_point[1]) / (last_point[0] - first_point[0])
        angle_first_last = np.degrees(np.arctan(slope_first_last))
    else:
        angle_first_last = 90.0 if (last_point[1] - first_point[1]) > 0 else -90.0
    middle_point = green_points[round(len(green_points) /3.5)]
    # 计算第一个和中间一个绿色点的斜率并转换为角度
    if first_point[0] != middle_point[0]:
        
        slope_first_middle = (middle_point[1] - first_point[1]) / (middle_point[0] - first_point[0])
        angle_first_middle = np.degrees(np.arctan(slope_first_middle))
    else:
        angle_first_middle = 90.0 if (middle_point[1] - first_point[1]) > 0 else -90.0

    # 获取最后3个绿色点的x坐标并计算平均值
    last_3_x_coords = [point[0] for point in green_points[:3]]
    avg_last_3_x = np.mean(last_3_x_coords)

    return angle_first_last, angle_first_middle, avg_last_3_x


def new_get_yellow_lane_bin_img(frame, low_rh, high_rh, low_gs, high_gs, low_bv, high_bv):


    lower_array = np.array([low_rh, low_gs, low_bv])  # 第二个参数调节亮度（注释可能有误，实则V）
    upper_array = np.array([high_rh, high_gs, high_bv])
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # 使用HSV进行车道线识别
    mask = cv2.inRange(hsv, lowerb=lower_array, upperb=upper_array)


    _img = mask

    small_img = cv2.resize(_img, (pw, ph),  cv2.INTER_AREA)

    small_img = small_img.astype(np.uint8)
    retval, bin_img = cv2.threshold(small_img, 125, 255, cv2.THRESH_BINARY)

    small_img = small_img.astype(np.float32)
    img_3chanel = cv2.cvtColor(small_img, cv2.COLOR_GRAY2BGR)
    origin_img = cv2.resize(frame, (pw, ph),  cv2.INTER_AREA)
    
    return origin_img, bin_img,mask

previous_frame_red_points = []  # 全局变量保存上一帧的红色点信息



history = []
def blue_state(img):
    global state,results,history
# 检查图像是否为
    if img is None or img.size == 0:
        raise ValueError("输入的图像为空")

    # 初始化存储蓝色点坐标的列表
    left_x_coords, left_y_coords = [], []
    center_x_coords, center_y_coords = [], []
    right_x_coords, right_y_coords = [], []
    all_blue_points = []

    height, width = img.shape[0], img.shape[1]
    step = 15

    # 遍历图像列
    prev_mean_idx = None
    for col in range(width - 1, -1, -step):
        white_indices = np.where(img[:, col] == 255)[0]
        if len(white_indices) == 0:
            continue

        # 计算平均索引
        mean_idx = int(np.mean(white_indices)) if len(white_indices) > 1 else white_indices[0]

        # 滤波
        if prev_mean_idx is None or abs(mean_idx - prev_mean_idx) <= height // 5:
            all_blue_points.append((col, mean_idx))
            prev_mean_idx = mean_idx
    # 检查蓝色点数量
    if len(all_blue_points) <= (width // step) * 0.85:
        blue = len(all_blue_points) 
        print("blue",blue)
        return

    # 对所有蓝色点按x坐标进行排序
    all_blue_points.sort(key=lambda point: point[0])


    # angle_points = []
    # def calculate_slope(point1, point2):
    #     x_diff = point2[0] - point1[0]
    #     y_diff = point2[1] - point1[1]
    #     return y_diff / x_diff if x_diff != 0 else 0

    # for i in range(1, len(all_blue_points) - 1):
    #     slope1 = calculate_slope(all_blue_points[i - 1], all_blue_points[i])
    #     slope2 = calculate_slope(all_blue_points[i], all_blue_points[i + 1])
    #     if abs(slope2 - slope1) > 0.2:
    #         angle_points.append(all_blue_points[i])
    # if angle_points != []:
    #     for x, y in angle_points:
    #         if x < width // 2:
    #             results["right"] += 1
    #             print("前瞻右转")
    #         else:
    #             results["left"] += 1
    #             print("前瞻左转")

    # 分配点到左、中、右三块
    total_points = len(all_blue_points)
    left_points = all_blue_points[:4]
    center_start_idx = max((total_points - 4) // 2, 4)
    center_points = all_blue_points[center_start_idx:center_start_idx + 4] if total_points > 12 else []
    right_points = all_blue_points[-4:]

    

    for x, y in left_points:
        left_x_coords.append(x)
        left_y_coords.append(y)
    for x, y in center_points:
        center_x_coords.append(x)
        center_y_coords.append(y)
    for x, y in right_points:
        right_x_coords.append(x)
        right_y_coords.append(y)

    # 计算平均斜率
    def calculate_average_slope(x_coords, y_coords):
        
        if len(x_coords) < 2:
            return 0
        slopes = []
        for i in range(len(x_coords) - 1):
            x_diff = x_coords[i + 1] - x_coords[i]
            y_diff = y_coords[i + 1] - y_coords[i]
            if x_diff != 0:
                slopes.append(y_diff / x_diff)
        return sum(slopes) / len(slopes) if slopes else 0

    average_left_slope = calculate_average_slope(left_x_coords, left_y_coords)
    average_center_slope = calculate_average_slope(center_x_coords, center_y_coords) if center_points else 0
    average_right_slope = calculate_average_slope(right_x_coords, right_y_coords)

    # # 可视化蓝色点和局部极大值
    # vis3 = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # for x, y in all_blue_points:
    #     cv2.circle(vis3, (x, y), 3, (255, 0, 0), -1)
    # cv2.imshow("All Blue Points Visualization", vis3)

    left_slope, center_slope, right_slope = average_left_slope, average_center_slope, average_right_slope
    
    abs_left_slope = abs(left_slope)
    abs_center_slope = abs(center_slope)
    abs_right_slope = abs(right_slope)
    print("left_slope",left_slope)
    print("center_slope",center_slope)
    print("right_slope",right_slope)
    if left_slope != 0 and center_slope!=0 and right_slope!=0 :
        if abs(center_slope - right_slope) < 0.2 and abs_center_slope < abs_right_slope < abs_left_slope and abs(center_slope - left_slope) > 0.2 and abs_left_slope > 0.2: 
            state = "right"  # 左转
            print("history",history)
            
            if len(history)<7:
                history.append("right")
            else:
                for i in range(len(history),1,-1):
                   
                    history[i-1] = history[i-2]
                print("history",history)
                history[0] = "right"
            if results["left"] + results["right"]>=7:
                if results[history[-1]] > 0:
                    results[history[-1]] = results[history[-1]] -1
                    results["right"] = results["right"] + 1
                else:
            
                    results["right"] = results["right"]
            else:
                results["right"] = results["right"] +1
            print("前瞻左转")
        elif abs(center_slope - left_slope) < 0.2 and abs_center_slope < abs_left_slope < abs_right_slope and abs(center_slope - right_slope) > 0.2 and abs_right_slope>0.2:
            state =  "left"   # 右转
            if len(history)<7:
                history.append("left")
            else:
                for i in range(len(history),1,-1):
                    history[i-1] = history[i-2]
                history[0] = "left"
            if results["left"] + results["right"]>=7:
                if results[history[-1]] > 0:
                    results[history[-1]] = results[history[-1]] -1
                    results["left"] = results["left"] + 1
                else:
            
                    results["left"] = results["left"]
            else:
                results["left"] = results["left"] +1
            print("前瞻右转")
        else:
            state =  "normal"
    else:
        state = "normal"
 

def calculate_blue_slopes(img):
    # 检查图像是否为空
    global state,results
# 检查图像是否为
    

    turn_direction = None
    if img is None or img.size == 0:
            raise ValueError("输入的图像为空")
    

        # 初始化存储蓝色点坐标的列表
    left_x_coords, left_y_coords = [], []
    center_x_coords, center_y_coords = [], []
    right_x_coords, right_y_coords = [], []
    all_blue_points = []

    height, width = img.shape[0], img.shape[1]
    step = 15

    # 遍历图像列
    prev_mean_idx = None
    for col in range(width - 1, -1, -step):
        white_indices = np.where(img[:, col] == 255)[0]
        if len(white_indices) == 0:
            continue

        # 计算平均索引
        mean_idx = int(np.mean(white_indices)) if len(white_indices) > 1 else white_indices[0]

        # 滤波
        if prev_mean_idx is None or abs(mean_idx - prev_mean_idx) <= height // 5:
            all_blue_points.append((col, mean_idx))
            prev_mean_idx = mean_idx
    # 检查蓝色点数量
    if len(all_blue_points) >= (width // step) * 0.85:
        
        
        if results["left"] >= results["right"] :
            turn_direction = "left"
            print("靠近左转")
        elif results["left"] < results["right"] :
            turn_direction = "right"
            print("靠近右转")
        else:
            

            # 对所有蓝色点按x坐标进行排序
            all_blue_points.sort(key=lambda point: point[0])
            # 分配点到左、中、右三块
            total_points = len(all_blue_points)
            left_points = all_blue_points[:4]
            center_start_idx = max((total_points - 4) // 2, 4)
            center_points = all_blue_points[center_start_idx:center_start_idx + 4] if total_points > 12 else []
            right_points = all_blue_points[-4:]

            

            for x, y in left_points:
                left_x_coords.append(x)
                left_y_coords.append(y)
            for x, y in center_points:
                center_x_coords.append(x)
                center_y_coords.append(y)
            for x, y in right_points:
                right_x_coords.append(x)
                right_y_coords.append(y)

            # 计算平均斜率
            def calculate_average_slope(x_coords, y_coords):
                if len(x_coords) < 2:
                    return 0
                slopes = []
                for i in range(len(x_coords) - 1):
                    x_diff = x_coords[i + 1] - x_coords[i]
                    y_diff = y_coords[i + 1] - y_coords[i]
                    if x_diff != 0:
                        slopes.append(y_diff / x_diff)
                return sum(slopes) / len(slopes) if slopes else 0

            average_left_slope = calculate_average_slope(left_x_coords, left_y_coords)
            average_center_slope = calculate_average_slope(center_x_coords, center_y_coords) if center_points else 0
            average_right_slope = calculate_average_slope(right_x_coords, right_y_coords)

            # # 可视化蓝色点和局部极大值
            # vis3 = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            # for x, y in all_blue_points:
            #     cv2.circle(vis3, (x, y), 3, (255, 0, 0), -1)
            #cv2.imshow("All Blue Points Visualization", vis3)

            left_slope, center_slope, right_slope = average_left_slope, average_center_slope, average_right_slope
            
            abs_left_slope = abs(left_slope)
            abs_center_slope = abs(center_slope)
            abs_right_slope = abs(right_slope)
            print("left_slope",left_slope)
            print("center_slope",center_slope)
            print("right_slope",right_slope)
            if left_slope != 0 and center_slope!=0 and right_slope!=0 :
                if abs(center_slope - right_slope) < 0.2 and abs_center_slope < abs_right_slope < abs_left_slope and abs(center_slope - left_slope) > 0.2 and abs_left_slope > 0.2: 
                    turn_direction = "right"  # 左转
                    print("靠近左转")
                elif abs(center_slope - left_slope) < 0.2 and abs_center_slope < abs_left_slope < abs_right_slope and abs(center_slope - right_slope) > 0.2 and abs_right_slope>0.2:
                    turn_direction =  "left"   # 右转
                    print("靠近右转")
                elif abs(center_slope - left_slope) < 0.2 and abs_center_slope < abs_left_slope and  abs_center_slope < abs_right_slope and abs(center_slope - right_slope) < 0.2:
                    turn_direction =  "stop"   # 右转
                    print("停停停")
                    print("停停停")
                    print("停停停")
                    print("停停停")
                    print("停停停")
                    print("停停停")
                    print("停停停")
                    print("停停停")
                else:
                    if len(all_blue_points) >= (width // step) * 0.95:
                        turn_direction = "left"
                        print("瞎****左转")
                    else:
                        turn_direction = "normal"
                        print("沙也没有")

            else:
                turn_direction = "normal"
                print("沙也没有")
    else:
        turn_direction = "normal"
    return turn_direction

def determine_path(average_left_slope, average_center_slope, average_right_slope):
    global state,change_flag,results
    left_slope, center_slope, right_slope = average_left_slope, average_center_slope, average_right_slope
    
    abs_left_slope = abs(left_slope)
    abs_center_slope = abs(center_slope)
    abs_right_slope = abs(right_slope)
    print("left_slope",left_slope)
    print("center_slope",center_slope)
    print("right_slope",right_slope)
    if left_slope!= 0 and center_slope!=0 and right_slope!=0:
        if abs(center_slope - right_slope) < 0.1 and abs(center_slope - left_slope) > 0.1 and abs_left_slope > 0.2: 
            return "right"  # 左转
        elif abs(center_slope - left_slope) < 0.1 and abs(center_slope - right_slope) > 0.1 and abs_right_slope>0.2:
            return "left"   # 右转
        elif abs(center_slope - right_slope) < 0.1 and abs(center_slope - left_slope) < 0.1 and abs_center_slope<0.2:

            if state == "left" :
                return "left"    
            elif state == "right" :
                return "right"    
            elif state == "normal" :
                return "left"  
            else:
                return "else"  # 其他情况
              
    else:
        return "else"  # 其他情况



def find_white_pixel_indices(img):
    # 确定图片的高度和宽度
    current_red_points_zuixiamian =[]
    height, width = img.shape
    global kanbujian , sigle , double
    sigle , double = 0,0
    # 初始化处理结果的列表
    green_points = []  # 存储所有绿色点坐标
    red_points = []  # 存储所有红色点坐标

    # 从图片的最底部开始向上扫描，每5行处理一次
    for y in range(height - 1, -1, -4):
        
        # 找出该行中所有白色像素的索引
        white_indices = np.where(img[y] == 255)[0]
        if len(white_indices) == 0:
            continue

        # 聚合连续的索引并计算它们的平均值
        diff = np.diff(white_indices)
        breaks = np.where(diff > 1)[0] + 1
        clusters = np.split(white_indices, breaks)
        mean_indices = [np.mean(cluster) for cluster in clusters]

        # 根据情况添加红色点，并计算绿色点
        current_red_points = []
        if len(mean_indices) == 1:
            sigle += 1
            # 当前行只有一个实际聚类点
            red_x = int(mean_indices[0])
            current_red_points.append(red_x)

            # 根据绿色点的趋势决定虚拟红点的位置
            if len(green_points) > 1:
                last_green_x = green_points[-1][0]
                second_last_green_x = green_points[-2][0]
                if last_green_x < second_last_green_x:
                    virtual_red_x = 0
                else:
                    virtual_red_x = width - 1
            else:
                virtual_red_x = width - 1 if red_x < width // 2 else 0

            current_red_points.append(virtual_red_x)

            # 计算红色点和虚拟红色点的中心为绿色点
            avg_index = np.mean(current_red_points)
            new_green_point = (int(avg_index), y)




        elif len(mean_indices) > 1:
            # 当前行有多个聚类点
            double += 1
            for idx in mean_indices:
                current_red_points.append(int(idx))

            # 检查两个红点之间的距离
            if len(current_red_points) == 2:
                if abs(current_red_points[0] - current_red_points[1]) < width / 3:
                    # 舍弃靠近中央的红点，补充虚拟红点
                    if abs(current_red_points[0] - width // 2) > abs(current_red_points[1] - width // 2):
                        current_red_points = [current_red_points[1], width - 1 if current_red_points[1] < width // 2 else 0]
                    else:
                        current_red_points = [current_red_points[0], width - 1 if current_red_points[0] < width // 2 else 0]

                avg_index = np.mean(current_red_points)
                new_green_point = (int(avg_index), y)
            # else:
            #     # 如果超过两个红点，则以图像中线为中心向左右寻找两个距离中心最近的红点
            #     mid_x = width // 2
            #     sorted_points = sorted(current_red_points, key=lambda x: abs(x - mid_x))
            #     closest_points = sorted_points[:2]
            #     avg_index = np.mean(closest_points)
            #     new_green_point = (int(avg_index), y)
            else:
                # 如果超过两个红点，则以图像中线为中心向左右寻找两个距离中心最近的红点
                mid_x = width // 2
                left_red_points = [pt for pt in current_red_points if pt < mid_x]
                right_red_points = [pt for pt in current_red_points if pt >= mid_x]

                if left_red_points:
                    left_nearest = min(left_red_points, key=lambda x: abs(x - mid_x))
                else:
                    left_nearest = 0  # 补充虚拟红点

                if right_red_points:
                    right_nearest = min(right_red_points, key=lambda x: abs(x - mid_x))
                else:
                    right_nearest = width - 1  # 补充虚拟红点

                current_red_points = [left_nearest, right_nearest]
                avg_index = np.mean(current_red_points)
                new_green_point = (int(avg_index), y)

        # 保存所有红点
        if current_red_points != []:
            current_red_points_zuixiamian = current_red_points
        for rp in current_red_points:
            red_points.append((rp, y))
        
        if sigle / (sigle + double) > 0.9 :
            kanbujian = 1
        else:
            kanbujian = 0


        # 如果当前绿色点与前一个绿色点之间的距离小于图像宽度的1/6，则添加到绿色点列表
        if len(green_points) == 0 or abs(new_green_point[0] - green_points[-1][0]) < width / 8:
            green_points.append(new_green_point)

    # 可视化结果
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for x, y in green_points:
        cv2.circle(vis, (x, y), 3, (0, 255, 0), -1)  # 绿色圆点
    for x, y in red_points:
        cv2.circle(vis, (x, y), 3, (0, 0, 255), -1)  # 红色圆点

    # cv2.imshow("White Pixel Clusters", vis)
    # cv2.waitKey(10)
    #create_video_front(vis)
    return green_points, red_points,current_red_points_zuixiamian








#将这三个函数复制到一起，放到主程序中的函数为new_get_results，输入是原图，输出为图像当前位置，，左波最大位置，右波最大位置，左波宽度，右波宽度，大小都在0-480之间
def log_scale(value, min_value=0, max_value=3.6):
    """
    将0-10的值压缩至0-1，曲线变化趋势符合log函数
    """
    if value < min_value or value > max_value:
        raise ValueError("Value should be within the range [0, 10]")
    
    # 压缩0-10至一个小的范围，避免log(0)问题
    epsilon = 1e-9
    normalized_value = (value - min_value) / (max_value - min_value) * (1 - epsilon) + epsilon
    
    # 应用log函数，将范围映射到0-1
    scaled_value = np.log(normalized_value) / np.log(1 - epsilon)
    
    return scaled_value

kanbujian = 0
zhuanxiang_time = 0
def xunxian_2(angle_first_last,angle_first_middle, avg_last_3_x,zhuanxiang,current_red_points_zuixiamian):
    vel = Twist()
    global state
    global last_v_z
    global delat_v_z
    global chongci
    global results
    global kp_z 
    global kp_y 
    global kd_z 
    global odom_yaw
    global goal_yaw
    global sound_finish
    global time
    global xunxian_input
    global zhuanxiang_time
    kp_z = 0
    kp_y = 0
    kd_z = 0
    chongci = 0
    global zuozhuan_flag
    global youzhuan_flag
    
    # if angle_first_last < 0:
    #     angle_first_last = -angle_first_last
    # else : angle_first_last = angle_first_last
    # if angle_first_middle < 0:
    #     angle_first_middle = -angle_first_middle
    # else : angle_first_middle = angle_first_middle
    if angle_first_middle > 0:
        error = -90 + (angle_first_middle)
    if angle_first_middle <= 0:
        error = 90 + (angle_first_middle)
    print("result",results)
    print("error",error)
    zhijiao = 0
    print("state",state)
    print("zhuanxiang",zhuanxiang)
    print("odom_yaw",odom_yaw)

    print("zhuanxiang_time",zhuanxiang_time)
    if zhuanxiang == "left" or zhuanxiang == "right" or zhuanxiang == "stop":


        if zhuanxiang_time <= len(xunxian_input) -2:
            zhuanxiang = xunxian_input[zhuanxiang_time]
        if zhuanxiang_time > len(xunxian_input) -2:
            zhuanxiang_time =len(xunxian_input)
            zhuanxiang = "stop"
        

    
    if zhuanxiang == "left" or zuozhuan_flag ==1 and zhuanxiang_time <= len(xunxian_input) -2:

        print("右转") 
        print("右转") 
        print("右转") 
        print("右转") 
        print("右转") 
        print("goal_yaw",goal_yaw)
        vel.linear.x = 0
        vel.linear.y = 0
        if zuozhuan_flag == 0:
            
            goal_yaw = odom_yaw - 3.14159/2
            if goal_yaw < -3.14:
                goal_yaw = goal_yaw +3.14*2

            print("goal_yaw",goal_yaw)
            zuozhuan_flag = zuozhuan_flag+1
            vel_puber.publish(vel)
        if zuozhuan_flag ==1:
            if abs(odom_yaw-goal_yaw)<0.4:
                zuozhuan_flag == 2
                vel.linear.x = 0
                vel.linear.y = 0
                vel.angular.z =0
                results["left"] = 0
                results["right"] = 0
                zhuanxiang_time = zhuanxiang_time + 1
                zuozhuan_flag =0
                vel_puber.publish(vel)
                
            else:
                vel.angular.z = -0.6
                delat_xuanzuhan = odom_yaw-goal_yaw
                if abs(delat_xuanzuhan)>math.pi:
                    if delat_xuanzuhan<0:
                        delat_xuanzuhan+=math.pi*2
                    else :
                        delat_xuanzuhan-=math.pi*2
                # delat_xuanzuhan = math.atan2(math.sin(odom_yaw-goal_yaw), math.cos(odom_yaw-goal_yaw))
                # vel.linear.x = 0.0 + abs(math.sin(delat_xuanzuhan)*0.1)
                # vel.linear.y = -abs(math.cos(delat_xuanzuhan)*0.1)
                vel.linear.x = 0.04
                if abs(delat_xuanzuhan)>1.4:

                    vel.linear.y = -0.035
                else:
                    vel.linear.y = 0
                #vel_puber.publish(vel)
                vel_puber.publish(vel)
                
            
            #time.sleep(0.1)
    elif zhuanxiang == "right" or youzhuan_flag ==1 and zhuanxiang_time <= len(xunxian_input) -2:
        print("左转") 
        print("左转") 
        print("左转") 
        print("左转") 
        print("左转") 
        print("goal_yaw",goal_yaw)
        vel.linear.x = 0
        vel.linear.y = 0
        if youzhuan_flag == 0:
            
            goal_yaw = odom_yaw + 3.14159/2
            if goal_yaw > 3.14:
                goal_yaw = goal_yaw -3.14*2
            print("goal_yaw",goal_yaw)
            youzhuan_flag = youzhuan_flag+1
            vel_puber.publish(vel)
        if youzhuan_flag ==1:
            if abs(odom_yaw-goal_yaw)<0.4:
                
                vel.linear.x = 0
                vel.linear.y = 0
                vel.angular.z =0
                results["left"] = 0
                results["right"] = 0
                zhuanxiang_time = zhuanxiang_time + 1
                youzhuan_flag =0
                vel_puber.publish(vel)
            else:
                
                vel.angular.z = 0.6
                delat_xuanzuhan = odom_yaw-goal_yaw
                if abs(delat_xuanzuhan)>math.pi:
                    if delat_xuanzuhan<0:
                        delat_xuanzuhan+=math.pi*2
                    else :
                        delat_xuanzuhan-=math.pi*2
                # delat_xuanzuhan = math.atan2(math.sin(odom_yaw-goal_yaw), math.cos(odom_yaw-goal_yaw))
                # vel.linear.x = 0.03 + abs(math.sin(delat_xuanzuhan)*0.05)
                # vel.linear.y = abs(math.cos(delat_xuanzuhan)*0.05)
                vel.linear.x = 0.04
                if abs(delat_xuanzuhan)>1.4:

                    vel.linear.y = 0.035
                else:
                    vel.linear.y = 0
                vel_puber.publish(vel)
    elif zhuanxiang == "stop" :
       # if (abs(odom_x-goal_x)<0.6)and abs(odom_y-goal_y)<1:
        
            vel = Twist()
            vel_puber.publish(vel)
            if sound_finish == 0:
                # playsound('完成人质解救工作.mp3')
                sound_finish = 1
            print("停车") 
            print("停车") 
            print("停车") 
            print("停车") 
            print("停车") 
            print(odom_x,odom_y)
            print(xunxian_ting)
            
    else:
        if zhijiao == 0: 
            if kanbujian == 1:       #视野中大部分只有一根线
                if 51 > abs(error) > 33.5:     #小大弯                                 #一般情况下，极弯error大于51，小于51即小大弯
                    kp_z = 0.024#
                    kp_y = 0.00005#0.00005
                    kd_z = 0.22#0.12
                    print("看不见：小大弯！")
                elif 62 > abs(error) > 51:
                    kp_z = 0.026        #中弯
                    kp_y = 0.00005#0.00005
                    kd_z = 0.2#0.12
                    print("看不见：中弯！")
                elif 64 > abs(error) > 62:
                    kp_z = 0.029          #极弯
                    kp_y = 0.00005#0.00005
                    kd_z = 0.1#0.12
                    print("看不见：极弯！")
                elif abs(error) > 64:          #大极弯
                    kp_z = 0.033              
                    kp_y = 0.00005#0.00005
                    kd_z = 0.25#0.12
                    print("看不见：大极弯！")
            else:
                if 34 > abs(error) > 30:       #小直线
                    kp_z = 0.016#
                    kp_y = 0.00005#0.00005
                    kd_z = 0.25#0.12
                    print("小直线！")
                elif 55 > abs(error) > 34:     #小弯，一般情况
                    kp_z = 0.022#
                    kp_y = 0.00005#0.00005
                    kd_z = 0.2#0.12
                    print("小弯，一般情况！")
                elif 60 > abs(error) > 55:     #中弯 
                    kp_z = 0.024
                    kp_y = 0.00005#0.00005
                    kd_z = 0.2#0.12
                    print("中弯！")
                elif abs(error) > 60:          #大弯
                    kp_z = 0.0265#
                    kp_y = 0.00005#0.00005
                    kd_z = 0.15#0.12
                    print("大弯！")
                else:                          #大直线
                    chongci = 0.1
                    kp_z = 0.013#0.023
                    kp_y = 0.0005#0.00005
                    kd_z = 0.3#0.12
                    print("大直线！")

            kp_z = kp_z*1.2
            kp_y = kp_y*15
            kd_z = kd_z*1.2
            vel.angular.z = kp_z * error - kd_z * delat_v_z
            delat_v_z = vel.angular.z - last_v_z 
            
            #vel.linear.x = 0.51-vel.angular.z*0.08 + chongci*0
            vel.linear.x = 0.36

            if results["left"] or results["right"] > 5:
                vel.linear.x = vel.linear.x * 1


            error_y = 0
            vel.linear.y = 0

            if current_red_points_zuixiamian != []:
                if current_red_points_zuixiamian[0] != 0 and current_red_points_zuixiamian[0] != 639:
                    if current_red_points_zuixiamian[1] != 0 and current_red_points_zuixiamian[1] != 639:
                        error_y = (current_red_points_zuixiamian[0] + current_red_points_zuixiamian[1])/2 
            if error_y != 0: # 车偏左为负

                error_y_contorl = error_y - 320
                print("error_y",error_y_contorl)
                vel.linear.y = kp_y * error_y_contorl*0.0005
            else:
                vel.linear.y = 0
            


        elif zhijiao == 1:
            vel.angular.z = 0
            vel.linear.x = 0
            vel.linear.y = 0

    # print("vel.linear.x ",vel.linear.x)
    # print("vel.linear.y ",vel.linear.y)
    # print("vel.angular.z ",vel.angular.z)
    vel_puber.publish(vel)
def check(red):
    li=[]
    green_poi=[]
    for i in red:
        if x[0]==0 or x[0]==479:continue
        green_poi.append(i)
    if len(green_poi)<5:return 0

    for i in range(0,len(green_poi)-3):
        x,y=green_poi[i]
        xx,yy=green_poi[i+3]
        slope_first_last = (yy - y) / (xx - x+0.00001)
        angle_first_last = np.degrees(np.arctan(slope_first_last))
        li.append(angle_first_last)
    print(li)
    for i in li:
        if abs(i)>(20*3.14/180):return 0
    return li[1]/abs(li[1])
def callback_read_current_position(data):  # amcl定位回调
        global odom_yaw
        # global odom_x
        # global odom_y
        
        odom_x = data.pose.pose.position.x      # 在地图中的x,y坐标 
        odom_y = data.pose.pose.position.y
        qx = data.pose.pose.orientation.x            # 当前位姿四元数
        qy = data.pose.pose.orientation.y
        qz = data.pose.pose.orientation.z
        qw = data.pose.pose.orientation.w
        quaternion = (qx, qy, qz, qw)
        euler = euler_from_quaternion(quaternion)    # 获取当前角度
        odom_yaw = euler[2]

history = deque(maxlen=3)
def find_white_centers(bin_img):
        # 获取图像的高度和宽度
        height, width = bin_img.shape
        
        # 分割图像为左右两部分
        left_img = bin_img[:, :width // 2]
        right_img = bin_img[:, width // 2:]
        
        def find_center(binary_img):
            # 查找所有白色像素的坐标
            center_y, center_x = (0,0),(0,0)
            white_pixels = np.argwhere(binary_img == 255)
            if white_pixels.size == 0:
                return None
            
            # 计算白色区域的中心

            center_y, center_x = np.mean(white_pixels, axis=0)
            return int(center_x), int(center_y)
        
        # 找到左右两部分白色区域的中心
        left_center = find_center(left_img)
        right_center = find_center(right_img)
        
        return left_center, right_center
# Helper function to maintain the sum constraint in the results dictionary
def update_results(results, key, value):
    global history
    while results["left"] + results["right"] + value > 7:
        if history:
            oldest_key = history.popleft()
            results[oldest_key] -= 1
    results[key] += value
    history.append(key)

    # Check if there are three consecutive same values
    if len(history) == 3 and all(k == key for k in history):
        results[key] += 1

def scan_callback(scan):
    # 左侧60 90 120度雷达距离
    global distance_left
    global distance_left_qian
    global distance_left_hou
    count = int(scan.scan_time / scan.time_increment)
    for i in range(count):
        degree = math.degrees(scan.angle_min + scan.angle_increment * i)
        if 88 < degree < 92:
            temp = 1
            while math.isinf(scan.ranges[i+temp])or scan.ranges[i+temp]==0:temp+=1
            distance_left = float(scan.ranges[i])
            temp = 0
        elif 108 < degree < 112:
            temp = 1
            while math.isinf(scan.ranges[i+temp])or scan.ranges[i+temp]==0:temp+=1
            distance_left_hou = float(scan.ranges[i])
            temp = 0
          
        elif 68 < degree < 72:
            temp = 1
            while math.isinf(scan.ranges[i+temp])or scan.ranges[i+temp]==0:temp+=1
            distance_left_qian = float(scan.ranges[i])
            temp = 0


def find_and_draw_nearest_white_points(img, step=4):
    # 确定图片的高度和宽度
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

    # # 可视化结果
    # vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # for x, y in left_yellow_points:
    #     cv2.circle(vis, (x, y), 3, (0, 255, 255), -1)  # 黄色圆点
    # for x, y in right_yellow_points:
    #     cv2.circle(vis, (x, y), 3, (0, 255, 255), -1)  # 黄色圆点
   # cv2.imshow("Visualization", vis)
   # cv2.waitKey(10)

    left_x_coords = [x for x, y in left_yellow_points]
    right_x_coords = [x for x, y in right_yellow_points]

    avg_left_x = np.mean(left_x_coords) if left_x_coords else None
    avg_right_x = np.mean(right_x_coords) if right_x_coords else None

    if len(left_yellow_points) <= 2:
        avg_left_x = None
    if len(right_yellow_points) <= 2:
        avg_right_x = None

    return avg_left_x, avg_right_x
  

def odom_x_callback(data):
    global odom_x
    odom_x = float(data.data)

def odom_y_callback(data):
    global odom_y
    odom_y = float(data.data)

from tf.transformations import euler_from_quaternion, quaternion_from_euler
from nav_msgs.msg import Path, Odometry
from sensor_msgs.msg import LaserScan
# from playsound import playsound
#yuan_image = cv2.imread("banyuan.mp4") #   zhixing2  323*321 zuozhuan2 439*441 24480  youzhuan 605*441
#capture = cv2.VideoCapture("/dev/ucar_video")
xunxian_flag_sub =  rospy.Subscriber('/xunxian_flag', String, xunxian_flag_callback, queue_size=1)
rospy.init_node('detector', anonymous=True)
vel_puber = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
leida_left = rospy.Subscriber("/scan", LaserScan, scan_callback, queue_size=1)
odom_x_sub =  rospy.Subscriber('/odom_x', String, odom_x_callback, queue_size=1) 
odom_y_sub =  rospy.Subscriber('/odom_y', String, odom_y_callback, queue_size=1) 
current_pose = rospy.Subscriber(
        '/after_final_and_amcl_odom',
        Odometry,   # 消息类型
        callback_read_current_position,
        queue_size=1)
bu = 0
zhuanxiang = 0
odom_yaw = 0

goal_yaw = 0


last_v_z = 0
delat_v_z = 0
global H
global W
global last_green_points
global change_flag 
global results
global youzhuan_flag
global zuozhuan_flag
duizhun_finisah = 0
zuozhuan_flag = 0
youzhuan_flag = 0
red_y_error = 0


results = {"left": 0, "right": 0}

change_flag = 0


state = "else"
last_green_points = []
# finish_flag = image_perspective_init()

goal_x = 1.8 
goal_y = 0.15
sound_finish = 0
xunxian_ting = 0
kong_flag = 0
xunxian_input = ['stop'] #z左 是 right
#xunxian_input = ["left","left","stop"] #z左 是 right
while not rospy.is_shutdown():
    if xunxian_flag == 1 and sound_finish == 0:
        capture = cv2.VideoCapture("/dev/video0")
        if capture.isOpened():
            while True:
                time1 = time.time()
                if kong_flag == 0:
                    for i in range(3):
                        ret, yuan_image = capture.read()
                    kong_flag = 1
                if duizhun_finisah == 0:

                    ret, yuan_image = capture.read()
                    height, width, channels = yuan_image.shape


                    print(f"Height: {height}, Width: {width}, Channels: {channels}")

                    green_points, red_points, last_green_points, bin_img_rectangle_ROI, bin_img_rectangle_ROI_2, current_red_points_zuixiamian = new_get_results(
                        yuan_image)
                    print("current_red_points", current_red_points_zuixiamian)

                    avg_left_x, avg_right_x = find_and_draw_nearest_white_points(bin_img_rectangle_ROI_2)
                    if avg_left_x == None:
                        avg_left_x = 0
                    if avg_right_x == None:
                        avg_right_x = 640 - 1
                    print("avg_left_x, avg_right_x", avg_left_x, avg_right_x)
                    white_center = (avg_left_x + avg_right_x) / 2
                    if white_center < 310:
                        vel = Twist()
                        vel.angular.z = 0
                        vel.linear.x = 0
                        vel.linear.y = -0.06
                        vel_puber.publish(vel)
                    elif white_center > 330:
                        vel = Twist()
                        vel.angular.z = 0
                        vel.linear.x = 0
                        vel.linear.y = 0.06
                        vel_puber.publish(vel)
                    else:
                        duizhun_finisah = 1
                else:


                    ret, yuan_image = capture.read()  # img 就是一帧图片
                    height, width, channels = yuan_image.shape  

                    green_points, red_points, last_green_points, bin_img_rectangle_ROI,bin_img_rectangle_ROI_2,current_red_points_zuixiamian = new_get_results(yuan_image)
                    print("current_red_points",current_red_points_zuixiamian)
                    #blue_state(bin_img_rectangle_ROI)


                    #average_left_slope, average_center_slope, average_right_slope = calculate_blue_slopes(bin_img_rectangle_ROI_2)
                    zhuanxiang = calculate_blue_slopes(bin_img_rectangle_ROI_2)
                    #zhuanxiang = determine_path(average_left_slope, average_center_slope, average_right_slope)
                # IfRightAngle, LeftOrRight,PointList = new_get_results_713_0203(show)

                    angle_first_last, angle_first_middle, avg_last_3_x = calculate_metrics(green_points,last_green_points,bin_img_rectangle_ROI)
                    
                #print(angle_first_last, angle_first_middle, avg_last_3_x)
                    # print("PointList",PointList)
                    # print("IfRightAngle",IfRightAngle)
                    # print("LeftOrRight",LeftOrRight)
                    
                    
                    #xunxian_2(angle_first_last,angle_first_middle, avg_last_3_x,IfRightAngle, LeftOrRight,PointList)
                    if duizhun_finisah == 1:
                        #print()
                        xunxian_2(angle_first_last,angle_first_middle, avg_last_3_x,zhuanxiang,current_red_points_zuixiamian)


                #xunxian_without_pid(bias,left_near, right_near, k_left, k_righ t)
                time2 = time.time()
                FPS = 1 / (time2-time1)
                print("FPS:",FPS)
                if distance_left >= 0.5 and distance_left_hou >= 0.5 and distance_left_qian >= 0.5 and (distance_left + distance_left_hou + distance_left_qian) < 4:
                    xunxian_ting = 1 
                    print(distance_left_qian,distance_left,distance_left_hou)
                else:
                    xunxian_ting=0
                if (abs(odom_x-goal_x)<0.4)and( -0.05<odom_y-goal_y< 0.3):
                    print(distance_left_qian,distance_left,distance_left_hou)
                    print('00000000000000')
                if (abs(odom_x-goal_x)<0.4)and( -0.05<odom_y-goal_y< 1) and xunxian_ting == 1:
                    vel = Twist()
                    vel_puber.publish(vel)
                    flag = 3
                    if sound_finish == 0:
                        # playsound('完成人质解救工作.mp3')
                        sound_finish = 1
                    print(odom_x,odom_y)
                    print(xunxian_ting)
      