#!/usr/bin/env python3
"""
Standalone Diagnostic & Data Recording script adapted from record.py.
Saves motion, odometry, IMU, TF, and line following diagnostic data.
Placed in Xunxian_standalone workspace.
"""
import os
import sys
import time
import csv
import math
import select
import termios
import tty

import rospy
import tf2_ros
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan, Imu
from tf.transformations import euler_from_quaternion


class LocalizationDiagnosticsV7:
    def __init__(self):
        rospy.init_node('standalone_record_node', anonymous=True)

        self.recording = False
        self.manual_trigger = False
        self.idle_start = None
        self.row_count = 0

        # CSV 目录与文件名设置
        record_dir = os.path.expanduser('~/Xunxian_standalone/records')
        os.makedirs(record_dir, exist_ok=True)
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        self.csv_path = os.path.join(record_dir, f'diag_record_{timestamp}.csv')

        self.csv_file = open(self.csv_path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)

        # 标头
        header = [
            'ros_time', 'wall_time',
            'cmd_vx', 'cmd_vy', 'cmd_wz',
            'wheel_vx', 'wheel_vy', 'wheel_wz',
            'icp_msg_age', 'icp_healthy', 'icp_frozen_count',
            'icp_vx', 'icp_vy', 'icp_wz',
            'icp_px', 'icp_py', 'icp_pyaw',
            'icp_cov_xx', 'icp_cov_yy', 'icp_cov_aa',
            'imu_wz', 'imu_cov', 'imu_msg_age',
            'wheel_px', 'wheel_py', 'wheel_pyaw',
            'amcl_px', 'amcl_py', 'amcl_pyaw',
            'map2odom_x', 'map2odom_y', 'map2odom_yaw',
            'particle_count',
            'scan_valid_pts', 'scan_near_ratio'
        ]
        self.csv_writer.writerow(header)
        self.csv_file.flush()

        # 话题数据缓存
        self.cmd_vel = Twist()
        self.wheel_odom = None
        self.icp_odom = None
        self.icp_last_stamp = rospy.Time(0)
        self.icp_healthy = False
        self.icp_msg_age = -1.0
        self.icp_consecutive_frozen = 0
        self.prev_icp_pose = None

        self.imu_msg = None
        self.imu_last_stamp = rospy.Time(0)
        self.imu_msg_age = -1.0

        self.amcl_pose = None
        self.latest_scan = None

        # TF buffer
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # 订阅者
        rospy.Subscriber('/cmd_vel', Twist, self.cmd_vel_cb)
        rospy.Subscriber('/odom', Odometry, self.wheel_odom_cb)
        rospy.Subscriber('/icp_odom', Odometry, self.icp_odom_cb)
        rospy.Subscriber('/imu/data', Imu, self.imu_cb)
        rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, self.amcl_cb)
        rospy.Subscriber('/scan', LaserScan, self.scan_cb)

        # 定时器 20Hz 记录
        self.timer = rospy.Timer(rospy.Duration(0.05), self.timer_cb)

        # 终端键盘监控
        self.settings = termios.tcgetattr(sys.stdin)

        print("==================================================")
        print(f"[数据记录节点] 已启动！保存目录: {self.csv_path}")
        print("键盘按键控制说明:")
        print("  's' / 'S' : 手动开始 / 暂停记录")
        print("  'q' / 'Q' : 退出记录")
        print("==================================================")

    def cmd_vel_cb(self, msg):
        self.cmd_vel = msg

    def wheel_odom_cb(self, msg):
        self.wheel_odom = msg

    def icp_odom_cb(self, msg):
        self.icp_odom = msg
        self.icp_last_stamp = msg.header.stamp if msg.header.stamp.to_sec() > 0 else rospy.Time.now()

    def imu_cb(self, msg):
        self.imu_msg = msg
        self.imu_last_stamp = msg.header.stamp if msg.header.stamp.to_sec() > 0 else rospy.Time.now()

    def amcl_cb(self, msg):
        self.amcl_pose = msg

    def scan_cb(self, msg):
        self.latest_scan = msg

    def _get_tf(self, target_frame, source_frame):
        try:
            t = self.tf_buffer.lookup_transform(
                target_frame, source_frame, rospy.Time(0), rospy.Duration(0.02))
            x = t.transform.translation.x
            y = t.transform.translation.y
            _, _, yaw = euler_from_quaternion([
                t.transform.rotation.x, t.transform.rotation.y,
                t.transform.rotation.z, t.transform.rotation.w])
            return (x, y, yaw)
        except:
            return None

    def _get_scan_quality(self):
        if self.latest_scan is None:
            return 0, 0.0
        valid = [r for r in self.latest_scan.ranges
                 if r > self.latest_scan.range_min and r < self.latest_scan.range_max]
        if not valid:
            return 0, 0.0
        valid_count = len(valid)
        near = sum(1 for r in valid if r < 0.5)
        near_ratio = near / len(valid) if len(valid) > 0 else 0.0
        return valid_count, near_ratio

    def _get_particle_count(self):
        try:
            if self.amcl_pose is not None:
                return 2000
            return 0
        except:
            return 0

    def _check_keyboard(self):
        if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
            tty.setraw(sys.stdin.fileno())
            key = sys.stdin.read(1)
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
            if key in ('s', 'S'):
                self.manual_trigger = not self.manual_trigger
                self.recording = self.manual_trigger
                print(f"\n[数据记录] 手动{'开始' if self.recording else '暂停'}记录")
            elif key in ('q', 'Q'):
                rospy.signal_shutdown("用户退出")
                return False
        return True

    def timer_cb(self, event):
        if not self._check_keyboard():
            return

        now = rospy.Time.now()
        now_sec = now.to_sec()
        wall_t = time.time()

        vel = abs(self.cmd_vel.linear.x) + abs(self.cmd_vel.angular.z)
        if not self.recording and not self.manual_trigger:
            if vel > 0.01:
                self.recording = True
                self.idle_start = None
                print("\n[数据记录] 检测到运动，自动开始记录")
        elif self.recording and not self.manual_trigger:
            if vel < 0.01:
                if self.idle_start is None:
                    self.idle_start = now_sec
                elif now_sec - self.idle_start > 5.0:
                    self.recording = False
                    self.idle_start = None
                    print(f"\n[数据记录] 静止超过5秒，自动停止记录（共{self.row_count}行）")
            else:
                self.idle_start = None

        if not self.recording:
            return

        cvx = self.cmd_vel.linear.x
        cvy = self.cmd_vel.linear.y
        cvw = self.cmd_vel.angular.z

        if self.wheel_odom is not None:
            tw = self.wheel_odom.twist.twist
            wvx, wvy, wvw = tw.linear.x, tw.linear.y, tw.angular.z
        else:
            wvx = wvy = wvw = 0.0

        if self.icp_last_stamp.to_sec() > 0:
            self.icp_msg_age = (now - self.icp_last_stamp).to_sec()
        else:
            self.icp_msg_age = -1.0
        self.icp_healthy = (self.icp_msg_age >= 0 and self.icp_msg_age < 0.5)

        if self.icp_odom is not None:
            p = self.icp_odom.pose.pose.position
            q = self.icp_odom.pose.pose.orientation
            _, _, icp_yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
            icp_px, icp_py, icp_pyaw = p.x, p.y, icp_yaw

            tw = self.icp_odom.twist.twist
            icp_vx, icp_vy, icp_vw = tw.linear.x, tw.linear.y, tw.angular.z

            cov = self.icp_odom.pose.covariance
            icp_cxx = cov[0] if len(cov) > 0 else 0.0
            icp_cyy = cov[7] if len(cov) > 7 else 0.0
            icp_caa = cov[35] if len(cov) > 35 else 0.0

            if self.prev_icp_pose is not None:
                dx = abs(icp_px - self.prev_icp_pose[0])
                dy = abs(icp_py - self.prev_icp_pose[1])
                dyaw = abs(icp_pyaw - self.prev_icp_pose[2])
                if dx < 0.0005 and dy < 0.0005 and dyaw < 0.0005:
                    self.icp_consecutive_frozen += 1
                else:
                    self.icp_consecutive_frozen = 0
            self.prev_icp_pose = (icp_px, icp_py, icp_pyaw)
        else:
            icp_px = icp_py = icp_pyaw = 0.0
            icp_vx = icp_vy = icp_vw = 0.0
            icp_cxx = icp_cyy = icp_caa = 0.0
            self.icp_consecutive_frozen = 0

        if self.imu_msg is not None:
            imu_wz = -self.imu_msg.angular_velocity.z
            imu_cov = self.imu_msg.angular_velocity_covariance[8] \
                      if len(self.imu_msg.angular_velocity_covariance) > 8 else -1.0
        else:
            imu_wz = 0.0
            imu_cov = -1.0
        if self.imu_last_stamp.to_sec() > 0:
            self.imu_msg_age = (now - self.imu_last_stamp).to_sec()
        else:
            self.imu_msg_age = -1.0

        if self.wheel_odom is not None:
            p = self.wheel_odom.pose.pose.position
            q = self.wheel_odom.pose.pose.orientation
            _, _, wyaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
            wpx, wpy, wpyaw = p.x, p.y, wyaw
        else:
            wpx = wpy = wpyaw = 0.0

        amcl_tf = self._get_tf('map', 'base_footprint')
        if amcl_tf:
            ax, ay, ayaw = amcl_tf
        else:
            ax = ay = ayaw = 0.0

        map_to_odom = self._get_tf('map', 'odom')
        if map_to_odom:
            mx, my, myaw = map_to_odom
        else:
            mx = my = myaw = 0.0

        pc = self._get_particle_count()
        valid_pts, near_ratio = self._get_scan_quality()

        row = [
            now_sec, wall_t,
            cvx, cvy, cvw,
            wvx, wvy, wvw,
            self.icp_msg_age,
            1 if self.icp_healthy else 0,
            self.icp_consecutive_frozen,
            icp_vx, icp_vy, icp_vw,
            icp_px, icp_py, icp_pyaw,
            icp_cxx, icp_cyy, icp_caa,
            imu_wz, imu_cov, self.imu_msg_age,
            wpx, wpy, wpyaw,
            ax, ay, ayaw,
            mx, my, myaw,
            pc,
            valid_pts, near_ratio,
        ]
        self.csv_writer.writerow(row)
        self.row_count += 1

        if self.row_count % 20 == 0:
            self.csv_file.flush()
            print(f"\r[数据记录] 已记录 {self.row_count} 行 | cmd_vx={cvx:.2f} cmd_wz={cvw:.2f}", end='', flush=True)

    def shutdown(self):
        self.csv_file.close()
        try:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
        except Exception:
            pass
        print(f"\n\n[数据记录] 数据已保存至: {self.csv_path} (共 {self.row_count} 条记录)")


if __name__ == '__main__':
    try:
        node = LocalizationDiagnosticsV7()
        rospy.on_shutdown(node.shutdown)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
