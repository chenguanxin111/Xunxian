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
import json
import urllib.request

import cv2
import rospy
import tf2_ros
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan, Imu, Image
from cv_bridge import CvBridge
from tf.transformations import euler_from_quaternion


class LocalizationDiagnosticsV7:
    def __init__(self):
        rospy.init_node('standalone_record_node', anonymous=True)

        # 默认立即记录，避免小车已经处于 STOPPED/FAULT 时因“检测运动”条件
        # 不成立而生成只有表头的 CSV。按 s 可暂停/恢复，按 q 退出。
        self.recording = True
        self.manual_trigger = True
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
            'scan_valid_pts', 'scan_near_ratio',
            'lf_mode', 'lf_error_deg', 'lf_heading_error_deg', 'lf_center_error_px',
            'lf_center_count', 'lf_kanbujian',
            'lf_vision_valid', 'lf_lane_tracks', 'lf_pair_valid',
            'lf_pair_slope_diff', 'lf_half_width', 'lf_half_width_samples',
            'lf_far_turn', 'lf_far_confidence', 'lf_far_points',
            'lf_stop_line_detected', 'lf_stop_line_y', 'lf_stop_line_enabled', 'lf_creep_started', 'lf_stop_line_stopped',
            'lf_image_age', 'wz_sign_flip_event'
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

        # 摄像头画面录制（用于排查停止线/巡线感知）
        self.bridge = CvBridge()
        self.latest_overlay = None
        self.latest_mask = None
        self.latest_raw = None
        self.frame_dir = os.path.join(record_dir, f'frames_{timestamp}')
        os.makedirs(self.frame_dir, exist_ok=True)
        self.frame_count = 0
        self.frame_save_interval = 0.1      # 每 0.1s 保存一帧（约10fps）
        self.last_frame_save = 0.0

        self.imu_msg = None
        self.imu_last_stamp = rospy.Time(0)
        self.imu_msg_age = -1.0

        self.amcl_pose = None
        self.latest_scan = None
        self.line_status = {}
        self.last_status_poll = 0.0
        self.previous_cmd_wz = 0.0
        self.prev_stop_line = False
        self.prev_creep_started = False
        self.prev_mode = ''

        # TF buffer
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # 订阅者
        rospy.Subscriber('/cmd_vel', Twist, self.cmd_vel_cb)
        rospy.Subscriber('/odom', Odometry, self.wheel_odom_cb)
        rospy.Subscriber('/icp_odom', Odometry, self.icp_odom_cb)
        rospy.Subscriber('/imu/physically_inverted', Imu, self.imu_cb)
        rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, self.amcl_cb)
        rospy.Subscriber('/scan', LaserScan, self.scan_cb)

        # 摄像头调试话题（polyline_following_ipm_0804 节点发布）
        rospy.Subscriber('/polyline/debug/overlay', Image, self.overlay_cb, queue_size=1, buff_size=2**24)
        rospy.Subscriber('/polyline/debug/mask', Image, self.mask_cb, queue_size=1, buff_size=2**24)
        rospy.Subscriber('/usb_cam/image_raw', Image, self.raw_cb, queue_size=1, buff_size=2**24)

        # 定时器 20Hz 记录
        self.timer = rospy.Timer(rospy.Duration(0.05), self.timer_cb)

        # 终端键盘监控（stdin 不是 TTY 时跳过，避免 termios 崩溃，如 ssh 无 TTY / 管道）
        self.keyboard_enabled = False
        try:
            if sys.stdin is not None and sys.stdin.isatty():
                self.settings = termios.tcgetattr(sys.stdin)
                self.keyboard_enabled = True
        except Exception:
            self.keyboard_enabled = False

        print("==================================================")
        print(f"[数据记录节点] 已启动！保存目录: {self.csv_path}")
        print("键盘按键控制说明:")
        print("  's' / 'S' : 暂停 / 恢复记录")
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

    def overlay_cb(self, msg):
        try:
            self.latest_overlay = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception:
            pass

    def mask_cb(self, msg):
        try:
            self.latest_mask = self.bridge.imgmsg_to_cv2(msg, 'mono8')
        except Exception:
            pass

    def raw_cb(self, msg):
        try:
            self.latest_raw = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception:
            pass

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
        except Exception:
            return 0

    def _poll_line_status(self, wall_time):
        """轮询巡线 Web 状态；优先 5010 (polyline 0804)，备用 5007/5004。"""
        if wall_time - self.last_status_poll < 0.045:
            return self.line_status
        self.last_status_poll = wall_time
        for port in [5010, 5007, 5004]:
            try:
                response = urllib.request.urlopen(
                    f'http://127.0.0.1:{port}/api/status', timeout=0.03
                )
                self.line_status = json.load(response)
                break
            except Exception:
                pass
        return self.line_status

    def _check_keyboard(self):
        if not self.keyboard_enabled:
            return True
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
        lf = self._poll_line_status(wall_t)

        stop_line = bool(lf.get('stop_line_detected', False))
        stop_y = lf.get('stop_line_y', -1)
        creep = bool(lf.get('creep_started', False))
        stop_stopped = bool(lf.get('stop_line_stopped', False))
        mode = lf.get('mode', '')
        msg = lf.get('message', '')

        # ------------------ 关键节点喊一声 (大声输出日志) ------------------
        if stop_line and not self.prev_stop_line:
            print(f"\n🚨 [!!! 发现停止线 !!!] y={stop_y}px | cmd_vx={cvx:.2f} cmd_wz={cvw:.3f} | odom=({wpx:.3f}, {wpy:.3f})")
        if creep and not self.prev_creep_started:
            print(f"\n🚗 [!!! 进入停止线蠕动阶段 !!!] cmd_vx={cvx:.2f} cmd_wz={cvw:.3f} | odom=({wpx:.3f}, {wpy:.3f}) | msg='{msg}'")
        if mode and mode != self.prev_mode and self.prev_mode != '':
            print(f"\n📌 [状态切换] {self.prev_mode} -> {mode} | msg='{msg}'")

        self.prev_stop_line = stop_line
        self.prev_creep_started = creep
        self.prev_mode = mode

        # 角速度换向/摆头监测
        sign_flip = int(
            abs(self.previous_cmd_wz) > 0.05 and abs(cvw) > 0.05
            and self.previous_cmd_wz * cvw < 0
        )
        if sign_flip:
            stage_str = "【蠕动阶段摆头摇晃】" if creep else "【巡线角速度换向】"
            print(f"\n⚠️ {stage_str} cmd_wz 突变: {self.previous_cmd_wz:.3f} -> {cvw:.3f} | "
                  f"error={lf.get('error_deg')} | stop_line={stop_line} (y={stop_y})")
        self.previous_cmd_wz = cvw

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
            lf.get('mode', ''), lf.get('error_deg', ''),
            lf.get('heading_error_deg', ''), lf.get('center_error_px', ''),
            lf.get('center_count', ''), int(bool(lf.get('kanbujian', False))),
            int(bool(lf.get('vision_valid', False))),
            lf.get('lane_track_count', ''), int(bool(lf.get('lane_pair_valid', False))),
            lf.get('lane_pair_slope_diff', ''), lf.get('ipm_half_width', ''),
            lf.get('ipm_half_width_samples', ''),
            lf.get('far_turn_direction', ''), lf.get('far_turn_confidence', ''),
            lf.get('far_turn_points', ''),
            int(stop_line), stop_y, int(bool(lf.get('stop_line_enabled', False))), int(creep), int(stop_stopped),
            lf.get('image_age_s', ''), sign_flip,
        ]
        self.csv_writer.writerow(row)
        self.row_count += 1

        # 定期保存摄像头画面（overlay 含检测到停止线的红线标注）
        if wall_t - self.last_frame_save >= self.frame_save_interval:
            self.last_frame_save = wall_t
            self._save_frames(now_sec, stop_line, stop_y)

        if self.row_count % 20 == 0:
            self.csv_file.flush()
            print(f"\r[数据记录] 已记录 {self.row_count} 行 | cmd_vx={cvx:.2f} cmd_wz={cvw:.2f}", end='', flush=True)

    def _save_frames(self, ros_sec, stop_line, stop_y):
        """保存当前 overlay / mask / raw 三幅画面到 frames_ 目录。"""
        try:
            ts = f'{ros_sec:08.3f}'
            tag = 'STOP' if stop_line else 'NO'
            if self.latest_overlay is not None:
                cv2.imwrite(os.path.join(self.frame_dir, f'f{self.frame_count:05d}_{ts}_{tag}_overlay.jpg'),
                            self.latest_overlay)
            if self.latest_mask is not None:
                cv2.imwrite(os.path.join(self.frame_dir, f'f{self.frame_count:05d}_{ts}_{tag}_mask.jpg'),
                            self.latest_mask)
            if self.latest_raw is not None:
                cv2.imwrite(os.path.join(self.frame_dir, f'f{self.frame_count:05d}_{ts}_{tag}_raw.jpg'),
                            self.latest_raw)
            self.frame_count += 1
        except Exception as e:
            rospy.logwarn_throttle(2, f"保存帧失败: {e}")

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
