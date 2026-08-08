# 车载参数一览表

本文档对照 `config/*.json` 逐项说明参数含义与调参方向。改 JSON 里的数字即可，重启节点生效。

---

## 1. steering.json — 巡线 PID 转向控制

| 参数 | 值 | 含义 | 调参建议 |
|------|----|------|----------|
| kp_heading_lo | 0.8 | 小方位角(直线段)转向增益 | 越大越跟线、易摆；越小越稳但可能偏 |
| kp_heading_mid | 0.8 | 中等方位角转向增益 | 中弯用，介于 lo/hi 之间插值 |
| kp_heading_hi | 0.75 | 大方位角(急弯)增益 | 越大越敢打方向 |
| hk1_deg | 5.0 | 增益分段阈值1：\|偏差\|≤此值用 lo | 阈值越小越早进大增益 |
| hk2_deg | 10.0 | 增益分段阈值2：\|偏差\|≥此值开始向 hi 靠拢 | — |
| hk3_deg | 18.0 | \|偏差\|≥此值完全用 hi 增益 | — |
| kp_center | 0.018 | 横向偏移(像素)的比例增益 | 偏太多时加大 |
| kd_heading | 0.10 | 方位角微分(变化率)阻尼 | 越大越抑制抖动，过大发滞 |
| kd_center | 0.005 | 横向偏移微分阻尼 | — |
| heading_spike_max_deg | 5.0 | 单帧方位角突变超此视为噪声并抑制；0=关闭 | 噪声大时调小 |
| ema_h | 0.35 | 方位角低通滤波系数 | 越大越跟手、噪声越多 |
| ema_c | 0.55 | 横向偏移低通滤波系数 | — |
| wz_slew | 3.0 | 角速度变化率限幅(rad/s/0.05s) | 越大转向越激进 |
| wz_max | 0.55 | 角速度绝对上限(rad/s) | 转向过猛限小 |
| speed_h_deg | 12.0 | 偏差超此值开始降速 | 弯道降速触发点 |
| speed_k_deg | 0.004 | 偏差每多1° 速度下降量 | 越大急弯降速越多 |
| kp_lat | 0.0 | 侧向平移增益（一般 0 不用） | — |
| heading_bias_deg | 0.0 | 方位角零偏补偿 | 相机装歪时用 |
| deadband_center_px | 3.0 | 横向死区：偏差小且方位角小则不转 | 防轻微抖动一直打 |
| deadband_heading_deg | 1.2 | 方位角死区 | — |

---

## 2. lane_detect.json — 车道线检测（图像处理）

| 参数 | 值 | 含义 | 调参建议 |
|------|----|------|----------|
| roi_bottom_ratio | 0.30 | ROI 只取画面下 30%（近车头） | 视野拉太远易噪 |
| min_center_pts | 3 | 中线拟合最少点数, 太少判丢线 | 抗干扰就调大 |
| min_center_span | 25.0 | 中线最小纵向跨度(px) | 太短视作噪声 |
| poly_max_resid | 20.0 | 中线拟合最大残差 | 拟合差时判失效 |
| lane_width_min | 110.0 | 车道宽下限(px) | 排除窄噪声 |
| lane_width_max | 230.0 | 车道宽上限(px) | 排除宽/双线干扰 |
| lane_half_width | 84.0 | 左右线距中线最大半宽(px) | 配对左右线用 |
| max_slope_diff | 0.55 | 左右线斜率差上限 | 防误配 |
| clean_MM_H | 12 | 轮廓最小高度(px) | — |
| clean_min_area | 40 | 轮廓最小面积(px²) | — |
| clean_min_ratio | 0.35 | 轮廓宽高比下限 | 细长条才算线 |
| lookahead_px | 120.0 | 中线前瞻距离(px)，控制取此点偏差 | 大= 提前转向 |
| track_stale_frames | 10 | 跟踪线持续无更新判失效帧数 | — |
| max_width_dev | 35.0 | 与历史平均宽最大偏差(px) | 防宽度跳变 |
| max_cluster_w | 48 | 单簇聚类最大宽度(px) | — |
| match_min_overlap | 18.0 | 上下帧匹配最小重叠(px) | — |

---

## 3. align.json — 中线平移对准（路口/方向识别前）

| 参数 | 值 | 含义 | 调参建议 |
|------|----|------|----------|
| speed | 0.04 | 前进速度上限（本阶段禁止前进来） | — |
| center_tol_px | 18.0 | 判定对准完成的横向误差容差(px) | 调小更精准 |
| heading_tol_deg | 12.0 | 判定对准完成的方位角容差(deg) | — |
| confirm_frames | 10 | 连续满足容差的帧数 | 越大越稳定 |
| kp_y | 0.0022 | 平移速度 = kp_y×横向误差 | 横移快慢 |
| kp_z | 0.0015 | 角速度 = kp_z×横向误差 | 转角纠偏 |
| wz_clamp | 0.10 | 角速度上限(rad/s) | 转向上限 |
| max_distance | 0.40 | 累计平移超此放弃进下一阶段(m) | — |
| timeout | 20.0 | 对准超时(s)进下一阶段 | — |
| hsv.low_h/high_h | 0 / 179 | H 色相范围 | 目标颜色色相 |
| hsv.low_s/high_s | 0 / 45 | S 饱和度范围 | 白线低饱和 |
| hsv.low_v/high_v | 170 / 255 | V 明度范围 | 白线很亮 |
| roi_top/bottom | 0.45 / 1.0 | ROI 上下比例 | — |
| roi_left/right | 0.0 / 1.0 | ROI 左右比例 | — |
| blur_ksize | 3 | 高斯模糊核 | — |
| erode_iter / ksize | 0 / 3 | 腐蚀次数/核 | — |
| dilate_iter / ksize | 2 / 3 | 膨胀次数/核 | — |

---

## 4. turn.json — 岔口转弯

| 参数 | 值 | 含义 | 调参建议 |
|------|----|------|----------|
| f_yaw_max_time | 0.6 | 首段纯角度校准最大耗时(s) | — |
| f_yaw_tol_deg | 1.5 | 首段到位容差(deg) | 更准调小 |
| f_yaw_kp | 1.2 | 首段角度比例增益 | — |
| f_yaw_wz | 0.15 | 首段角速度上限(rad/s) | — |
| advance_speed | 0.15 | 前进段速度(m/s) | — |
| advance_distance | 0.30 | 前进段目标距离(m) | 前冲距离 |
| advance_timeout | 10.0 | 前进段超时(s) | — |
| rotate_speed | 0.35 | 旋转速度(rad/s) | — |
| rotate_target_deg | 65.0 | 旋转目标角度(deg) | 转多少 |
| rotate_timeout | 12.0 | 旋转段超时(s) | — |
| search_rotate_wz | 0.15 | 搜索旋转角速度(rad/s) | 慢速找线 |
| search_accum_limit_deg | 85.0 | 搜索累计角度上限，超过放弃 | — |
| search_timeout | 10.0 | 搜索总超时(s) | — |
| search_confirm_frames | 8 | 找到候选需连续确认帧数 | — |

---

## 5. stopline.json — 停止线检测 / 蠕动 / 丢线恢复

| 参数 | 值 | 含义 | 调参建议 |
|------|----|------|----------|
| stop_line_roi_top_ratio | 0.80 | 停止线 ROI 扫画面下 80%~100% | — |
| stop_line_width_ratio | 0.40 | 横线像素≥宽×此值判定停止线 | 太灵敏调大 |
| stop_line_thin_ratio | 0.40 | 行白像像素占比阈值 | — |
| creep_speed | 0.10 | 蠕动速度(m/s) | — |
| creep_distance | 0.10 | 蠕动目标距离(m)，走满结束弧线 | — |
| creep_timeout | 10.0 | 蠕动最长耗时(s)，超时停车 | — |
| camera_timeout | 0.8 | 相机画面超时判定(s) | 越紧越易误停 |
| lost_creep_timeout | 2.0 | 丢线后先蠕动 2s 再进搜索 | 越大越晚搜 |
| search_rotate_wz | 0.35 | SEARCH 原地旋转角速度(rad/s) | — |
| search_confirm_frames | 8 | SEARCH 找到确认帧数 | — |
| search_first_sec | 5.0 | 第一方向搜索时长(s) | — |
| search_first_deg | 75.0 | 第一方向累计旋转角上限 | — |
| search_second_sec | 6.0 | 反向搜索时长(s) | — |
| search_second_deg | 90.0 | 反向搜索累计旋转上限 | — |

---

## 6. polyline.json — 折线巡线段

| 参数 | 值 | 含义 | 调参建议 |
|------|----|------|----------|
| target_speed | 0.2 | 直线段巡线速度(m/s) | — |
| advance_speed | 0.15 | 起始直行速度(m/s) | — |
| advance_distance | 0.50 | 起始直行目标距离(m) | 起步走多远 |
| advance_timeout | 8.0 | 起始直行超时(s) | — |
| advance_kp_yaw | 0.8 | 起始直行航向增益 | — |
| advance_wz_clamp | 0.08 | 起始直行角速度上限 | — |
| lane_stable_frames | 10 | 直线段进入前稳定帧数 | — |
| roi_wide_bottom_ratio | 0.48 | 起步宽 ROI 底部 | 起步视野开 |
| roi_tight_bottom_ratio | 0.30 | 正常窄 ROI 底部 | — |
| roi_wide_frames_after_follow | 15 | 进 LINE 后保留宽 ROI 帧数 | — |
| advance_roi_switch_dist | 0.30 | 提前切 ROI 的距离(m) | — |
| lane_bias_right_px | 10.0 | 中线基准右偏(px) | 车偏基准修正 |
| camera_timeout | 0.8 | 相机超时(s) | — |
| creep_speed | 0.10 | 蠕动速度 | — |
| creep_distance | 0.10 | 蠕动距离 | — |
| creep_timeout | 10.0 | 蠕动超时 | — |
| lost_creep_timeout | 2.0 | 丢线蠕动时长，仍丢则转弯 | — |
| stop_line_enable_delay_sec | 1.0 | 停止线检测启用延迟(s) | 起步不误判 |
| left_yaw_limit_deg | 15.0 | 左偏航角上限(deg)，超异常 | — |
| turn_advance_speed | 0.131 | 转弯前微小前进速度 | — |
| turn_drive_wz | -0.29 | 转弯角速度(rad/s, 负=右) | — |
| turn_yaw_deg | 47.0 | 转弯目标偏航角(deg) | — |
| turn_timeout | 25.0 | 转弯总超时(s) | — |
| right_trust_nx_min | 340.0 | 右线可信最近x下限(px) | — |
| right_trust_span_min | 60.0 | 右线可信纵向跨度下限 | — |
| right_trust_frames | 5 | 右线可信连续帧数 | — |
| right_trust_jitter_px | 25.0 | 右线抖动容差(px) | — |
| search_rotate_wz | -0.25 | 雷达搜索旋转角速度 | — |
| search_angle_deg | 70.0 | 雷达搜索最大角 | — |
| search_timeout | 15.0 | 雷达搜索超时 | — |
| keep_wall_dist | 0.26 | 目标侧墙距离(m) | 贴墙多远 |
| turn_trigger_dist | 0.36 | 触发转弯的前墙距离 | — |
| turn_done_dist | 0.28 | 转弯完成判定前墙距 | — |
| parallel_angle_deg | 15.0 | 与墙平行判定角(deg) | — |
| forward_speed | 0.20 | 贴墙前进速度 | — |
| turn_wz_max / min | -0.40 / -0.18 | 转弯角速上下限 | — |
| kp_turn_ang | 0.008 | 转弯角度增益 | — |
| turn_guard_dist | 0.18 | 转弯保护距离 | — |
| turn_guard_wz | -0.10 | 转弯保护角速度 | — |
| kp_parallel / ki_parallel | 0.55 / 0.35 | 平行段侧距 P/I 增益 | — |
| parallel_int_max | 0.15 | 平行段积分上限 | — |
| kp_ang_head | 0.02 | 平行段航向增益 | — |
| parallel_wz_clamp | 0.30 | 平行段角速度上限 | — |
| front_slow_dist / front_stop_dist | 0.60 / 0.50 | 前墙减速/停下距离 | — |
| front_slow_speed | 0.08 | 前墙减速最低速 | — |
| min_safe_dist | 0.13 | 全局安全距离 | — |
| safe_clamp_dist | 0.18 | 雷达制动钳制距离 | — |
| align_target_deg | 135.0 | 转向目标角 | — |
| align_stop_err_deg | 2.0 | 转向到位误差容差 | — |
| kp_align | 0.06 | 转向比例增益 | — |
| align_wz_max / min | 0.35 / 0.15 | 转向角速上限/下限 | — |
| align_timeout | 5.0 | 转向超时 | — |
| skip_prealign | true | 跳过预对齐 | — |
| left_ang_lo / hi | 25.0 / 155.0 | 左墙扇区(deg) | — |
| front_ang_lo / hi | -30.0 / 30.0 | 前墙扇区(deg) | — |
| scan_timeout | 0.6 | 雷达数据超时 | — |
| break_gap | 0.25 | 雷达断口间隙(m) 墙角用 | — |
| radar_handoff_perp_min/max | 0.25 / 0.40 | 接管左墙垂距上下限 | — |
| radar_handoff_frames | 5 | 接管连续帧数 | — |
| radar_wall_lost_frames | 12 | 左墙丢失连续帧→回退视觉 | — |
| radar_retry_cooldown | 3.0 | 回退重试雷达冷却(s) | — |

---

## 7. perspective_params.json — 图像去畸变（IPM 标定）

由 `calib_page.py` 网页标定写入，一般**不要手改**：
- image_width / image_height：输入图像尺寸
- src_points：图像上 4 个标定点（梯形四角）
- dst_points：IPM 平面映射的 4 个点（矩形四角，S=400px/m）
> 改错会导致拟定 / 拐弯失真，只能重新标定页面生成。

---

## 8. white_lane.json / white_lane_right.json / white_line.json — HSV 颜色阈值 + ROI

三个文件结构相同，分别是"主车道线 / 右侧车道线 / 巡线"配色参数：

| 参数 | 值 | 含义 | 调参建议 |
|------|----|------|----------|
| name | — | 名称（仅展示） | — |
| low_h / high_h | 42 / 179 | H 色相范围 | 目标是不同颜色则改 |
| low_s / high_s | 5 / 71 | S 饱和度范围 | 白色低饱和 |
| low_v / high_v | 116 / 255 | V 明度范围 | 亮面 |
| roi_top / bottom | 0.45 / 1.0 | ROI 上下比例 | — |
| roi_left / right | 0.0 / 1.0 | ROI 左右比例 | — |
| blur_ksize | 4 | 高斯模糊核 | — |
| erode_iter / ksize | 0 / 3 | 腐蚀 | — |
| dilate_iter / ksize | 2 / 3 | 膨胀 | — |

> 白线识别：S 调低、V 调高，追线更干净；日光/灯光不同需现场微调 HSV。