def first():
    def extract_features(window, is_padded_window):
    features = []
    feature_names = [] # for debug purposes we store the names of each feature at the same index
    
    # Assume sensor column order from SENSOR_COLUMNS
    # Indices: button=0, motor=1, touch_pos=[2,5,8,11,14], touch_press=[3,6,9,12,15]
    button_col = 0
    motor_col = 1
    touch_channel_cols = [4, 7, 10, 13, 16]  # touch_X_channel indices
    touch_pressure_cols = [3, 6, 9, 12, 15]  # touch_X_pressure indices
    touch_position_cols = [2, 5, 8, 11, 14]  # touch_X_position indices
    original_length_col = len(SENSOR_COLUMNS)  # last column is original length

    button_press_data = window[:, button_col]
    motor_data = window[:, motor_col]
    touch_pressures = window[:, touch_pressure_cols]
    touch_positions = window[:, touch_position_cols]
    # Only consider touch positions where the pressure > 0
    valid_mask = touch_pressures > 0
    touch_positions = touch_positions[valid_mask]
    original_length = int(window[0, original_length_col]) if is_padded_window else window.shape[0]

    # A 1 signalizes the button was pressed, this can happen multiple times in a window, so we first have to extract each consecutive time a button was pressed, plus the amount of samples for each time
    binary_button_press = (button_press_data > 0).astype(int)
    padded_button_press = np.concatenate(([0], binary_button_press, [0]))
    button_press_diff = np.diff(padded_button_press)
    button_press_starts = np.where(button_press_diff == 1)[0]
    button_press_ends = np.where(button_press_diff == -1)[0]
    button_press_durations = button_press_ends - button_press_starts

    def circular_range(angles):
        # 1. Sort the angles
        sorted_angles = np.sort(angles)
        # 2. Calculate gaps between neighbors
        gaps = np.diff(sorted_angles)
        # 3. Calculate the wrap-around gap (last to first)
        wrap_gap = (2 * np.pi - sorted_angles[-1]) + sorted_angles[0]
        # 4. Combine all gaps
        all_gaps = np.append(gaps, wrap_gap)
        # 5. Range is the full circle minus the biggest empty space
        return 2 * np.pi - np.max(all_gaps)
    
    def add_feature(name, value):
        features.append(value)
        feature_names.append(name)

    # Button Features
    # 1. Button pressed at all?
    # add_feature("button_pressed_any", np.any(binary_button_press))
    # # 2. How often was the button pressed (changed from 0 to 1 and vice versa) and how often was it let go
    # add_feature("button_press_count", len(button_press_starts))
    # # 3. Mean (raw + durations)
    # add_feature("button_press_mean", np.mean(binary_button_press))
    # add_feature("button_press_duration_mean", np.mean(button_press_durations) if len(button_press_durations) > 0 else 0)
    # # 4. Std Dev (raw + durations)
    # add_feature("button_press_std", np.std(binary_button_press))
    # add_feature("button_press_duration_std", np.std(button_press_durations) if len(button_press_durations) > 0 else 0)
    # # 5. Median (raw + durations)
    # add_feature("button_press_median", np.median(binary_button_press))
    # add_feature("button_press_duration_median", np.median(button_press_durations) if len(button_press_durations) > 0 else 0)
    # # 6. Max duration
    # add_feature("button_press_max_duration", np.max(button_press_durations) if len(button_press_durations) > 0 else 0)
    # # 7. Min duration
    # add_feature("button_press_min_duration", np.min(button_press_durations) if len(button_press_durations) > 0 else 0)
    # # 8. Range duration
    # add_feature("button_press_range_duration", circular_range(button_press_durations) if len(button_press_durations) > 0 else 0)
    # # 9. Total duration
    # add_feature("button_press_total_duration", np.sum(button_press_durations) if len(button_press_durations) > 0 else 0)

    # Motor Angle Features
    # 1. Mean
    # add_feature("motor_angle_mean", np.mean(motor_data))
    # # 2. Std Dev
    # add_feature("motor_angle_std", np.std(motor_data))
    # # 3. Min Max
    # add_feature("motor_angle_min", np.min(motor_data))
    # add_feature("motor_angle_max", np.max(motor_data))
    # # 4. Range
    # add_feature("motor_angle_range", circular_range(motor_data))
    # # 5. Median
    # add_feature("motor_angle_median", np.median(motor_data))
    # # 6. Q25 / Q75
    # add_feature("motor_angle_q25", np.percentile(motor_data, 25))
    # add_feature("motor_angle_q75", np.percentile(motor_data, 75))
    # # 7. Skewness / Kurtosis
    # add_feature("motor_angle_skewness", stats.skew(motor_data))
    # add_feature("motor_angle_kurtosis", stats.kurtosis(motor_data))
    # # 8. Velocity
    # if len(motor_data) > 1:
        # motor_velocity = np.diff(motor_data) * SAMPLING_RATE_HZ
        # add_feature("motor_velocity_mean", np.mean(motor_velocity))
        # add_feature("motor_velocity_std", np.std(motor_velocity))
        # add_feature("motor_velocity_max", np.max(motor_velocity))
    # else:
        # add_feature("motor_velocity_mean", 0)
        # add_feature("motor_velocity_std", 0)
        # add_feature("motor_velocity_max", 0)

    # General Touch Activity Features
    # 1. Max simultaneous touches
    active_touches = (touch_pressures > 0).astype(int)
    add_feature("max_simultaneous_touches", np.max(np.sum(active_touches, axis=1)))
    # 2. Avg active touches
    add_feature("avg_active_touches", np.mean(np.sum(active_touches, axis=1)))
    # 3. Any touch at all?
    add_feature("any_touch", np.any(active_touches))
    # 4. Total pressure across all sensors
    add_feature("total_active_touch_samples", np.sum(active_touches))
    add_feature("total_pressure", np.sum(touch_pressures))
    # 5. Max pressure
    add_feature("max_pressure", np.max(touch_pressures))
    # 5.1 Min non-zero pressure
    non_zero_pressures = touch_pressures[touch_pressures > 0]
    add_feature("min_non_zero_pressure", np.min(non_zero_pressures) if len(non_zero_pressures) > 0 else 0)
    # 6. Touch centroid
    touch_position_sin_mean = np.sin(touch_positions).mean()
    touch_position_cos_mean = np.cos(touch_positions).mean()
    touch_position_mean_angle = np.arctan2(touch_position_sin_mean, touch_position_cos_mean)
    add_feature("touch_centroid", touch_position_mean_angle % (2 * np.pi))
    # 7. Touch Spread
    touch_position_R = np.sqrt(touch_position_sin_mean**2 + touch_position_cos_mean**2)
    add_feature("touch_spread", 1 - touch_position_R)
    add_feature("touch_spread_log", -np.log(np.clip(touch_position_R, 1e-9, 1.0)))

    # Distances between each finger (including wrapping)
    def short_dist(a, b):
        return np.abs(np.arctan2(np.sin(a - b), np.cos(a - b)))
    
    i = 0
    for p1, p2 in combinations(touch_position_cols, 2):
        i += 1
        finger1_positions = window[:, p1]
        finger2_positions = window[:, p2]
        # Only consider positions where the pressure > 0
        finger1_pressures = window[:, touch_pressure_cols[touch_position_cols.index(p1)]]
        finger2_pressures = window[:, touch_pressure_cols[touch_position_cols.index(p2)]]
        valid_mask = (finger1_pressures > 0) & (finger2_pressures > 0)
        finger1_positions = finger1_positions[valid_mask]
        finger2_positions = finger2_positions[valid_mask]

        distances = short_dist(finger1_positions, finger2_positions)

        # Calculate the distance values in 5 buckets to get a better temporal understanding of what is happening at the start middle and end
        # buckets = np.array_split(distances, 5)
        # for b_idx, bucket in enumerate(buckets):
        #     has_distance = len(bucket) != 0
        #     add_feature(f"finger_distance_{i}_mean_bucket_{b_idx+1}", np.mean(bucket) if has_distance else 0)
        #     add_feature(f"finger_distance_{i}_std_bucket_{b_idx+1}", np.std(bucket) if has_distance else 0)
        #     add_feature(f"finger_distance_{i}_max_bucket_{b_idx+1}", np.max(bucket) if has_distance else 0)
        #     add_feature(f"finger_distance_{i}_min_bucket_{b_idx+1}", np.min(bucket) if has_distance else 0)
        #     add_feature(f"finger_distance_{i}_range_bucket_{b_idx+1}", circular_range(bucket) if has_distance else 0)
        #     add_feature(f"finger_distance_{i}_q25_bucket_{b_idx+1}", np.percentile(bucket, 25) if has_distance else 0)
        #     add_feature(f"finger_distance_{i}_q75_bucket_{b_idx+1}", np.percentile(bucket, 75) if has_distance else 0)

        # If the distances are empty add 0 for all
        has_distance = len(distances) != 0
        add_feature(f"finger_distance_{i}_mean", np.mean(distances) if has_distance else 0)
        add_feature(f"finger_distance_{i}_std", np.std(distances) if has_distance else 0)
        add_feature(f"finger_distance_{i}_max", np.max(distances) if has_distance else 0)
        add_feature(f"finger_distance_{i}_min", np.min(distances) if has_distance else 0)
        add_feature(f"finger_distance_{i}_range", circular_range(distances) if has_distance else 0)
        add_feature(f"finger_distance_{i}_q25", np.percentile(distances, 25) if has_distance else 0)
        add_feature(f"finger_distance_{i}_q75", np.percentile(distances, 75) if has_distance else 0)


    amount_active_moving_fingers = 0
    # Per Finger Features
    for i, (pos_col, press_col, channel_col) in enumerate(zip(touch_position_cols, touch_pressure_cols, touch_channel_cols)):
        finger_positions = window[:, pos_col]
        finger_pressures = window[:, press_col]
        finger_channels = window[:, channel_col]

        # Fill nan with 0
        finger_positions = np.nan_to_num(finger_positions, nan=0.0)
        finger_pressures = np.nan_to_num(finger_pressures, nan=0.0)

        # Only consider positions where the pressure > 0
        valid_mask = finger_pressures > 0
        finger_positions = finger_positions[valid_mask]
        finger_channels = finger_channels[valid_mask]

        # General
        # 1. Any touch activity?
        is_touching = finger_pressures > 0
        add_feature(f"finger_{i+1}_any_touch", np.any(is_touching))
        # 3. Has moved? (this has a threshold to avoid noise)
        circle_range = circular_range(finger_positions) if len(finger_positions) > 0 else 0
        has_moved = circle_range > TOUCH_MOVE_THRESHOLD
        add_feature(f"finger_{i+1}_has_moved", has_moved)

        # Create 5 moved buckets
        moved_buckets = np.array_split(finger_positions, 5)
        for b_idx, bucket in enumerate(moved_buckets):
            bucket_range = circular_range(bucket) if len(bucket) > 0 else 0
            add_feature(f"finger_{i+1}_moved_range_bucket_{b_idx+1}", bucket_range)

        # 4. Amount of reactivations (so how often was the finger lifted and put down again)
        touch_reactivations = 0
        for t in range(1, len(is_touching)):
            if is_touching[t] and not is_touching[t - 1]:
                touch_reactivations += 1
        add_feature(f"finger_{i+1}_touch_reactivations", touch_reactivations)

        # Get the time between reactivations
        touch_reactivation_times = []
        last_touch_end = None
        for t in range(1, len(is_touching)):
            if is_touching[t] and not is_touching[t - 1]:
                if last_touch_end is not None:
                    touch_reactivation_times.append(t - last_touch_end)
            if not is_touching[t] and is_touching[t - 1]:
                last_touch_end = t
        # Mean time between reactivations
        add_feature(f"finger_{i+1}_mean_time_between_reactivations", np.mean(touch_reactivation_times) if len(touch_reactivation_times) > 0 else 0)
        # Std Dev time between reactivations
        add_feature(f"finger_{i+1}_std_time_between_reactivations", np.std(touch_reactivation_times) if len(touch_reactivation_times) > 0 else 0)
        # Max time between reactivations
        add_feature(f"finger_{i+1}_max_time_between_reactivations", np.max(touch_reactivation_times) if len(touch_reactivation_times) > 0 else 0)
        # Min time between reactivations
        add_feature(f"finger_{i+1}_min_time_between_reactivations", np.min(touch_reactivation_times) if len(touch_reactivation_times) > 0 else 0)
        # Range time between reactivations
        add_feature(f"finger_{i+1}_range_time_between_reactivations", np.abs(np.max(touch_reactivation_times) - np.min(touch_reactivation_times)) if len(touch_reactivation_times) > 0 else 0)

        # Create 5 reactivation buckets to get a better temporal understanding of what is happening at the start middle and end
        # reactivation_buckets = np.array_split(touch_reactivation_times, 5)
        # for b_idx, bucket in enumerate(reactivation_buckets):
        #     has_reactivation = len(bucket) != 0
        #     add_feature(f"finger_{i+1}_mean_time_between_reactivations_bucket_{b_idx+1}", np.mean(bucket) if has_reactivation else 0)
        #     add_feature(f"finger_{i+1}_std_time_between_reactivations_bucket_{b_idx+1}", np.std(bucket) if has_reactivation else 0)
        #     add_feature(f"finger_{i+1}_max_time_between_reactivations_bucket_{b_idx+1}", np.max(bucket) if has_reactivation else 0)
        #     add_feature(f"finger_{i+1}_min_time_between_reactivations_bucket_{b_idx+1}", np.min(bucket) if has_reactivation else 0)
        #     add_feature(f"finger_{i+1}_range_time_between_reactivations_bucket_{b_idx+1}", np.abs(np.max(bucket) - np.min(bucket)) if has_reactivation else 0)

        # Touch Times and Amount (Pressure is 0 or null if no touch) There might be multiple touches in a window for a given finger, so we have to extract each consecutive time a touch was active, plus the amount of samples for each time.
        # start_of_touches = is_touching & (~is_touching.shift(1, fill_value=False))
        # end_of_touches = is_touching & (~is_touching.shift(-1, fill_value=False))
        start_of_touches = is_touching & ~np.insert(is_touching[:-1], 0, False)
        end_of_touches = is_touching & ~np.append(is_touching[1:], False)
        touch_starts = np.where(start_of_touches)[0]
        touch_ends = np.where(end_of_touches)[0]
        touch_durations = touch_ends - touch_starts

        max_touch_duration = np.max(touch_durations) if len(touch_durations) > 0 else 0
        min_touch_duration = np.min(touch_durations) if len(touch_durations) > 0 else 0
        
        # Is continuously pressed during the window? (If the min touch duration is the length of the window)
        is_continuous_touch = min_touch_duration >= original_length - 1
        add_feature(f"finger_{i+1}_continuous_touch_duration", min_touch_duration if is_continuous_touch else 0)
        add_feature(f"finger_{i+1}_non_continuous_touch_duration", 0 if is_continuous_touch else np.sum(touch_durations) if len(touch_durations) > 0 else 0)
        add_feature(f"finger_{i+1}_total_touch_duration", np.sum(touch_durations) if len(touch_durations) > 0 else 0)
        add_feature(f"finger_{i+1}_max_touch_duration", max_touch_duration)
        add_feature(f"finger_{i+1}_min_touch_duration", min_touch_duration)
        add_feature(f"finger_{i+1}_mean_touch_duration", np.mean(touch_durations) if len(touch_durations) > 0 else 0)
        add_feature(f"finger_{i+1}_std_touch_duration", np.std(touch_durations) if len(touch_durations) > 0 else 0)
        add_feature(f"finger_{i+1}_is_continuous_touch", is_continuous_touch)

        if np.any(is_touching) and (has_moved or not is_continuous_touch):
            amount_active_moving_fingers += 1

        # 4. How many times was the finger touched?
        add_feature(f"finger_{i+1}_touch_starts", len(touch_starts))
        add_feature(f"finger_{i+1}_touch_ends", len(touch_ends))
        # 5. Mean touch duration
        add_feature(f"finger_{i+1}_mean_touch_duration", np.mean(touch_durations) if len(touch_durations) > 0 else 0)
        # 6. Std Dev touch duration
        add_feature(f"finger_{i+1}_std_touch_duration", np.std(touch_durations) if len(touch_durations) > 0 else 0)
        # 7. Max touch duration
        add_feature(f"finger_{i+1}_max_touch_duration", max_touch_duration)
        # 8. Min touch duration
        add_feature(f"finger_{i+1}_min_touch_duration", min_touch_duration)
        # 9. Range touch duration
        add_feature(f"finger_{i+1}_range_touch_duration", np.abs(max_touch_duration - min_touch_duration) if len(touch_durations) > 0 else 0)
        # 10. Total touch duration
        add_feature(f"finger_{i+1}_total_touch_duration", np.sum(touch_durations) if len(touch_durations) > 0 else 0)

        # Create 5 duration buckets to get a better temporal understanding of what is happening at the start middle and end
        # duration_buckets = np.array_split(touch_durations, 5)
        # for b_idx, bucket in enumerate(duration_buckets):
        #     has_duration = len(bucket) != 0
        #     add_feature(f"finger_{i+1}_mean_touch_duration_bucket_{b_idx+1}", np.mean(bucket) if has_duration else 0)
        #     add_feature(f"finger_{i+1}_std_touch_duration_bucket_{b_idx+1}", np.std(bucket) if has_duration else 0)
        #     add_feature(f"finger_{i+1}_max_touch_duration_bucket_{b_idx+1}", np.max(bucket) if has_duration else 0)
        #     add_feature(f"finger_{i+1}_min_touch_duration_bucket_{b_idx+1}", np.min(bucket) if has_duration else 0)
        #     add_feature(f"finger_{i+1}_range_touch_duration_bucket_{b_idx+1}", np.abs(np.max(bucket) - np.min(bucket)) if has_duration else 0)
        #     add_feature(f"finger_{i+1}_total_touch_duration_bucket_{b_idx+1}", np.sum(bucket) if has_duration else 0)

        # Position, Pressure Features
        for data_type, data in [("positions", finger_positions), ("pressures", finger_pressures), ("channels", finger_channels)]:
            # If the data is empty, add zeros for all features
            has_data = len(data) != 0
            # 1. Mean pressure
            add_feature(f"finger_{i+1}_{data_type}_mean", np.mean(data) if has_data else 0)
            # 2. Std Dev pressure
            add_feature(f"finger_{i+1}_{data_type}_std", np.std(data) if has_data else 0)
            # 3. Max pressure
            add_feature(f"finger_{i+1}_{data_type}_max", np.max(data) if has_data else 0)
            # 4. Min pressure
            add_feature(f"finger_{i+1}_{data_type}_min", np.min(data) if has_data else 0)
            # 5. Range pressure
            add_feature(f"finger_{i+1}_{data_type}_range", circular_range(data) if has_data else 0)
            # 7. Q25 / Q75
            add_feature(f"finger_{i+1}_{data_type}_q25", np.percentile(data, 25) if has_data else 0)
            add_feature(f"finger_{i+1}_{data_type}_q75", np.percentile(data, 75) if has_data else 0)
            # 8. Skewness / Kurtosis
            add_feature(f"finger_{i+1}_{data_type}_skewness", stats.skew(data) if has_data else 0)
            add_feature(f"finger_{i+1}_{data_type}_kurtosis", stats.kurtosis(data) if has_data else 0)
            if data_type == "channels":
                continue
            # 9. Velocity
            has_velocity = len(data) > 1
            pressure_velocity = np.diff(data) * SAMPLING_RATE_HZ
            add_feature(f"finger_{i+1}_{data_type}_velocity_mean", np.mean(pressure_velocity) if has_velocity else 0)
            add_feature(f"finger_{i+1}_{data_type}_velocity_std", np.std(pressure_velocity) if has_velocity else 0)
            add_feature(f"finger_{i+1}_{data_type}_velocity_max", np.max(pressure_velocity) if has_velocity else 0)
            if data_type == "pressures":
                continue
            # 10. Acceleration
            has_acceleration = len(data) > 2
            pressure_acceleration = np.diff(data, n=2) * (SAMPLING_RATE_HZ ** 2)
            add_feature(f"finger_{i+1}_{data_type}_acceleration_mean", np.mean(pressure_acceleration) if has_acceleration else 0)
            add_feature(f"finger_{i+1}_{data_type}_acceleration_std", np.std(pressure_acceleration) if has_acceleration else 0)
            add_feature(f"finger_{i+1}_{data_type}_acceleration_max", np.max(pressure_acceleration) if has_acceleration else 0)

            # Create 5 buckets to get a better temporal understanding of what is happening at the start middle and end
            # buckets = np.array_split(data, 5)
            # for b_idx, bucket in enumerate(buckets):
            #     has_bucket_data = len(bucket) != 0
            #     add_feature(f"finger_{i+1}_{data_type}_mean_bucket_{b_idx+1}", np.mean(bucket) if has_bucket_data else 0)
            #     add_feature(f"finger_{i+1}_{data_type}_std_bucket_{b_idx+1}", np.std(bucket) if has_bucket_data else 0)
            #     add_feature(f"finger_{i+1}_{data_type}_max_bucket_{b_idx+1}", np.max(bucket) if has_bucket_data else 0)
            #     add_feature(f"finger_{i+1}_{data_type}_min_bucket_{b_idx+1}", np.min(bucket) if has_bucket_data else 0)
            #     add_feature(f"finger_{i+1}_{data_type}_range_bucket_{b_idx+1}", circular_range(bucket) if has_bucket_data else 0)
            #     add_feature(f"finger_{i+1}_{data_type}_q25_bucket_{b_idx+1}", np.percentile(bucket, 25) if has_bucket_data else 0)
            #     add_feature(f"finger_{i+1}_{data_type}_q75_bucket_{b_idx+1}", np.percentile(bucket, 75) if has_bucket_data else 0)

    # 8. Amount of fingers minus the amount of fingers that are constantly touching and not moving
    add_feature("active_moving_fingers", amount_active_moving_fingers)

    return np.array(features, dtype=np.float32), feature_names

def second():
    def extract_features(window, is_padded_window):
    features = []
    feature_names = [] # for debug purposes we store the names of each feature at the same index
    
    # Assume sensor column order from SENSOR_COLUMNS
    # Indices: button=0, motor=1, touch_pos=[2,5,8,11,14], touch_press=[3,6,9,12,15]
    button_col = 0
    motor_col = 1
    touch_channel_cols = [4, 7, 10, 13, 16]  # touch_X_channel indices
    touch_pressure_cols = [3, 6, 9, 12, 15]  # touch_X_pressure indices
    touch_position_cols = [2, 5, 8, 11, 14]  # touch_X_position indices
    original_length_col = len(SENSOR_COLUMNS)  # last column is original length

    button_press_data = window[:, button_col]
    motor_data = window[:, motor_col]
    touch_pressures = window[:, touch_pressure_cols]
    touch_positions = window[:, touch_position_cols]
    # Only consider touch positions where the pressure > 0
    valid_mask = touch_pressures > 0
    touch_positions = touch_positions[valid_mask]
    original_length = int(window[0, original_length_col]) if is_padded_window else window.shape[0]

    # A 1 signalizes the button was pressed, this can happen multiple times in a window, so we first have to extract each consecutive time a button was pressed, plus the amount of samples for each time
    binary_button_press = (button_press_data > 0).astype(int)
    padded_button_press = np.concatenate(([0], binary_button_press, [0]))
    button_press_diff = np.diff(padded_button_press)
    button_press_starts = np.where(button_press_diff == 1)[0]
    button_press_ends = np.where(button_press_diff == -1)[0]
    button_press_durations = button_press_ends - button_press_starts

    def circular_range(angles):
        # 1. Sort the angles
        sorted_angles = np.sort(angles)
        # 2. Calculate gaps between neighbors
        gaps = np.diff(sorted_angles)
        # 3. Calculate the wrap-around gap (last to first)
        wrap_gap = (2 * np.pi - sorted_angles[-1]) + sorted_angles[0]
        # 4. Combine all gaps
        all_gaps = np.append(gaps, wrap_gap)
        # 5. Range is the full circle minus the biggest empty space
        return 2 * np.pi - np.max(all_gaps)
    
    def add_feature(name, value):
        features.append(value)
        feature_names.append(name)

    # Button Features
    # 1. Button pressed at all?
    add_feature("button_pressed_any", np.any(binary_button_press))
    # # 2. How often was the button pressed (changed from 0 to 1 and vice versa) and how often was it let go
    add_feature("button_press_count", len(button_press_starts))
    # # 3. Mean (raw + durations)
    add_feature("button_press_mean", np.mean(binary_button_press))
    add_feature("button_press_duration_mean", np.mean(button_press_durations) if len(button_press_durations) > 0 else 0)
    # # 4. Std Dev (raw + durations)
    add_feature("button_press_std", np.std(binary_button_press))
    add_feature("button_press_duration_std", np.std(button_press_durations) if len(button_press_durations) > 0 else 0)
    # # 5. Median (raw + durations)
    add_feature("button_press_median", np.median(binary_button_press))
    add_feature("button_press_duration_median", np.median(button_press_durations) if len(button_press_durations) > 0 else 0)
    # # 6. Max duration
    add_feature("button_press_max_duration", np.max(button_press_durations) if len(button_press_durations) > 0 else 0)
    # # 7. Min duration
    add_feature("button_press_min_duration", np.min(button_press_durations) if len(button_press_durations) > 0 else 0)
    # # 8. Range duration
    add_feature("button_press_range_duration", circular_range(button_press_durations) if len(button_press_durations) > 0 else 0)
    # # 9. Total duration
    add_feature("button_press_total_duration", np.sum(button_press_durations) if len(button_press_durations) > 0 else 0)

    # Motor Angle Features
    # 1. Mean
    add_feature("motor_angle_mean", np.mean(motor_data))
    # # 2. Std Dev
    add_feature("motor_angle_std", np.std(motor_data))
    # # 3. Min Max
    add_feature("motor_angle_min", np.min(motor_data))
    add_feature("motor_angle_max", np.max(motor_data))
    # # 4. Range
    add_feature("motor_angle_range", circular_range(motor_data))
    # # 5. Median
    add_feature("motor_angle_median", np.median(motor_data))
    # # 6. Q25 / Q75
    add_feature("motor_angle_q25", np.percentile(motor_data, 25))
    add_feature("motor_angle_q75", np.percentile(motor_data, 75))
    # # 7. Skewness / Kurtosis
    add_feature("motor_angle_skewness", stats.skew(motor_data))
    add_feature("motor_angle_kurtosis", stats.kurtosis(motor_data))
    # # 8. Velocity
    if len(motor_data) > 1:
        motor_velocity = np.diff(motor_data) * SAMPLING_RATE_HZ
        add_feature("motor_velocity_mean", np.mean(motor_velocity))
        add_feature("motor_velocity_std", np.std(motor_velocity))
        add_feature("motor_velocity_max", np.max(motor_velocity))
    else:
        add_feature("motor_velocity_mean", 0)
        add_feature("motor_velocity_std", 0)
        add_feature("motor_velocity_max", 0)

    # General Touch Activity Features
    # 1. Max simultaneous touches
    active_touches = (touch_pressures > 0).astype(int)
    add_feature("max_simultaneous_touches", np.max(np.sum(active_touches, axis=1)))
    # 2. Avg active touches
    add_feature("avg_active_touches", np.mean(np.sum(active_touches, axis=1)))
    # 3. Any touch at all?
    add_feature("any_touch", np.any(active_touches))
    # 4. Total pressure across all sensors
    add_feature("total_active_touch_samples", np.sum(active_touches))
    add_feature("total_pressure", np.sum(touch_pressures))
    # 5. Max pressure
    add_feature("max_pressure", np.max(touch_pressures))
    # 5.1 Min non-zero pressure
    non_zero_pressures = touch_pressures[touch_pressures > 0]
    add_feature("min_non_zero_pressure", np.min(non_zero_pressures) if len(non_zero_pressures) > 0 else 0)
    # 6. Touch centroid
    touch_position_sin_mean = np.sin(touch_positions).mean()
    touch_position_cos_mean = np.cos(touch_positions).mean()
    touch_position_mean_angle = np.arctan2(touch_position_sin_mean, touch_position_cos_mean)
    add_feature("touch_centroid", touch_position_mean_angle % (2 * np.pi))
    # 7. Touch Spread
    touch_position_R = np.sqrt(touch_position_sin_mean**2 + touch_position_cos_mean**2)
    add_feature("touch_spread", 1 - touch_position_R)
    add_feature("touch_spread_log", -np.log(np.clip(touch_position_R, 1e-9, 1.0)))

    # Distances between each finger (including wrapping)
    def short_dist(a, b):
        return np.abs(np.arctan2(np.sin(a - b), np.cos(a - b)))
    
    i = 0
    for p1, p2 in combinations(touch_position_cols, 2):
        i += 1
        finger1_positions = window[:, p1]
        finger2_positions = window[:, p2]
        # Only consider positions where the pressure > 0
        finger1_pressures = window[:, touch_pressure_cols[touch_position_cols.index(p1)]]
        finger2_pressures = window[:, touch_pressure_cols[touch_position_cols.index(p2)]]
        valid_mask = (finger1_pressures > 0) & (finger2_pressures > 0)
        finger1_positions = finger1_positions[valid_mask]
        finger2_positions = finger2_positions[valid_mask]

        distances = short_dist(finger1_positions, finger2_positions)

        # Calculate the distances for the first and last third of the window
        third_length = len(distances) // 3
        first_third_distances = distances[:third_length]
        last_third_distances = distances[-third_length:]
        add_feature("finger_distance_mean_first_third_" + str(i), np.mean(first_third_distances) if len(first_third_distances) > 0 else 0)
        add_feature("finger_distance_mean_last_third_" + str(i), np.mean(last_third_distances) if len(last_third_distances) > 0 else 0)

        # If the distances are empty add 0 for all
        has_distance = len(distances) != 0
        add_feature("finger_distance_mean_" + str(i), np.mean(distances) if has_distance else 0)
        add_feature("finger_distance_std_" + str(i), np.std(distances) if has_distance else 0)
        add_feature("finger_distance_max_" + str(i), np.max(distances) if has_distance else 0)
        add_feature("finger_distance_min_" + str(i), np.min(distances) if has_distance else 0)
        add_feature("finger_distance_range_" + str(i), circular_range(distances) if has_distance else 0)
        add_feature("finger_distance_q25_" + str(i), np.percentile(distances, 25) if has_distance else 0)
        add_feature("finger_distance_q75_" + str(i), np.percentile(distances, 75) if has_distance else 0)


    amount_active_moving_fingers = 0
    # Per Finger Features
    for i, (pos_col, press_col, channel_col) in enumerate(zip(touch_position_cols, touch_pressure_cols, touch_channel_cols)):
        finger_positions = window[:, pos_col]
        finger_pressures = window[:, press_col]
        finger_channels = window[:, channel_col]

        # Fill nan with 0
        finger_positions = np.nan_to_num(finger_positions, nan=0.0)
        finger_pressures = np.nan_to_num(finger_pressures, nan=0.0)

        # Only consider positions where the pressure > 0
        valid_mask = finger_pressures > 0
        finger_positions = finger_positions[valid_mask]
        finger_channels = finger_channels[valid_mask]

        # General
        # 1. Any touch activity?
        is_touching = finger_pressures > 0
        add_feature(f"finger_{i+1}_any_touch", np.any(is_touching))
        # 3. Has moved? (this has a threshold to avoid noise)
        circle_range = circular_range(finger_positions) if len(finger_positions) > 0 else 0
        has_moved = circle_range > TOUCH_MOVE_THRESHOLD
        add_feature(f"finger_{i+1}_has_moved", has_moved)

        # 4. Amount of reactivations (so how often was the finger lifted and put down again)
        touch_reactivations = 0
        for t in range(1, len(is_touching)):
            if is_touching[t] and not is_touching[t - 1]:
                touch_reactivations += 1
        add_feature(f"finger_{i+1}_touch_reactivations", touch_reactivations)

        # Get the time between reactivations
        touch_reactivation_times = []
        last_touch_end = None
        for t in range(1, len(is_touching)):
            if is_touching[t] and not is_touching[t - 1]:
                if last_touch_end is not None:
                    touch_reactivation_times.append(t - last_touch_end)
            if not is_touching[t] and is_touching[t - 1]:
                last_touch_end = t
        # Mean time between reactivations
        add_feature(f"finger_{i+1}_mean_time_between_reactivations", np.mean(touch_reactivation_times) if len(touch_reactivation_times) > 0 else 0)
        # Std Dev time between reactivations
        add_feature(f"finger_{i+1}_std_time_between_reactivations", np.std(touch_reactivation_times) if len(touch_reactivation_times) > 0 else 0)
        # Max time between reactivations
        add_feature(f"finger_{i+1}_max_time_between_reactivations", np.max(touch_reactivation_times) if len(touch_reactivation_times) > 0 else 0)
        # Min time between reactivations
        add_feature(f"finger_{i+1}_min_time_between_reactivations", np.min(touch_reactivation_times) if len(touch_reactivation_times) > 0 else 0)
        # Range time between reactivations
        add_feature(f"finger_{i+1}_range_time_between_reactivations", np.abs(np.max(touch_reactivation_times) - np.min(touch_reactivation_times)) if len(touch_reactivation_times) > 0 else 0)

        # Touch Times and Amount (Pressure is 0 or null if no touch) There might be multiple touches in a window for a given finger, so we have to extract each consecutive time a touch was active, plus the amount of samples for each time.
        # start_of_touches = is_touching & (~is_touching.shift(1, fill_value=False))
        # end_of_touches = is_touching & (~is_touching.shift(-1, fill_value=False))
        start_of_touches = is_touching & ~np.insert(is_touching[:-1], 0, False)
        end_of_touches = is_touching & ~np.append(is_touching[1:], False)
        touch_starts = np.where(start_of_touches)[0]
        touch_ends = np.where(end_of_touches)[0]
        touch_durations = touch_ends - touch_starts

        max_touch_duration = np.max(touch_durations) if len(touch_durations) > 0 else 0
        min_touch_duration = np.min(touch_durations) if len(touch_durations) > 0 else 0
        
        # Is continuously pressed during the window? (If the min touch duration is the length of the window)
        is_continuous_touch = min_touch_duration >= original_length - 1
        add_feature(f"finger_{i+1}_continuous_touch_duration", min_touch_duration if is_continuous_touch else 0)
        add_feature(f"finger_{i+1}_non_continuous_touch_duration", 0 if is_continuous_touch else np.sum(touch_durations) if len(touch_durations) > 0 else 0)
        add_feature(f"finger_{i+1}_total_touch_duration", np.sum(touch_durations) if len(touch_durations) > 0 else 0)
        add_feature(f"finger_{i+1}_max_touch_duration", max_touch_duration)
        add_feature(f"finger_{i+1}_min_touch_duration", min_touch_duration)
        add_feature(f"finger_{i+1}_mean_touch_duration", np.mean(touch_durations) if len(touch_durations) > 0 else 0)
        add_feature(f"finger_{i+1}_std_touch_duration", np.std(touch_durations) if len(touch_durations) > 0 else 0)
        add_feature(f"finger_{i+1}_is_continuous_touch", is_continuous_touch)

        if np.any(is_touching) and (has_moved or not is_continuous_touch):
            amount_active_moving_fingers += 1

        # 4. How many times was the finger touched?
        add_feature(f"finger_{i+1}_touch_starts", len(touch_starts))
        add_feature(f"finger_{i+1}_touch_ends", len(touch_ends))
        # 5. Mean touch duration
        add_feature(f"finger_{i+1}_mean_touch_duration", np.mean(touch_durations) if len(touch_durations) > 0 else 0)
        # 6. Std Dev touch duration
        add_feature(f"finger_{i+1}_std_touch_duration", np.std(touch_durations) if len(touch_durations) > 0 else 0)
        # 7. Max touch duration
        add_feature(f"finger_{i+1}_max_touch_duration", max_touch_duration)
        # 8. Min touch duration
        add_feature(f"finger_{i+1}_min_touch_duration", min_touch_duration)
        # 9. Range touch duration
        add_feature(f"finger_{i+1}_range_touch_duration", np.abs(max_touch_duration - min_touch_duration) if len(touch_durations) > 0 else 0)
        # 10. Total touch duration
        add_feature(f"finger_{i+1}_total_touch_duration", np.sum(touch_durations) if len(touch_durations) > 0 else 0)

        # Position, Pressure Features
        for data_type, data in [("positions", finger_positions), ("pressures", finger_pressures), ("channels", finger_channels)]:
            # If the data is empty, add zeros for all features
            has_data = len(data) != 0
            # 1. Mean pressure
            add_feature(f"finger_{i+1}_{data_type}_mean", np.mean(data) if has_data else 0)
            # 2. Std Dev pressure
            add_feature(f"finger_{i+1}_{data_type}_std", np.std(data) if has_data else 0)
            # 3. Max pressure
            add_feature(f"finger_{i+1}_{data_type}_max", np.max(data) if has_data else 0)
            # 4. Min pressure
            add_feature(f"finger_{i+1}_{data_type}_min", np.min(data) if has_data else 0)
            # 5. Range pressure
            add_feature(f"finger_{i+1}_{data_type}_range", circular_range(data) if has_data else 0)
            # 7. Q25 / Q75
            add_feature(f"finger_{i+1}_{data_type}_q25", np.percentile(data, 25) if has_data else 0)
            add_feature(f"finger_{i+1}_{data_type}_q75", np.percentile(data, 75) if has_data else 0)
            # 8. Skewness / Kurtosis
            add_feature(f"finger_{i+1}_{data_type}_skewness", stats.skew(data) if has_data else 0)
            add_feature(f"finger_{i+1}_{data_type}_kurtosis", stats.kurtosis(data) if has_data else 0)
            # 9. Velocity
            has_velocity = len(data) > 1
            pressure_velocity = np.diff(data) * SAMPLING_RATE_HZ
            add_feature(f"finger_{i+1}_{data_type}_velocity_mean", np.mean(pressure_velocity) if has_velocity else 0)
            add_feature(f"finger_{i+1}_{data_type}_velocity_std", np.std(pressure_velocity) if has_velocity else 0)
            add_feature(f"finger_{i+1}_{data_type}_velocity_max", np.max(pressure_velocity) if has_velocity else 0)
            # 10. Acceleration
            has_acceleration = len(data) > 2
            pressure_acceleration = np.diff(data, n=2) * (SAMPLING_RATE_HZ ** 2)
            add_feature(f"finger_{i+1}_{data_type}_acceleration_mean", np.mean(pressure_acceleration) if has_acceleration else 0)
            add_feature(f"finger_{i+1}_{data_type}_acceleration_std", np.std(pressure_acceleration) if has_acceleration else 0)
            add_feature(f"finger_{i+1}_{data_type}_acceleration_max", np.max(pressure_acceleration) if has_acceleration else 0)

    # 8. Amount of fingers minus the amount of fingers that are constantly touching and not moving
    add_feature("active_moving_fingers", amount_active_moving_fingers)

    return np.array(features, dtype=np.float32), feature_names