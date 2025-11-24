import sqlite3
import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from order_fingers import TouchTracker

tracker = TouchTracker()
db_file = 'merged.sqlite'
output_dir = 'gesture-images'
touch_colors = [
    (255, 0, 0),    # Red
    (0, 255, 0),    # Green
    (0, 0, 255),    # Blue
    (255, 255, 0),  # Yellow
    (255, 0, 255),   # Magenta
]

# create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

with sqlite3.connect(db_file) as conn:
    participant_ids = conn.execute("SELECT DISTINCT participant_id FROM session").fetchall()
    participant_ids = [row[0] for row in participant_ids]

    for pid in participant_ids:
        # Create pid directory
        pid_dir = os.path.join(output_dir, str(pid))
        os.makedirs(pid_dir, exist_ok=True)

        # Get all session Tasks for this participant
        session_tasks = conn.execute("""
            SELECT st.id, rt.gesture, rt.title
            FROM sessionTask st
                LEFT JOIN session s ON st.session_id = s.id
                LEFT JOIN recordingTask rt ON rt.id = st.recording_task_id
            WHERE s.participant_id = ?
        """, (pid,)).fetchall()
        # for test in ["distance", "normal"]:
        for test in ["normal"]:
            for st_id, gesture, title in session_tasks:
                tracker.clear()
                if gesture is None:
                    gesture = title.lower().replace(' ', '_')
                # Get all sensor data and sessiontaskmarkers for this session task
                sensor_data = conn.execute("""
                    SELECT sdm.*
                    FROM sensorData sdm
                    WHERE sdm.session_task_id = ?
                    ORDER BY sdm.timestamp
                """, (st_id,)).fetchall()
                markers = conn.execute("""
                    SELECT stm.timestamp, stm.marker
                    FROM sessionTaskMarker stm
                    WHERE stm.session_task_id = ?
                    ORDER BY stm.timestamp
                """, (st_id,)).fetchall()

                if not sensor_data:
                    continue
                # Convert to DataFrame for easier processing
                columns = [desc[1] for desc in conn.execute("PRAGMA table_info(sensorData)").fetchall()]
                df = pd.DataFrame(sensor_data, columns=columns)
                marker_df = pd.DataFrame(markers, columns=['timestamp', 'marker'])

                # Convert touches to floats
                touch_columns = [col for col in df.columns if col.startswith('touch_') and ('position' in col or 'pressure' in col or 'channel' in col)]
                for col in touch_columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

                # Create a picture array that is 360 pixels wide and as tall as there are timestamps
                img_width = 360
                min_start = 0
                max_end = len(df)
                img_height = max_end - min_start
                img_array = np.zeros((img_height, img_width, 3), dtype=np.uint8)

                # There are 5 Fingers in the sensor data with touch_X_position and touch_X_pressure and touch_X_channels (the width of the touch)
                # These touches have to be mapped to the image, where the position is the rotation around the circle (0-360 degrees), the pressure the opacity and the channels the width of the touch
                # Each row/timestamp is one horizontal line of pixels in the image

                for i, row in df.iterrows():
                    if i < min_start or i >= max_end:
                        continue
                    motor_angle = row['motor_angle']
                    if np.isnan(motor_angle):
                        motor_angle = 0.0

                    new_touches = []
                    for finger in range(1, 6):
                        pos = row[f'touch_{finger}_position']
                        pressure = row[f'touch_{finger}_pressure']
                        channels = row[f'touch_{finger}_channel']
                        new_touches.append((pos, pressure, channels))

                    new_fingers = tracker.assign_ids(new_touches)
                    for finger, pos, pressure, channels in new_fingers:
                        finger += 1  # finger index from 1 to 5
                        if np.isnan(pos) or np.isnan(pressure) or np.isnan(channels):
                            continue

                        # Calculate the shortest distance to the motor rotation and use that as the position
                        if not np.isnan(motor_angle) and (test == "distance"):
                            pos = (math.pi - pos + math.pi) % (2 * math.pi)
                        # The pos has to be mapped from 0-2 PI to 0-360
                        pos = math.degrees(pos) % 360

                        # normalize pressure between 0 and 4000 to 0-1
                        pressure = pressure / 2000.0

                        center = int(pos) % img_width
                        width = int(channels)
                        intensity = min(max(int(pressure * 255), 0), 255)

                        start = max(center - width // 2, 0)
                        end = min(center + width // 2, img_width - 1)

                        color = touch_colors[finger%len(touch_colors)]
                        color_with_intensity = [min(int(c * pressure), 255) for c in color]

                        img_array[i - min_start, start:end+1, :] = color_with_intensity

                    # Add a red line where the motor angle was
                    if not np.isnan(motor_angle):
                        motor_pos = math.degrees(motor_angle) % img_width
                        motor_center = int(motor_pos)
                        img_array[i - min_start, motor_center, :] = [255, 0, 0]

                    button_pressed = row['button_pressed']
                    # Add a faint purple line if the button was pressed
                    if button_pressed:
                        img_array[i - min_start, :, 0] = np.clip(img_array[i - min_start, :, 0] + 100, 0, 255)
                        img_array[i - min_start, :, 2] = np.clip(img_array[i - min_start, :, 2] + 100, 0, 255)

                    # Get markers at this timestamp
                    timestamp = row['timestamp']
                    marker_rows = marker_df[marker_df['timestamp'] == timestamp]
                    for _, marker_row in marker_rows.iterrows():
                        marker = marker_row['marker']
                        if marker == 'start':
                            img_array[i - min_start, 0:5, :] = [0, 255, 0]  # Green for start
                        elif marker == 'end':
                            img_array[i - min_start, -5:, :] = [0, 0, 255] # Blue for end
                        else:
                            img_array[i - min_start, 0:5, :] = [255, 255, 0] # Yellow for other markers

                # Convert array to image
                img = Image.fromarray(img_array)
                img_path = os.path.join(pid_dir, f'gesture_{gesture}_{test}.png')
                img.save(img_path)
                print(f'Saved image for participant {pid}, sessionTask {st_id}, gesture {gesture} at {img_path}')
print('All gesture images have been generated.')
            
