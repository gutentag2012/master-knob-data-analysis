# --- Touch Tracking ---
next_touch_id = 0
active_touches = {}  # id -> (rel_pos, pressure)

def assign_ids(new_touches):
    """
    Match new touches to old ones by nearest relative position.
    new_touches: list of (rel_pos, pressure)
    Returns: list of (id, rel_pos, pressure)
    """
    global next_touch_id, active_touches

    matched = []
    used_old_ids = set()

    for rel_pos, pressure in new_touches:
        # Find nearest old touch by position
        best_id = None
        best_dist = float("inf")
        for old_id, (old_pos, _) in active_touches.items():
            dist = abs(old_pos - rel_pos)
            if dist < best_dist and old_id not in used_old_ids:
                best_dist = dist
                best_id = old_id

        if best_id is not None and best_dist < 5:  # Threshold (degrees)
            # Reuse existing ID
            matched.append((best_id, rel_pos, pressure))
            used_old_ids.add(best_id)
        else:
            # New touch → assign new ID
            tid = next_touch_id
            next_touch_id += 1
            matched.append((tid, rel_pos, pressure))
            used_old_ids.add(tid)

    # Update active_touches with current frame
    active_touches = {tid: (pos, pr) for tid, pos, pr in matched}
    return matched

# --- Main Processing ---
iteration = 0
for rows in chunks:
    # Reset tracker state per chunk
    active_touches.clear()
    next_touch_id = 0
    id_to_color.clear()

    height = len(rows)
    image = np.zeros((height, IMAGE_WIDTH, 3), dtype=np.uint8)

    # --- First pass: track spans + positions ---
    touch_spans = {}      # tid -> (first_row, last_row)
    touch_positions = {}  # tid -> [positions]
    touch_pressures = {}  # tid -> [pressures]

    for y, row in enumerate(rows):
        new_touches = []
        for i in range(5):
            pressure = row[3 + i*2]
            rel_pos = row[4 + i*2]
            if pressure is not None and pressure > 200 and rel_pos is not None:
                new_touches.append((rel_pos, pressure))

        tracked = assign_ids(new_touches)

    # --- Second pass: render image ---
    active_touches.clear()
    next_touch_id = 0

    for y, row in enumerate(rows):
        motor_angle = row[1]

        # alternate background color per 2 rows
        bg = (255, 255, 255) if y % 4 < 2 else (0, 0, 0)
        image[y, :, :] = bg

        # Draw motor angle line
        motor_x = int(round(motor_angle % 360)) + IMAGE_PADDING
        image[y, motor_x - 1:motor_x + 1] = (255, 0, 0)

        # Collect valid touches again
        new_touches = []
        for i in range(5):
            pressure = row[3 + i*2]
            rel_pos = row[4 + i*2]
            if pressure is not None and pressure > 200 and rel_pos is not None:
                new_touches.append((rel_pos, pressure))

        tracked = assign_ids(new_touches)

        for tid, rel_pos, pressure in tracked:
            first_row, last_row = touch_spans[tid]

            # Transition coloring
            if y - first_row < 3:
                color = (0, 255, 0)  # green start
            elif last_row - y < 3:
                color = (255, 255, 0)  # yellow end
            else:
                color = (0, 0, 255)# id_to_color[tid]

            # Draw touch
            angle = (rel_pos - motor_angle) % 360
            center_x = int(round(angle)) + IMAGE_PADDING
            size = max(1, int(round(pressure * PRESSURE_SCALE)))

            for dx in range(-size // 2, size // 2 + 1):
                x = (center_x + dx) % (IMAGE_WIDTH - 2 * IMAGE_PADDING)
                image[y, x] = color

    # Save image
    img = Image.fromarray(image, mode='RGB')