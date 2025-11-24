import math

class TouchTracker:
    def __init__(self, match_threshold: float = math.pi / 8):
        """
        Initializes a touch tracker to assign consistent IDs across frames.
        :param match_threshold: Max distance (in relative position units, e.g. degrees)
                                to consider a touch the same as in the previous frame.
        """
        self.next_touch_id = 0
        self.active_touches = {}  # id -> (rel_pos, pressure)
        self.match_threshold = match_threshold

        """Resets the tracker state."""
    def clear(self):
        self.active_touches.clear()

    def assign_ids_old(self, new_touches):
        matched = []
        used_old_ids = set()

        for rel_pos, pressure, *rest in new_touches:
            if rel_pos is None or math.isnan(rel_pos):
                continue
            best_id = None
            best_dist = float("inf")

            # Find nearest old touch
            for old_id, (old_pos, _) in self.active_touches.items():
                dist = abs(old_pos - rel_pos)
                print(old_id, dist)
                if dist < best_dist and old_id not in used_old_ids:
                    best_dist = dist
                    best_id = old_id

            print(f"New touch at {rel_pos} best matches old ID {best_id} at distance {best_dist}")
            if best_id is not None and best_dist < self.match_threshold:
                # Reuse existing ID
                matched.append((best_id, rel_pos, pressure, *rest))
                used_old_ids.add(best_id)
            else:
                # New touch
                tid = self.next_touch_id
                self.next_touch_id += 1
                matched.append((tid, rel_pos, pressure, *rest))
                used_old_ids.add(tid)

        # Update active touches
        self.active_touches = {tid: (pos, pr) for tid, pos, pr, *rest in matched}
        return matched

    def assign_ids(self, new_touches):
        """
        For each new touch, all old touches will be sorted by their distance to the new touch.
        All old touches outside of the match_threshold will be ignored.
        Then the old touch that is closest to any new touch will be matched first, removed from the pool, and assigned to that new touch.
        """
        matched = [None] * len(new_touches)
        new_to_old = []
        for new_index, (rel_pos, pressure, *rest) in enumerate(new_touches):
            if rel_pos is None or math.isnan(rel_pos):
                continue
            sorted_old = []
            for old_id, old_pos in self.active_touches.items():
                dist = abs(old_pos - rel_pos)
                if dist < self.match_threshold:
                    sorted_old.append((dist, old_id))
            sorted_old.sort()
            new_to_old.append((new_index, sorted_old))
        
        # Sort new touches by their closest old touch distance
        new_to_old.sort(key=lambda x: x[1][0][0] if x[1] else float("inf"))
        used_old_ids = set()
        for new_index, sorted_old in new_to_old:
            rel_pos, pressure, *rest = new_touches[new_index]
            if matched[new_index] is not None:
                continue
            for dist, old_id in sorted_old:
                if old_id not in used_old_ids:
                    matched[new_index] = (old_id, rel_pos, pressure, *rest)
                    used_old_ids.add(old_id)
                    break
            if matched[new_index] is None:
                tid = self.next_touch_id
                self.next_touch_id += 1
                matched[new_index] = (tid, rel_pos, pressure, *rest)
                used_old_ids.add(tid)

        # Update active touches
        matched = [m for m in matched if m is not None]
        self.active_touches = {tid: pos for tid, pos, *rest in matched}

        return matched
