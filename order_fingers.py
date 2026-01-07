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
        self.printed = 0

        """Resets the tracker state."""
    def clear(self):
        self.active_touches.clear()

    def assign_ids(self, new_touches):
        """
        For each new touch, all old touches will be sorted by their distance to the new touch.
        All old touches outside of the match_threshold will be ignored.
        Then the old touch that is closest to any new touch will be matched first, removed from the pool, and assigned to that new touch.
        """
        
        def calculate_wrapped_distance(phi1, phi2):
            """
            Calculates the shortest angular distance between two points on a circle.
            Returns a value between 0 and pi.
            """
            # Use modulo to ensure points are within [0, 2pi]
            two_pi = 2 * math.pi
            phi1 = phi1 % two_pi
            phi2 = phi2 % two_pi
            
            # Calculate absolute difference
            diff = abs(phi1 - phi2)
            
            # Return the smaller of the two possible paths
            return min(diff, two_pi - diff)

        matched = [None] * len(new_touches)
        new_to_old = []
        for new_index, (rel_pos, pressure, *rest) in enumerate(new_touches):
            if rel_pos is None or math.isnan(rel_pos):
                continue
            sorted_old = []
            for old_id, old_pos in self.active_touches.items():
                dist = calculate_wrapped_distance(old_pos, rel_pos)
                if dist < self.match_threshold:
                    sorted_old.append((dist, old_id))
            sorted_old.sort()
            new_to_old.append((new_index, sorted_old))
        
        # Sort new touches by their closest old touch distance
        new_to_old.sort(key=lambda x: x[1][0][0] if x[1] else float("inf"))
        used_old_ids = set()
        for new_index, sorted_old in new_to_old:
            rel_pos, pressure, *rest = new_touches[new_index]
            assigned = False
            for dist, old_id in sorted_old:
                if old_id not in used_old_ids:
                    matched[new_index] = (old_id, rel_pos, pressure, *rest)
                    used_old_ids.add(old_id)
                    assigned = True
                    break
                    
            if not assigned:
                tid = self.next_touch_id
                self.next_touch_id += 1
                matched[new_index] = (tid, rel_pos, pressure, *rest)
                used_old_ids.add(tid) # Technically new IDs don't need to be in used_old_ids for this loop, but it's safe.

        matched = [m for m in matched if m is not None]
        self.active_touches = {tid: pos for tid, pos, *rest in matched}

        return matched
