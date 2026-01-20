import math

class TouchTracker:
    def __init__(self, match_threshold: float = math.pi / 8, max_age=200):
        """
        Initializes a touch tracker to assign consistent IDs across frames.
        :param match_threshold: Max distance (in relative position units, e.g. degrees)
                                to consider a touch the same as in the previous frame.
        """
        self.next_touch_id = 0
        self.match_threshold = match_threshold
        self.max_age = max_age
        
        # Currently active: {id: position}
        self.active_touches = {}
        
        # Inactive but remembered: {id: {"pos": pos, "age": 0}}
        self.persistence_buffer = {}

        """Resets the tracker state."""
    def clear(self):
        self.active_touches.clear()
        self.persistence_buffer.clear()

    def assign_ids(self, new_touches):
        def calculate_wrapped_distance(phi1, phi2):
            two_pi = 2 * math.pi
            phi1, phi2 = phi1 % two_pi, phi2 % two_pi
            diff = abs(phi1 - phi2)
            return min(diff, two_pi - diff)

        matched = [None] * len(new_touches)
        new_to_candidates = []

        # 1. Combine Active and Persisted touches as candidates for matching
        candidate_pool = {}
        # Active touches get priority (implicit in distance check or order)
        for tid, pos in self.active_touches.items():
            candidate_pool[tid] = pos
        # Persisted touches (the "graveyard")
        for tid, data in self.persistence_buffer.items():
            candidate_pool[tid] = data["pos"]

        # 2. Build candidate list for each new input
        for new_index, (rel_pos, pressure, *rest) in enumerate(new_touches):
            if rel_pos is None or (isinstance(rel_pos, float) and math.isnan(rel_pos)):
                continue
            
            sorted_candidates = []
            for tid, old_pos in candidate_pool.items():
                dist = calculate_wrapped_distance(old_pos, rel_pos)
                if dist < self.match_threshold:
                    sorted_candidates.append((dist, tid))
            
            sorted_candidates.sort()
            new_to_candidates.append((new_index, sorted_candidates))

        # 3. Greedy Matching (Closest pairs first)
        new_to_candidates.sort(key=lambda x: x[1][0][0] if x[1] else float("inf"))
        used_candidate_ids = set()
        
        for new_index, sorted_candidates in new_to_candidates:
            rel_pos, pressure, *rest = new_touches[new_index]
            assigned = False
            for dist, tid in sorted_candidates:
                if tid not in used_candidate_ids:
                    matched[new_index] = (tid, rel_pos, pressure, *rest)
                    used_candidate_ids.add(tid)
                    assigned = True
                    break
            
            if not assigned:
                tid = self.next_touch_id
                self.next_touch_id += 1
                matched[new_index] = (tid, rel_pos, pressure, *rest)
                used_candidate_ids.add(tid)

        # 4. Update state for the NEXT frame
        valid_matches = [m for m in matched if m is not None]
        new_active_ids = {m[0] for m in valid_matches}
        
        # Move currently active touches that weren't matched into persistence
        for tid, pos in self.active_touches.items():
            if tid not in new_active_ids:
                self.persistence_buffer[tid] = {"pos": pos, "age": 0}

        # Update ages in persistence buffer and remove old ones
        expired_ids = []
        for tid in list(self.persistence_buffer.keys()):
            if tid in new_active_ids:
                # Successfully resurrected, remove from graveyard
                del self.persistence_buffer[tid]
            else:
                self.persistence_buffer[tid]["age"] += 1
                if self.persistence_buffer[tid]["age"] > self.max_age:
                    expired_ids.append(tid)
        
        for tid in expired_ids:
            del self.persistence_buffer[tid]

        # Finalize active_touches for the next row
        self.active_touches = {m[0]: m[1] for m in valid_matches}

        return valid_matches
