# explore_landmarks.py
from RobotUtils.CalibratedRobot import CalibratedRobot
import camera
import time
import numpy as np
import time
import math

class LocalizationPathing:
    def __init__(self, robot, camera, required_landmarks, step_cm=20, rotation_deg=20):
        self.robot = robot
        self.camera = camera
        self.required_landmarks = set(required_landmarks)
        self.step_cm = float(step_cm)      # default forward step per call (cm)
        self.rotation_deg = float(rotation_deg)

        self.observed_landmarks = set()
        self.all_seen = False

    def explore_step(self, drive=False, min_dist=400):
        dist = 0.0
        angle_deg = self.rotation_deg
        angle_rad = np.radians(angle_deg)

        if self.all_seen:
            return 0.0, 0.0

        if not drive:
            self.robot.turn_angle(angle_deg)
            time.sleep(0.2)

        if drive:
            dist = self.step_cm
            left, center, right = self.robot.proximity_check()

            if left < min_dist or center < min_dist or right < min_dist:
                self.robot.stop()
            if left > right:
                self.robot.turn_angle(45)
                angle_rad = np.radians(45)
            else:
                self.robot.turn_angle(-45)
                angle_rad = np.radians(-45)

            self.robot.drive_distance_cm(dist)

        frame = self.camera.get_next_frame()
        objectIDs, dists, angles = self.camera.detect_aruco_objects(frame)
        if objectIDs is not None:
            self.observed_landmarks.update(objectIDs)

        self.all_seen = self.required_landmarks.issubset(self.observed_landmarks)

        return dist, angle_rad

    def seen_all_landmarks(self):
        return self.all_seen

    def move_towards_goal_step(self, est_pose, center, step_cm=None):
        """
        Small, safe adjustments:
        - use class step_cm by default,
        - rotate a little first if misaligned (no drive in same iteration),
        - smaller forward steps when close to target.
        """
        # ---- tiny tunables (just for final approach) ----
        CENTER_TOL_CM = 10.0                 # "arrived" threshold
        ALIGN_THRESH_RAD = np.radians(8.0)   # rotate-only if heading error > this
        MAX_TURN_STEP_RAD = np.radians(12.0) # cap per-call turn magnitude
        NEAR_RADIUS_CM = 80.0                # start fine steps inside this
        MIN_STEP_CM = 2.0                    # don't command < 2 cm
        # -------------------------------------------------

        if step_cm is None:
            step_cm = self.step_cm  # use your class default

        # current pose and goal (cm, rad)
        rx, ry, rth = est_pose.getX(), est_pose.getY(), est_pose.getTheta()
        gx, gy = float(center[0]), float(center[1])

        dx, dy = gx - rx, gy - ry
        dist_to_center = float(np.hypot(dx, dy))
        angle_to_center = math.atan2(dy, dx) - rth
        # normalize to [-pi, pi]
        angle_to_center = math.atan2(math.sin(angle_to_center), math.cos(angle_to_center))

        # 1) close enough? (no drive; caller will confirm across frames)
        if dist_to_center < CENTER_TOL_CM:
            print("reached center (within tolerance)")
            return 0.0, 0.0

        # 2) if misaligned, rotate a small, capped amount and return (don't drive this tick)
        if abs(angle_to_center) > ALIGN_THRESH_RAD:
            turn = max(-MAX_TURN_STEP_RAD, min(MAX_TURN_STEP_RAD, angle_to_center))
            self.robot.turn_angle(np.degrees(turn))
            return 0.0, turn

        # 3) move forward; smaller steps when near the goal
        if dist_to_center < NEAR_RADIUS_CM:
            # take at most 25% of remaining distance, but not less than MIN_STEP_CM
            move_distance = min(step_cm, max(MIN_STEP_CM, 0.25 * dist_to_center))
        else:
            # farther out: take up to 50% of remaining distance but bounded by step_cm
            move_distance = min(step_cm, 0.5 * dist_to_center)

        print(f"distance moved: {move_distance:.1f} cm, heading error: {np.degrees(angle_to_center):.1f}°")
        self.robot.drive_distance_cm(move_distance)

        # we already aligned above, so no extra turn here
        return move_distance, 0.0
