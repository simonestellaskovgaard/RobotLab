# explore_landmarks.py
from RobotUtils.CalibratedRobot import CalibratedRobot
import camera
import time
import numpy as np
import math

class LocalizationPathing:
    def __init__(self, robot, camera, required_landmarks, step_cm=20, rotation_deg=20):
        """
        robot.turn_angle(deg): +deg turns RIGHT/CW on this platform.
        math radians: +rad is CCW. So we convert: math_angle = -np.radians(deg_sent_to_robot)
        """
        self.robot = robot
        self.camera = camera
        self.required_landmarks = set(required_landmarks)
        self.step_cm = float(step_cm)        # default forward step per call (cm)
        self.rotation_deg = float(rotation_deg)

        self.observed_landmarks = set()
        self.all_seen = False

    # ----------------------------- Exploration -----------------------------
    def explore_step(self, drive=False, min_dist=400):
        """
        Spin to look for landmarks; optionally take a forward step while avoiding proximity hits.
        Returns: (distance_cm, math_angle_rad)
                 math_angle_rad is the actual change in heading in mathematical convention (CCW positive).
        """
        dist = 0.0
        angle_deg_cmd = self.rotation_deg

        if self.all_seen:
            return 0.0, 0.0

        # Rotate in place to scan
        if not drive:
            # Robot API: +deg => RIGHT/CW. Math: CW is negative radians.
            self.robot.turn_angle(angle_deg_cmd)
            angle_rad_math = -np.radians(angle_deg_cmd)
            time.sleep(0.2)
        else:
            # Take a small forward step with a bias away from closer side
            dist = self.step_cm
            left, center, right = self.robot.proximity_check()

            if left < min_dist or center < min_dist or right < min_dist:
                self.robot.stop()

            if left > right:
                self.robot.turn_angle(45)          # robot turns right/CW by +45
                angle_rad_math = -np.radians(45)   # math negative
            else:
                self.robot.turn_angle(-45)         # robot turns left/CCW by -45
                angle_rad_math = -np.radians(-45)  # math positive

            self.robot.drive_distance_cm(dist)

        # Sense and update which landmarks we’ve seen
        frame = self.camera.get_next_frame()
        objectIDs, dists, angles = self.camera.detect_aruco_objects(frame)
        if objectIDs is not None:
            self.observed_landmarks.update(objectIDs)

        self.all_seen = self.required_landmarks.issubset(self.observed_landmarks)

        return dist, float(angle_rad_math)

    def seen_all_landmarks(self):
        """True if all required landmarks have been observed at least once."""
        return self.all_seen

    # ------------------------- Go-to-midpoint step -------------------------
    def move_towards_goal_step(self, est_pose, center, step_cm=None):
        """
        Rotate-towards then take a forward step (both small and gentle).
        Returns (distance_cm, math_angle_rad_applied)
        """
        # ---- tiny tunables (minimal!) ----
        CENTER_TOL_CM       = 5.0                 # consider arrived if closer than this
        ALIGN_DEADBAND_RAD  = math.radians(5.0)   # don't rotate for tiny errors
        MAX_TURN_STEP_RAD   = math.radians(15.0)  # cap per-call turn magnitude
        # ----------------------------------

        if step_cm is None:
            step_cm = self.step_cm

        # Current pose and goal (cm, rad)
        rx, ry = float(est_pose.getX()), float(est_pose.getY())
        rth    = float(est_pose.getTheta())
        gx, gy = float(center[0]), float(center[1])

        dx, dy = gx - rx, gy - ry
        distance_to_center = float(np.hypot(dx, dy))

        # Heading error in math convention (CCW positive)
        heading_error = math.atan2(dy, dx) - rth
        heading_error = math.atan2(math.sin(heading_error), math.cos(heading_error))  # wrap [-pi, pi]

        # Close enough? (no motion)
        if distance_to_center < CENTER_TOL_CM:
            print("reached center")
            return 0.0, 0.0

        # KEY: Convert math desired turn to robot command degrees (robot +deg = right/CW = math negative)
        # We cap and apply in *robot* space, then return the *math* angle actually applied.
        applied_turn_math = 0.0

        if abs(heading_error) > ALIGN_DEADBAND_RAD:
            # We want to reduce heading_error toward 0 in math space.
            # Robot needs: turn_deg_cmd = -applied_turn_math_deg
            # Choose a small turn toward reducing the error:
            turn_step_math = max(-MAX_TURN_STEP_RAD, min(MAX_TURN_STEP_RAD, heading_error))
            turn_deg_cmd = -math.degrees(turn_step_math)  # convert to robot's sign
            self.robot.turn_angle(turn_deg_cmd)
            applied_turn_math = turn_step_math  # what the filter should integrate
        else:
            applied_turn_math = 0.0

        # Drive forward a modest step toward the goal
        move_distance = min(step_cm, distance_to_center)
        self.robot.drive_distance_cm(move_distance)

        # Return EXACTLY what we commanded in math convention
        return float(move_distance), float(applied_turn_math)
