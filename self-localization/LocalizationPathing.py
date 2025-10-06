# explore_landmarks.py
from RobotUtils.CalibratedRobot import CalibratedRobot
import camera
import time
import numpy as np

import time

class LocalizationPathing:
    """
    Exploration strategy for searching landmarks.
    Allows step-by-step exploration while tracking which landmarks have been seen.
    """

    def __init__(self, robot, camera, required_landmarks, step_cm=20, rotation_deg=15):
        self.robot = robot
        self.camera = camera
        self.required_landmarks = set(required_landmarks)
        self.step_cm = step_cm
        self.rotation_deg = rotation_deg

        self.observed_landmarks = set()
        self.all_seen = False

    def explore_step(self, drive = False):
        """
        Moves the robot one step forward and rotates slightly.
        Updates the set of observed landmarks and all_seen flag.
        """
        dist = 0
        angle = self.rotation_deg

        if self.all_seen:
            return  0, 0 # Already done

        # Rotate slightly
        self.robot.turn_angle(self.rotation_deg)
        time.sleep(0.4)
        
        # Move forward
        if drive:
            dist = self.step_cm
            self.robot.drive_distance_cm(dist)

        # Capture image and detect landmarks
        frame = self.camera.get_next_frame()
        objectIDs, dists, angles = self.camera.detect_aruco_objects(frame)
        if objectIDs is not None:
            self.observed_landmarks.update(objectIDs)

        # Update boolean
        self.all_seen = self.required_landmarks.issubset(self.observed_landmarks)

        return dist, angle
        


    def seen_all_landmarks(self):
        """
        Returns True if all required landmarks have been observed.
        """
        return self.all_seen
    
    def move_towards_goal_step(self, est_pose, center, step_m=0.2):
        """
        Move a single step toward the center point between landmarks.
        Updates the robot orientation slightly and moves forward a small distance.
        """
        
        # Compute direction vector
        robot_pos = np.array([est_pose.getX(), est_pose.getY()])
        direction = center - robot_pos
        distance_to_center = np.linalg.norm(direction)
        angle_to_center = np.arctan2(direction[1], direction[0]) - est_pose.getTheta()
        
        # Normalize angle to [-pi, pi]
        angle_to_center = (angle_to_center + np.pi) % (2 * np.pi) - np.pi
        
        # Limit movement step to avoid overshooting
        move_distance = min(step_m, distance_to_center)

        print(f"distance: {move_distance}")
        print(f"angle: {angle_to_center}")
        
        # Rotate robot toward center
        self.robot.turn_angle(np.degrees(angle_to_center))
        
        # Move a small step forward
        self.robot.drive_distance_cm(move_distance)

        return move_distance, angle_to_center
        


