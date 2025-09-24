import time
import numpy as np
import math
import RobotUtils.Robot as robot

class CalibratedRobot:
    def __init__(self):
        self.arlo = robot.Robot()

        self.TRANSLATION_TIME = 2.5 # time to drive 1 meter at default speed (64)
        self.TURN_TIME = 0.89 # time to turn 90 degrees at default speed (64)
        
        #ratio for adjusting the wheels to have the same power
        self.CAL_KL = 0.97
        self.CAL_KR = 1.0
        
        self.MIN_PWR = 40
        self.MAX_PWR = 127

        self.default_speed = 64  
        self.FORWARD = 1
        self.BACKWARD = 0
        
        self.stop_distance = 200

    def clamp_power(self, p):
        return max(self.MIN_PWR, min(self.MAX_PWR, int(round(p))))
    
    def drive(self, leftSpeed, rightSpeed, leftDir, rightDir):
        l = self.clamp_power(leftSpeed * self.CAL_KL) if leftSpeed > 0 else 0
        r = self.clamp_power(rightSpeed * self.CAL_KR) if rightSpeed > 0 else 0
        self.arlo.go_diff(l, r, leftDir, rightDir)

    def drive_distance(self, meters, direction=None, speed=None,):
        """Drive a certain amount of meters at a given speed."""
        if speed is None:
            speed = self.default_speed
        if direction is None:
            direction = self.FORWARD
        #The formula for the duration to drive the desired meters: duration = TRANSLATION_TIME * meters​ * (default speed /current speed​)
        duration = self.TRANSLATION_TIME * meters * (self.default_speed / speed)
        self.drive(speed, speed, direction, direction)
        time.sleep(duration)
        self.arlo.stop()

    def turn_angle(self, angleDeg, speed=None):
        """Turn a given angle in degrees at a given speed. Positive = left, negative = right."""
        if speed is None:
            speed = self.default_speed
        #The formula for the duration to turn the desired angle: duration = TURN_TIME *  (abs(angle) / 90.0) * (default_speed /current speed)
        duration = self.TURN_TIME * (abs(angleDeg) / 90.0) * (self.default_speed / speed)
        if angleDeg > 0:
            self.drive(speed, speed, self.BACKWARD, self.FORWARD)  # left
            time.sleep(duration)
            self.arlo.stop()
        else:
            self.drive(speed, speed, self.FORWARD, self.BACKWARD)  # right
            time.sleep(duration)
            self.arlo.stop()
            
    def forward_proximity_check(self, center_dist = 0, left_dist = 0, right_dist = 0):
        left = self.arlo.read_left_ping_sensor()
        center = self.arlo.read_front_ping_sensor()
        right = self.arlo.read_right_ping_sensor()
        if center < center_dist or left < left_dist or right < right_dist:
            self.arlo.stop()

    def follow_path(self, path, start_theta=0.0):
        """
        Follow a path of points [(x0, y0), (x1, y1), ...].
        Robot turns in place towards the next point, then drives straight.
        start_theta: initial orientation of the robot in radians (0 = +x axis)
        """
        current_theta = start_theta  # robot's current heading in radians

        for i in range(len(path) - 1):
            p1 = np.array(path[i])
            p2 = np.array(path[i + 1])
            delta = p2 - p1

            # Calculate dot product
            dot_product = np.dot(p1, p2)

            # Calculate magnitudes (lengths of the vectors)
            magnitude_A = np.linalg.norm(p1)
            magnitude_B = np.linalg.norm(p2)

            # Calculate angle in radians
            angle_radians = np.arccos(dot_product / (magnitude_A * magnitude_B))

            # Convert radians to degrees
            angle_degrees = np.degrees(angle_radians)
            distance = np.linalg.norm(delta)

            print(f"desired angle (rad): {angle_degrees}")
            print(f"distance: {distance}")

            # Turn robot by the relative angle
            #self.turn_angle(-math.degrees(angle_to_turn))  # convert to degrees if needed
            # Drive forward
            #self.drive_distance(distance)

               
    def stop(self):
        self.arlo.stop()