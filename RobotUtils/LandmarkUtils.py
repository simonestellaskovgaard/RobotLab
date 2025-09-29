import numpy as np
import matplotlib.pyplot as plt

class LandmarkUtils:
    def __init__(self, cam, aruco, cube_side=0.05, robot_radius=0.1):
        self.cam = cam
        self.aruco = aruco
        self.cube_side = cube_side
        self.robot_radius = robot_radius
        self.landmarks = []

    def map_landmarks(self):
        frame = self.cam.get_frame()
        cv2.imwrite("frame.png", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        plt.imsave("frame.png", frame)
        corners, ids = self.aruco.detect_markers(frame)
        self.landmarks = []

        if ids is not None:
            rvecs, tvecs = self.aruco.estimate_pose(corners, self.cam.camera_matrix)
            for i, marker_id in enumerate(ids.flatten()):
                tvec = tvecs[i][0]
                position_2d = (tvec[0], tvec[2])
                self.landmarks.append((marker_id, position_2d))

        return self.landmarks

    def plot_landmark_map(self, landmark_map):
        plt.figure()
        for marker_id, (x, z) in landmark_map:
            plt.scatter(x, z, label=f"ID {marker_id}")
            plt.text(x + 0.02, z + 0.02, str(marker_id))
        plt.scatter(0, 0, c='red', marker='x', s=100, label='Camera')
        plt.xlabel("x (m) — sideways")
        plt.ylabel("z (m) — forward")
        plt.title("Landmark Map")
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.show()

    


