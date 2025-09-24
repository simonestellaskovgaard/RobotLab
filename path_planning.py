import matplotlib.pyplot as plt
from RobotUtils.CameraUtils import CameraUtils, ArucoUtils
from RobotUtils.LandmarkUtils import LandmarkUtils
from LandmarkOccupancyGrid import LandmarkOccupancyGrid

cam = CameraUtils()
cam.start_camera()
aruco = ArucoUtils()
landmark_utils = LandmarkUtils(cam, aruco, cube_side=0.05, robot_radius=0.1)
grid_map = LandmarkOccupancyGrid(low=(-2,-2), high=(2,2), res=0.05)

landmarks = landmark_utils.map_landmarks()
print(f"landmarks detected: {landmarks}")

landmark_radius = 0.15
landmarks_for_grid = [(pos[0], pos[1], landmark_radius) for _, pos in landmarks]
                      
grid_map.add_landmarks(landmarks_for_grid)

robot_pos = (0.0, 0.0)
robot_radius = 0.225

plt.figure(figsize=(5,5))
grid_map.draw_map(robot_pos=robot_pos, robot_radius=robot_radius)
plt.savefig("/home/robot/occupancy_grid.png", bbox_inches='tight')
plt.close()  
