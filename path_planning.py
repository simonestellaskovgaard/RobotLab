import matplotlib.pyplot as plt
from RobotUtils.CameraUtils import CameraUtils, ArucoUtils
from RobotUtils.LandmarkUtils import LandmarkUtils
from LandmarkOccupancyGrid import LandmarkOccupancyGrid
from robot_model import RobotModel
from robot_RRT import robot_RRT

cam = CameraUtils()
cam.start_camera()
aruco = ArucoUtils()
landmark_utils = LandmarkUtils(cam, aruco, cube_side=0.05, robot_radius=0.1)
grid_map = LandmarkOccupancyGrid(low=(-3,-3), high=(3, 3), res=0.05)

landmarks = landmark_utils.map_landmarks()
print(f"landmarks detected: {landmarks}")

landmark_radius = 0.165
landmarks_for_grid = [(pos[0], pos[1], landmark_radius) for _, pos in landmarks]
                      
grid_map.add_landmarks(landmarks_for_grid)



plt.figure(figsize=(5,5))
grid_map.draw_map()
plt.savefig("occupancy_grid.png", bbox_inches='tight')
plt.close()  

robot = RobotModel()

path_res = 0.05

rrt = robot_RRT(
    start=[0, 0],
    goal=[0, 1.9],
    robot_model=robot,
    map=grid_map,
    expand_dis=0.2,
    path_resolution=path_res,
    )

map.add_landma