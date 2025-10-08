import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import particle
import camera
import numpy as np
import time
from timeit import default_timer as timer
from RobotUtils.CalibratedRobot import CalibratedRobot
from scipy.stats import norm
import math
from LocalizationPathing import LocalizationPathing
import random
import cv2
from LandmarkOccupancyGrid import LandmarkOccupancyGrid

# Flags
showGUI = False  # Whether or not to open GUI windows
onRobot = True   # Whether or not we are running on the Arlo robot

def isRunningOnArlo():
    """Return True if we are running on Arlo, otherwise False."""
    return onRobot

try:
    from RobotUtils.Robot import Robot
except ImportError:
    print("selflocalize.py: robot module not present - forcing not running on Arlo!")
    onRobot = False

# ------------------ TUNING KNOBS (added) ------------------
STABILIZE_STEPS = 2          # short settle after landmarks acquired
CENTER_TOL_CM = 10.0         # how close counts as "at center"
GOAL_HITS_REQUIRED = 5       # require K consecutive confirmations
SLOW_DOWN_RADIUS_CM = 80.0   # start taking small steps when closer than this
MIN_STEP_CM = 2.0            # never smaller than this (cm)

# Spin slowdown (slightly slower when turning-in-place)
SPIN_SLOWDOWN_THRESHOLD = 0.25  # rad; if commanded turn exceeds this, slow it a bit
SPIN_SLOWDOWN_SCALE = 0.7       # scale big turns down slightly
MAX_TURN_RAD = 0.12             # cap the per-iteration turn magnitude (rad)
# -----------------------------------------------------------

# Some color constants in BGR format
CRED = (0, 0, 255)
CGREEN = (0, 255, 0)
CBLUE = (255, 0, 0)
CCYAN = (255, 255, 0)
CYELLOW = (0, 255, 255)
CMAGENTA = (255, 0, 255)
CWHITE = (255, 255, 255)
CBLACK = (0, 0, 0)

# Landmarks in centimeters
landmarkIDs = [6, 7]
landmarks = {
    6: (0.0, 0.0),
    7: (300.0, 0.0)
}

center = np.array([(landmarks[6][0] + landmarks[7][0]) / 2,
                   (landmarks[6][1] + landmarks[7][1]) / 2])

landmark_colors = [CRED, CGREEN]  # Colors used when drawing the landmarks

def jet(x):
    """Colour map for drawing particles."""
    r = (x >= 3.0/8.0 and x < 5.0/8.0) * (4.0 * x - 3.0/2.0) + (x >= 5.0/8.0 and x < 7.0/8.0) + (x >= 7.0/8.0) * (-4.0 * x + 9.0/2.0)
    g = (x >= 1.0/8.0 and x < 3.0/8.0) * (4.0 * x - 1.0/2.0) + (x >= 3.0/8.0 and x < 5.0/8.0) + (x >= 5.0/8.0 and x < 7.0/8.0) * (-4.0 * x + 7.0/2.0)
    b = (x < 1.0/8.0) * (4.0 * x + 1.0/2.0) + (x >= 1.0/8.0 and x < 3.0/8.0) + (x >= 3.0/8.0 and x < 5.0/8.0) * (-4.0 * x + 5.0/2.0)
    return (255.0*r, 255.0*g, 255.0*b)

def draw_world(est_pose, particles, world):
    """Draw robots position in the world coordinate system."""
    offsetX = 100
    offsetY = 250
    ymax = world.shape[0]
    world[:] = CWHITE

    max_weight = 0
    for p in particles:
        max_weight = max(max_weight, p.getWeight())

    for p in particles:
        x = int(p.getX() + offsetX)
        y = ymax - (int(p.getY() + offsetY))
        colour = jet(p.getWeight() / max_weight if max_weight > 0 else 0.0)
        cv2.circle(world, (x, y), 2, colour, 2)
        b = (int(p.getX() + 15.0*np.cos(p.getTheta())) + offsetX,
             ymax - (int(p.getY() + 15.0*np.sin(p.getTheta())) + offsetY))
        cv2.line(world, (x, y), b, colour, 2)

    for i in range(len(landmarkIDs)):
        ID = landmarkIDs[i]
        lm = (int(landmarks[ID][0] + offsetX), int(ymax - (landmarks[ID][1] + offsetY)))
        cv2.circle(world, lm, 5, landmark_colors[i], 2)

    a = (int(est_pose.getX()) + offsetX, ymax - (int(est_pose.getY()) + offsetY))
    b = (int(est_pose.getX() + 15.0*np.cos(est_pose.getTheta())) + offsetX,
         ymax - (int(est_pose.getY() + 15.0*np.sin(est_pose.getTheta())) + offsetY))
    cv2.circle(world, a, 5, CMAGENTA, 2)
    cv2.line(world, a, b, CMAGENTA, 2)

def initialize_particles(num_particles):
    particles = []
    for i in range(num_particles):
        p = particle.Particle(600.0*np.random.ranf() - 100.0,
                              600.0*np.random.ranf() - 250.0,
                              np.mod(2.0*np.pi*np.random.ranf(), 2.0*np.pi),
                              1.0/num_particles)
        particles.append(p)
    return particles

def sample_motion_model(particles_list, distance, angle, sigma_d, sigma_theta):
    for p in particles_list:
        delta_x = distance * np.cos(p.getTheta() + angle)
        delta_y = distance * np.sin(p.getTheta() + angle)
        particle.move_particle(p, delta_x, delta_y, angle)
    particle.add_uncertainty_von_mises(particles_list, sigma_d, sigma_theta)

def motion_model_with_map(particle_obj, distance, angle, sigma_d, sigma_theta, grid):
    indices, valid = grid.world_to_grid([particle_obj.getX(), particle_obj.getY()])
    p_map = 0 if (not valid or grid.in_collision(indices)) else 1
    if p_map:
        sample_motion_model([particle_obj], distance, angle, sigma_d, sigma_theta)
        return 1
    return 0

def sample_motion_model_with_map(particles_list, distance, angle, sigma_d, sigma_theta, grid, max_tries=10):
    for p in particles_list:
        for attempt in range(max_tries):
            pi = motion_model_with_map(p, distance, angle, sigma_d, sigma_theta, grid)
            if pi > 0:
                break
        else:
            p.setWeight(0.01)  # low weight to indicate invalid

def measurement_model(particle_list, landmarkIDs, dists, angles, sigma_d, sigma_theta):
    for p in particle_list:
        x_i = p.getX()
        y_i = p.getY()
        theta_i = p.getTheta()
        p_obs = 1.0
        for landmarkID, dist, ang in zip(landmarkIDs, dists, angles):
            if landmarkID in landmarkIDs:
                l_x, l_y = landmarks[landmarkID]
                d_i = np.sqrt((l_x - x_i)**2 + (l_y - y_i)**2)
                p_d_m = norm.pdf(dist, loc=d_i, scale=sigma_d)
                e_theta = np.array([np.cos(theta_i), np.sin(theta_i)])
                e_theta_hat = np.array([-np.sin(theta_i), np.cos(theta_i)])
                e_l = np.array([l_x - x_i, l_y - y_i]) / max(d_i, 1e-9)
                phi_i = np.sign(np.dot(e_l, e_theta_hat)) * np.arccos(np.clip(np.dot(e_l, e_theta), -1.0, 1.0))
                p_phi_m = norm.pdf(ang, loc=phi_i, scale=sigma_theta)
                p_obs *= (p_d_m * p_phi_m)
        p.setWeight(p_obs)

def resample_particles(particle_list, weights, w_fast, w_slow):
    cdf = np.cumsum(weights)
    resampled = []
    for _ in range(len(particle_list)):
        ratio = (w_fast / w_slow) if w_slow > 0 else 1.0
        if random.random() < max(0.0, 1.0 - ratio):
            p_new = initialize_particles(1)[0]
            resampled.append(p_new)
        else:
            z = np.random.rand()
            idx = np.searchsorted(cdf, z)
            p_src = particle_list[idx]
            p_res = particle.Particle(p_src.getX(), p_src.getY(), p_src.getTheta(), 1.0/(len(particle_list)))
            resampled.append(p_res)
    return resampled

def filter_landmarks_by_distance(objectIDs, dists, angles):
    """Keep only the closest measurement for each landmark ID."""
    min_dist = {}
    for id_, d, a in zip(objectIDs, dists, angles):
        if id_ not in min_dist or d < min_dist[id_][0]:
            min_dist[id_] = (d, a)
    filtered_ids = list(min_dist.keys())
    filtered_dists = [min_dist[ID][0] for ID in filtered_ids]
    filtered_angles = [min_dist[ID][1] for ID in filtered_ids]
    return filtered_ids, filtered_dists, filtered_angles

# Main program #
try:
    if showGUI:
        WIN_RF1 = "Robot view"
        cv2.namedWindow(WIN_RF1)
        cv2.moveWindow(WIN_RF1, 50, 50)
        WIN_World = "World view"
        cv2.namedWindow(WIN_World)
        cv2.moveWindow(WIN_World, 500, 50)

    # Initialize particles
    num_particles = 1000
    particles = initialize_particles(num_particles)

    est_pose = particle.estimate_pose(particles)  # The estimate of the robots current pose
    print(f"estimated pose: {est_pose}")

    # Driving parameters
    distance = 0.0  # distance driven at this time step (cm)
    angle = 0.0     # angle turned at this timestep (rad)

    # Motion / measurement noise
    sigma_d = 10
    sigma_theta = 0.2

    # Adaptive resampling accumulators (guarded)
    w_slow = 1e-9
    w_fast = 1e-9
    alpha_slow = 0.01
    alpha_fast = 0.10

    # Initialize the robot
    if isRunningOnArlo():
        arlo = CalibratedRobot()

    # world view canvas
    world = np.zeros((500, 500, 3), dtype=np.uint8)
    draw_world(est_pose, particles, world)

    print("Opening and initializing camera")
    if isRunningOnArlo():
        cam = camera.Camera(1, robottype='arlo', useCaptureThread=False)
        pathing = LocalizationPathing(arlo, cam, landmarkIDs)

        # --- added state ---
        stabilization_counter = 0
        landmarks_acquired = False
        goal_hits = 0
    else:
        cam = camera.Camera(0, robottype='macbookpro', useCaptureThread=False)

    while True:
        action = cv2.waitKey(10)
        if action == ord('q'):
            break

        if not isRunningOnArlo():
            if action == ord('w'):
                distance = 10.0
            elif action == ord('x'):
                distance = -10.0
            elif action == ord('a'):
                angle = 0.2
            elif action == ord('d'):
                angle = -0.2
            else:
                distance = 0.0
                angle = 0.0

        # --- Arlo motion policy (with tiny local fixes) ---
        if isRunningOnArlo():
            if not landmarks_acquired:
                if pathing.seen_all_landmarks():
                    landmarks_acquired = True
                    stabilization_counter = 0
                    distance, angle = 0.0, 0.0
                else:
                    drive = random.random() < (1/18)
                    distance, angle = pathing.explore_step(drive)
            else:
                if stabilization_counter < STABILIZE_STEPS:
                    stabilization_counter += 1
                    distance, angle = 0.0, 0.0
                else:
                    # distance to goal from current estimate
                    dx = float(center[0] - est_pose.getX())
                    dy = float(center[1] - est_pose.getY())
                    dist_to_goal = math.hypot(dx, dy)

                    if dist_to_goal <= CENTER_TOL_CM:
                        goal_hits += 1
                        distance, angle = 0.0, 0.0
                        if goal_hits >= GOAL_HITS_REQUIRED:
                            print("[INFO] Reached center (confirmed).")
                            # Keep position (or break to stop program)
                            # break
                    else:
                        goal_hits = 0
                        distance, angle = pathing.move_towards_goal_step(est_pose, center)

                        # Slow down linear step near goal to avoid overshoot
                        if dist_to_goal < SLOW_DOWN_RADIUS_CM:
                            max_allowed = max(MIN_STEP_CM, 0.25 * dist_to_goal)
                            distance = min(distance, max_allowed)

            # Slightly slow spinning: cap/scale large turn commands
            if abs(angle) > SPIN_SLOWDOWN_THRESHOLD:
                angle = math.copysign(abs(angle) * SPIN_SLOWDOWN_SCALE, angle)
            angle = math.copysign(min(abs(angle), MAX_TURN_RAD), angle)

        # Apply motion model to particles
        sample_motion_model(particles, distance, angle, sigma_d, sigma_theta)

        # Fetch next frame
        colour = cam.get_next_frame()

        # Detect objects
        objectIDs, dists, angles = cam.detect_aruco_objects(colour)
        if not isinstance(objectIDs, type(None)):
            objectIDs, dists, angles = filter_landmarks_by_distance(objectIDs, dists, angles)
            for i in range(len(objectIDs)):
                print("Object ID = ", objectIDs[i], ", Distance = ", dists[i], ", angle = ", angles[i])

            # Compute particle weights
            measurement_model(particles, objectIDs, dists, angles, sigma_d, sigma_theta)

            # Normalize + adaptive resampling
            weights = np.array([p.getWeight() for p in particles], dtype=np.float64)
            w_sum = weights.sum()
            if w_sum <= 0 or not np.isfinite(w_sum):
                weights[:] = 1.0 / len(weights)
            else:
                weights /= w_sum

            w_avg = float(weights.mean())
            w_slow += alpha_slow * (w_avg - w_slow)
            w_fast += alpha_fast * (w_avg - w_fast)

            particles = resample_particles(particles, weights, w_fast, w_slow)

            # Draw detected objects
            cam.draw_aruco_objects(colour)
        else:
            # No observation - reset weights to uniform distribution
            for p in particles:
                p.setWeight(1.0/num_particles)

        est_pose = particle.estimate_pose(particles)

        if showGUI:
            draw_world(est_pose, particles, world)
            cv2.imshow("Robot view", colour)
            cv2.imshow("World view", world)

finally:
    cv2.destroyAllWindows()
    try:
        cam.terminateCaptureThread()
    except:
        pass
