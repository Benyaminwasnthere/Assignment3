import argparse
import math

import numpy as np
import argparse
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.transforms import Affine2D
import random

dt = 0.1
length = 0.1
width = 0.2
#------------------------------------------
def global_landmark_positions(robot_x, robot_y, robot_theta, landmarks_local):
    landmarks_global = []

    for t in landmarks_local:
        distance, angle_rad = t
        landmark_x = robot_x + distance * np.cos(robot_theta + angle_rad)
        landmark_y = robot_y + distance * np.sin(robot_theta + angle_rad)
        landmarks_global.append([landmark_x, landmark_y])

    return landmarks_global





# ------------------------------------------
def convert_structure_controls(controls):
    initial_location = controls[0]
    control_sequence = controls[1:]
    q = initial_location
    trajectory = []
    trajectory.append([q[0], q[1], q[2]])

    # Simulate robot movement
    for i in range(len(control_sequence)):
        dq = differential_drive_model(q, control_sequence[i])
        q += dq
        trajectory.append([q[0], q[1], q[2]])

    Odometry = np.array(trajectory, dtype=object)

    return Odometry


# ------------------------------------------
def next_position(control, q):
    new_pos = differential_drive_model(q, control)
    return new_pos


# ------------------------------------------


def differential_drive_model(q, u):
    dq = np.zeros_like(q)
    dq[0] = u[0] * np.cos(q[2]) * dt
    dq[1] = u[0] * np.sin(q[2]) * dt
    dq[2] = u[1] * dt
    return dq


def draw_rotated_rectangle(ax, center, width, height, angle_degrees, color='r'):
    x, y = center
    rect = patches.Rectangle((x - width / 2, y - height / 2), width, height, linewidth=1, edgecolor=color,
                             facecolor='none')
    t = Affine2D().rotate_deg_around(x, y, angle_degrees) + ax.transData
    rect.set_transform(t)
    ax.add_patch(rect)


# ------------------------------------------


def seperate_data(sensing):
    controls = []
    sensor = []
    data = np.load(sensing, allow_pickle=True)
    for i in range(len(data)):
        if (i % 2 == 0):
            controls.append(data[i])
        else:
            sensor.append(data[i])

    return controls, sensor


# -----------------------------------------------
def sort_particles_by_weight(particles):
    sorted_particles = sorted(particles, key=lambda x: x[1], reverse=True)
    return sorted_particles
# -----------------------------------------------
def landmark_sensor(ground_truth_x, ground_truth_y, ground_truth_theta, landmarks):
    robot_position = np.array([ground_truth_x, ground_truth_y])
    robot_orientation = ground_truth_theta
    t = []
    distances, angles = [], []
    for landmark in landmarks:
        delta = landmark - robot_position
        distance = np.linalg.norm(delta)
        angle = np.degrees(np.arctan2(delta[1], delta[0]) - robot_orientation)
        angle = (angle + 360) % 360  # Wrap the angle to the range [0, 360]

        distances.append(distance)
        angles.append(np.radians(angle))
        t.append([distance, np.radians(angle)])
    return t

# -----------------------------------------------
def particle_filter(prior_particle, control, measurement,landmarks):
    particles = []
    n = 0
    sigma_distance = 0.02
    sigma_direction = 0.02

    prior_particle = sort_particles_by_weight(prior_particle)
    prior = np.array(prior_particle, dtype=object)
    for i in range(len(prior_particle)):
        particle = np.random.choice(prior[:, 0], 1, p=prior[:, 1].astype(np.float64))[0]
        new_pos = particle + next_position([control[0] + np.random.normal(0, sigma_distance), control[1] + np.random.normal(0, sigma_direction)],particle)
        particle_observation = landmark_sensor(new_pos[0], new_pos[1], new_pos[2], landmarks)

        # Calculate joint Gaussian probability (likelihood) for all landmarks
        likelihood = 1.0
        for j in range(len(measurement)):
            distance_diff = measurement[j][0] - particle_observation[j][0]
            direction_diff = measurement[j][1] - particle_observation[j][1]

            likelihood *= (
                    (1.0 / (np.sqrt(2 * np.pi) * sigma_distance)) *
                    np.exp(-0.5 * ((distance_diff / sigma_distance) ** 2))
            )
            likelihood *= (
                    (1.0 / (np.sqrt(2 * np.pi) * sigma_direction)) *
                    np.exp(-0.5 * ((direction_diff / sigma_direction) ** 2))
            )

        # Update particle weight based on joint likelihood
        new_weight = likelihood
        new_weight+=0.00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001
        n += new_weight
        particles.append([new_pos, new_weight])



    for i in range(len(particles)):
        particles[i][1] /= n
    particles=sort_particles_by_weight(particles)
    return particles


# ------------------------------------------
def animate(controls, measurements, initial, landmarks_map, length, width, particles):
    fig, ax = plt.subplots(figsize=(6, 6))
    trajectory = []
    trajectory_estimate = []
    q = initial.copy()
    for i in range(len(controls)):
        plt.clf()
        ax = plt.gca()
        plt.xlim(0, 2)
        plt.ylim(0, 2)
        q += next_position(controls[i], q)
        trajectory.append([q[0], q[1], q[2]])
        particles = particle_filter(particles, controls[i], measurements[i],landmarks_map)
        plt.scatter(landmarks_map[:, 0], landmarks_map[:, 1], marker='o', label='Landmarks Map 1')

        # Plot Particles
        particle_positions = np.array([particle[0] for particle in particles])
        plt.scatter(particle_positions[:, 0], particle_positions[:, 1], marker='.', color='orange')
        average_position = np.mean(particle_positions, axis=0)
        trajectory_estimate.append([average_position[0], average_position[1], average_position[2]])
        # Draw path
        path_array = np.array(trajectory)
        plt.plot(path_array[:, 0], path_array[:, 1], color='red', linestyle='--', linewidth=2)

        # Draw path
        path_array_estimate = np.array(trajectory_estimate)
        plt.plot(path_array_estimate[:, 0], path_array_estimate[:, 1], color='black', linestyle='--', linewidth=2)
        # Draw line segments between average position and landmarks
        if i > 1:
            landmarks_previous = global_landmark_positions(average_position[0], average_position[1],
                                                           average_position[2],
                                                           measurements[i])
            for landmark in landmarks_previous:
                plt.plot([average_position[0], landmark[0]], [average_position[1], landmark[1]], color='blue',
                         linestyle=':', linewidth=0.5)
                plt.scatter(np.array(landmarks_previous)[:, 0], np.array(landmarks_previous)[:, 1], color='red', marker='x', label='Landmarks')

        # Draw robot body
        draw_rotated_rectangle(ax, [q[0], q[1]], length, width, np.degrees(q[2]))
        draw_rotated_rectangle(ax, [average_position[0], average_position[1]], length, width,
                               np.degrees(average_position[2]), 'black')
        plt.pause(0.05)
    plt.show()


# ------------------------------------------
def create_initial_particle(initial, n):
    particle = [[initial, 1 / n] for i in range(n)]
    return particle


# ------------------------------------------
if __name__ == "__main__":
    #--map maps/landmark_0.npy --sensing readings/readings_0_1_H.npy --num_particles 20
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Particle Filter for Robot Pose Estimation")
    parser.add_argument("--map", type=str, help="Path to the map file")
    parser.add_argument("--sensing", type=str, help="Path to the readings file")
    parser.add_argument("--num_particles", type=int, help="Number of particles")
    parser.add_argument("--estimates", type=str, help="Path to save pose estimates")

    args = parser.parse_args()
    controls, measurements = seperate_data(args.sensing)
    initial_location = controls[0]
    control_sequence = controls[1:]
    particle = create_initial_particle(initial_location, args.num_particles)
    landmarks_map = np.load(args.map)
    animate(control_sequence, measurements, initial_location, landmarks_map, length, width, particle)
