import argparse
import argparse
import math
import os

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



#-----------------------------------------------distance equations
def euclidean_distance(point1, point2):
    point1 = np.array(point1)
    point2 = np.array(point2)

    distance = np.linalg.norm(point1 - point2)

    return distance

def circular_distance(angle1, angle2):

    # Calculate the absolute angular difference
    angular_difference = abs(angle1 - angle2)

    # Ensure circularity by taking the minimum difference considering the circular boundary
    circular_distance = min(angular_difference, 2 * np.pi - angular_difference)

    return circular_distance




#-----------------------------------------------

def animate(landmarks_map,trajectory_a,estimate):
    fig, ax = plt.subplots(figsize=(6, 6))
    plt.clf()
    ax = plt.gca()
    plt.xlim(0, 2)
    plt.ylim(0, 2)
    translational=[]
    circle=[]

    for i in(range(len(execution)-1)):
        euclidean_distances=euclidean_distance([estimate[i][0],estimate[i][1]], [trajectory_a[i][0],trajectory_a[i][1]])
        circular_distances=circular_distance(estimate[i][2],trajectory_a[i][2])
        translational.append(euclidean_distances)
        circle.append(circular_distances)
        plt.clf()
        ax = plt.gca()
        plt.xlim(0, 2)
        plt.ylim(0, 2)
        q=estimate[i].copy()
        plt.scatter(landmarks_map[:, 0], landmarks_map[:, 1], marker='o')
        # Draw path
        path_array_a = np.array(trajectory_a[:i])
        plt.plot(path_array_a[:, 0], path_array_a[:, 1], color='blue', linestyle='--', linewidth=1)

        # Draw path
        path_array_estimate = np.array(estimate[:i])
        plt.plot(path_array_estimate[:, 0], path_array_estimate[:, 1], color='black', linestyle='--', linewidth=1)

        draw_rotated_rectangle(ax, [q[0], q[1]], length, width, np.degrees(q[2]),'black')
        plt.pause(0.05)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))

    ax1.set_title('Circular Distance')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Circular Distance')
    ax1.plot(range(len(execution) - 1), circle, marker='o', color='blue')
    ax1.grid(True)

    ax2.set_title('Euclidean Distance')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Euclidean Distance')
    ax2.plot(range(len(execution) - 1), translational, marker='o', color='red')
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


    plt.tight_layout()




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Evaluation Script")

    parser.add_argument("--map", type=str, help="Path to the map file")
    parser.add_argument("--execution", type=str, help="Path to the execution file")
    parser.add_argument("--estimates", type=str, help="Path to the estimates file")

    args = parser.parse_args()
    landmarks_map = np.load(args.map)
    execution = np.load(args.execution, allow_pickle=True)
    estimate = np.load(args.estimates, allow_pickle=True)
    animate(landmarks_map,execution,estimate)