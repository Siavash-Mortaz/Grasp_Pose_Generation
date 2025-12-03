import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.transform import Rotation as R

import os
import pickle

# 3D Visualizing Functions
def plot_hand_3d(handJoints3D, handTrans, handPose,objname):
    # Applying translation
    handJoints3D_transformed = handJoints3D + handTrans

    # Converting axis-angle to rotation matrix and applying rotation
    rotation = R.from_rotvec(handPose[:3])
    handJoints3D_transformed = rotation.apply(handJoints3D_transformed)

    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Extracting coordinates
    xs = handJoints3D_transformed[:, 0]
    ys = handJoints3D_transformed[:, 1]
    zs = handJoints3D_transformed[:, 2]

    # Plotting joints
    ax.scatter(xs, ys, zs, c='r', marker='o')

    # Connecting joints to form the skeleton (assuming MANO joint order)
    connections = [
        (0, 7), (7, 8), (8, 9), (9, 20),  # Little finger
        (0, 10), (10, 11), (11, 12), (12, 19),  # Ring finger
        (0, 4), (4, 5), (5, 6), (6, 18),  # Middle finger
        (0, 1), (1, 2), (2, 3), (3, 17),  # Index finger
        (0, 13), (13, 14), (14, 15), (15, 16)  # Thumb
    ]

    for (i, j) in connections:
        ax.plot([xs[i], xs[j]], [ys[i], ys[j]], [zs[i], zs[j]], 'b')

    # Setting labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'3D Scatter Plot of Transformed Hand pose \n Based on Object File Name: ({objname})')

    plt.show()


# Load Data
filepath = r"D:\UNI\Sem3\Dissertation\My effort\HOnnotate\ho3d-master\Dataset\HO3D_v3\train\SMu42\meta\1000.pkl"

hand_poses = []
object_translations = []
object_rotations = []
hand_translations = []
object_corners_3d = []
hand_joints_3d = []

with open(filepath, 'rb') as f:
    data = pickle.load(f)
    # Extract required fields from the loaded data
    hand_pose = data.get('handPose', np.zeros((48, 1))) # Assuming this is a 48x1 vector
    hand_trans = data.get('handTrans', np.zeros(3))  # Optional field, default to zeros if not available
    hand_joints_3d_value = data.get('handJoints3D',np.zeros((21, 3)))  # Optional field, default to zeros if not available
    obj_name=data['objName']


# Example data


handJoints3D = hand_joints_3d_value
handTrans = hand_trans
handPose =  hand_pose


# Visualize the Hand
plot_hand_3d(handJoints3D, handTrans, handPose,obj_name)




