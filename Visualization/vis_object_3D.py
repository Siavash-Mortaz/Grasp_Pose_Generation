import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.spatial.transform import Rotation as R

# Function to load the points from an XYZ file
def load_xyz(file_path):
    points = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.strip():  # Ensure the line is not empty
                x, y, z = map(float, line.split())
                points.append([x, y, z])
    return np.array(points)

# Function to apply rotation and translation to points
def apply_transformations(points, obj_trans, obj_rot_matrix):
    # Apply rotation
    rotated_points = np.dot(points, obj_rot_matrix.T)
    # Apply translation
    translated_points = rotated_points + obj_trans
    return translated_points


# Load the data from the pickle file
filepath = r"D:\UNI\Sem3\Dissertation\My effort\HOnnotate\ho3d-master\Dataset\HO3D_v3\train\SS1\meta\0055.pkl"

with open(filepath, 'rb') as f:
    data = pickle.load(f)

    obj_trans = data.get('objTrans', np.zeros(3)) # Assuming this is a 3-element vector
    obj_rot = data.get('objRot', np.zeros(3)) # Assuming this is a 3-element vector
    obj_name = data['objName']

# Path to the points.xyz file
xyz_file_path = rf"D:\UNI\Sem3\Dissertation\My effort\HOnnotate\ho3d-master\Dataset\models\{obj_name}\points.xyz"

# Load the points
points = load_xyz(xyz_file_path)

# Apply transformations to the points
rotation = R.from_rotvec(obj_rot.flatten()).as_matrix()
transformed_points = apply_transformations(points, obj_trans, rotation)

# Extract x, y, z coordinates for transformed points
x = transformed_points[:, 0]
y = transformed_points[:, 1]
z = transformed_points[:, 2]

# Plot the points in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Create a 3D scatter plot
ax.scatter(x, y, z, c='b', marker='o')

# Set labels and title
ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.set_zlabel('Z Coordinate')
ax.set_title(f'3D Scatter Plot of Transformed Object Points \n Object File Name: ({obj_name})')

# Set the aspect ratio to be equal
ax.set_box_aspect([1, 1, 1])  # Aspect ratio is 1:1:1


plt.show()


