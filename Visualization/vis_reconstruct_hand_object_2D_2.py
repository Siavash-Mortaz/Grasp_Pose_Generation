import pickle
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from Models_Trains.all_models import CVAE_02_2
from Models_Trains.all_models import CVAE_02_3
from Models_Trains.loading_data import loading_data

def mean_squared_error(pred, target):
    return F.mse_loss(pred, target)

def mean_absolute_error(pred, target):
    return F.l1_loss(pred, target)

def visualize_two_hands_and_object_2d(hand1Trans, hand1Joints3D, hand2Trans, hand2Joints3D, objTrans, objRot, points,objname,testNumber):
    """
    Visualize two hands and an object in 2D given their respective transformations and coordinates.

    Parameters:
    hand1Trans (np.array): A 3x1 vector representing the first hand's translation.
    hand1Pose (np.array): A 48x1 vector representing the first hand's rotation in axis-angle format.
    hand1Joints3D (np.array): A 21x3 matrix representing the 3D coordinates of the first hand's joints.
    hand2Trans (np.array): A 3x1 vector representing the second hand's translation.
    hand2Pose (np.array): A 48x1 vector representing the second hand's rotation in axis-angle format.
    hand2Joints3D (np.array): A 21x3 matrix representing the 3D coordinates of the second hand's joints.
    objTrans (np.array): A 3x1 vector representing the object's translation.
    objRot (np.array): A 3x1 vector representing the object's rotation in axis-angle format.
    points (np.array): A Nx3 matrix representing the 3D coordinates of the object's points.
    """
    # Apply hand translations
    hand1Joints3D_transformed = hand1Joints3D + hand1Trans
    hand2Joints3D_transformed = hand2Joints3D + hand2Trans

    # Apply object rotation and translation
    rotation_matrix = R.from_rotvec(objRot.flatten()).as_matrix()
    rotated_points = np.dot(points, rotation_matrix.T)
    translated_points = rotated_points + objTrans

    # Extract coordinates for transformed points
    obj_x = translated_points[:, 0]
    obj_y = translated_points[:, 1]

    # Extract coordinates for transformed hand joints
    hand1_x = hand1Joints3D_transformed[:, 0]
    hand1_y = hand1Joints3D_transformed[:, 1]

    hand2_x = hand2Joints3D_transformed[:, 0]
    hand2_y = hand2Joints3D_transformed[:, 1]

    fig, axs = plt.subplots(1, 2, figsize=(14, 7))

    # Plot Hand 1 and Object on X-Y plane
    axs[0].scatter(obj_x, obj_y, c='b', marker='o', label='Object')
    axs[0].scatter(hand1_x, hand1_y, c='r', marker='o', label='Generated Hand Joints')
    for (i, j) in [(0, 7), (7, 8), (8, 9), (9, 20), (0, 10), (10, 11), (11, 12), (12, 19),
                   (0, 4), (4, 5), (5, 6), (6, 18), (0, 1), (1, 2), (2, 3), (3, 17),
                   (0, 13), (13, 14), (14, 15), (15, 16)]:
        axs[0].plot([hand1_x[i], hand1_x[j]], [hand1_y[i], hand1_y[j]], 'b')
    axs[0].set_xlabel('X Coordinate')
    axs[0].set_ylabel('Y Coordinate')
    axs[0].set_title('Reconstructed Hand Pose - X-Y Plane')
    axs[0].legend()
    axs[0].grid(True)

    # Plot Hand 2 and Object on X-Y plane
    axs[1].scatter(obj_x, obj_y, c='b', marker='o', label='Object')
    axs[1].scatter(hand2_x, hand2_y, c='r', marker='o', label='Actual Hand Joints')
    for (i, j) in [(0, 7), (7, 8), (8, 9), (9, 20), (0, 10), (10, 11), (11, 12), (12, 19),
                   (0, 4), (4, 5), (5, 6), (6, 18), (0, 1), (1, 2), (2, 3), (3, 17),
                   (0, 13), (13, 14), (14, 15), (15, 16)]:
        axs[1].plot([hand2_x[i], hand2_x[j]], [hand2_y[i], hand2_y[j]], 'b')
    axs[1].set_xlabel('X Coordinate')
    axs[1].set_ylabel('Y Coordinate')
    axs[1].set_title('Actual Hand Pose - X-Y Plane')
    axs[1].legend()
    axs[1].grid(True)

    fig.suptitle(f'Test Data[{testNumber}] Plot of Hand pose \n Based on Object File Name: ({objname})')
    # fig.text(0.5, 0.03, f'Mean Squared Error = {mse}', ha='center', va='center')
    # fig.text(0.5, 0.005, f'Mean Absolute Error = {mae}', ha='center', va='center')

    plt.tight_layout()
    plt.show()


def load_xyz(file_path):
    points = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.strip():  # Ensure the line is not empty
                x, y, z = map(float, line.split())
                points.append([x, y, z])
    return np.array(points)



# PARAMETERS
input_dim = 114
latent_dim=64
condition_dim = 30

# LOAD DATA
hand_train, hand_val, hand_test, obj_train, obj_val, obj_test, train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader,obj_names=loading_data('../PreprocessData/hand_object_data.pkl', 64)

# LOAD MODEL AND SCALERS
model = CVAE_02_3(input_dim, latent_dim, condition_dim)
model.load_state_dict(torch.load('../Models_Trains\cvae_02_3_weights.pth'))

with open('../PreprocessData/scalers.pkl', 'rb') as scalers_file:
    scalers= pickle.load(scalers_file)

# Extract the individual scalers
scaler_hand_joints = scalers['scaler_hand_joints']
scaler_obj_corners = scalers['scaler_obj_corners']
scaler_hand_pose = scalers['scaler_hand_pose']
scaler_hand_trans = scalers['scaler_hand_trans']
scaler_obj_trans = scalers['scaler_obj_trans']
scaler_obj_rot = scalers['scaler_obj_rot']


# Example inference
model.eval()

with torch.no_grad():
    file=14050
    object_data = obj_test[file].unsqueeze(0)
    object_name=obj_names[file]
    hand_data=hand_test[file].unsqueeze(0)
    z = torch.randn(1, latent_dim)
    generated_hand_pose = model.decode(z, object_data)
    mse = mean_squared_error(generated_hand_pose, hand_data)
    mae = mean_absolute_error(generated_hand_pose, hand_data)
    print("Generated hand pose:", generated_hand_pose)
    print(generated_hand_pose.shape[1])

#------------------- GENERATED HAND ----------
#RECONSTRUCTED HAND INFORMATION
generated_hand_pose_np = generated_hand_pose.numpy().reshape(-1)
hand_pose_generated = generated_hand_pose_np[:48]  # First 48 elements for hand pose
hand_trans_generated = generated_hand_pose_np[48:51]  # Next 3 elements for hand translation
hand_joints_generated = generated_hand_pose_np[51:]  # Remaining 63 elements for hand joints (21x3)
hand_joints_generated = hand_joints_generated.reshape(-1, 3) # Reshape hand joints back to (21, 3)
# Inverse transform each component using the corresponding scaler
hand_pose_original = scaler_hand_pose.inverse_transform(hand_pose_generated.reshape(1, -1)).reshape(-1)
hand_trans_original = scaler_hand_trans.inverse_transform(hand_trans_generated.reshape(1, -1)).reshape(-1)
hand_joints_original = scaler_hand_joints.inverse_transform(hand_joints_generated.reshape(1, -1)).reshape(-1, 3)
#----------------------------------------------

#------------------- ACTUAL HAND ----------
hand_data_np = hand_data.numpy().reshape(-1)
hand_pose_actual = hand_data_np[:48]  # First 48 elements for hand pose
hand_trans_actual = hand_data_np[48:51]  # Next 3 elements for hand translation
hand_joints_actual = hand_data_np[51:]  # Remaining 63 elements for hand joints (21x3)
hand_joints_actual = hand_joints_actual.reshape(-1, 3)# Reshape hand joints back to (21, 3)
# Inverse transform each component using the corresponding scaler
hand_pose_act_original = scaler_hand_pose.inverse_transform(hand_pose_actual.reshape(1, -1)).reshape(-1)
hand_trans_act_original = scaler_hand_trans.inverse_transform(hand_trans_actual.reshape(1, -1)).reshape(-1)
hand_joints_act_original = scaler_hand_joints.inverse_transform(hand_joints_actual.reshape(1, -1)).reshape(-1, 3)


#------------------- OBJECT  ----------
object_data_np = object_data.numpy().reshape(-1)
obj_trans=object_data_np[:3] # First 3 elements for object translation
obj_rot=object_data_np[3:6] #Next 3 elements for object rotation
# Inverse transform each component using the corresponding scaler
obj_trans_original = scaler_obj_trans.inverse_transform(obj_trans.reshape(1, -1)).reshape(-1)
obj_rot_original = scaler_obj_rot.inverse_transform(obj_rot.reshape(1, -1)).reshape(-1)



xyz_file_path = fr"D:\UNI\Sem3\Dissertation\My effort\HOnnotate\ho3d-master\Dataset\models\{object_name}\points.xyz"
points = load_xyz(xyz_file_path)

print(f'Object Name: {object_name}')
print(f'The test number: {file}')
print(f'Mean Square Error: {mse}')
print(f'Mean Absolute Error: {mae}')

# Visualize hand and object in 2D

visualize_two_hands_and_object_2d(hand_trans_original, hand_joints_original, hand_trans_act_original, hand_joints_act_original, obj_trans_original, obj_rot_original, points,object_name,file)
