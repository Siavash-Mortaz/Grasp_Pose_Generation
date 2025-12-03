"""Preprocess and prepare data for training."""

import pickle
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from .load import load_saved_data


def preprocess_data(hand_poses, object_infos):
    """
    Preprocess hand and object data.
    
    It extracts, normalizes, and combines the necessary features.
    It splits the data into appropriate sets for training, validation, and testing.
    Finally, it saves the processed data and the normalization parameters to disk for later use.
    
    Args:
        hand_poses: List of hand pose dictionaries
        object_infos: List of object information dictionaries
    
    Returns:
        hand_data: Combined hand data array
        obj_data: Combined object data array
        obj_names: List of object names
        scalers: Dictionary of scaler objects
    """
    # Extract hand pose information
    hand_pose_data = np.array([pose['handPose'] for pose in hand_poses])
    hand_trans_data = np.array([pose['handTrans'] for pose in hand_poses])
    hand_joints_data = np.array([pose['handJoints3D'] for pose in hand_poses])

    # Extract object information
    obj_trans_data = np.array([info['objTrans'] for info in object_infos])
    obj_rot_data = np.array([info['objRot'] for info in object_infos])
    obj_corners_data = np.array([info['objCorners3D'] for info in object_infos])

    # Extract object names (this will not be used for training)
    obj_names = [info['objName'] for info in object_infos]

    # Reshape objRot if necessary
    num_objRot_reshaped = 0
    for i in range(len(obj_rot_data)):
        if obj_rot_data[i].shape != (3,):
            obj_rot_data[i] = obj_rot_data[i].reshape(3)
            num_objRot_reshaped += 1
    print(f'The number of reshaped: {num_objRot_reshaped}')

    # Data is normalized using StandardScaler from sklearn,
    # which standardizes features by removing the mean and scaling to unit variance.
    # Normalize hand joints and object corners
    # Hand joints and object corners (reshaped for 3D data).
    # Hand pose, hand translation, object translation, and object rotation.

    scaler_hand_joints = StandardScaler()
    hand_joints_data = scaler_hand_joints.fit_transform(hand_joints_data.reshape(-1, 63)).reshape(-1, 21, 3)

    scaler_obj_corners = StandardScaler()
    obj_corners_data = scaler_obj_corners.fit_transform(obj_corners_data.reshape(-1, 24)).reshape(-1, 8, 3)

    # Normalize other parameters
    scaler_hand_pose = StandardScaler()
    hand_pose_data = scaler_hand_pose.fit_transform(hand_pose_data)

    scaler_hand_trans = StandardScaler()
    hand_trans_data = scaler_hand_trans.fit_transform(hand_trans_data)

    scaler_obj_trans = StandardScaler()
    obj_trans_data = scaler_obj_trans.fit_transform(obj_trans_data)

    scaler_obj_rot = StandardScaler()
    obj_rot_data = scaler_obj_rot.fit_transform(obj_rot_data)

    # Combine data
    # Hand data is combined into a single array, consisting of hand pose, translation, and joints data.
    # Object data is similarly combined into a single array, consisting of object translation, rotation, and corner data.
    hand_data = np.hstack((hand_pose_data, hand_trans_data, hand_joints_data.reshape(-1, 63)))
    obj_data = np.hstack((obj_trans_data, obj_rot_data, obj_corners_data.reshape(-1, 24)))

    scalers = {
        'scaler_hand_joints': scaler_hand_joints,
        'scaler_obj_corners': scaler_obj_corners,
        'scaler_hand_pose': scaler_hand_pose,
        'scaler_hand_trans': scaler_hand_trans,
        'scaler_obj_trans': scaler_obj_trans,
        'scaler_obj_rot': scaler_obj_rot
    }

    return hand_data, obj_data, obj_names, scalers

