import pickle
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from b_Load_ExtractedData import load_saved_data


#============== LOADING DATA ===========================
hand_poses_file = r'hand_poses.pkl'
object_infos_file = r'object_infos.pkl'
hand_poses, object_infos = load_saved_data(hand_poses_file, object_infos_file)


#============== Reshape objRot IF NECESSARY ==============
num_objRot_reshaped = 0
for i in range(len(object_infos)):
    if object_infos[i]['objRot'].shape != (3,):
        object_infos[i]['objRot'] = object_infos[i]['objRot'].reshape(3)
        num_objRot_reshaped += 1
print(f'The number of reshaped: {num_objRot_reshaped}')


#============== PREPROCESSING & PREPARING  ===========================
# It extracts, normalizes, and combines the necessary features.
# It splits the data into appropriate sets for training, validation, and testing.
# Finally, it saves the processed data and the normalization parameters to disk for later use.
def preprocess_data(hand_poses, object_infos):
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

    return hand_data, obj_data, obj_names, scaler_hand_joints, scaler_obj_corners, scaler_hand_pose, scaler_hand_trans, scaler_obj_trans, scaler_obj_rot

# Preprocess the data
hand_data, obj_data, obj_names, scaler_hand_joints, scaler_obj_corners, scaler_hand_pose, scaler_hand_trans, scaler_obj_trans, scaler_obj_rot = preprocess_data(hand_poses, object_infos)

#============== SPLITING DATA  ===========================
# Split data into training, validation, and test sets
# 20% of the data is set aside for testing
# The remaining 80% is split again, with 25% of it used for validation
# (25% of 80% is 0.25 * 0.8 = 0.2, or 20% of the original dataset),
# leaving 60% of the original data for training
hand_train, hand_test, obj_train, obj_test = train_test_split(hand_data, obj_data, test_size=0.2, random_state=42)
hand_train, hand_val, obj_train, obj_val = train_test_split(hand_train, obj_train, test_size=0.25, random_state=42)  # 0.25 * 0.8 = 0.2

print("Training data shape:", hand_train.shape, obj_train.shape)
print("Validation data shape:", hand_val.shape, obj_val.shape)
print("Test data shape:", hand_test.shape, obj_test.shape)

#============== SAVE DATA AND SCALERS =============
data_files = {
    'hand_train': hand_train,
    'hand_val': hand_val,
    'hand_test': hand_test,
    'obj_train': obj_train,
    'obj_val': obj_val,
    'obj_test': obj_test,
    'obj_names': obj_names  # Save the object names separately
}

scalers = {
    'scaler_hand_joints': scaler_hand_joints,
    'scaler_obj_corners': scaler_obj_corners,
    'scaler_hand_pose': scaler_hand_pose,
    'scaler_hand_trans': scaler_hand_trans,
    'scaler_obj_trans': scaler_obj_trans,
    'scaler_obj_rot': scaler_obj_rot
}

# Save data into hand_object_data.pkl
with open('hand_object_data.pkl', 'wb') as data_file:
    pickle.dump(data_files, data_file)

# Save scalers into scalers.pkl
with open('scalers.pkl', 'wb') as scalers_file:
    pickle.dump(scalers, scalers_file)

print("Data and scalers saved successfully!")
