import pickle

#============ LOAD PREPROCESSED DATA ===================
with open('hand_object_data.pkl', 'rb') as data_file:
    data_files = pickle.load(data_file)

hand_train = data_files['hand_train']
hand_val = data_files['hand_val']
hand_test = data_files['hand_test']
obj_train = data_files['obj_train']
obj_val = data_files['obj_val']
obj_test = data_files['obj_test']

#================= LOAD SAVED SCALERS ======================
with open('scalers.pkl', 'rb') as scalers_file:
    scalers = pickle.load(scalers_file)

scaler_hand_joints = scalers['scaler_hand_joints']
scaler_obj_corners = scalers['scaler_obj_corners']
scaler_hand_pose = scalers['scaler_hand_pose']
scaler_hand_trans = scalers['scaler_hand_trans']
scaler_obj_trans = scalers['scaler_obj_trans']
scaler_obj_rot = scalers['scaler_obj_rot']

print("Data and scalers loaded successfully!")
print("Training data shape:", hand_train.shape, obj_train.shape)
print("Validation data shape:", hand_val.shape, obj_val.shape)
print("Test data shape:", hand_test.shape, obj_test.shape)
print('-----------------------------------------------------')
print('----- Print A Sample Of Preprocessed And Combined Information ----------')
print(f"1st Object Information From Training Part:  {obj_train[0]}")
print(f"1st Hand Information From Training Part:  {hand_train[0]}")