import pickle
import matplotlib.pyplot as plt


# Load losses and errors for the Model 02(1st)
with open('../Models_Trains/cvae_02_losses_errors.pkl', 'rb') as f:
    losses_errors_1 = pickle.load(f)
t_loss_1 = losses_errors_1['t_loss']
v_loss_1 = losses_errors_1['v_loss']
v_rec_loss_1 = losses_errors_1['v_rec_loss']
v_pose_error_1 = losses_errors_1['v_pose_error']
v_joints_error_1 = losses_errors_1['v_joints_error']
v_trans_error_1 = losses_errors_1['v_trans_error']

# Load losses and errors for the Model 02(2nd)
with open('../Models_Trains/cvae_02_losses_errors_1.pkl', 'rb') as f:
    losses_errors_2 = pickle.load(f)
t_loss_2 = losses_errors_2['t_loss']
v_loss_2 = losses_errors_2['v_loss']
v_rec_loss_2 = losses_errors_2['v_rec_loss']
v_pose_error_2 = losses_errors_2['v_pose_error']
v_joints_error_2 = losses_errors_2['v_joints_error']
v_trans_error_2 = losses_errors_2['v_trans_error']

# Load losses and errors for the Model 02(3rd)
with open('../Models_Trains/cvae_02_losses_errors_2.pkl', 'rb') as f:
    losses_errors_03 = pickle.load(f)
t_loss_3 = losses_errors_03['t_loss']
v_loss_3 = losses_errors_03['v_loss']
v_rec_loss_3 = losses_errors_03['v_rec_loss']
v_pose_error_3 = losses_errors_03['v_pose_error']
v_joints_error_3 = losses_errors_03['v_joints_error']
v_trans_error_3 = losses_errors_03['v_trans_error']



fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()

# Plotting training loss
axes[0].plot(t_loss_1, label='CVAE_02(1st)', color='green')
axes[0].plot(t_loss_2, label='CVAE_02(2nd)', color='blue')
axes[0].plot(t_loss_3, label='CVAE_02(3rd)', color='red')
axes[0].set_title('Train Loss')
axes[0].set_xlabel('Number of Epochs')
axes[0].set_ylabel('Loss Value')
axes[0].legend()

# Plotting validation loss
axes[1].plot(v_loss_1, label='CVAE_02(1st)', color='green')
axes[1].plot(v_loss_2, label='CVAE_02(2nd)', color='blue')
axes[1].plot(v_loss_3, label='CVAE_02(3rd)', color='red')
axes[1].set_title('Validation Loss')
axes[1].set_xlabel('Number of Epochs')
axes[1].set_ylabel('Loss Value')
axes[1].legend()

# Plotting reconstruction error
axes[2].plot(v_rec_loss_1, label='CVAE_02(1st)', color='green')
axes[2].plot(v_rec_loss_2, label='CVAE_02(2nd)', color='blue')
axes[2].plot(v_rec_loss_3, label='CVAE_02(3rd)', color='red')
axes[2].set_title('Reconstruct Error')
axes[2].set_xlabel('Number of Epochs')
axes[2].set_ylabel('Error Value (Millimeter)')
axes[2].legend()

# Plotting hand pose error
axes[3].plot(v_pose_error_1, label='CVAE_02(1st)', color='green')
axes[3].plot(v_pose_error_2, label='CVAE_02(2nd)', color='blue')
axes[3].plot(v_pose_error_3, label='CVAE_02(3rd)', color='red')
axes[3].set_title('HandPose Error')
axes[3].set_xlabel('Number of Epochs')
axes[3].set_ylabel('Error Value (Millimeter)')
axes[3].legend()

# Plotting hand joints error
axes[4].plot(v_joints_error_1, label='CVAE_02(1st)', color='green')
axes[4].plot(v_joints_error_2, label='CVAE_02(2nd)', color='blue')
axes[4].plot(v_joints_error_3, label='CVAE_02(3rd)', color='red')
axes[4].set_title('HandJoints Error')
axes[4].set_xlabel('Number of Epochs')
axes[4].set_ylabel('Error Value (Millimeter)')
axes[4].legend()

# Plotting hand translation error
axes[5].plot(v_trans_error_1, label='CVAE_02(1st)', color='green')
axes[5].plot(v_trans_error_2, label='CVAE_02(2nd)', color='blue')
axes[5].plot(v_trans_error_3, label='CVAE_02(3rd)', color='red')
axes[5].set_title('HandTrans Error')
axes[5].set_xlabel('Number of Epochs')
axes[5].set_ylabel('Error Value (Millimeter)')
axes[5].legend()

plt.tight_layout()
plt.show()
