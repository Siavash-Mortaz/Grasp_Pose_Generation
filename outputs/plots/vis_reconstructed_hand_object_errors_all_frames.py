import pickle
import torch
from Models_Trains.all_models import CVAE_02_2
from Models_Trains.all_models import CVAE_02_3
from Models_Trains.loading_data import loading_data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch.nn.functional as F

import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

from mpl_toolkits.mplot3d import Axes3D


def load_xyz(file_path):
    points = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.strip():  # Ensure the line is not empty
                x, y, z = map(float, line.split())
                points.append([x, y, z])
    return np.array(points)



def mean_squared_error(pred, target):
    return F.mse_loss(pred, target)

def mean_absolute_error(pred, target):
    return F.l1_loss(pred, target)


input_dim = 114
latent_dim=64
condition_dim = 30

#Load Data
hand_train, hand_val, hand_test, obj_train, obj_val, obj_test, train_dataset, val_dataset, test_dataset, \
    train_loader, val_loader, test_loader,obj_names=loading_data('../PreprocessData/hand_object_data.pkl', 64)

# Load the model and scalers
model = CVAE_02_3(input_dim, latent_dim, condition_dim)
model.load_state_dict(torch.load('../Models_Trains/cvae_02_3_weights.pth'))


print(f'OBJECTS SHAPE: {len(obj_test)}')

mse_list = []
mae_list = []
# Example inference
model.eval()
with torch.no_grad():
    for i in range(len(obj_test)):
        # if obj_names[i]=='003_cracker_box':
        #     print(f'003_cracker_box: [{i}]')
        # if obj_names[i]=='006_mustard_bottle':
        #     print(f'006_mustard_bottle: [{i}]')
        # if obj_names[i]=='004_sugar_box':
        #     print(f'004_sugar_box: [{i}]')
        # if obj_names[i]=='025_mug':
        #     print(f'025_mug: [{i}]')
        # if obj_names[i]=='035_power_drill':
        #     print(f'035_power_drill: [{i}]')

        object_data = obj_test[i].unsqueeze(0)
        object_name=obj_names[i]
        hand_data=hand_test[i].unsqueeze(0)
        z = torch.randn(1, latent_dim)
        generated_hand_pose = model.decode(z, object_data)
        mse = mean_squared_error(generated_hand_pose, hand_data)
        mae = mean_absolute_error(generated_hand_pose, hand_data)

        mse_list.append(mse.item())
        mae_list.append(mae.item())

# print(f'1: {mse_list[0:4]}')
# print(f'2: {mse_list[0:2]}')
# print(f'3: {mse_list[3:4]}')







# ------ PLOT ERRORS
plt.figure(figsize=(15,5))
plt.plot(mse_list[:4245], color='blue')
plt.xlabel('Number of Frames')
plt.ylabel('Error Value (Millimeter)')
plt.title("Mean Squared Error For Reconstructed Hand pose \n Based on 003_cracker_box")
plt.tight_layout()
plt.show()

plt.figure(figsize=(15,5))
plt.plot(mse_list[4246:7814], color='red')
plt.xlabel('Number of Frames')
plt.ylabel('Error Value (Millimeter)')
plt.title("Mean Squared Error For Reconstructed Hand pose \n Based on 006_mustard_bottle")
plt.tight_layout()
plt.show()

plt.figure(figsize=(15,5))
plt.plot(mse_list[7815:11361], color='orange')
plt.xlabel('Number of Frames')
plt.ylabel('Error Value (Millimeter)')
plt.title("Mean Squared Error For Reconstructed Hand pose \n Based on 004_sugar_box")
plt.tight_layout()
plt.show()

plt.figure(figsize=(15,5))
plt.plot(mse_list[11362:13138], color='purple')
plt.xlabel('Number of Frames')
plt.ylabel('Error Value (Millimeter)')
plt.title("Mean Squared Error For Reconstructed Hand pose \n Based on 025_mug")
plt.tight_layout()
plt.show()

plt.figure(figsize=(15,5))
plt.plot(mae_list[13139:], color='black')
plt.xlabel('Number of Frames')
plt.ylabel('Error Value (Millimeter)')
plt.title("Mean Squared Error For Reconstructed Hand pose \n Based on 035_power_drill")
plt.tight_layout()
plt.show()

# MAE
plt.figure(figsize=(15,5))
plt.plot(mae_list[:4245], color='blue')
plt.xlabel('Number of Frames')
plt.ylabel('Error Value (Millimeter)')
plt.title("Mean Absolute Error For Reconstructed Hand pose \n Based on 003_cracker_box")
plt.tight_layout()
plt.show()

plt.figure(figsize=(15,5))
plt.plot(mae_list[4246:7814], color='red')
plt.xlabel('Number of Frames')
plt.ylabel('Error Value (Millimeter)')
plt.title("Mean Absolute Error For Reconstructed Hand pose \n Based on 006_mustard_bottle")
plt.tight_layout()
plt.show()

plt.figure(figsize=(15,5))
plt.plot(mae_list[7815:11361], color='orange')
plt.xlabel('Number of Frames')
plt.ylabel('Error Value (Millimeter)')
plt.title("Mean Absolute Error For Reconstructed Hand pose \n Based on 004_sugar_box")
plt.tight_layout()
plt.show()

plt.figure(figsize=(15,5))
plt.plot(mae_list[11362:13138], color='purple')
plt.xlabel('Number of Frames')
plt.ylabel('Error Value (Millimeter)')
plt.title("Mean Absolute Error For Reconstructed Hand pose \n Based on 025_mug")
plt.tight_layout()
plt.show()

plt.figure(figsize=(15,5))
plt.plot(mae_list[13139:], color='black')
plt.xlabel('Number of Frames')
plt.ylabel('Error Value (Millimeter)')
plt.title("Mean Absolute Error For Reconstructed Hand pose \n Based on 035_power_drill")
plt.tight_layout()
plt.show()




#--------- BAR CHART For Average
#Average MSE for all
avg_mse_03 = sum(mse_list[:4245]) / len(mse_list[:4245])
avg_mse_06 = sum(mse_list[4246:7814]) / len(mse_list[4246:7814])
avg_mse_04 = sum(mse_list[7815:11361]) / len(mse_list[7815:11361])
avg_mse_025 = sum(mse_list[11362:13138]) / len(mse_list[11362:13138])
avg_mse_035 = sum(mse_list[13139:]) / len(mse_list[13139:])
#Average MAE for all

avg_mae_03 = sum(mae_list[:4245]) / len(mae_list[:4245])
avg_mae_06 = sum(mae_list[4246:7814]) / len(mae_list[4246:7814])
avg_mae_04 = sum(mae_list[7815:11361]) / len(mae_list[7815:11361])
avg_mae_025 = sum(mae_list[11362:13138]) / len(mae_list[11362:13138])
avg_mae_035 = sum(mae_list[13139:]) / len(mae_list[13139:])

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].bar('Cracker Box', avg_mse_03, color='blue')
axes[0].bar('Mustard Bottle', avg_mse_06, color='red')
axes[0].bar('Sugar Box', avg_mse_04, color='orange')
axes[0].bar('Mug', avg_mse_025, color='purple')
axes[0].bar('Power Drill', avg_mse_035, color='black')


axes[0].set_xlabel('Objects')
axes[0].set_ylabel('Average Errors (Millimeter)')
axes[0].set_title('Average of MSE On Test Dataset')

axes[1].bar('Cracker Box', avg_mae_03, color='blue')
axes[1].bar('Mustard Bottle', avg_mae_06, color='red')
axes[1].bar('Sugar Box', avg_mae_04, color='orange')
axes[1].bar('Mug', avg_mae_025, color='purple')
axes[1].bar('Power Drill', avg_mae_035, color='black')


axes[1].set_xlabel('Objects')
axes[1].set_ylabel('Average Errors (Millimeter)')
axes[1].set_title('Average of MAE On Test Dataset')

# plt.legend()
plt.tight_layout()
plt.show()


# -----------------------------------------

