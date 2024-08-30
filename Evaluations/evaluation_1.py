import torch
import torch.nn as nn
import pickle
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import matplotlib.pyplot as plt

from Models_Trains.all_models import CVAE_01
from Models_Trains.all_models import CVAE_02
from Models_Trains.all_models import CVAE_03

from Models_Trains.loading_data import loading_data

# Load the model weights
input_dim = 114
latent_dim = 32
condition_dim = 30



#LOAD TEST DATA
# LOAD DATA
hand_train, hand_val, hand_test, obj_train, obj_val, obj_test, train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader,obj_names=loading_data('../PreprocessData/hand_object_data.pkl',64)




def mean_squared_error(pred, target):
    return F.mse_loss(pred, target)

def mean_absolute_error(pred, target):
    return F.l1_loss(pred, target)



weights_path=['cvae_01_weights.pth','cvae_02_weights.pth','cvae_03_weights.pth','cvae_03Noise_weights.pth']
mse_models=[]
mae_models=[]
for i in range(len(weights_path)):
    if weights_path[i]=='cvae_01_weights.pth':
        model = CVAE_01(input_dim, latent_dim, condition_dim)
        model_name='Model_01'
    elif weights_path[i]=='cvae_02_weights.pth':
        model = CVAE_02(input_dim, latent_dim, condition_dim)
        model_name='Model_02'
    elif weights_path[i]=='cvae_03_weights.pth' or weights_path[i]=='cvae_03Noise_weights.pth':
        model = CVAE_03(input_dim, latent_dim, condition_dim)
        model_name='Model_03'


    model.load_state_dict(torch.load(f'../Models_Trains/{weights_path[i]}'))
    model.eval()
    mse_list = []
    mae_list = []

    with torch.no_grad():
        for i in range(len(obj_val)):
            object_data = obj_test[i].unsqueeze(0)  # Example object data
            actual_hand_pose = hand_test[i].unsqueeze(0)  # Actual hand pose data

            z = torch.randn(1, latent_dim)
            generated_hand_pose = model.decode(z, object_data)

            mse = mean_squared_error(generated_hand_pose, actual_hand_pose)
            mae = mean_absolute_error(generated_hand_pose, actual_hand_pose)

            mse_list.append(mse.item())
            mae_list.append(mae.item())

    # Calculate average metrics
    avg_mse = sum(mse_list) / len(mse_list)
    avg_mae = sum(mae_list) / len(mae_list)

    mse_models.append(avg_mse)
    mae_models.append(avg_mae)

    print(f'Average MSE_{model_name}: {avg_mse:.4f}')
    print(f'Average MAE_{model_name}: {avg_mae:.4f}')




fig, axes = plt.subplots(1, 2, figsize=(12, 6))



axes[0].bar('CVAE_01', mse_models[0], color='blue')
axes[0].bar('CVAE_02', mse_models[1], color='green')
axes[0].bar('CVAE_03', mse_models[2], color='yellow')
axes[0].bar('CVAE_03_Noise', mse_models[3], color='orange')

axes[0].set_xlabel('Models')
axes[0].set_ylabel('Average Value (Millimeter)')
axes[0].set_title('Average of MSE On Test Dataset')

axes[1].bar('CVAE_01', mae_models[0], color='blue')
axes[1].bar('CVAE_02', mae_models[1], color='green')
axes[1].bar('CVAE_03', mae_models[2], color='yellow')
axes[1].bar('CVAE_03_Noise', mae_models[3], color='orange')

axes[1].set_xlabel('Models')
axes[1].set_ylabel('Average Value (Millimeter)')
axes[1].set_title('Average of MAE On Test Dataset')


plt.tight_layout()
plt.show()



