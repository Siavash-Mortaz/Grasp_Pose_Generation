import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pickle
import matplotlib.pyplot as plt

from Models_Trains.all_models import CVAE_02
from Models_Trains.loading_data import loading_data

hand_train, hand_val, hand_test, obj_train, obj_val, obj_test, train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader,obj_names=loading_data('../PreprocessData/hand_object_data.pkl',64)



# Hyperparameters
input_dim = hand_train.shape[1]  # Adjust according to your input data dimensions
latent_dim = 32 #32#64
condition_dim = obj_train.shape[1]  # Adjust according to your condition data dimensions
num_epochs = 50 #100
learning_rate=0.001 #0.001 #0.0005

model = CVAE_02(input_dim, latent_dim, condition_dim)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

t_loss=[]
v_loss=[]
v_rec_loss=[]
v_pose_error=[]
v_joints_error=[]
v_trans_error=[]


#Learning Rate Scheduling
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10)
# Training loop
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for batch in train_loader:
        hand_pose_data, object_data = batch
        optimizer.zero_grad()
        reconstructed, mean, log_var = model(hand_pose_data, object_data)
        recon_loss = nn.functional.mse_loss(reconstructed, hand_pose_data, reduction='sum')
        kl_divergence = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        loss = recon_loss + kl_divergence
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader.dataset)
    t_loss.append(train_loss)

    # Validation loop
    model.eval()
    val_loss = 0
    val_recon_loss = 0
    val_pose_error = 0
    val_joints_error = 0
    val_trans_error = 0

    with torch.no_grad():
        for batch in val_loader:
            hand_pose_data, object_data = batch
            reconstructed, mean, log_var = model(hand_pose_data, object_data)
            recon_loss = nn.functional.mse_loss(reconstructed, hand_pose_data, reduction='sum')
            kl_divergence = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
            loss = recon_loss + kl_divergence
            val_loss += loss.item()
            val_recon_loss += recon_loss.item()

            # Calculate errors for hand pose, joints, and translation
            hand_pose_dim = 48
            joints_dim = 21 * 3
            trans_dim = 3

            pose_error = nn.functional.mse_loss(reconstructed[:, :hand_pose_dim], hand_pose_data[:, :hand_pose_dim],
                                                reduction='sum')
            joints_error = nn.functional.mse_loss(reconstructed[:, hand_pose_dim:hand_pose_dim + joints_dim],
                                                  hand_pose_data[:, hand_pose_dim:hand_pose_dim + joints_dim],
                                                  reduction='sum')
            trans_error = nn.functional.mse_loss(
                reconstructed[:, hand_pose_dim + joints_dim:hand_pose_dim + joints_dim + trans_dim],
                hand_pose_data[:, hand_pose_dim + joints_dim:hand_pose_dim + joints_dim + trans_dim], reduction='sum')

            val_pose_error += pose_error.item()
            val_joints_error += joints_error.item()
            val_trans_error += trans_error.item()

    val_loss /= len(val_loader.dataset)
    val_recon_loss /= len(val_loader.dataset)
    val_pose_error /= len(val_loader.dataset)
    val_joints_error /= len(val_loader.dataset)
    val_trans_error /= len(val_loader.dataset)

    v_loss.append(val_loss)
    v_rec_loss.append(val_recon_loss)
    v_pose_error.append(val_pose_error)
    v_joints_error.append(val_joints_error)
    v_trans_error.append(val_trans_error)

    print(
        f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Recon Loss: {val_recon_loss:.4f}')
    print(
        f'Val Pose Error: {val_pose_error:.4f}, Val Joints Error: {val_joints_error:.4f}, Val Trans Error: {val_trans_error:.4f}')
    scheduler.step(val_loss)

# Save the model
torch.save(model.state_dict(), '/cvae_02_weights.pth')
print("Model saved successfully!")

losses_errors = {
    't_loss': t_loss,
    'v_loss': v_loss,
    'v_rec_loss': v_rec_loss,
    'v_pose_error': v_pose_error,
    'v_joints_error': v_joints_error,
    'v_trans_error': v_trans_error
}

with open('/cvae_02_losses_errors.pkl', 'wb') as f:
    pickle.dump(losses_errors, f)

print("Losses And Errors saved successfully!")


fig, axes = plt.subplots(2, 3, figsize=(12, 6))
axes = axes.flatten()


axes[0].plot(t_loss)
axes[0].set_title('Train Loss ')

axes[1].plot(v_loss)
axes[1].set_title('Validation Loss')

axes[2].plot(v_rec_loss)
axes[2].set_title('Reconstruct Error')

axes[3].plot(v_pose_error)
axes[3].set_title('HandPose Error')

axes[4].plot(v_joints_error)
axes[4].set_title('HandJoints Error')

axes[5].plot(v_trans_error)
axes[5].set_title('HandTrans Error')

for i in range(6, len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.show()
