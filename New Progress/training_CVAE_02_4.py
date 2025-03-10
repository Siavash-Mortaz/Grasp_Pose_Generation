import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pickle
import matplotlib.pyplot as plt

from Models_Trains.all_models import CVAE_02_4
from Models_Trains.loading_data import loading_data
import numpy as np
from lion_pytorch import Lion






# Load data
hand_train, hand_val, hand_test, obj_train, obj_val, obj_test, train_dataset, val_dataset, test_dataset, \
    train_loader, val_loader, test_loader,obj_names, folder_names, frame_numbers, idx_train, idx_val, \
    idx_test = loading_data(
    '../PreprocessData/hand_object_data.pkl', 64)

# Hyperparameters
input_dim = hand_train.shape[1]  # Adjust according to your input data dimensions
latent_dim = 16  # 32 or 64 depending on your experiments
condition_dim = obj_train.shape[1]  # Adjust according to your condition data dimensions
num_epochs = 200  # Adjust according to your experiments
# learning_rate = 0.0005  # 0.001 or 0.0005 depending on your experiments
# learning_rate = 1e-6  # 0.001 or 0.0005 depending on your experiments
learning_rate = 2e-5  # Slight increase for better training dynamics
num_gaussians = 10  # Number of Gaussian components in MoG


# Initialize the model, optimizer, and scheduler
model = CVAE_02_4(input_dim, latent_dim, condition_dim,num_gaussians)
# optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
# optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
# optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-6)
# optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-6)
optimizer = Lion(model.parameters(), lr=learning_rate, weight_decay=5e-7)
# optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=5e-7)


# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5, min_lr=1e-6)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True, min_lr=1e-7)







# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10)

# Lists for tracking loss and errors
t_loss = []
v_loss = []
v_rec_loss = []
v_pose_error = []
v_joints_error = []
v_trans_error = []


# MoG-VAE KL Divergence Loss Function
def mog_kl_loss(mean, log_var, mixing_coeffs):
    """
    Compute the KL divergence between the learned posterior (MoG) and the prior (MoG).
    """
    prior_mean = torch.zeros_like(mean)
    prior_log_var = torch.zeros_like(log_var)

    # Compute KL divergence for each Gaussian component
    kl = 0.5 * (prior_log_var - log_var + (log_var.exp() + (mean - prior_mean).pow(2)) / prior_log_var.exp() - 1)

    # Weight KL by mixing coefficients
    kl_weighted = torch.sum(kl * mixing_coeffs.unsqueeze(-1), dim=1)

    return torch.sum(kl_weighted)

# Set beta value for KL divergence weight
# beta = 5.0  # You can experiment with values like 2.0, 4.0, 10.0

best_loss = float("inf")
patience = 50  # Stop if no improvement for 5 epochs
min_delta = 0.001  # Stop if validation loss doesn’t improve by at least this much
counter = 0

# Training loop
for epoch in range(num_epochs):
    # beta = min(5.0, epoch / 150)  # KL Annealing: Gradually increase β
    # beta = min(10.0, epoch / 50)  # Faster KL Annealing
    # beta = 5 * (1 / (1 + np.exp(-0.1 * (epoch - 100)))) ++
    # beta = 10 * (1 / (1 + np.exp(-0.05 * (epoch - 50))))
    # beta = 5 * (1 - np.exp(-0.1 * epoch))  # Slow growth at the beginning
    # beta = 5 * (epoch / (num_epochs * 2))  # Very slow linear increase
    # beta = min(10.0, epoch / 50)
    # beta = 10 * (1 - np.exp(-0.05 * epoch))  # Slower growth over epochs ++
    # beta = min(10.0, epoch / 100)
    # beta = min(15.0, epoch / 50)  # Increase KL effect faster ++
    # beta = 5 + 10 * np.sin((np.pi / num_epochs) * epoch)
    # beta = min(10.0, epoch / 50)  # Smoother KL growth
    beta = 5 * (1 - np.exp(-0.05 * epoch))
    # beta = 10 * (1 - np.exp(-0.05 * epoch))  final_6

    model.train()
    train_loss = 0

    for batch in train_loader:
        hand_pose_data, object_data = batch
        optimizer.zero_grad()

        # Encode hand pose
        mean, log_var, mixing_coeffs = model.encode(hand_pose_data)
        z = model.reparameterize(mean, log_var, mixing_coeffs)
        reconstructed = model.decode(z, object_data)

        # Compute losses
        recon_loss = nn.functional.mse_loss(reconstructed, hand_pose_data, reduction='sum')
        kl_divergence = mog_kl_loss(mean, log_var, mixing_coeffs)

        # loss = recon_loss + beta * kl_divergence  # Apply KL Annealing
        loss = recon_loss + beta * (kl_divergence / kl_divergence.detach().mean())

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
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
            mean, log_var, mixing_coeffs = model.encode(hand_pose_data)
            z = model.reparameterize(mean, log_var, mixing_coeffs)
            reconstructed = model.decode(z, object_data)

            recon_loss = nn.functional.mse_loss(reconstructed, hand_pose_data, reduction='sum')
            kl_divergence = mog_kl_loss(mean, log_var, mixing_coeffs)
            loss = recon_loss + beta * (kl_divergence / kl_divergence.detach().mean())

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
        f'Epoch {epoch + 1}/{num_epochs}, Beta: {beta:.2f}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Recon Loss: {val_recon_loss:.4f}')
    print(
        f'Val Pose Error: {val_pose_error:.4f}, Val Joints Error: {val_joints_error:.4f}, Val Trans Error: {val_trans_error:.4f}')
    # Check for improvement
    if val_loss < best_loss:
        best_loss = val_loss
        counter = 0  # Reset counter if loss improves
        torch.save(model.state_dict(), 'best_model.pth')  # Save best model
        print(f"New best model saved at epoch {epoch + 1} with Val Loss: {val_loss:.4f}")
    else:
        counter += 1  # Increase counter if no improvement
        if counter >= patience:
            print(f"Early stopping triggered at epoch {epoch + 1}. Training stopped.")
            break  # Stop training if no improvement for 'patience' epochs

    scheduler.step(val_loss)  # Adjust LR if needed

# Final model save after training (if early stopping didn't trigger)
torch.save(model.state_dict(), 'final_model_5.pth')
print("Final model saved successfully!")

# Save loss and error metrics
losses_errors = {
    't_loss': t_loss,
    'v_loss': v_loss,
    'v_rec_loss': v_rec_loss,
    'v_pose_error': v_pose_error,
    'v_joints_error': v_joints_error,
    'v_trans_error': v_trans_error
}

with open('cvae_02_4_losses_errors_mog_5.pkl', 'wb') as f:
    pickle.dump(losses_errors, f)

print("Losses And Errors saved successfully!")



# Plot the loss and error metrics
fig, axes = plt.subplots(2, 3, figsize=(12, 6))
axes = axes.flatten()

axes[0].plot(t_loss)
axes[0].set_title('Train Loss ')

axes[1].plot(v_loss)
axes[1].set_title('Validation Loss')

axes[2].plot(v_rec_loss)
axes[2].set_title('Reconstruction Error')

axes[3].plot(v_pose_error)
axes[3].set_title('Hand Pose Error')

axes[4].plot(v_joints_error)
axes[4].set_title('Hand Joints Error')

axes[5].plot(v_trans_error)
axes[5].set_title('Hand Translation Error')

plt.tight_layout()
plt.show()