"""Training script for CVAE models.

Example usage:
    python scripts/train.py --model cvae_01 --epochs 50 --batch_size 64
"""

import argparse
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import matplotlib.pyplot as plt

from grasp_pose_generation.models import CVAE_01, CVAE_02, CVAE_03, CVAE_02_1, CVAE_02_2, CVAE_02_3
from grasp_pose_generation.training import loading_data
from grasp_pose_generation.training import loading_data


def train_model(model_name, data_path, batch_size, num_epochs, latent_dim, learning_rate, output_dir):
    """Train a CVAE model."""
    # Load data
    hand_train, hand_val, hand_test, obj_train, obj_val, obj_test, train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader, obj_name = loading_data(
        data_path, batch_size
    )

    # Hyperparameters
    input_dim = hand_train.shape[1]
    condition_dim = obj_train.shape[1]

    # Model selection
    model_classes = {
        'cvae_01': CVAE_01,
        'cvae_02': CVAE_02,
        'cvae_03': CVAE_03,
        'cvae_02_1': CVAE_02_1,
        'cvae_02_2': CVAE_02_2,
        'cvae_02_3': CVAE_02_3,
    }

    if model_name.lower() not in model_classes:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(model_classes.keys())}")

    model = model_classes[model_name.lower()](input_dim, latent_dim, condition_dim)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training metrics
    t_loss = []
    v_loss = []
    v_rec_loss = []
    v_pose_error = []
    v_joints_error = []
    v_trans_error = []

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

                pose_error = nn.functional.mse_loss(
                    reconstructed[:, :hand_pose_dim], 
                    hand_pose_data[:, :hand_pose_dim],
                    reduction='sum'
                )
                joints_error = nn.functional.mse_loss(
                    reconstructed[:, hand_pose_dim:hand_pose_dim + joints_dim],
                    hand_pose_data[:, hand_pose_dim:hand_pose_dim + joints_dim],
                    reduction='sum'
                )
                trans_error = nn.functional.mse_loss(
                    reconstructed[:, hand_pose_dim + joints_dim:hand_pose_dim + joints_dim + trans_dim],
                    hand_pose_data[:, hand_pose_dim + joints_dim:hand_pose_dim + joints_dim + trans_dim],
                    reduction='sum'
                )

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
            f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Recon Loss: {val_recon_loss:.4f}'
        )
        print(
            f'Val Pose Error: {val_pose_error:.4f}, Val Joints Error: {val_joints_error:.4f}, Val Trans Error: {val_trans_error:.4f}'
        )

    # Save the model
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, f'{model_name}_weights.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Save the losses and errors to logs directory
    # If output_dir is models/checkpoints, logs go to outputs/logs
    if 'checkpoints' in output_dir:
        logs_dir = os.path.join(os.path.dirname(os.path.dirname(output_dir)), 'outputs', 'logs')
    else:
        logs_dir = os.path.join(output_dir, '..', 'outputs', 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    losses_errors = {
        't_loss': t_loss,
        'v_loss': v_loss,
        'v_rec_loss': v_rec_loss,
        'v_pose_error': v_pose_error,
        'v_joints_error': v_joints_error,
        'v_trans_error': v_trans_error
    }

    losses_path = os.path.join(logs_dir, f'{model_name}_losses_errors.pkl')
    with open(losses_path, 'wb') as f:
        pickle.dump(losses_errors, f)
    print(f"Losses and errors saved to {losses_path}")

    # Plot Losses and Errors
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    axes = axes.flatten()

    axes[0].plot(t_loss)
    axes[0].set_title('Train Loss')

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

    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'{model_name}_training_plot.png')
    plt.savefig(plot_path)
    print(f"Training plot saved to {plot_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Train CVAE models')
    parser.add_argument('--model', type=str, default='cvae_01',
                        choices=['cvae_01', 'cvae_02', 'cvae_03', 'cvae_02_1', 'cvae_02_2', 'cvae_02_3'],
                        help='Model to train')
    parser.add_argument('--data_path', type=str, default='data/hand_object_data.pkl',
                        help='Path to preprocessed data')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--latent_dim', type=int, default=32, help='Latent dimension')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--output_dir', type=str, default='models/checkpoints',
                        help='Output directory for models and logs')

    args = parser.parse_args()
    train_model(args.model, args.data_path, args.batch_size, args.epochs, args.latent_dim, args.lr, args.output_dir)


if __name__ == '__main__':
    main()

