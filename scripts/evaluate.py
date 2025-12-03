"""Evaluation script for CVAE models.

Example usage:
    python scripts/evaluate.py --model cvae_02_3 --checkpoint models/checkpoints/cvae_02_3_weights.pth
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from grasp_pose_generation.models import CVAE_01, CVAE_02, CVAE_03, CVAE_02_1, CVAE_02_2, CVAE_02_3
from grasp_pose_generation.training import loading_data
from grasp_pose_generation.evaluation import mean_squared_error, mean_absolute_error


def evaluate_model(model_name, checkpoint_path, data_path, batch_size, latent_dim):
    """Evaluate a trained CVAE model."""
    # Load data
    hand_train, hand_val, hand_test, obj_train, obj_val, obj_test, train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader, obj_names = loading_data(
        data_path, batch_size
    )

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

    # Initialize model
    input_dim = hand_test.shape[1]
    condition_dim = obj_test.shape[1]
    model = model_classes[model_name.lower()](input_dim, latent_dim, condition_dim)

    # Load checkpoint
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()

    # Evaluate on test set
    test_mse = 0
    test_mae = 0
    with torch.no_grad():
        for batch in test_loader:
            hand_pose_data, object_data = batch
            reconstructed, mean, log_var = model(hand_pose_data, object_data)
            test_mse += mean_squared_error(reconstructed, hand_pose_data).item()
            test_mae += mean_absolute_error(reconstructed, hand_pose_data).item()

    test_mse /= len(test_loader)
    test_mae /= len(test_loader)

    print(f"Test MSE: {test_mse:.4f}")
    print(f"Test MAE: {test_mae:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate CVAE models')
    parser.add_argument('--model', type=str, required=True,
                        choices=['cvae_01', 'cvae_02', 'cvae_03', 'cvae_02_1', 'cvae_02_2', 'cvae_02_3'],
                        help='Model to evaluate')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data_path', type=str, default='data/hand_object_data.pkl',
                        help='Path to preprocessed data')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--latent_dim', type=int, default=32, help='Latent dimension')

    args = parser.parse_args()
    evaluate_model(args.model, args.checkpoint, args.data_path, args.batch_size, args.latent_dim)


if __name__ == '__main__':
    main()

