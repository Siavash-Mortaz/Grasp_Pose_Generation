"""Tests for evaluation metrics."""

import pytest
import torch
from grasp_pose_generation.evaluation import mean_squared_error, mean_absolute_error


def test_mean_squared_error():
    """Test mean squared error calculation."""
    pred = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    target = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    
    mse = mean_squared_error(pred, target)
    
    # Perfect prediction should give zero error
    assert mse.item() == 0.0


def test_mean_squared_error_with_difference():
    """Test MSE with known difference."""
    pred = torch.tensor([[2.0, 3.0], [4.0, 5.0]])
    target = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    
    mse = mean_squared_error(pred, target)
    
    # Difference is 1.0 for all elements, so MSE should be 1.0
    assert abs(mse.item() - 1.0) < 1e-6


def test_mean_absolute_error():
    """Test mean absolute error calculation."""
    pred = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    target = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    
    mae = mean_absolute_error(pred, target)
    
    # Perfect prediction should give zero error
    assert mae.item() == 0.0


def test_mean_absolute_error_with_difference():
    """Test MAE with known difference."""
    pred = torch.tensor([[2.0, 3.0], [4.0, 5.0]])
    target = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    
    mae = mean_absolute_error(pred, target)
    
    # Difference is 1.0 for all elements, so MAE should be 1.0
    assert abs(mae.item() - 1.0) < 1e-6


def test_metrics_with_different_shapes():
    """Test that metrics work with different tensor shapes."""
    pred = torch.randn(32, 114)
    target = torch.randn(32, 114)
    
    mse = mean_squared_error(pred, target)
    mae = mean_absolute_error(pred, target)
    
    # Should return scalar tensors
    assert mse.dim() == 0
    assert mae.dim() == 0
    
    # Values should be positive
    assert mse.item() >= 0
    assert mae.item() >= 0

