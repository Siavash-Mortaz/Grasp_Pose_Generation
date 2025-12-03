"""Evaluation metrics for model performance."""

import torch
import torch.nn.functional as F


def mean_squared_error(pred, target):
    """Calculate mean squared error between predictions and targets."""
    return F.mse_loss(pred, target)


def mean_absolute_error(pred, target):
    """Calculate mean absolute error between predictions and targets."""
    return F.l1_loss(pred, target)

