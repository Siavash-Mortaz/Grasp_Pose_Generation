"""Tests for CVAE models."""

import pytest
import torch
from grasp_pose_generation.models import (
    CVAE_01,
    CVAE_02,
    CVAE_03,
    CVAE_02_1,
    CVAE_02_2,
    CVAE_02_3,
)


@pytest.fixture
def sample_data():
    """Create sample input data for testing."""
    batch_size = 32
    input_dim = 114
    condition_dim = 30
    x = torch.randn(batch_size, input_dim)
    c = torch.randn(batch_size, condition_dim)
    return x, c, input_dim, condition_dim


@pytest.mark.parametrize("model_class", [
    CVAE_01,
    CVAE_02,
    CVAE_03,
    CVAE_02_1,
    CVAE_02_2,
])
def test_cvae_forward_pass(model_class, sample_data):
    """Test that CVAE models can perform forward pass."""
    x, c, input_dim, condition_dim = sample_data
    latent_dim = 32
    
    model = model_class(input_dim, latent_dim, condition_dim)
    model.eval()
    
    with torch.no_grad():
        reconstructed, mean, log_var = model(x, c)
    
    assert reconstructed.shape == x.shape
    assert mean.shape == (x.shape[0], latent_dim)
    assert log_var.shape == (x.shape[0], latent_dim)


def test_cvae_02_3_forward_pass(sample_data):
    """Test CVAE_02_3 which has different encoder signature."""
    x, c, input_dim, condition_dim = sample_data
    latent_dim = 32
    
    model = CVAE_02_3(input_dim, latent_dim, condition_dim)
    model.eval()
    
    with torch.no_grad():
        reconstructed, mean, log_var = model(x, c)
    
    assert reconstructed.shape == x.shape
    assert mean.shape == (x.shape[0], latent_dim)
    assert log_var.shape == (x.shape[0], latent_dim)


@pytest.mark.parametrize("model_class", [
    CVAE_01,
    CVAE_02,
    CVAE_03,
    CVAE_02_1,
    CVAE_02_2,
])
def test_cvae_encode_decode(model_class, sample_data):
    """Test encode and decode methods separately."""
    x, c, input_dim, condition_dim = sample_data
    latent_dim = 32
    
    model = model_class(input_dim, latent_dim, condition_dim)
    model.eval()
    
    with torch.no_grad():
        mean, log_var = model.encode(x, c)
        z = model.reparameterize(mean, log_var)
        decoded = model.decode(z, c)
    
    assert mean.shape == (x.shape[0], latent_dim)
    assert log_var.shape == (x.shape[0], latent_dim)
    assert z.shape == (x.shape[0], latent_dim)
    assert decoded.shape == x.shape


def test_cvae_02_3_encode_decode(sample_data):
    """Test CVAE_02_3 encode and decode methods (encoder doesn't take condition)."""
    x, c, input_dim, condition_dim = sample_data
    latent_dim = 32
    
    model = CVAE_02_3(input_dim, latent_dim, condition_dim)
    model.eval()
    
    with torch.no_grad():
        mean, log_var = model.encode(x)  # Note: no condition for encoder
        z = model.reparameterize(mean, log_var)
        decoded = model.decode(z, c)
    
    assert mean.shape == (x.shape[0], latent_dim)
    assert log_var.shape == (x.shape[0], latent_dim)
    assert z.shape == (x.shape[0], latent_dim)
    assert decoded.shape == x.shape


def test_reparameterize(sample_data):
    """Test reparameterization trick."""
    x, c, input_dim, condition_dim = sample_data
    latent_dim = 32
    
    model = CVAE_01(input_dim, latent_dim, condition_dim)
    mean = torch.zeros(10, latent_dim)
    log_var = torch.zeros(10, latent_dim)
    
    z = model.reparameterize(mean, log_var)
    
    assert z.shape == (10, latent_dim)
    # With zero mean and log_var, z should be approximately zero (with some noise)
    assert torch.allclose(z, torch.zeros_like(z), atol=1.0)


def test_model_training_mode(sample_data):
    """Test that model can switch between train and eval modes."""
    x, c, input_dim, condition_dim = sample_data
    latent_dim = 32
    
    model = CVAE_01(input_dim, latent_dim, condition_dim)
    
    # Test training mode
    model.train()
    assert model.training
    
    # Test eval mode
    model.eval()
    assert not model.training

