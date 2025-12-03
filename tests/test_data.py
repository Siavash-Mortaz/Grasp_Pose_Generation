"""Tests for data loading and preprocessing utilities."""

import pytest
import pickle
import numpy as np
import torch
import tempfile
import os
from grasp_pose_generation.data.load import load_saved_data
from grasp_pose_generation.training.data_loader import loading_data


def test_load_saved_data():
    """Test loading saved data from pickle files."""
    # Create temporary pickle files
    hand_poses = [
        {'handPose': np.array([1, 2, 3]), 'handTrans': np.array([0.1, 0.2, 0.3])},
        {'handPose': np.array([4, 5, 6]), 'handTrans': np.array([0.4, 0.5, 0.6])},
    ]
    object_infos = [
        {'objTrans': np.array([1.0, 2.0, 3.0]), 'objRot': np.array([0.1, 0.2, 0.3])},
        {'objTrans': np.array([4.0, 5.0, 6.0]), 'objRot': np.array([0.4, 0.5, 0.6])},
    ]
    
    with tempfile.TemporaryDirectory() as tmpdir:
        hand_file = os.path.join(tmpdir, 'hand_poses.pkl')
        obj_file = os.path.join(tmpdir, 'object_infos.pkl')
        
        with open(hand_file, 'wb') as f:
            pickle.dump(hand_poses, f)
        with open(obj_file, 'wb') as f:
            pickle.dump(object_infos, f)
        
        loaded_hand, loaded_obj = load_saved_data(hand_file, obj_file)
        
        assert len(loaded_hand) == 2
        assert len(loaded_obj) == 2
        assert np.array_equal(loaded_hand[0]['handPose'], hand_poses[0]['handPose'])


def test_loading_data():
    """Test loading preprocessed data for training."""
    # Create a temporary preprocessed data file
    data_dict = {
        'hand_train': np.random.randn(100, 114).astype(np.float32),
        'hand_val': np.random.randn(20, 114).astype(np.float32),
        'hand_test': np.random.randn(20, 114).astype(np.float32),
        'obj_train': np.random.randn(100, 30).astype(np.float32),
        'obj_val': np.random.randn(20, 30).astype(np.float32),
        'obj_test': np.random.randn(20, 30).astype(np.float32),
        'obj_names': ['object1', 'object2', 'object3'],
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        data_file = os.path.join(tmpdir, 'hand_object_data.pkl')
        
        with open(data_file, 'wb') as f:
            pickle.dump(data_dict, f)
        
        batch_size = 32
        result = loading_data(data_file, batch_size)
        
        hand_train, hand_val, hand_test, obj_train, obj_val, obj_test, train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader, obj_names = result
        
        # Check shapes
        assert hand_train.shape == (100, 114)
        assert hand_val.shape == (20, 114)
        assert hand_test.shape == (20, 114)
        assert obj_train.shape == (100, 30)
        assert obj_val.shape == (20, 30)
        assert obj_test.shape == (20, 30)
        
        # Check that data loaders are created
        assert train_loader is not None
        assert val_loader is not None
        assert test_loader is not None
        
        # Check that datasets are created
        assert len(train_dataset) == 100
        assert len(val_dataset) == 20
        assert len(test_dataset) == 20
        
        # Check object names
        assert len(obj_names) == 3


def test_data_loader_batch_size():
    """Test that data loader uses correct batch size."""
    data_dict = {
        'hand_train': np.random.randn(100, 114).astype(np.float32),
        'hand_val': np.random.randn(20, 114).astype(np.float32),
        'hand_test': np.random.randn(20, 114).astype(np.float32),
        'obj_train': np.random.randn(100, 30).astype(np.float32),
        'obj_val': np.random.randn(20, 30).astype(np.float32),
        'obj_test': np.random.randn(20, 30).astype(np.float32),
        'obj_names': [],
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        data_file = os.path.join(tmpdir, 'hand_object_data.pkl')
        
        with open(data_file, 'wb') as f:
            pickle.dump(data_dict, f)
        
        batch_size = 16
        _, _, _, _, _, _, _, _, _, train_loader, _, _, _ = loading_data(data_file, batch_size)
        
        # Check first batch
        for batch in train_loader:
            hand_batch, obj_batch = batch
            assert hand_batch.shape[0] <= batch_size
            assert obj_batch.shape[0] <= batch_size
            break

