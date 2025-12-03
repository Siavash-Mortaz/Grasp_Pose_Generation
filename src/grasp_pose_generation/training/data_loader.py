"""Data loading utilities for training."""

import torch
import pickle
from torch.utils.data import DataLoader, TensorDataset


def loading_data(hand_object_data_path, batch_size):
    """
    Load preprocessed data and create data loaders.
    
    It loads preprocessed data from the pickle file (hand_object_data.pkl),
    converts it into PyTorch tensors,
    and prepares it for use in a machine learning model by organizing it into datasets and data loaders,
    which are then returned for further use in model training, validation, and testing.
    
    Args:
        hand_object_data_path: Path of the preprocessed data pickle file
        batch_size: Specifies how many samples per batch to load.
    
    Returns:
        Tuple containing:
        - hand_train, hand_val, hand_test: Training, validation, and test hand data tensors
        - obj_train, obj_val, obj_test: Training, validation, and test object data tensors
        - train_dataset, val_dataset, test_dataset: Dataset objects
        - train_loader, val_loader, test_loader: DataLoader objects
        - obj_names: List of object names
    """
    # LOAD PREPROCESSED DATA
    with open(hand_object_data_path, 'rb') as data_file:
        data_files = pickle.load(data_file)

    hand_train = torch.tensor(data_files['hand_train'], dtype=torch.float32)
    hand_val = torch.tensor(data_files['hand_val'], dtype=torch.float32)
    hand_test = torch.tensor(data_files['hand_test'], dtype=torch.float32)
    obj_train = torch.tensor(data_files['obj_train'], dtype=torch.float32)
    obj_val = torch.tensor(data_files['obj_val'], dtype=torch.float32)
    obj_test = torch.tensor(data_files['obj_test'], dtype=torch.float32)
    obj_names = data_files['obj_names']

    # COMBINATION DATA
    # Combines the hand and object tensors for each dataset
    # (train, validation, test) into a single dataset object
    # Each element in these datasets will consist of a pair (hand_data, obj_data)
    train_dataset = TensorDataset(hand_train, obj_train)
    val_dataset = TensorDataset(hand_val, obj_val)
    test_dataset = TensorDataset(hand_test, obj_test)

    # CREATE DATALOADER
    # Creates data loaders for the train, validation, and test datasets
    # shuffle=True =>  Randomly shuffles the data in the training set at every epoch
    # to prevent the model from learning the order of the data
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # shuffle=False => For validation and test datasets, data is not shuffled
    # since shuffling is not usually necessary or desirable for these sets.
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print('Data is loaded successfully')

    return hand_train, hand_val, hand_test, obj_train, obj_val, obj_test, train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader, obj_names

