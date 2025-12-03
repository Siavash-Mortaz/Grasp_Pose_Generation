"""Extract data from HO-3D dataset."""

import os
import pickle
import numpy as np
from tqdm import tqdm


def load_ho3d_best_info(data_dir, split='train'):
    """
    Load hand and object information from HO-3D dataset.
    
    It reads sequence information from text files,
    extracts relevant annotations from corresponding pickle files,
    and then saves the processed data into new pickle files.
    The progress of loading the data is displayed using a progress bar,
    and finally prints the count of loaded entries.
    
    Args:
        data_dir: Directory where the dataset is stored.
        split: Specifies the dataset split to be loaded, default is 'train'
    
    Returns:
        hand_poses: List of hand pose dictionaries
        object_infos: List of object information dictionaries
    """
    hand_poses = []
    object_infos = []

    split_file = os.path.join(data_dir, f'{split}.txt')
    with open(split_file, 'r') as f:
        sequences = f.readlines()

    # Initialize the progress bar
    total_sequences = len(sequences)
    with tqdm(total=total_sequences, desc="Loading data", unit="sequence") as pbar:
        for sequence in sequences:
            seq_name, file_id = sequence.strip().split('/')
            meta_file = os.path.join(
                data_dir, 'train', seq_name, 'meta', f'{file_id}.pkl')

            # Load annotations
            with open(meta_file, 'rb') as mf:
                annotations = pickle.load(mf)

            if annotations['handPose'] is not None and annotations['objTrans'] is not None:
                hand_poses.append({
                    'handPose': annotations['handPose'],
                    'handTrans': annotations['handTrans'],
                    'handJoints3D': annotations['handJoints3D']
                })

                object_infos.append({
                    'objTrans': annotations['objTrans'],
                    'objRot': annotations['objRot'],
                    'objName': annotations['objName'],
                    'objLabel': annotations['objLabel'],
                    'objCorners3D': annotations['objCorners3D']
                })

            # Update the progress bar
            pbar.update(1)

    return hand_poses, object_infos

