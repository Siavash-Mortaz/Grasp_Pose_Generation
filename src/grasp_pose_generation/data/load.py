"""Load extracted data from pickle files."""

import pickle


def load_saved_data(hand_poses_file, object_infos_file):
    """
    Load and deserialize data from hand_poses.pkl and object_infos.pkl files.
    
    Args:
        hand_poses_file: The location of hand extracted data
        object_infos_file: The location of object extracted data
    
    Returns:
        hand_poses: List of hand pose dictionaries
        object_infos: List of object information dictionaries
    """
    with open(hand_poses_file, 'rb') as hp_file:
        hand_poses = pickle.load(hp_file)

    with open(object_infos_file, 'rb') as oi_file:
        object_infos = pickle.load(oi_file)
    
    return hand_poses, object_infos

