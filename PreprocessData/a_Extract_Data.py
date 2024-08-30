import os
import pickle
import numpy as np
from tqdm import tqdm



#------load_ho3d_best_info FUNCTION-----
#It reads sequence information from text files,
#extracts relevant annotations from corresponding pickle files,
#and then saves the processed data into new pickle files.
#The progress of loading the data is displayed using a progress bar,
#and finally print the count of loaded entries.


#data_dir=Directory where the dataset is stored.
#split= Specifies the dataset split to be loaded, default is 'train'
def load_ho3d_best_info(data_dir, split='train'):
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
                data_dir+r'\train', seq_name, 'meta', f'{file_id}.pkl')

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


# Load data
data_dir = r'D:\UNI\Sem3\Dissertation\My effort\HOnnotate\ho3d-master\Dataset\HO3D_v3'
hand_poses, object_infos = load_ho3d_best_info(data_dir, split='train')

# Save hand data into hand_poses.pkl
# Save object data into object_infos.pkl
hand_poses_file = 'hand_poses.pkl'
object_infos_file = 'object_infos.pkl'

with open(hand_poses_file, 'wb') as hp_file:
    pickle.dump(hand_poses, hp_file)

with open(object_infos_file, 'wb') as oi_file:
    pickle.dump(object_infos, oi_file)

print("Data saved successfully!")


print("Data loaded successfully!")
print("Number of hand poses loaded:", len(hand_poses))
print("Number of object infos loaded:", len(object_infos))
