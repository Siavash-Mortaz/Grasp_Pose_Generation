import os
import pickle
import numpy as np

hand_poses_file= r'hand_poses.pkl'
object_infos_file= r'object_infos.pkl'


# -- load_saved_data FUNCTION --
# It's loads and deserializes data from hand_poses.pkl and object_infos.pkl files
#hand_poses_file = The location of hand extracted data
#object_infos_file = The location of object extracted data
def load_saved_data(hand_poses_file,object_infos_file):
    with open(hand_poses_file, 'rb') as hp_file:
        hand_poses = pickle.load(hp_file)

    with open(object_infos_file, 'rb') as oi_file:
        object_infos = pickle.load(oi_file)
    return hand_poses,object_infos


hand_poses,object_infos=load_saved_data(hand_poses_file,object_infos_file)

#-----PRINTS OUT HOW MANY ITEMS WERE LOADED FROM EACH FILE -------
print("Number of hand poses loaded:", len(hand_poses))
print("Number of object infos loaded:", len(object_infos))



# ----------- AND PRINT A SAMPLE OF HAND AND OBJECT INFORMATION ------------------
print('===============================================')
print('--Print The 1st Object Info As A Sample--')
print(f"The Object Name: {object_infos[0]['objName']}")
print(f"The Object Label: {object_infos[0]['objLabel']}")
print(f"The Object Rotation: {object_infos[0]['objRot']}")
print(f"The Object Translation: {object_infos[0]['objTrans']}")
print(f"The Object 3D Bounding Box: {object_infos[0]['objCorners3D']}")

print('===============================================')
print('--Print The 1st Hand Info As A Sample--')
print(f"The Hand Pose: {hand_poses[0]['handPose']}")
print(f"The Hand Translation: {hand_poses[0]['handTrans']}")
print(f"The Object 3D Hand Joints: {hand_poses[0]['handJoints3D']}")
