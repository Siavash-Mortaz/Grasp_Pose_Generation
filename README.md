
# Grasp Pose Generation: Engineering Generative Models to Synthesize Human Hand Actions Based on Object Affordances for Robots

This project focuses on developing generative models to synthesize human-like hand actions for robotic systems using knowledge of object affordances. It primarily utilizes Conditional Variational Autoencoders (CVAE) to autonomously generate hand motions based on the perceived functions of objects, without explicit programming. By training these models on the HO-3D_v3 dataset, which includes detailed 3D hand-object interactions, the research aims to enhance the adaptability and efficiency of robots in performing real-world tasks, such as human-robot interaction and autonomous object manipulation. The work presents a robust CVAE architecture that effectively incorporates object information, demonstrating improved generalization across various objects and scenarios. It highlights the potential for generative models to enable more dynamic, flexible, and autonomous robotic behaviors, marking a significant advancement in the field of robotics.


## Dataset
Dataset is taken from (Hampali, Shreyas; Rad, Mahdi; Oberweger, Markus; Lepetit, Vincent, 2020). HO-3D_v3 is specifically designed for 3D pose annotation for interacting hand–object. The dataset contains 103,462 RGB images along with their 3D hand–object poses and corresponding depth maps. The dataset contains 10 human subjects, 3 females, and 7 men; further, it contains 10 objects selected from the YCB dataset. MANO model is selected for estimation of hand pose. The Dataset is available in https://www.tugraz.at/institute/icg/research/team-lepetit/research-projects/hand-object-3d-pose-annotation/

```bash
  @INPROCEEDINGS{hampali2020honnotate,
title={HOnnotate: A method for 3D Annotation of Hand and Object Poses},
author={Shreyas Hampali and Mahdi Rad and Markus Oberweger and Vincent Lepetit},
booktitle = {CVPR},
year = {2020}
}
```
To gain a better understanding of the dataset, random objects from various frames of the HO-3D_v3 dataset were applied to the 3D point data of objects in the YCB dataset. This approach helps explore how object rotation and translation work in combination between the YCB dataset and the HO-3D_v3 dataset. Similarly, it allows for a deeper insight into the 3D hand joint poses, along with their rotation and translation, across different objects and frames.






## Extracting, pre-processing and preparation Data
The process started with selecting suitable data from the HO-3D_v3 dataset, containing 3D hand-object interaction data essential for training generative models. Key features, including hand and object poses, were extracted and saved in pickle files ("hand_poses.pkl" and "object_infos.pkl") for future use. The data was then preprocessed using the StandardScaler from sklearn.preprocessing to standardize it, crucial for effective model training. After splitting the data into training, validation, and testing sets (60:20:20 ratio), the normalized data and scalers were saved for consistent input scaling during predictions, ensuring model accuracy.
![Preprocessing Data](https://github.com/Siavash-Mortaz/Grasp_Pose_Generation/blob/main/slides/Pre_Data01.JPG)
![Splitting Data](https://github.com/Siavash-Mortaz/Grasp_Pose_Generation/blob/main/slides/Pre_Data02.JPG)

## Models Architecture
The goal was to develop an autoencoder capable of encoding information about hand poses and objects, which could later be reconstructed from the encoded representation. The underlying assumption was that, after sufficient training, the model would automatically extract hand pose information from the object data.

Initially, three different Conditional Variational Autoencoder models—referred to as CVAE_01, CVAE_02, and CVAE_03—were created. All three models were trained under the same conditions, with identical latent space dimensionality, learning rate, and number of epochs. Among them, CVAE_02 demonstrated superior performance, prompting further enhancements based on this model. Two new variations, CVAE_02_1 and CVAE_02_2, were developed by adding additional layers to CVAE_02. Additionally, CVAE_02_3 was designed by removing the conditional input from the encoder, simplifying the model while preserving its performance, with only the decoder utilizing the conditional input.
![CVAE_01](https://github.com/Siavash-Mortaz/Grasp_Pose_Generation/blob/main/slides/CVAE_01.JPG)
![CVAE_02](https://github.com/Siavash-Mortaz/Grasp_Pose_Generation/blob/main/slides/CVAE_02.JPG)
![CVAE_03](https://github.com/Siavash-Mortaz/Grasp_Pose_Generation/blob/main/slides/CVAE_03.JPG)
![CVAE_02_1](https://github.com/Siavash-Mortaz/Grasp_Pose_Generation/blob/main/slides/CVAE_02_1.JPG)
![CVAE_02_2](https://github.com/Siavash-Mortaz/Grasp_Pose_Generation/blob/main/slides/CVAE_02_2.JPG)
![CVAE_02_3](https://github.com/Siavash-Mortaz/Grasp_Pose_Generation/blob/main/slides/CVAE_02_3.JPG)


## Training Progress
The CVAE models—CVAE_01, CVAE_02, and CVAE_03—were trained with a latent space of 32, a learning rate of 0.001, and for 50 epochs. CVAE_03 was specifically trained using hand and object data, later incorporating random noise into the hand information. The models' weights were stored in corresponding pickle files. Key metrics, such as train loss, validation loss, and various hand pose errors, were captured and saved.

Among the models, CVAE_02 demonstrated the best performance, showing the lowest loss and error metrics. In contrast, CVAE_03 showed minimal performance variation, even when noise was added, suggesting the noise had little impact. This highlighted CVAE_02 as the most effective model.

Further refinement of CVAE_02 involved increasing the latent space to 64 and extending the training epochs, which further reduced errors. This fine-tuning yielded strong results, encouraging continued development and optimization.

Subsequent runs of CVAE_02 with additional layers (CVAE_02_1 and CVAE_02_2) showed similar performance, indicating that the added complexity did not significantly affect the model's behavior. CVAE_02_2 showed a slight performance improvement.

In a final experiment, CVAE_02_2 was tested without the conditional decoder component, revealing that its removal had minimal impact on performance. This led to the adoption of CVAE_02_3 as the final model for regenerating hand postures based on object information, successfully simulating human-like hand movements and demonstrating the model's effectiveness. The model's predictions were further validated using the YCB dataset embedded in the HO3D dataset, allowing for accurate visualization of hand-object interactions.
## Visualization the Results

