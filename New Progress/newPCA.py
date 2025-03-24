import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial.transform import Rotation as R
import torch.nn as nn
import torch.nn.functional as F
# import seaborn as sns
from sklearn.decomposition import PCA
from Models_Trains.all_models import CVAE_02_4_3


def visualize_hand_2d(handTrans, handJoints3D):

    # Apply hand translations
    handJoints3D_transformed = handJoints3D + handTrans


    # Extract coordinates for transformed hand joints
    hand_x = handJoints3D_transformed[:, 0]
    hand_y = handJoints3D_transformed[:, 1]


    fig, ax = plt.subplots(figsize=(10, 8))


    # Plot Hand 1 (Reconstructed) on X-Y plane
    ax.scatter(hand_x, hand_y, c='r', marker='o', label='Hand Joints')
    for (i, j) in [(0, 7), (7, 8), (8, 9), (9, 20), (0, 10), (10, 11), (11, 12), (12, 19),
                   (0, 4), (4, 5), (5, 6), (6, 18), (0, 1), (1, 2), (2, 3), (3, 17),
                   (0, 13), (13, 14), (14, 15), (15, 16)]:
        ax.plot([hand_x[i], hand_x[j]], [hand_y[i], hand_y[j]], 'b')



    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title(f'Sample of Hand Pose Data - X-Y Plane ')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.show()

def visualize_hand_3d(handJoints3D, handTrans, handPose):
    # Applying translation
    handJoints3D_transformed = handJoints3D + handTrans

    # Converting axis-angle to rotation matrix and applying rotation
    rotation = R.from_rotvec(handPose[:3])
    handJoints3D_transformed = rotation.apply(handJoints3D_transformed)

    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Extracting coordinates
    xs = handJoints3D_transformed[:, 0]
    ys = handJoints3D_transformed[:, 1]
    zs = handJoints3D_transformed[:, 2]

    # Plotting joints
    ax.scatter(xs, ys, zs, c='r', marker='o')

    # Connecting joints to form the skeleton (MANO joint order)
    connections = [
        (0, 7), (7, 8), (8, 9), (9, 20),  # Little finger
        (0, 10), (10, 11), (11, 12), (12, 19),  # Ring finger
        (0, 4), (4, 5), (5, 6), (6, 18),  # Middle finger
        (0, 1), (1, 2), (2, 3), (3, 17),  # Index finger
        (0, 13), (13, 14), (14, 15), (15, 16)  # Thumb
    ]

    for (i, j) in connections:
        ax.plot([xs[i], xs[j]], [ys[i], ys[j]], [zs[i], zs[j]], 'b')

    # Setting labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'3D Scatter Plot of Transformed Hand pose')

    plt.show()

model_weight=r"D:\UNI\Sem3\Dissertation\My effort\effort_02 - Git - Copy\Models_Trains\final_model_8_8.pth"
hand_object_data=r"D:\UNI\Sem3\Dissertation\My effort\effort_02 - Git - Copy\PreprocessData\hand_object_data.pkl"
scaler_file=r"D:\UNI\Sem3\Dissertation\My effort\effort_02 - Git - Copy\PreprocessData\scalers.pkl"



# Load dataset
with open(hand_object_data, "rb") as f:
    data = pickle.load(f)

# Extract hand pose data (only hand parameters)
hand_pose_data = data['hand_train']
object_data = data['obj_train']
object_names = np.array(data['obj_names'])[data['train_indices']]  # Filtered object names

with open(scaler_file, 'rb') as scalers_file:
    scalers= pickle.load(scalers_file)

# Extract the individual scalers
scaler_hand_joints = scalers['scaler_hand_joints']
scaler_hand_pose = scalers['scaler_hand_pose']
scaler_hand_trans = scalers['scaler_hand_trans']

hand_pose= hand_pose_data[0][:48]  # First 48 elements for hand pose
hand_trans= hand_pose_data[48:51]  # Next 3 elements for hand translation
hand_joints= hand_pose_data[51:]  # Remaining 63 elements for hand joints (21x3)
print(f'First 48 elements for hand pose: {hand_pose.shape}')
print(f'Next 3 elements for hand translation: {hand_trans.shape}')
print(f'Remaining 63 elements for hand joints: {hand_joints.shape}')

sample_data=hand_pose_data[1300]
hand_pose= sample_data[:48]  # First 48 elements for hand pose
hand_trans= sample_data[48:51]  # Next 3 elements for hand translation
hand_joints= sample_data[51:]  # Remaining 63 elements for hand joints (21x3)
hand_joints= hand_joints.reshape(-1, 3) # Reshape hand joints back to (21, 3)
# Inverse transform each component using the corresponding scaler
hand_pose_original = scaler_hand_pose.inverse_transform(hand_pose.reshape(1, -1)).reshape(-1)
hand_trans_original = scaler_hand_trans.inverse_transform(hand_trans.reshape(1, -1)).reshape(-1)
hand_joints_original = scaler_hand_joints.inverse_transform(hand_joints.reshape(1, -1)).reshape(-1, 3)

visualize_hand_2d(hand_trans_original, hand_joints_original)
visualize_hand_3d(hand_joints_original, hand_trans_original, hand_pose_original)

# Initialize the CVAE_02_4 model with correct parameters
input_dim = hand_pose_data.shape[1]
latent_dim = 16  # Adjust based on the latest model training
condition_dim = data['obj_train'].shape[1]  # Object condition dimensions
num_gaussians = 10 #10  # Number of Gaussians used in MoG
print(f'Input(Hand) Dim: {input_dim}')
print(f'Latent Dim: {latent_dim}')
print(f'Condition(Object) Dim: {condition_dim}')

# Define and load the CVAE_02_4 model
model = CVAE_02_4_3(input_dim, latent_dim, condition_dim,num_gaussians)
model.load_state_dict(torch.load(model_weight))  # Load latest pre-trained weights
model.eval()

# Convert hand pose data to tensor
hand_pose_tensor = torch.tensor(hand_pose_data, dtype=torch.float32)

# Extract latent representations
with torch.no_grad():
    mean, log_var, mixing_coeffs = model.encode(hand_pose_tensor)
    latent_vectors = model.reparameterize(mean, log_var, mixing_coeffs).numpy()


# Perform PCA on latent space
pca = PCA(n_components=latent_vectors.shape[1])  # Keep all components
pca.fit(latent_vectors)
# Plot explained variance ratio
plt.figure(figsize=(8, 5))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o', linestyle='--')
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA Explained Variance Ratio")
plt.grid(True)
plt.show()

pca = PCA(n_components=16)  # Try with All components
pca_results = pca.fit_transform(latent_vectors)
explained_variance = pca.explained_variance_ratio_

# Print cumulative explained variance
cumulative_variance = np.cumsum(explained_variance)
for i, var in enumerate(cumulative_variance):
    print(f"First {i+1} components explain {var:.2%} of the variance")


pca = PCA(n_components=10)  # Try with 10 components
pca_results = pca.fit_transform(latent_vectors)
explained_variance = pca.explained_variance_ratio_

# Print cumulative explained variance
cumulative_variance = np.cumsum(explained_variance)
for i, var in enumerate(cumulative_variance):
    print(f"First {i+1} components explain {var:.2%} of the variance")

# Plot Cumulative explained variance ratio
plt.figure(figsize=(8, 5))
plt.bar(range(1, 11), cumulative_variance, tick_label=[f'First{i}' for i in range(1, 11)])
plt.xlabel("Components")
plt.ylabel("Cumulative Explained Variance Ratio")
plt.title("Cumulative Explained Variance of Components")
plt.show()




# Perform PCA
pca = PCA(n_components=16)  # Reduce to 16 components for visualization
pca_results = pca.fit_transform(latent_vectors)

# Get explained variance ratio
explained_variance = pca.explained_variance_ratio_


# Display the PCA Result
print("PCA Results:")
print("\nExplained Variance Ratio:")
for i, var in enumerate(explained_variance):
    print(f"Principal Component {i+1} explains {var:.2%} of the variance")

# Plot explained variance ratio
plt.figure(figsize=(8, 5))
plt.bar(range(1, 17), explained_variance, tick_label=[f'PC{i}' for i in range(1, 17)])
plt.xlabel("Principal Component")
plt.ylabel("Explained Variance Ratio")
plt.title("Explained Variance of Each Principal Component")
plt.show()


# Convert PCA results to a DataFrame
pca_df = pd.DataFrame(pca_results[:,:], columns=['PC1', 'PC2','PC3','PC4','PC5',
                                                  'PC6','PC7','PC8','PC9','PC10',
                                                  'PC11','PC12','PC13',
                                                  'PC14','PC15','PC16'])
pca_df['Object Name'] = object_names  # Add object names for grouping

# print(pca_df)

# Unique object names for plotting
unique_objects = np.unique(object_names)

# Plot PCA results for each object
plt.figure(figsize=(12, 8))
for obj_name in unique_objects:
    # if obj_name=='035_power_drill'or obj_name=='003_cracker_box':

    subset = pca_df[pca_df['Object Name'] == obj_name]
    plt.scatter(subset['PC1'], subset['PC2'], label=obj_name, alpha=0.6)

# Add plot legend and labels
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Object Names")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA of Latent Space Grouped by Object Names")
plt.tight_layout()
plt.show()



from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plotting

# Convert PCA results to a DataFrame
pca_df = pd.DataFrame(
    pca_results[:, :],
    columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5',
             'PC6', 'PC7', 'PC8', 'PC9', 'PC10',
             'PC11', 'PC12', 'PC13', 'PC14', 'PC15', 'PC16']
)
pca_df['Object Name'] = object_names  # Add object names for grouping

# Unique object names for plotting
unique_objects = np.unique(object_names)

# Create 3D figure
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot PCA results for each object
for obj_name in unique_objects:
    subset = pca_df[pca_df['Object Name'] == obj_name]
    ax.scatter(subset['PC1'], subset['PC2'], subset['PC3'], label=obj_name, alpha=0.6)

# Add axis labels, legend, and title
ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
ax.set_zlabel("Principal Component 3")
plt.title("3D PCA of Latent Space Grouped by Object Names")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Object Names")
plt.tight_layout()
plt.show()



from sklearn.cluster import KMeans

# Apply K-Means clustering to PCA results (choosing 3 clusters)
num_clusters = 4
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
clusters = kmeans.fit_predict(pca_results)

# Plot PCA with cluster colors
plt.figure(figsize=(10, 6))
scatter = plt.scatter(pca_results[:, 0], pca_results[:, 1], c=clusters, cmap='Set1', alpha=0.7)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA of Latent Space with Clusters")
plt.legend(*scatter.legend_elements(), title="Clusters")
plt.tight_layout()
plt.show()

#=============================

# First 3 PCs to use
n_components_to_use = 3
pca_data = pca_results[:, :n_components_to_use]

wcss = []
for i in range(1, 11):  # Test k from 1 to 10
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(pca_data)
    wcss.append(kmeans.inertia_)  # Inertia is the WCSS

# Plot the elbow graph
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Based on the elbow, choose k and perform k-means
k = 4
kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
labels = kmeans.fit_predict(pca_data)

# Visualize in 3D (using the first 3 PCs)
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pca_data[:, 0], pca_data[:, 1], pca_data[:, 2], c=labels)
plt.show()
#======================


# Apply K-Means clustering to PCA results (choosing 9 clusters)
num_clusters = 9
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
clusters = kmeans.fit_predict(pca_results)

# Get centroids
centroids = kmeans.cluster_centers_

# Plot PCA with cluster colors
plt.figure(figsize=(10, 6))
scatter = plt.scatter(pca_results[:, 0], pca_results[:, 1], c=clusters, cmap='Set1', alpha=0.7, label="Data Points")

# Plot centroids in the same color as their respective clusters
plt.scatter(centroids[:, 0], centroids[:, 1], c=range(num_clusters), cmap='Set1', marker='X', s=200, edgecolors='k', label="Centroids")

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA of Latent Space with Clusters and Centroids")
plt.legend()
plt.tight_layout()
plt.show()

# Print centroid coordinates
print("Centroids of K-Means clusters:")
print(centroids)

# # 3D : 3 PC
# from sklearn.cluster import KMeans
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D  # for 3D plotting
# import numpy as np
#
# # Assume 'pca_results' is an array with shape (n_samples, n_components).
# # In this example, we only use the first 3 components for clustering and plotting.
# pca_results_3d = pca_results[:, :3]
#
# # Number of clusters
# num_clusters = 9
#
# # Apply K-Means clustering to the first three principal components
# kmeans = KMeans(n_clusters=num_clusters, random_state=42)
# clusters = kmeans.fit_predict(pca_results_3d)
#
# # Get cluster centroids
# centroids = kmeans.cluster_centers_
#
# # Create a 3D figure
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')
#
# # Plot the data points with cluster colors
# scatter = ax.scatter(
#     pca_results_3d[:, 0],
#     pca_results_3d[:, 1],
#     pca_results_3d[:, 2],
#     c=clusters,
#     cmap='Set1',
#     alpha=0.7,
#     label="Data Points"
# )
#
# # Plot the centroids
# ax.scatter(
#     centroids[:, 0],
#     centroids[:, 1],
#     centroids[:, 2],
#     c=range(num_clusters),
#     cmap='Set1',
#     marker='X',
#     s=200,
#     edgecolors='k',
#     label="Centroids"
# )
#
# # Label axes and add a title
# ax.set_xlabel("Principal Component 1")
# ax.set_ylabel("Principal Component 2")
# ax.set_zlabel("Principal Component 3")
# plt.title("3D PCA of Latent Space with Clusters and Centroids")
#
# # Show legend and final plot
# plt.legend()
# plt.tight_layout()
# plt.show()

# Print centroid coordinates
print("Centroids of K-Means clusters:")
print(centroids)




latent_centroids = pca.inverse_transform(centroids)  # Convert centroids back to latent space
latent_centroids.shape
latent_centroids[0]

# Select an object condition vector from the dataset (mean condition vector)
mean_condition = object_data.mean(axis=0)  # Take mean condition across the dataset
mean_condition = torch.tensor(mean_condition, dtype=torch.float32).unsqueeze(0)  # Convert to tensor

# Convert centroids to PyTorch tensor
latent_centroids_tensor = torch.tensor(latent_centroids, dtype=torch.float32)

# Pass each centroid through the decoder
model.eval()  # Set model to evaluation mode

reconstructed_poses = []
with torch.no_grad():
    for i in range(num_clusters):
        latent_vector = latent_centroids_tensor[i].unsqueeze(0)  # Select one centroid
        reconstructed_pose = model.decode(latent_vector, mean_condition)  # Decode
        reconstructed_poses.append(reconstructed_pose.cpu().numpy())  # Convert to NumPy

# Convert list to NumPy array for further analysis
reconstructed_poses = np.array(reconstructed_poses)

print(reconstructed_poses.shape)

for rec_pose in reconstructed_poses:
  rec_pose=rec_pose.reshape(-1)
  hand_pose_generated = rec_pose[:48] # First 48 elements for hand pose
  hand_pose_generated = scaler_hand_pose.inverse_transform(hand_pose_generated.reshape(1, -1)).reshape(-1) # Inverse transform each component using the corresponding scaler
  print(f'Shape hand_pose_np:{hand_pose_generated.shape}')

  hand_trans_generated = rec_pose[48:51] # Next 3 elements for hand translation
  hand_trans_generated = scaler_hand_trans.inverse_transform(hand_trans_generated.reshape(1, -1)).reshape(-1) # Inverse transform each component using the corresponding scaler
  print(f'Shape hand_trans_np:{hand_trans_generated.shape}')

  hand_joints_generated = rec_pose[51:] # Remaining 63 elements for hand joints (21x3)
  hand_joints_generated = hand_joints_generated.reshape(-1, 3) # Reshape hand joints back to (21, 3)
  hand_joints_generated = scaler_hand_joints.inverse_transform(hand_joints_generated.reshape(1, -1)).reshape(-1, 3)# Inverse transform each component using the corresponding scaler
  print(f'Shape hand_joints_np:{hand_joints_generated.shape}')

  visualize_hand_2d(hand_trans_generated, hand_joints_generated)
  visualize_hand_3d(hand_joints_generated, hand_trans_generated, hand_pose_generated)

