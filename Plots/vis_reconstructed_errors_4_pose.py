import matplotlib.pyplot as plt

# 003_cracker_box
obj1 = [10, 1400, 1550, 3020]
mse1_1 = [0.02214, 0.00402, 0.03320, 0.26042]
mae1_1=[0.07350, 0.04886, 0.11325, 0.28807]

mse1_2 = [0.01803, 0.00346, 0.02656, 0.19452]
mae1_2=[0.06721, 0.04600, 0.10665, 0.24028]

mse1_3 = [0.02168, 0.00468, 0.03316, 0.16994]
mae1_3=[0.06853, 0.05642, 0.10174, 0.23654]
#-------------------------------------------------

# 006_mustard_bottle
obj2 = [4500, 5215, 7600, 7750]
mse2_1 = [0.06275, 0.02468, 0.06269, 0.08407]
mae2_1=[0.10893, 0.07551, 0.10862, 0.14601]

mse2_2 = [0.04420, 0.04772, 0.04460, 0.09299]
mae2_2=[0.09111, 0.13946, 0.08429, 0.14788]

mse2_3 = [0.04374, 0.03355, 0.08153, 0.06805]
mae2_3=[0.09016, 0.11307, 0.12405, 0.12760]
#-------------------------------------------------

# 004_sugar_box
obj3 = [8000, 10000, 10525, 11221]
mse3_1 = [0.18972, 0.02652, 0.09557, 0.02668]
mae3_1=[0.15505, 0.07554, 0.11846, 0.06919]

mse3_2 = [0.06486, 0.02350, 0.09420, 0.03053]
mae3_2=[0.09280, 0.06315, 0.11787, 0.07746]

mse3_3 = [0.05801, 0.02405, 0.10815, 0.03103]
mae3_3=[0.08891, 0.06330, 0.13472, 0.07737]
#-------------------------------------------------

# 025_mug
obj4 = [12500, 12600, 12900, 13052]
mse4_1 = [0.03833, 0.00185, 0.05709, 0.10840]
mae4_1=[0.09860, 0.03277, 0.07561, 0.11590]

mse4_2 = [0.04844, 0.00282, 0.05632, 0.09769]
mae4_2=[0.12349, 0.04323, 0.06738, 0.09024]

mse4_3 = [0.04394, 0.00162, 0.05268, 0.10308]
mae4_3=[0.10613, 0.03291, 0.07012, 0.09461]
#-------------------------------------------------

# 035_power_drill
obj5 = [14000, 14050, 16600, 16653]
mse5_1 = [0.24712, 0.09182, 0.03831, 0.06431]
mae5_1=[0.30400, 0.13309, 0.09022, 0.08015]

mse5_2 = [0.13530, 0.09024, 0.04211, 0.05800]
mae5_2=[0.22366, 0.12743, 0.08903, 0.07534]

mse5_3 = [0.05241, 0.09198, 0.04188, 0.06953]
mae5_3=[0.13036, 0.12706, 0.08894, 0.09124]
#-------------------------------------------------


#============PLOT FOR 003_cracker_box
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes = axes.flatten()

axes[0].plot(obj1, mse1_1, 'bo-', label='CVAE_02_1', color='orange')
axes[0].plot(obj1, mse1_2, 'bo-', label='CVAE_02_2', color='purple')
axes[0].plot(obj1, mse1_3, 'bo-', label='CVAE_02_3', color='black')
axes[0].set_title("MSE")
axes[0].set_xlabel('Test Data Number')
axes[0].set_ylabel('Mean Squared Error (Millimeter)')
axes[0].grid(True)
axes[0].legend()

axes[1].plot(obj1, mae1_1, 'bo-', label='CVAE_02_1', color='orange')
axes[1].plot(obj1, mae1_2, 'bo-', label='CVAE_02_2', color='purple')
axes[1].plot(obj1, mae1_3, 'bo-', label='CVAE_02_3', color='black')
axes[1].set_title("MAE")
axes[1].set_xlabel('Test Data Number')
axes[1].set_ylabel('Mean Absolute Error (Millimeter)')
axes[1].grid(True)
axes[1].legend()

fig.suptitle("Plot MSE an MAE For Hand Pose Generated \n Based On Object: '003_cracker_box'")

plt.grid(True)
plt.show()

#============PLOT FOR 006_mustard_bottle
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes = axes.flatten()

axes[0].plot(obj2, mse2_1, 'bo-', label='CVAE_02_1', color='orange')
axes[0].plot(obj2, mse2_2, 'bo-', label='CVAE_02_2', color='purple')
axes[0].plot(obj2, mse2_3, 'bo-', label='CVAE_02_3', color='black')
axes[0].set_title("MSE")
axes[0].set_xlabel('Test Data Number')
axes[0].set_ylabel('Mean Squared Error (Millimeter)')
axes[0].grid(True)
axes[0].legend()

axes[1].plot(obj2, mae2_1, 'bo-', label='CVAE_02_1', color='orange')
axes[1].plot(obj2, mae2_2, 'bo-', label='CVAE_02_2', color='purple')
axes[1].plot(obj2, mae2_3, 'bo-', label='CVAE_02_3', color='black')
axes[1].set_title("MAE")
axes[1].set_xlabel('Test Data Number')
axes[1].set_ylabel('Mean Absolute Error (Millimeter)')
axes[1].grid(True)
axes[1].legend()

fig.suptitle("Plot MSE an MAE For Hand Pose Generated \n Based On Object: '006_mustard_bottle'")

plt.grid(True)
plt.show()

#============PLOT FOR 004_sugar_box
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes = axes.flatten()

axes[0].plot(obj3, mse3_1, 'bo-', label='CVAE_02_1', color='orange')
axes[0].plot(obj3, mse3_2, 'bo-', label='CVAE_02_2', color='purple')
axes[0].plot(obj3, mse3_3, 'bo-', label='CVAE_02_3', color='black')
axes[0].set_title("MSE")
axes[0].set_xlabel('Test Data Number')
axes[0].set_ylabel('Mean Squared Error (Millimeter)')
axes[0].grid(True)
axes[0].legend()

axes[1].plot(obj3, mae1_3, 'bo-', label='CVAE_02_1', color='orange')
axes[1].plot(obj3, mae1_3, 'bo-', label='CVAE_02_2', color='purple')
axes[1].plot(obj3, mae1_3, 'bo-', label='CVAE_02_3', color='black')
axes[1].set_title("MAE")
axes[1].set_xlabel('Test Data Number')
axes[1].set_ylabel('Mean Absolute Error (Millimeter)')
axes[1].grid(True)
axes[1].legend()

fig.suptitle("Plot MSE an MAE For Hand Pose Generated \n Based On Object: '004_sugar_box'")

plt.grid(True)
plt.show()

#============PLOT FOR 025_mug
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes = axes.flatten()

axes[0].plot(obj4, mse4_1, 'bo-', label='CVAE_02_1', color='orange')
axes[0].plot(obj4, mse4_2, 'bo-', label='CVAE_02_2', color='purple')
axes[0].plot(obj4, mse4_3, 'bo-', label='CVAE_02_3', color='black')
axes[0].set_title("MSE")
axes[0].set_xlabel('Test Data Number')
axes[0].set_ylabel('Mean Squared Error (Millimeter)')
axes[0].grid(True)
axes[0].legend()

axes[1].plot(obj4, mae4_1, 'bo-', label='CVAE_02_1', color='orange')
axes[1].plot(obj4, mae4_2, 'bo-', label='CVAE_02_2', color='purple')
axes[1].plot(obj4, mae4_3, 'bo-', label='CVAE_02_3', color='black')
axes[1].set_title("MAE")
axes[1].set_xlabel('Test Data Number')
axes[1].set_ylabel('Mean Absolute Error (Millimeter)')
axes[1].grid(True)
axes[1].legend()

fig.suptitle("Plot MSE an MAE For Hand Pose Generated \n Based On Object: '025_mug'")

plt.grid(True)
plt.show()

#============PLOT FOR 035_power_drill
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes = axes.flatten()

axes[0].plot(obj5, mse5_1, 'bo-', label='CVAE_02_1', color='orange')
axes[0].plot(obj5, mse5_2, 'bo-', label='CVAE_02_2', color='purple')
axes[0].plot(obj5, mse5_3, 'bo-', label='CVAE_02_3', color='black')
axes[0].set_title("MSE")
axes[0].set_xlabel('Test Data Number')
axes[0].set_ylabel('Mean Squared Error (Millimeter)')
axes[0].grid(True)
axes[0].legend()

axes[1].plot(obj5, mae5_1, 'bo-', label='CVAE_02_1', color='orange')
axes[1].plot(obj5, mae5_2, 'bo-', label='CVAE_02_2', color='purple')
axes[1].plot(obj5, mae5_3, 'bo-', label='CVAE_02_3', color='black')
axes[1].set_title("MAE")
axes[1].set_xlabel('Test Data Number')
axes[1].set_ylabel('Mean Absolute Error (Millimeter)')
axes[1].grid(True)
axes[1].legend()

fig.suptitle("Plot MSE an MAE For Hand Pose Generated \n Based On Object: '035_power_drill'")

plt.grid(True)
plt.show()