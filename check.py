import pickle
import pandas as pd
with open('/home/kyber/Desktop/unused/TNT/trajectories_targets_smarts.pickle', "rb") as f:
    vehicle_data = pickle.load(f)
breakpoint()
# import matplotlib.pyplot as plt
# plt.plot(vehicle_data['gt_trajectories'][6][:, 0],vehicle_data['gt_trajectories'][6][:, 1], color='red', alpha=1, linewidth=1, zorder=15)
# for i in range(len(vehicle_data['forecasted_trajectories'][1])):
#     plt.plot(vehicle_data['forecasted_trajectories'][6][i][:, 0], vehicle_data['forecasted_trajectories'][6][i][:, 1], color='blue', alpha=1, linewidth=1, zorder=15)
# x = vehicle_data['predicted_targets'][6][:,0]
# y = vehicle_data['predicted_targets'][6][:,1]
# plt.scatter(x,y)
# plt.show()

# with open('/home/kyber/Desktop/unused/TNT/dataset/interm_data_younwoo_3/val_intermediate/raw/features_1.pkl', "rb") as f:
# with open('/home/kyber/Desktop/unused/TNT/dataset/interm_data_original/val_intermediate/raw/features_1.pkl', "rb") as f:
#     vehicle_data = pd.read_pickle(f)
# breakpoint()