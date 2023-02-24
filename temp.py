# from calendar import EPOCH
# import torch
# from torch import nn, tensor
# import os
# from torchviz import make_dot

# rnn = nn.LSTM(2, 1, 1)  # 2 input, 4 hidden, 1 layer
# rnn.bias = False
# input = tensor([[1, 2]], dtype=torch.float32)
# # h0 = tensor([[0, 0, 0, 0]], dtype=torch.float32)
# # c0 = tensor([[0, 0, 0, 0]], dtype=torch.float32)
# output, (hn, cn) = rnn(input)
# print(output)

import os 
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
action_input_folder_path = 'data/action_input'
action_input_file_list = os.listdir(action_input_folder_path)
action_input_file_list.sort() 

ground_truth_folder_path = 'data/ground_truth'
ground_truth_file_list = os.listdir(ground_truth_folder_path)
ground_truth_file_list.sort()

fordward_velocity = []
steer = []
position_x = []
position_y = []
predict_x = []
predict_y = []
error_x = []
error_y = []

# for i in range(len(action_input_file_list)):
# for i in range(0,5):
#     action_input_files = action_input_file_list[i]
#     action_input_temp_path = os.path.join(action_input_folder_path, action_input_files)       # path for action input file                
#     action_input_temp = pd.read_csv(action_input_temp_path, header=None).values                  # a 2*8 ndarray, 1st row is linear velocity, 2nd row is angular velocity, column is timestep
#     ground_truth_files = ground_truth_file_list[i]
#     ground_truth_temp_path = os.path.join(ground_truth_folder_path, ground_truth_files)       # path for action input file
#     ground_truth_temp = pd.read_csv(ground_truth_temp_path, header=None).values   
#     for j in range(8):
#         fordward_velocity.append(action_input_temp[0][j])
#         steer.append(action_input_temp[1][j])
#         position_x.append(ground_truth_temp[1][j])
#         position_y.append(ground_truth_temp[2][j])
    
# fig, ax = plt.subplots(2, 2)
# ax[0, 0].plot(fordward_velocity)
# ax[0, 1].plot(steer)
# ax[1, 0].plot(position_x)
# ax[1, 1].plot(position_y)

# ax[0, 0].set(xlabel='timestep', ylabel='m/s')
# ax[0, 1].set(xlabel='timestep', ylabel='not sure, no influence')
# ax[1, 0].set(xlabel='timestep', ylabel='m')
# ax[1, 1].set(xlabel='timestep', ylabel='m')
# ax[0, 0].set_title('Fordward Velocity')
# ax[0, 1].set_title('Steer')
# ax[1, 0].set_title('Position X')
# ax[1, 1].set_title('Position Y')

# plt.show()


i = 0
action_input_files = action_input_file_list[i]
action_input_temp_path = os.path.join(action_input_folder_path, action_input_files)       # path for action input file                
action_input_temp = pd.read_csv(action_input_temp_path, header=None).values                  # a 2*8 ndarray, 1st row is linear velocity, 2nd row is angular velocity, column is timestep
ground_truth_files = ground_truth_file_list[i]
ground_truth_temp_path = os.path.join(ground_truth_folder_path, ground_truth_files)       # path for action input file
ground_truth_temp = pd.read_csv(ground_truth_temp_path, header=None).values   
for j in range(9):
    position_x.append(ground_truth_temp[1][j])
    position_y.append(ground_truth_temp[2][j])
    
    if j == 0:
        predict_x.append(ground_truth_temp[1][0])
        predict_y.append(ground_truth_temp[2][0])
    else:    
        position_x_pre = predict_x[-1]
        position_y_pre = predict_y[-1]
        yaw = ground_truth_temp[5][j-1]
        fordward_velocity = action_input_temp[0][j-1]
        velocity_x = fordward_velocity * (np.cos(yaw))
        velocity_y = fordward_velocity * (np.sin(yaw))
        predict_x.append(position_x_pre + velocity_x * 0.15)
        predict_y.append(position_y_pre + velocity_y * 0.15)

error_x = np.array(position_x) - np.array(predict_x)
error_y = np.array(position_y) - np.array(predict_y)
fig, ax = plt.subplots(3, 2)
ax[0,0].plot(predict_x, label='predict')
ax[0,1].plot(predict_y, label='predict')
ax[0,0].plot(position_x, label='ground truth')
ax[0,1].plot(position_y, label='ground truth')
ax[0,0].set(xlabel='timestep', ylabel='m')
ax[0,1].set(xlabel='timestep', ylabel='m')
ax[0,0].set_title('X')
ax[0,1].set_title('Y')
ax[0,0].legend()
ax[0,1].legend()

ax[1,0].plot(error_x, label='error')
ax[1,1].plot(error_y, label='error')
ax[1,0].set(xlabel='timestep', ylabel='m')
ax[1,1].set(xlabel='timestep', ylabel='m')
ax[1,0].set_title('Error X')
ax[1,1].set_title('Error Y')
ax[1,0].legend()
ax[1,1].legend()

ax[2,0].plot(position_x, position_y, 'ro', label='ground truth')
ax[2,0].plot(predict_x, predict_y, 'g^', label='predict')
ax[2,0].set(xlabel='X', ylabel='Y')
ax[2,0].set_title('X-Y')
ax[2,0].legend()
ax[2,0].axis('equal')
ax[2,1].remove()

# plt.legend()
plt.tight_layout()
plt.show()