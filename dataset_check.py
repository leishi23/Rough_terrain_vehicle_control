from calendar import EPOCH
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from dataset import carla_rgb
import os
import pandas as pd
import math
import matplotlib.pyplot as plt
import random
import json

# hyperparameters
horizon = 8

# to load the dataset from the json file
datapoints_folder_path = '/home/lshi23/carla_test/data/raw_data'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
datapoints_files_list = os.listdir(datapoints_folder_path)
datapoint_num = len(datapoints_files_list)
print('Number of datapoints: ', datapoint_num)
ground_truth_position = torch.randn(datapoint_num, 3, device=device)  
angular_velocity = torch.randn(datapoint_num, device=device) 

for i in range(datapoint_num):
    datapoint_path = os.path.join(datapoints_folder_path, datapoints_files_list[i])
    with open(datapoint_path, 'rb') as f: data = json.load(f)
    ground_truth_position_temp = data[1:4]
    angular_velocity_temp = data[6]
    ground_truth_position[i] = torch.FloatTensor(ground_truth_position_temp).to(device)
    angular_velocity[i] = torch.FloatTensor([angular_velocity_temp]).to(device)


# ax = plt.figure().add_subplot(projection='3d',)
ax = plt.figure().add_subplot()
ax.set_xlabel('m')

gx = ground_truth_position[:, 0].detach().cpu().numpy()
gy = -ground_truth_position[:, 1].detach().cpu().numpy()
gz = ground_truth_position[:, 2].detach().cpu().numpy()

ax.plot(gx, gy, 'r*', label='path for ground truth')
ax.set_xlabel('x (m)', fontsize=10)
ax.set_ylabel('y (m)', fontsize=10)
ax.axis('equal')
# ax.set_zlabel('z (m)', fontsize=10)
ax.title.set_text('Test dataset whole path')

ax.legend()
ax.set_xlabel('X')
ax.set_ylabel('Y')
# ax.set_zlabel('Z')

ax_w = plt.figure().add_subplot()
ax_w.set_xlabel('time frame')
ax_w.set_ylabel('angular velocity (rad/s)')

ax_w.plot(angular_velocity[:500].detach().cpu().numpy(), 'b*', label='angular velocity')
ax_w.title.set_text('Angular velocity for the whole dataset')

plt.show()