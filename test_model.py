# %%
from calendar import EPOCH
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import os
import pandas as pd
import math
import matplotlib.pyplot as plt
import random
import json
from model import combined_model
# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyperparameters
BATCH_SIZE = 1
horizon = 8

# to load the dataset from the json file
datapoint_path = '/home/lshi23/carla_test/data/datapoints'
files = []
for file in os.listdir(datapoint_path):
    files.append(os.path.join(datapoint_path, file))
files.sort()
# %%
# PATH = '/home/lshi23/mar14.pt'
model_path = '/home/lshi23/carla_test/saved_models/0.419257model.pt'

model = combined_model().to(device)
model.load_state_dict(torch.load(model_path))
model.eval()
# %%
model_errors = []
manual_errors = []
for f in files[:1]:
    with open(f, 'rb') as fj:
        data = json.load(fj)

    img_data = torch.tensor(data['image_input'], dtype=torch.float32).to(device)/255
    img_data.unsqueeze_(0)
    img_data = img_data.permute(0, 3, 1, 2)

    linear_velocity = torch.tensor(data['action_input']['linear_velocity'], dtype=torch.float32).to(device)
    steer = torch.tensor([i for i in data['action_input']['steer']], dtype=torch.float32).to(device)
    action_input = torch.stack([linear_velocity, steer], dim=1).to(device).unsqueeze_(0)
    ground_truth_collision_temp = np.array(data['ground_truth']['collision'])
    ground_truth_location_temp = np.array(data['ground_truth']['location'])
    ground_truth_location_x_temp = ground_truth_location_temp[:,0]
    ground_truth_location_y_temp = ground_truth_location_temp[:,1]
    ground_truth_location_z_temp = ground_truth_location_temp[:,2]
    ground_truth_velocity_temp = np.array(data['ground_truth']['velocity'])
    ground_truth_yaw_temp = np.array(data['ground_truth']['yaw'])
    ground_truth_steer_temp = np.array(data['ground_truth']['steer'])
    ground_truth_latlong_temp = np.array(data['ground_truth']['latlong'])
    ground_truth_lat_temp = ground_truth_latlong_temp[:,0]
    ground_truth_long_temp = ground_truth_latlong_temp[:,1]    
    ground_truth_reset_temp = np.array(data['ground_truth']['reset'])
    ground_truth_data_temp = np.stack((ground_truth_collision_temp, ground_truth_location_x_temp, ground_truth_location_y_temp, ground_truth_location_z_temp, 
                                        ground_truth_velocity_temp, ground_truth_yaw_temp, ground_truth_steer_temp, ground_truth_lat_temp, ground_truth_long_temp, 
                                        ground_truth_reset_temp), axis=1)
    ground_truth_data_temp = torch.FloatTensor(ground_truth_data_temp).to(device)
    ground_truth = ground_truth_data_temp.unsqueeze(0)

    # action_input = torch.rand((1, 8, 2)).to(device)
    # ground_truth = torch.rand((1, 9, 10)).to(device)
    # ground_truth[0, 0, 1] = ground_truth_location_x_temp[0]
    # ground_truth[0, 0, 2] = ground_truth_location_y_temp[0]
    # ground_truth[0, 0, 5] = ground_truth_yaw_temp[0]
    with torch.no_grad():
        model_output, local_pos = model(img_data, action_input, ground_truth, BATCH_SIZE, horizon)
    if np.any(ground_truth_reset_temp):
        model_error = -1
        manual_error = -1
    else:
        print(local_pos)
        model_error = np.linalg.norm(model_output[0, :, :2].detach().cpu().numpy() - ground_truth_location_temp[1:, :2])
        dis = np.arange(9) * 0.3
        pred_pos = ground_truth_location_temp[0, :2] + np.expand_dims(dis, axis=1) * np.array([np.cos(ground_truth_yaw_temp[0]), np.sin(ground_truth_yaw_temp[0])])
        manual_error = np.linalg.norm(ground_truth_location_temp[1:, :2] - pred_pos[1:, :])
    
    model_errors.append(model_error)
    manual_errors.append(manual_error)
    
# %%
model_errors = np.array(model_errors)
manual_errors = np.array(manual_errors)
print(f'model error: {np.mean(model_errors[model_errors > 0])}')
print(f'manual error: {np.mean(manual_errors[manual_errors > 0])}')
# %%
