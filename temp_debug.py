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
from PIL import Image
from torchvision import transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyperparameters
BATCH_SIZE = 1
horizon = 8

# %%
# to load the dataset from the json file
datapoint_path = '/home/lshi23/carla_test/data/datapoints/000023f.json'
with open(datapoint_path, 'rb') as f:
    data = json.load(f)

image_temp_path = data['image_input']
image_data_temp = Image.open(image_temp_path)
image_tensor = transforms.ToTensor()(image_data_temp).to(device).unsqueeze_(0)

linear_velocity = torch.tensor(data['action_input']['linear_velocity'], dtype=torch.float32).to(device)
steer = torch.tensor([-i for i in data['action_input']['steer']], dtype=torch.float32).to(device)
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

# ground_truth[0, :, 1:4] = torch.rand(1, 9, 3)
# ground_truth[0, :, 5] = torch.rand(1, 9)
# %%
# PATH = '/home/lshi23/mar14.pt'
PATH = '/home/lshi23/carla_test/saved_models/0.647355model.pt'

model = combined_model().to(device)
model.load_state_dict(torch.load(PATH))
model.eval()
     
with torch.no_grad():
    model_output = model(image_tensor, action_input, ground_truth, BATCH_SIZE, horizon)
    
print("model output is: ", model_output[:, :, :2])
print("linear velocity is: ", linear_velocity)
print("steer is: ", steer)
# %%
