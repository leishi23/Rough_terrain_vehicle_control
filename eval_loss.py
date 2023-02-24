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
BATCH_SIZE = 5
horizon = 8

# to load the dataset from the json file
datapoints_folder_path = 'data/datapoints'
dataset_split_path = os.path.join('/home/lshi23/carla_test/data', 'dataset_split.json')
with open(dataset_split_path, 'rb') as f: data = json.load(f)
training_datapoints_file_list = data['training']
test_datapoints_file_list = data['test']
training_BATCH_NUM = int(len(training_datapoints_file_list)/BATCH_SIZE)
test_BATCH_NUM = int(len(test_datapoints_file_list)/BATCH_SIZE)

PATH = '/home/lshi23/carla_test/combined_model.pt'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def rotmat(yaw):
    zeros = torch.tensor(0, dtype=torch.float32).to(device)
    ones = torch.tensor(1, dtype=torch.float32).to(device)
    temp = torch.stack([torch.cos(yaw), -torch.sin(yaw), zeros, torch.sin(yaw), torch.cos(yaw), 
                        zeros, zeros, zeros, ones])
    return torch.reshape(temp, (3, 3))

class combined_model(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.obs_im_model = nn.Sequential(
        nn.Conv2d(3, 32, 5, stride=2),
        nn.ReLU(),
        nn.Conv2d(32, 64, 3, stride=2),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3, stride=2),
        nn.Flatten(start_dim=1),
        nn.Linear(42240, 256),
        nn.ReLU(),
        nn.Linear(256, 128) 
        )
        
        self.obs_lowd_model = nn.Sequential(
        nn.Linear(128, 128),       
        nn.ReLU(),
        nn.Linear(128, 128)
        )
        
        self.action_input_model = nn.Sequential(
        nn.Linear(2, 16),           # verified input (8,2), output(8,16) when nn.Linear(2,16)
        nn.ReLU(),
        nn.Linear(16, 16)
        )
        
        self.rnn_cell = nn.LSTM(16, 64, 8, batch_first = True)             # (input_size, hidden_size/num_units, num_layers)
        
        self.output_model = nn.Sequential(
        nn.Linear(64, 32),       # hidden layer features are 64
        nn.ReLU(),
        nn.Linear(32, 4)       # 4 is the output dimension, actually 8*4   
        )
        
    def forward(self, datapoints_file_list, BATCH_NUM):
        
        model_output = torch.randn((BATCH_NUM*BATCH_SIZE, horizon, 4), device=device)
        
        for j in range(BATCH_NUM):
            
            action_input_data = torch.randn(BATCH_SIZE, horizon, 2, device=device)                           # 2 is linear and angular velocity
            ground_truth_data = torch.randn(BATCH_SIZE, horizon+1, 10, device=device)                        # 10 is [collision, location 3, velocity, yaw, angular velocity, latlong 2, reset]
            image_data = torch.randn(BATCH_SIZE, 3, 192, 256, device=device)                                 # 3 is RGB channels, 192 is height, 256 is width
            
            for i in range(BATCH_SIZE):
                
                datapoint_path = os.path.join(datapoints_folder_path, datapoints_file_list[j*BATCH_SIZE+i])
                with open(datapoint_path, 'rb') as f: data = json.load(f)
                
                image_data_temp = torch.FloatTensor(data['image_input']).to(device)/255                          # [Height, Width, 3(C:channels)]
                image_data_temp = image_data_temp.permute(2, 0, 1)                                               # [3, Height, Width]
                image_data[i] = image_data_temp
            
                # img_raw_data, _ = next(iter(img_train_dataloader))  # [batch_size, 3, Height, Width]
                # img_raw_data = img_raw_data.to(device).type(torch.float32)/255
                
                # print(datapoint_path)trainer.py
                # img = transforms.ToPILImage()(image_data[i]).convert('RGB')
                # plt.imshow(img)
                # plt.show()
                # plt.pause(0.001)            
                  
                linear_velocity_temp = torch.FloatTensor(data['action_input']['linear_velocity']).to(device)
                steer_temp = torch.FloatTensor(data['action_input']['steer']).to(device)
                action_input_data_temp = torch.stack([linear_velocity_temp, steer_temp], dim=1)
                action_input_data[i] = action_input_data_temp
                
                ground_truth_collision_temp = torch.FloatTensor(data['ground_truth']['collision']).to(device)
                ground_truth_location_temp = torch.FloatTensor(data['ground_truth']['location']).to(device)
                ground_truth_location_x_temp = ground_truth_location_temp[:,0]
                ground_truth_location_y_temp = ground_truth_location_temp[:,1]
                ground_truth_location_z_temp = ground_truth_location_temp[:,2]
                ground_truth_velocity_temp = torch.FloatTensor(data['ground_truth']['velocity']).to(device)
                ground_truth_yaw_temp = torch.FloatTensor(data['ground_truth']['yaw']).to(device)
                ground_truth_steer_temp = torch.FloatTensor(data['ground_truth']['steer']).to(device)
                ground_truth_latlong_temp = torch.FloatTensor(data['ground_truth']['latlong']).to(device)
                ground_truth_lat_temp = ground_truth_latlong_temp[:,0]
                ground_truth_long_temp = ground_truth_latlong_temp[:,1]    
                ground_truth_reset_temp = torch.FloatTensor(data['ground_truth']['reset']).to(device)
                ground_truth_data_temp = torch.stack([ground_truth_collision_temp, ground_truth_location_x_temp, ground_truth_location_y_temp, ground_truth_location_z_temp,
                                                      ground_truth_velocity_temp, ground_truth_yaw_temp, ground_truth_steer_temp, ground_truth_lat_temp, ground_truth_long_temp, ground_truth_reset_temp], dim=1)
                ground_truth_data[i] = ground_truth_data_temp
                
                
            cnn_out1 = self.obs_im_model(image_data)
            cnn_out2 = self.obs_lowd_model(cnn_out1)
            action_input_processed = self.action_input_model(action_input_data)
            
            initial_state_c_temp, initial_state_h_temp = torch.split(cnn_out2, 64, dim=-1)
            initial_state_c = torch.randn(horizon, BATCH_SIZE, 64, device=device).copy_(initial_state_c_temp)
            initial_state_h = torch.randn(horizon, BATCH_SIZE, 64, device=device).copy_(initial_state_h_temp)
            
            # initial_state_c = torch.zeros(horizon, BATCH_SIZE, 64, device=device)
            # initial_state_c[0] = initial_state_c_temp
            # initial_state_h = torch.zeros(horizon, BATCH_SIZE, 64, device=device)
            # initial_state_h[0] = initial_state_h_temp       
            
            lstm_out, _ = self.rnn_cell(action_input_processed, (initial_state_h, initial_state_c))
            
            model_output_temp = self.output_model(lstm_out)           # [position x, position y, position z, collision], diff from ground_truth              
                    
            for i in range(BATCH_SIZE):
                yaw_data = ground_truth_data[i, 0, 5]      # yaw is from CARLA vehicle transform frame where clockwise is positive. In rotation matrix, yaw is from world frame where counter-clockwise is positive. When reverse the yaw, it's still counter-clockwise.
                rotmatrix = rotmat(yaw_data)
                model_output_temp_local_position = model_output_temp[i, :, :3]
                model_output_temp_global_position = torch.matmul(model_output_temp_local_position, rotmatrix)
                model_output[i+j*BATCH_SIZE, :,:3] = model_output_temp_global_position + ground_truth_data[i, 0, 1:4]     # add the initial position for timestep 0 to the output position 
                model_output[i+j*BATCH_SIZE, :, 3] = model_output_temp[i, :, 3]
            
        return model_output     # [batch_size, horizon+1, 4], horizon+1 is timestep, 4 is [position x, position y, position z, collision]

model = combined_model().to(device)
model.load_state_dict(torch.load(PATH))
model.eval()

model_output = model(test_datapoints_file_list, test_BATCH_NUM)

ground_truth_position = torch.randn(BATCH_SIZE*test_BATCH_NUM, horizon, 3, device=device)
ground_truth_collision = torch.randn(BATCH_SIZE*test_BATCH_NUM, horizon, device=device)
ground_truth_reset = torch.randn(BATCH_SIZE*test_BATCH_NUM, horizon, device=device)
for i in range(test_BATCH_NUM):
    for j in range(BATCH_SIZE):
        datapoint_path = os.path.join(datapoints_folder_path, test_datapoints_file_list[i*BATCH_SIZE+j])
        with open(datapoint_path, 'rb') as f: data = json.load(f)
        ground_truth_position_temp = data['ground_truth']['location'][1:horizon+1]          # shape: [horizon, 3]
        ground_truth_collision_temp = data['ground_truth']['collision'][1:horizon+1]        # shape: [horizon]
        ground_truth_reset_temp = data['ground_truth']['reset'][1:horizon+1]                # shape: [horizon]
        ground_truth_position[j+BATCH_SIZE*i] = torch.FloatTensor(ground_truth_position_temp).to(device)
        ground_truth_collision[j+BATCH_SIZE*i] = torch.FloatTensor(ground_truth_collision_temp).to(device)
        ground_truth_reset[j+BATCH_SIZE*i] = torch.FloatTensor(ground_truth_reset_temp).to(device)

ground_truth_reset_idx = torch.nonzero(ground_truth_reset, as_tuple=True)[0]
loss_position_pre = 0.5 * torch.square(model_output[:, :, :3] - ground_truth_position)
loss_position_pre = torch.sum(loss_position_pre, dim=(1, 2))
for idx in ground_truth_reset_idx:
    loss_position_pre[idx] = 0
    ground_truth_position[idx] = torch.zeros(horizon, 3, device=device)
    model_output[idx] = torch.zeros(horizon, 4, device=device)
loss_position = loss_position_pre.sum()
    
loss_cross_entropy = nn.CrossEntropyLoss(reduction='sum')        
loss_collision = loss_cross_entropy(model_output[:, :, 3], ground_truth_collision)
if loss_collision != 0:
    print('loss_collision', loss_collision)

loss = (loss_position + loss_collision)/(test_BATCH_NUM*BATCH_SIZE)
print('one datapoint loss is', loss.item())
print('approximate single axis position loss for one time stpe is', math.pow(loss.item(), 0.5)/80)

temp = torch.sum((ground_truth_position-model_output[:,:,:3]),(1,2))

# ax = plt.figure().add_subplot(projection='3d',)
ax = plt.figure().add_subplot()
ax.set_xlabel('m')

gx = np.array([])
gy = np.array([])
gz = np.array([])
for i in range(BATCH_SIZE*test_BATCH_NUM):
    gx = np.concatenate((gx, ground_truth_position[i, :, 0].detach().cpu().numpy()))
    gy = np.concatenate((gy, ground_truth_position[i, :, 1].detach().cpu().numpy()))
    gz = np.concatenate((gz, ground_truth_position[i, :, 2].detach().cpu().numpy()))
# ax.plot(gx, gy, gz, zdir='z', label='path for ground truth')
ax.plot(gx, gy, 'r*', label='path for ground truth')

mx = np.array([])
my = np.array([])
mz = np.array([])
for i in range(BATCH_SIZE*test_BATCH_NUM):
    mx = np.concatenate((mx, model_output[i, :, 0].detach().cpu().numpy()))
    my = np.concatenate((my, model_output[i, :, 1].detach().cpu().numpy()))
    mz = np.concatenate((mz, model_output[i, :, 2].detach().cpu().numpy()))
# ax.plot(mx, my, mz, zdir='z', label='path for model output')
ax.plot(mx, my, 'g*', label='path for model output')

ax.set_xlabel('x (m)', fontsize=10)
ax.set_ylabel('y (m)', fontsize=10)
ax.axis('equal')
# ax.set_zlabel('z (m)', fontsize=10)
ax.title.set_text('Test dataset whole path')

x_error = np.add.accumulate(abs(gx-mx))
y_error = np.add.accumulate(abs(gy-my))
z_error = np.add.accumulate(abs(gz-mz))

time_horizon = np.linspace(0.0, 1.2*BATCH_SIZE, num=horizon*BATCH_SIZE*test_BATCH_NUM)
figs, axs = plt.subplots(3)
figs.suptitle('Error in x, y, z direction')
axs[0].plot(time_horizon, x_error)
axs[0].set(xlabel='time step (s)', ylabel='x accumulated error (m)')
axs[1].plot(time_horizon, y_error)
axs[1].set(xlabel='time step (s)', ylabel='y accumulated error (m)')
axs[2].plot(time_horizon, z_error)
axs[2].set(xlabel='time step (s)', ylabel='z accumulated error (m)')

ax.legend()
ax.set_xlabel('X')
ax.set_ylabel('Y')
# ax.set_zlabel('Z')
plt.show()