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

# hyperparameters
BATCH_SIZE = 32
horizon = 8

# to load the dataset from the json file
datapoints_folder_path = 'data/datapoints'
# datapoints_folder_path = '/home/lshi23'
dataset_split_path = os.path.join('/home/lshi23/carla_test/data', 'dataset_split.json')
with open(dataset_split_path, 'rb') as f: data = json.load(f)
test_datapoints_file_list = data['test']
# test_datapoints_file_list = ['000014f.json']
test_BATCH_NUM = int(len(test_datapoints_file_list)/BATCH_SIZE)
# test_BATCH_NUM = 1

PATH = '/home/lshi23/carla_test/combined_model.pt'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = combined_model().to(device)
model.load_state_dict(torch.load(PATH))
model.eval()
loss_epoch = 0

gx = np.array([])
gy = np.array([])
gz = np.array([])

mx = np.array([])
my = np.array([])
mz = np.array([])

for j in range(test_BATCH_NUM):
        
        action_input_data = torch.randn(BATCH_SIZE, horizon, 2, device=device)                           # 2 is linear and angular velocity
        ground_truth_data = torch.randn(BATCH_SIZE, horizon+1, 10, device=device)                        # 10 is [collision, location 3, velocity, yaw, angular velocity, latlong 2, reset]
        image_data = torch.randn(BATCH_SIZE, 3, 96*2, 128*2, device=device)                                 # 3 is RGB channels, 192 is height, 256 is width
        
        for i in range(BATCH_SIZE):
            # load the data
            datapoint_path = os.path.join(datapoints_folder_path, test_datapoints_file_list[j*BATCH_SIZE+i])
            with open(datapoint_path, 'rb') as f: data = json.load(f)
            
            image_data_temp = torch.FloatTensor(data['image_input']).to(device)/255                          # [Height, Width, 3(C:channels)]
            image_data_temp = image_data_temp.permute(2, 0, 1)                                               # [3, Height, Width]
            image_data[i] = image_data_temp
            
            # print(datapoint_path)trainer.py
            # img = transforms.ToPILImage()(image_data[i]).convert('RGB')
            # plt.imshow(img)
            # plt.show()
            # plt.pause(0.001)            
                
            linear_velocity_temp = np.array(data['action_input']['linear_velocity'])
            steer_temp = np.array(data['action_input']['steer'])
            action_input_data_temp = np.stack((linear_velocity_temp, steer_temp), axis=1)
            action_input_data[i] = torch.FloatTensor(action_input_data_temp).to(device)
            
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
            ground_truth_data[i] = ground_truth_data_temp
            
        with torch.no_grad():
            model_output = model(image_data, action_input_data, ground_truth_data, BATCH_SIZE, horizon)
        
        ground_truth_position = torch.randn(BATCH_SIZE, horizon, 3, device=device)
        ground_truth_collision = torch.randn(BATCH_SIZE, horizon, device=device)
        ground_truth_reset = torch.randn(BATCH_SIZE, horizon, device=device)        
        for i in range(BATCH_SIZE):
            # load ground truth data
            datapoint_path = os.path.join(datapoints_folder_path, test_datapoints_file_list[j*BATCH_SIZE+i])
            with open(datapoint_path, 'rb') as f: data = json.load(f)
            ground_truth_position_temp = data['ground_truth']['location'][1:horizon+1]          # shape: [horizon, 3]
            ground_truth_collision_temp = data['ground_truth']['collision'][1:horizon+1]        # shape: [horizon]
            ground_truth_reset_temp = data['ground_truth']['reset'][1:horizon+1]                # shape: [horizon]
            ground_truth_position[i] = torch.FloatTensor(ground_truth_position_temp).to(device)
            ground_truth_collision[i] = torch.FloatTensor(ground_truth_collision_temp).to(device)
            ground_truth_reset[i] = torch.FloatTensor(ground_truth_reset_temp).to(device)
            
            
        ground_truth_reset_idx = torch.nonzero(ground_truth_reset, as_tuple=True)[0]
        loss_position_pre = 0.5 * torch.square(model_output[:, :, :3] - ground_truth_position)
        loss_position_pre = torch.sum(loss_position_pre, dim=(1, 2))
        for idx in ground_truth_reset_idx:
            loss_position_pre[idx] = 0
        loss_position = loss_position_pre.sum()
        
        loss_cross_entropy = nn.CrossEntropyLoss(reduction='none')
        loss_collision = loss_cross_entropy(model_output[:, :, 3], ground_truth_collision)
        
        for idx in ground_truth_reset_idx:
            loss_collision[idx] = 0
            ground_truth_position[idx] = torch.zeros(horizon, 3, device=device)
            model_output[idx] = torch.zeros(horizon, 4, device=device)
        loss_collision = loss_collision.sum()
        
        for i in range(BATCH_SIZE):
            gx = np.concatenate((gx, ground_truth_position[i, :, 0].detach().cpu().numpy()))
            gy = np.concatenate((gy, ground_truth_position[i, :, 1].detach().cpu().numpy()))
            gz = np.concatenate((gz, ground_truth_position[i, :, 2].detach().cpu().numpy()))
            
            mx = np.concatenate((mx, model_output[i, :, 0].detach().cpu().numpy()))
            my = np.concatenate((my, model_output[i, :, 1].detach().cpu().numpy()))
            mz = np.concatenate((mz, model_output[i, :, 2].detach().cpu().numpy()))
        
        loss = (loss_position + loss_collision)
        loss_epoch += loss.item()
        print(' Batch:', j,'test loss:', float(loss.item())/BATCH_SIZE)

loss_epoch = loss_epoch/(BATCH_SIZE*test_BATCH_NUM)
print('one datapoint loss is', loss_epoch)
print('approximate single axis position loss for one time stpe is', math.pow(loss_epoch, 0.5)/80)

temp = torch.sum((ground_truth_position-model_output[:,:,:3]),(1,2))

# ax = plt.figure().add_subplot(projection='3d',)
ax = plt.figure().add_subplot()
ax.set_xlabel('m')

# ax.plot(gx, gy, gz, zdir='z', label='path for ground truth')
ax.plot(gx, gy, 'r*', label='path for ground truth')

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