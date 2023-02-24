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

# hyperparameters
LEARNING_RATE = 4e-4
EPOCHS = int(1e7)
BATCH_SIZE = 5
horizon = 8
WEIGHT_DECAY = 1e-4

# ground truth dataset
ground_truth_folder_path = 'data/ground_truth'          # path to the ground truth of every datapoints folder
ground_truth_file_list = os.listdir(ground_truth_folder_path)
ground_truth_file_list.sort()

# action input dataset
action_input_folder_path = 'data/action_input'
action_input_file_list = os.listdir(action_input_folder_path)
action_input_file_list.sort()

# image input dataset
img_file_path = 'data/image'
img_dataset = carla_rgb(img_file_path)
img_train_dataloader = DataLoader(img_dataset, batch_size=BATCH_SIZE)

datapoints = len(os.listdir(ground_truth_folder_path))
BATCH_NUM = int(datapoints/BATCH_SIZE)       

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
        nn.Linear(8960, 256),
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
        
    def forward(self, img_train_dataloader, ground_truth_files_list, action_input_files_list):
        
        model_output = torch.randn((BATCH_NUM*BATCH_SIZE, horizon, 4), device=device)
        
        for j in range(BATCH_NUM):
            
            img_raw_data, _ = next(iter(img_train_dataloader))  # [batch_size, 3, Height, Width]
            img_raw_data = img_raw_data.to(device).type(torch.float32)/255
            
            # str list for ground truth files and action input files
            ground_truth_files = ground_truth_files_list[j*BATCH_SIZE:(j+1)*BATCH_SIZE]           
            action_input_files = action_input_files_list[j*BATCH_SIZE:(j+1)*BATCH_SIZE]
            
            # img_raw_data = img_raw_data[0]
            # img = transforms.ToPILImage()(img_raw_data).convert('RGB')
            # plt.imshow(img)
            # plt.show()
            # plt.pause(0.001)
            
            action_input_data = torch.randn(BATCH_SIZE, horizon, 2, device=device)                           # 2 is linear and angular velocity
            for i in range(BATCH_SIZE):   
                action_input_temp_path = os.path.join(action_input_folder_path, action_input_files[i])       # path for action input file                
                action_input_temp = pd.read_csv(action_input_temp_path, header=None).values                  # a 2*8 ndarray, 1st row is linear velocity, 2nd row is angular velocity, column is timestep
                action_input_temp = np.transpose(action_input_temp)                                          # 8*2 ndarray
                action_input_temp = torch.from_numpy(action_input_temp).to(device).type(torch.float32)
                action_input_data[i, :, :] = action_input_temp
                
            ground_truth_data = torch.randn(BATCH_SIZE, horizon+1, 9, device=device)                         # 9 is [collision, location 3, velocity, yaw, angular velocity, latlong 2]
            for i in range(BATCH_SIZE):
                ground_truth_temp_path = os.path.join(ground_truth_folder_path, ground_truth_files[i])
                ground_truth_temp = pd.read_csv(ground_truth_temp_path, header=None).values
                ground_truth_temp = np.transpose(ground_truth_temp)                                          # transpose is necessary, because the pd read data in row instead of column, row is feature, column is timestep
                ground_truth_temp = torch.from_numpy(ground_truth_temp).to(device).type(torch.float32)
                ground_truth_data[i, :, :] = ground_truth_temp
                
            cnn_out1 = self.obs_im_model(img_raw_data)
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

model_output = model(img_train_dataloader, ground_truth_file_list, action_input_file_list)

ground_truth_position = torch.randn(BATCH_SIZE*BATCH_NUM, horizon, 3, device=device)
ground_truth_collision = torch.randn(BATCH_SIZE*BATCH_NUM, horizon, device=device)
for i in range(BATCH_NUM):
    for j in range(BATCH_SIZE):
        ground_truth_file_path = os.path.join(ground_truth_folder_path, ground_truth_file_list[i*BATCH_SIZE+j])
        ground_truth_temp = pd.read_csv(ground_truth_file_path, header=None).values
        ground_truth_temp = np.transpose(ground_truth_temp)
        ground_truth_temp = torch.from_numpy(ground_truth_temp).to(device).type(torch.float32)
        ground_truth_position_temp = ground_truth_temp[1:horizon+1, 1:4]
        ground_truth_collision_temp = ground_truth_temp[1:horizon+1, 0]
        ground_truth_position[j+BATCH_SIZE*i] = ground_truth_position_temp
        ground_truth_collision[j+BATCH_SIZE*i] = ground_truth_collision_temp

loss_mse = nn.MSELoss(reduction='sum')
loss_position = loss_mse(model_output[:, :, :3], ground_truth_position)
loss_position.retain_grad()
    
loss_cross_entropy = nn.CrossEntropyLoss(reduction='sum')        
loss_collision = loss_cross_entropy(model_output[:, :, 3], ground_truth_collision)
if loss_collision != 0:
    print('loss_collision', loss_collision)

loss = (loss_position + loss_collision)/BATCH_NUM
print('test dataset loss', loss.item())
print('approximate single axis position loss for one time stpe', math.pow(loss.item(), 0.5)/80)

temp = torch.sum((ground_truth_position-model_output[:,:,:3]),(1,2))

ax = plt.figure().add_subplot(projection='3d')

i=6
gx = ground_truth_position[i, :, 0].detach().cpu().numpy()
gy = ground_truth_position[i, :, 1].detach().cpu().numpy()
gz = ground_truth_position[i, :, 2].detach().cpu().numpy()
ax.plot(gx, gy, gz, zdir='z', label='path for ground truth')

mx = model_output[i, :, 0].detach().cpu().numpy()
my = model_output[i, :, 1].detach().cpu().numpy()
mz = model_output[i, :, 2].detach().cpu().numpy()
ax.plot(mx, my, mz, zdir='z', label='path for model output')

x_error = gx-mx
y_error = gy-my
z_error = gz-mz

fig, axs = plt.subplots(3)
fig.suptitle('Error in x, y, z direction')
axs[0].plot(x_error)
axs[0].set(xlabel='time step', ylabel='x error')
axs[1].plot(y_error)
axs[1].set(xlabel='time step', ylabel='y error')
axs[2].plot(z_error)
axs[2].set(xlabel='time step', ylabel='z error')


ax.legend()
# ax.set_xlim(0, 10)
# ax.set_ylim(0, 1)
# ax.set_zlim(0, -1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()