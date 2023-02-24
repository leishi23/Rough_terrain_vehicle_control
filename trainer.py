from calendar import EPOCH
import numpy as np
import matplotlib.image as mpimg
import torch
from torch import nn, tensor
from torch.utils.data import DataLoader
from dataset import carla_rgb
import matplotlib.pyplot as plt
from torchvision import transforms
import os
import queue
import pandas as pd
import random
import json

# according to the nn.LSTM documentation, I need to set an environment variable here
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:2"

# hyperparameters
LEARNING_RATE = 1e-4
EPOCHS = int(1e7)
BATCH_SIZE = 5
horizon = 8
WEIGHT_DECAY = 1e-8

# to load the dataset from the json file
datapoints_folder_path = 'data/datapoints'
dataset_split_path = os.path.join('/home/lshi23/carla_test/data', 'dataset_split.json')
with open(dataset_split_path, 'rb') as f: data = json.load(f)
training_datapoints_file_list = data['training']
test_datapoints_file_list = data['test']
training_BATCH_NUM = int(len(training_datapoints_file_list)/BATCH_SIZE)
test_BATCH_NUM = int(len(test_datapoints_file_list)/BATCH_SIZE)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

PATH = '/home/lshi23/carla_test/combined_model.pt'

def loss_plot(data, loss_queue):
    loss_queue.put(data)
    
def rotmat(yaw):
    zeros = torch.tensor(0, dtype=torch.float32).to(device)
    ones = torch.tensor(1, dtype=torch.float32).to(device)
    temp = torch.stack([torch.cos(yaw), -torch.sin(yaw), zeros, torch.sin(yaw), torch.cos(yaw), 
                        zeros, zeros, zeros, ones])
    return torch.reshape(temp, (3, 3))

loss_queue = queue.Queue(maxsize=1)
test_loss_queue = queue.Queue(maxsize=1)

plt.ion()
fig, ax = plt.subplots(1, 2)
ax[0].set_title('Train  Loss Plot')
ax[0].set_ylabel('Training Loss')
ax[0].set_xlabel('Epochs (x10)')
ax[1].set_title('Test Loss Plot')
ax[1].set_ylabel('Test Loss')
ax[1].set_xlabel('Epochs (x10)')
loss_plot_list = list()
test_loss_plot_list = list()

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

# Adam optimizer (L2 regularization)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
loss_max = 0.0

for step in range(EPOCHS):
            
    model_output = model(training_datapoints_file_list, training_BATCH_NUM)
    
    # Loss function: MSE for position, cross entropy for collision
    # To get the ground truth data first for the model output
    
    ground_truth_position = torch.randn(BATCH_SIZE*training_BATCH_NUM, horizon, 3, device=device)
    ground_truth_collision = torch.randn(BATCH_SIZE*training_BATCH_NUM, horizon, device=device)
    ground_truth_reset = torch.randn(BATCH_SIZE*training_BATCH_NUM, horizon, device=device)
    for i in range(training_BATCH_NUM):
        for j in range(BATCH_SIZE):
            datapoint_path = os.path.join(datapoints_folder_path, training_datapoints_file_list[i*BATCH_SIZE+j])
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
    loss_position = loss_position_pre.sum()
     
    loss_cross_entropy = nn.CrossEntropyLoss(reduction='none')
    loss_collision = loss_cross_entropy(model_output[:, :, 3], ground_truth_collision)
    for idx in ground_truth_reset_idx:
        loss_collision[idx] = 0
    loss_collision = loss_collision.sum()
    
    loss = (loss_position + loss_collision)/(training_BATCH_NUM*BATCH_SIZE)
    loss_max = max(loss_max, loss)
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    
    optimizer.step()
    
    if step % 10 == 0 :
        print('Epoch: ', step, 'loss: ', loss.item())
        loss_plot(loss.item(), loss_queue)
        loss_data = loss_queue.get()
        loss_plot_list.append(loss_data)
        ax[0].plot(loss_plot_list, color='blue')
        plt.show()
        plt.tight_layout()
        plt.pause(0.001)
        
        print('save model at step', step)
        torch.save(model.state_dict(), '/home/lshi23/carla_test/combined_model.pt')
       
        # test the model
        with torch.no_grad():
            model.eval()
            test_model_output = model(test_datapoints_file_list, test_BATCH_NUM)
        model.train()
        
        test_ground_truth_position = torch.randn(BATCH_SIZE*test_BATCH_NUM, horizon, 3, device=device)
        test_ground_truth_collision = torch.randn(BATCH_SIZE*test_BATCH_NUM, horizon, device=device)
        test_ground_truth_reset = torch.randn(BATCH_SIZE*test_BATCH_NUM, horizon, device=device)
        for i in range(test_BATCH_NUM):
            for j in range(BATCH_SIZE):
                datapoint_path = os.path.join(datapoints_folder_path, test_datapoints_file_list[i*BATCH_SIZE+j])
                with open(datapoint_path, 'rb') as f: data = json.load(f)
                test_ground_truth_position_temp = data['ground_truth']['location'][1:horizon+1]          # shape: [horizon, 3]
                test_ground_truth_collision_temp = data['ground_truth']['collision'][1:horizon+1]        # shape: [horizon]
                test_ground_truth_reset_temp = data['ground_truth']['reset'][1:horizon+1]                # shape: [horizon]
                test_ground_truth_position[j+BATCH_SIZE*i] = torch.FloatTensor(test_ground_truth_position_temp).to(device)
                test_ground_truth_collision[j+BATCH_SIZE*i] = torch.FloatTensor(test_ground_truth_collision_temp).to(device)
                test_ground_truth_reset[j+BATCH_SIZE*i] = torch.FloatTensor(test_ground_truth_reset_temp).to(device)
        
        test_ground_truth_reset_idx = torch.nonzero(test_ground_truth_reset, as_tuple=True)[0]
        test_loss_position_pre = 0.5 * torch.square(test_model_output[:, :, :3] - test_ground_truth_position)
        test_loss_position_pre = torch.sum(test_loss_position_pre, dim=(1, 2))
        for idx in test_ground_truth_reset_idx:
            test_loss_position_pre[idx] = 0
        test_loss_position = test_loss_position_pre.sum()
        
        if step > 3000:
            print('test loss is', test_loss_position.item())
        
        test_loss_cross_entropy = nn.CrossEntropyLoss(reduction='sum')        
        test_loss_collision = test_loss_cross_entropy(test_model_output[:, :, 3], test_ground_truth_collision)

        test_loss = (test_loss_position + test_loss_collision)/(test_BATCH_NUM*BATCH_SIZE)
        loss_plot(test_loss.item(), test_loss_queue)
        test_loss_data = test_loss_queue.get()
        test_loss_plot_list.append(test_loss_data)
        ax[1].plot(test_loss_plot_list, color='blue')
        plt.show()
        plt.tight_layout()
        plt.pause(0.001)
        

    if step > 1000 and loss < 0.15:
        print('pause training, loss is', loss)
    
    if loss < loss_max*0.00001 or loss < 0.05 or step>1e5:
        print('save model, break')
        torch.save(model.state_dict(), '/home/lshi23/carla_test/combined_model.pt')
        break