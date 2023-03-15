from calendar import EPOCH
import numpy as np
import matplotlib.image as mpimg
import torch
from torch import nn, tensor
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision import transforms
import os
import queue
import pandas as pd
import random
import json
from model import combined_model

# according to the nn.LSTM documentation, I need to set an environment variable here
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:2"
# PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:200

# hyperparameters
LEARNING_RATE = 3e-4
EPOCHS = int(1e7)
BATCH_SIZE = 64
horizon = 8
WEIGHT_DECAY = 1e-3

try:
    # to delete the rgb images folder
    md_location = "/home/lshi23/carla_test/saved_models"
    for file in os.listdir(md_location):
        os.remove(os.path.join(md_location, file))

finally:
    print("saved models folder is empty now")

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
    
loss_queue = queue.Queue(maxsize=1)
test_loss_queue = queue.Queue(maxsize=1)

plt.ion()
fig, ax = plt.subplots(1, 2)
ax[0].set_title('Train  Loss Plot')
ax[0].set_ylabel('Training Loss')
ax[0].set_xlabel('Epoch')
ax[1].set_title('Test Loss Plot')
ax[1].set_ylabel('Test Loss')
ax[1].set_xlabel('Epoch')
loss_plot_list = list()
test_loss_plot_list = list()

model = combined_model().to(device)

# Adam optimizer (L2 regularization)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

for step in range(EPOCHS):     
            
    loss_epoch = 0
    for j in range(training_BATCH_NUM):
        
        action_input_data = torch.randn(BATCH_SIZE, horizon, 2, device=device)                           # 2 is linear and angular velocity
        ground_truth_data = torch.randn(BATCH_SIZE, horizon+1, 10, device=device)                        # 10 is [collision, location 3, velocity, yaw, angular velocity, latlong 2, reset]
        image_data = torch.randn(BATCH_SIZE, 3, 96*2, 128*2, device=device)                                 # 3 is RGB channels, 192 is height, 256 is width
        
        for i in range(BATCH_SIZE):
            # load the data
            datapoint_path = os.path.join(datapoints_folder_path, training_datapoints_file_list[j*BATCH_SIZE+i])
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
        
        model_output = model(image_data, action_input_data, ground_truth_data, BATCH_SIZE, horizon)
        
        ground_truth_position = torch.randn(BATCH_SIZE, horizon, 3, device=device)
        ground_truth_collision = torch.randn(BATCH_SIZE, horizon, device=device)
        ground_truth_reset = torch.randn(BATCH_SIZE, horizon, device=device)        
        for i in range(BATCH_SIZE):
            # load ground truth data
            datapoint_path = os.path.join(datapoints_folder_path, training_datapoints_file_list[j*BATCH_SIZE+i])
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
        loss_collision = loss_collision.sum()
        
        loss = (loss_position + loss_collision)
        optimizer.zero_grad()
        loss.backward()
        loss_epoch += float(loss.item())
        
        optimizer.step()

        print('Epoch:', step, ' Batch:', j,' training loss:', float(loss.item())/BATCH_SIZE)
    
    loss_epoch = loss_epoch / (training_BATCH_NUM*BATCH_SIZE)
    loss_plot(loss_epoch, loss_queue )
    loss_data = loss_queue.get()
    loss_plot_list.append(loss_data)
    ax[0].plot(loss_plot_list, color='blue')
    plt.show()
    plt.tight_layout()
    plt.pause(0.001)
    
    print('save model at step', step)
    torch.save(model.state_dict(), '/home/lshi23/carla_test/saved_models/%06fmodel.pt' %loss_epoch)
    
    if step % 100 == 0 :
       
        # test the model
        loss_epoch = 0.0
        for i in range(test_BATCH_NUM):
            
            action_input_data = torch.randn(BATCH_SIZE, horizon, 2, device=device)                           # 2 is linear and angular velocity
            ground_truth_data = torch.randn(BATCH_SIZE, horizon+1, 10, device=device)                        # 10 is [collision, location 3, velocity, yaw, angular velocity, latlong 2, reset]
            image_data = torch.randn(BATCH_SIZE, 3, 96*2, 128*2, device=device)                                 # 3 is RGB channels, 192 is height, 256 is width
            
            for j in range(BATCH_SIZE):
                # load the data
                datapoint_path = os.path.join(datapoints_folder_path, test_datapoints_file_list[i*BATCH_SIZE+j])
                with open(datapoint_path, 'rb') as f: data = json.load(f)
                
                image_data_temp = torch.FloatTensor(data['image_input']).to(device)/255                          # [Height, Width, 3(C:channels)]
                image_data_temp = image_data_temp.permute(2, 0, 1)                                               # [3, Height, Width]
                image_data[j] = image_data_temp
                
                # print(datapoint_path)trainer.py
                # img = transforms.ToPILImage()(image_data[i]).convert('RGB')
                # plt.imshow(img)
                # plt.show()
                # plt.pause(0.001)            
                    
                linear_velocity_temp = np.array(data['action_input']['linear_velocity'])
                steer_temp = np.array(data['action_input']['steer'])
                action_input_data_temp = np.stack((linear_velocity_temp, steer_temp), axis=1)
                action_input_data[j] = torch.FloatTensor(action_input_data_temp).to(device)
                
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
                ground_truth_data[j] = ground_truth_data_temp
            
            with torch.no_grad():
                model.eval()
                test_model_output = model(image_data, action_input_data, ground_truth_data, BATCH_SIZE, horizon)                                        # shape: [BATCH_SIZE, horizon, 4
            model.train()
        
            test_ground_truth_position = torch.randn(BATCH_SIZE, horizon, 3, device=device)
            test_ground_truth_collision = torch.randn(BATCH_SIZE, horizon, device=device)
            test_ground_truth_reset = torch.randn(BATCH_SIZE, horizon, device=device)      
            for j in range(BATCH_SIZE):
                # load ground truth data
                datapoint_path = os.path.join(datapoints_folder_path, test_datapoints_file_list[i*BATCH_SIZE+j])
                with open(datapoint_path, 'rb') as f: data = json.load(f)
                test_ground_truth_position_temp = data['ground_truth']['location'][1:horizon+1]          # shape: [horizon, 3]
                test_ground_truth_collision_temp = data['ground_truth']['collision'][1:horizon+1]        # shape: [horizon]
                test_ground_truth_reset_temp = data['ground_truth']['reset'][1:horizon+1]                # shape: [horizon]
                test_ground_truth_position[j] = torch.FloatTensor(test_ground_truth_position_temp).to(device)
                test_ground_truth_collision[j] = torch.FloatTensor(test_ground_truth_collision_temp).to(device)
                test_ground_truth_reset[j] = torch.FloatTensor(test_ground_truth_reset_temp).to(device)
    
            test_ground_truth_reset_idx = torch.nonzero(test_ground_truth_reset, as_tuple=True)[0]
            test_loss_position_pre = 0.5 * torch.square(test_model_output[:, :, :3] - test_ground_truth_position)
            test_loss_position_pre = torch.sum(test_loss_position_pre, dim=(1, 2))
            for idx in test_ground_truth_reset_idx:
                test_loss_position_pre[idx] = 0
            test_loss_position = test_loss_position_pre.sum()
            
            if step > 3000: print('test loss is', test_loss_position.item())
            
            test_loss_cross_entropy = nn.CrossEntropyLoss(reduction='sum')        
            test_loss_collision = test_loss_cross_entropy(test_model_output[:, :, 3], test_ground_truth_collision)

            test_loss = (test_loss_position + test_loss_collision)
            loss_epoch += float(test_loss.item())
            print('Epoch:', step, ' Batch:', i,' test loss:', float(test_loss.item())/BATCH_SIZE)
            
        loss_epoch = loss_epoch / (test_BATCH_NUM*BATCH_SIZE)
        loss_plot(loss_epoch, test_loss_queue)
        test_loss_data = test_loss_queue.get()
        test_loss_plot_list.append(test_loss_data)
        ax[1].plot(test_loss_plot_list, color='blue')
        plt.show()
        plt.tight_layout()
        plt.pause(0.001)
        

    if step > 1000 and loss < 0.15:
        print('pause training, loss is', loss)
    
    if loss < 0.05 or step>1e5:
        print('save model, break')
        torch.save(model.state_dict(), '/home/lshi23/carla_test/combined_model.pt')
        break