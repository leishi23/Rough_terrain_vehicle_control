from calendar import EPOCH
import torch
from torch import nn, tensor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        
    def forward(self, image_data, action_input_data, ground_truth_data, BATCH_SIZE, horizon):
        # image_data: (BATCH_SIZE, 3, height, width)
        # action_input_data: (BATCH_SIZE, horizon, 2)
        # ground_truth_data: (BATCH_SIZE, horizon+1, 8)

        model_output = torch.randn((BATCH_SIZE, horizon, 4), device=device)
        
        cnn_out1 = self.obs_im_model(image_data)
        cnn_out2 = self.obs_lowd_model(cnn_out1)
        action_input_processed = self.action_input_model(action_input_data)
        
        initial_state_c_temp, initial_state_h_temp = torch.split(cnn_out2, 64, dim=-1)
        initial_state_c = torch.randn(horizon, BATCH_SIZE, 64, device=device).copy_(initial_state_c_temp)
        initial_state_h = torch.randn(horizon, BATCH_SIZE, 64, device=device).copy_(initial_state_h_temp)   
        
        lstm_out, _ = self.rnn_cell(action_input_processed, (initial_state_h, initial_state_c))
          
        model_output_temp = self.output_model(lstm_out)           # [position x, position y, position z, collision], diff from ground_truth  
                
        for i in range(BATCH_SIZE):
            yaw_data = ground_truth_data[i, 0, 5]      # yaw is from CARLA vehicle transform frame where clockwise is positive. In rotation matrix, yaw is from world frame where counter-clockwise is positive. When reverse the yaw, it's still counter-clockwise.
            zeros = torch.tensor(0, dtype=torch.float32).to(device)
            ones = torch.tensor(1, dtype=torch.float32).to(device)
            temp = torch.stack([torch.cos(yaw_data), -torch.sin(yaw_data), zeros, torch.sin(yaw_data), torch.cos(yaw_data), 
                    zeros, zeros, zeros, ones])
            rotmatrix = torch.reshape(temp, (3, 3))
            model_output_temp_local_position = model_output_temp[i, :, :3]
            model_output_temp_global_position = torch.matmul(model_output_temp_local_position, rotmatrix)
            model_output[i, :,:3] = model_output_temp_global_position + ground_truth_data[i, 0, 1:4]     # add the initial position for timestep 0 to the output position 
            model_output[i, :, 3] = model_output_temp[i, :, 3]
            
        return model_output     # [batch_size, horizon+1, 4], horizon+1 is timestep, 4 is [position x, position y, position z, collision]
