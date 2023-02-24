import glob
import os
import sys
import queue
import carla
import numpy as np
import torch 
from torch import nn

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

def process_img(data, rgb_queue):
    rgb_queue.put(data)

# Connect to the client and retrieve the world object
client = carla.Client('localhost', 2000)
client.set_timeout(20.0)
world = client.get_world()
client.load_world('Map_Dec_31')
IM_WIDTH = 128
IM_HEIGHT = 96   

horizon = 8       
beta = 0.6
sigma = 1.0
N = 4096
gamma = 50   

settings = world.get_settings()
settings.synchronous_mode = True  # Enables synchronous mode
delta_time = 0.25
settings.fixed_delta_seconds = delta_time  # Sets the fixed time step 0.05 by default
world.apply_settings(settings) 

vehicle_blueprints = world.get_blueprint_library().filter('*vehicle*')

# spawn points for vehicles
spawn_points = carla.Transform(carla.Location(x=27, y=26, z=0.1), carla.Rotation(pitch=0, yaw=float(np.random.rand(1)*360), roll=0))  # carla.Location y value is negative of the roadrunner y value
final_position = carla.Transform(carla.Location(x=27, y=26, z=0.1), carla.Rotation(pitch=0, yaw=float(np.random.rand(1)*360), roll=0))
ego_bp = world.get_blueprint_library().find('vehicle.audi.tt')
ego_vehicle = world.spawn_actor(ego_bp, spawn_points)     # set z as 0.1 to avoid collision with the ground
print('created ego_%s' % ego_vehicle.type_id)

# Create a transform to place the camera on top of the vehicle
camera_transform = carla.Transform(carla.Location(x=0.5, z=2.5))

# We create sensors through a blueprint that defines its properties
camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
camera_bp.set_attribute("image_size_x", f"{IM_WIDTH}")
camera_bp.set_attribute("image_size_y", f"{IM_HEIGHT}")
camera_bp.set_attribute("fov", "110")

# We spawn the camera and attach it to our ego vehicle
camera = world.spawn_actor(camera_bp, camera_transform, attach_to=ego_vehicle)

# The sensor data will be saved in thread-safe Queues
rgb_image_queue = queue.Queue(maxsize=1)   
location_queue = queue.Queue(maxsize=1)
yaw_queue = queue.Queue(maxsize=1)
best_action_queue = queue.Queue(maxsize=1)

camera.listen(lambda data: process_img(data, rgb_image_queue))

# set the spectator
spectator = world.get_spectator()
spectator.set_transform(carla.Transform(ego_vehicle.get_location() + carla.Location(z=10), carla.Rotation(pitch=-90)))

# set the time counter
step_counter = 0

# set the model   
PATH = '/home/lshi23/carla_test/combined_model.pt'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# set the goal 
goal = {
    'position': ,
    'cost_weights.position_sigmoid_scale': ,
    'cost_weights.position_sigmoid_center': ,
    'cost_weights.collision': ,
    'cost_weights.position': ,
    'cost_weights.action_magnitude': ,
    'cost_weights.action_smooth': ,
}

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
        
    def forward(self, img_data, location_data, yaw_data_radians, actions):
        
        model_output = torch.randn((N, horizon, 4), device=device)
            
        img_raw_data = img_data                     # UNSURE FORMAT!!
        
        # img_raw_data = img_raw_data[0]
        # img = transforms.ToPILImage()(img_raw_data).convert('RGB')
        # plt.imshow(img)
        # plt.show()
        # plt.pause(0.001)
        
        action_input_data = actions                 # [ N, horizon, 2]
            
        cnn_out1 = self.obs_im_model(img_raw_data)
        cnn_out2 = self.obs_lowd_model(cnn_out1)
        action_input_processed = self.action_input_model(action_input_data)
        
        initial_state_c_temp, initial_state_h_temp = torch.split(cnn_out2, 64, dim=-1)
        initial_state_c = torch.randn(horizon, N, 64, device=device).copy_(initial_state_c_temp)
        initial_state_h = torch.randn(horizon, N, 64, device=device).copy_(initial_state_h_temp)
        
        # initial_state_c = torch.zeros(horizon, BATCH_SIZE, 64, device=device)
        # initial_state_c[0] = initial_state_c_temp
        # initial_state_h = torch.zeros(horizon, BATCH_SIZE, 64, device=device)
        # initial_state_h[0] = initial_state_h_temp       
        
        lstm_out, _ = self.rnn_cell(action_input_processed, (initial_state_h, initial_state_c))
        
        model_output_temp = self.output_model(lstm_out)           # [position x, position y, position z, collision], diff from ground_truth, shape is [N, horizon, 4]          
                
        for i in range(N):
            yaw_data = yaw_data_radians      # yaw is from CARLA vehicle transform frame where clockwise is positive. In rotation matrix, yaw is from world frame where counter-clockwise is positive. When reverse the yaw, it's still counter-clockwise.
            rotmatrix = rotmat(yaw_data)
            model_output_temp_local_position = model_output_temp[i, :, :3]
            model_output_temp_global_position = torch.matmul(model_output_temp_local_position, rotmatrix)
            model_output[i, :,:3] = model_output_temp_global_position + location_data     # add the initial position for timestep 0 to the output position 
            model_output[i, :, 3] = model_output_temp[i, :, 3]
        
        return model_output     # [N, horizon+1, 4], horizon+1 is timestep, 4 is [position x, position y, position z, collision]
    
def cost_function(model_outputs, actions, goal):
    #### collision cost
    collision_model_outputs = model_outputs[:, :, 3]     # [position x, position y, position z, collision]
    clamp_value = 0.02
    collision_model_outputs = torch.clamp(collision_model_outputs, min=clamp_value, max=1.-clamp_value)
    collision_model_outputs = (collision_model_outputs - clamp_value)/(1.-2.*clamp_value)
    cost_collision = collision_model_outputs

    #### distance cost
    position_model_outputs = model_outputs[:, :, 0:3]    # [position x, position y, position z, collision]
    position_goal = goal.get('position')
    dot_product = torch.sum(position_model_outputs * position_goal, dim=2)
    position_model_outputs_norm = torch.linalg.norm(position_model_outputs, dim=2)
    position_goal_norm = torch.linalg.norm(position_goal, dim=2)
    cos_theta = dot_product / torch.max(position_model_outputs_norm * position_goal_norm, torch.tensor(1e-6).to(device))
    cos_theta = torch.clamp(cos_theta, min=-1.+1e-4, max=1.-1e-4)
    theta = torch.acos(cos_theta)
    theta = (1. / np.pi) * torch.abs(theta)
    cost_position = torch.nn.Sigmoid()(goal.get('cost_weights.position_sigmoid_scale') * (theta - goal.get('cost_weights.position_sigmoid_center')))
    
    #### magnitude action cost
    angular_velocity = actions[:, :, 1]
    cost_action_magnitude = torch.square(angular_velocity)
    
    #### smooth action cost (TBD)
    cost_action_smooth = torch.cat([torch.square(angular_velocity[:, 1:] - angular_velocity[:, :-1]), torch.zeros(angular_velocity.shape[0], 1).to(device)], dim=1)
    
    total_cost = goal.get('cost_weights.collision') * cost_collision + \
                    goal.get('cost_weights.position') * cost_position + \
                    goal.get('cost_weights.action_magnitude') * cost_action_magnitude + \
                    goal.get('cost_weights.action_smooth') * cost_action_smooth
                    
    return total_cost

model = combined_model().to(device)
model.load_state_dict(torch.load(PATH))
model.eval()

while world is not None:
    
    # Get the ego vehicle location and yaw
    location_queue.put(ego_vehicle.get_location())
    ego_transform = ego_vehicle.get_transform()
    yaw = ego_transform.rotation.yaw
    yaw_queue.put(yaw)
    
    # Get the best action from the queue
    past_best_action = best_action_queue.get()
    past_mean = past_best_action[0]
    shifted_mean = torch.cat((past_best_action[1:], past_best_action[-1:]), dim=0)
    
    # Apply the best action to the ego vehicle
    forward_velocity = past_best_action[0][0]
    steer = past_best_action[0][1]
    ego_vehicle.enable_constant_velocity(carla.Vector3D(x=forward_velocity,y=0,z=0))
    ego_vehicle.apply_control(carla.VehicleControl(steer=steer))
    world.tick()
    step_counter += 1
    
    # Get the frame 
    world_snapshot = world.get_snapshot()
    frame = world_snapshot.frame
    
    # Whether arrive the final position
    location = ego_vehicle.get_location()
    goal_x = goal.get('position')[0]
    goal_y = goal.get('position')[1]
    if abs(location.x - goal_x) < 1 and abs(location.y - goal_y) < 1 and step_counter > 10:
        print("Arrive the final position")
        break
    
    # Generate the random action
    delta_limits = 0.5*([1,3] - [-1,3])
    eps = torch.normal(mean=0, std=delta_limits*sigma, size=(N, horizon, 2))
    actions = []
    lower_limits = np.array([-1,2.9])
    upper_limits = np.array([1,3.1])
    for h in range(horizon):
        if h == 0:
            action_h = beta * (shifted_mean[h, :] + eps[:, h, :]) + (1 - beta) * past_mean
        else:
            action_h = beta * (shifted_mean[h, :] + eps[:, h, :]) + (1 - beta) * actions[-1]
        action_h= np.array(action_h)                                    # action_h is with shape [N,2], i.e. [N, action_dim] (forward_velocity, steer), convert to numpy array
        action_h = np.clip(action_h, lower_limits, upper_limits)        # use np.clip to limit the action range
        action_h = action_h.tolist()                                    # convert to list
        actions.append(action_h)                                        # actions is a list with 8 elements, each element is with shape [N,2], i.e. [N, action_dim] (forward_velocity, steer) 
    actions = torch.stack(actions, dim=1)                               # actions is with shape [N,8,2], i.e. [N, horizon, action_dim] (forward_velocity, steer)
    
    # Get the model output using image data, yaw/position of step-0 and random action
    try:
        image_data = rgb_image_queue.get()
        location_data = location_queue.get()            # float [x,y,z]
        yaw_data = yaw_queue.get()                      # float
        yaw_data_radians = yaw_data * np.pi / 180       # convert to radians
    except queue.Empty:
        print("[Warning] Some sensor data has been missed")
        continue

    model_output = model(image_data, location_data, yaw_data_radians, actions)    # model_output is with shape [N,9,4], i.e. [N, horizon+1, 4], horizon+1 is timestep, 4 is [position x, position y, position z, collision]
    
    # Get the best action with the cost function
    costs = cost_function(model_output, actions, goal)  # costs is with shape [N]    
    cost_min, cost_min_index = torch.min(costs)    
    
    # Store the best action into queue
    best_action = actions[cost_min_index, :, :]         
    best_action_queue.put(best_action)                  # best_action is with shape [8,2], i.e. [horizon, action_dim] (forward_velocity, steer)