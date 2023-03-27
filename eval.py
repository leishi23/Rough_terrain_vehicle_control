import glob
import os
import sys
import queue
import carla
import numpy as np
import torch
from torch import nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import time 
from model import combined_model

os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:2"

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
IM_WIDTH = 128*2
IM_HEIGHT = 96*2

horizon = 8       
beta = 0.6
sigma = 1.0        # original/badgr is 1.0
N = 256             # original/badgr is 4096
gamma = 50   

settings = world.get_settings()
settings.synchronous_mode = True  # Enables synchronous mode
delta_time = 0.15
settings.fixed_delta_seconds = delta_time  # Sets the fixed time step 0.05 by default
settings.substepping = True
settings.max_substep_delta_time = 0.01      # 0.01*50 = 0.5 > 0.15  !!!!!!!!!!!
settings.max_substeps = 16
world.apply_settings(settings) 

traffic_manager = client.get_trafficmanager()    
traffic_manager.set_synchronous_mode(True)

vehicle_blueprints = world.get_blueprint_library().filter('*vehicle*')

# spawn points for vehicles
spawn_points = carla.Transform(carla.Location(x=27, y=26, z=0.1), carla.Rotation(pitch=0, yaw=float(235), roll=0))  # carla.Location y value is negative of the roadrunner y value
final_position = carla.Transform(carla.Location(x=5, y=16, z=0.1), carla.Rotation(pitch=0, yaw=float(-135), roll=0))
ego_bp = world.get_blueprint_library().find('vehicle.audi.tt')
ego_vehicle = world.spawn_actor(ego_bp, spawn_points)     # set z as 0.1 to avoid collision with the ground
print('created ego_%s' % ego_vehicle.type_id)

# Create a transform to place the camera on top of ssthe vehicle
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

# camera.listen(lambda data: process_img(data, rgb_image_queue))
# camera.listen(lambda image: image.save_to_disk('data/eval/eval.png'))
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
    'position': [5, 16],
    'cost_weights.position_sigmoid_scale': 100.,
    'cost_weights.position_sigmoid_center': 0.4,           # original/badgr is 0.4
    'cost_weights.collision': 0,
    'cost_weights.position': 1,
    'cost_weights.action_magnitude': 0.0,
    'cost_weights.action_smooth': 0.0
}

def cost_function(model_outputs, actions, goal):
    #### collision cost
    collision_model_outputs = model_outputs[:, :, 3]     # [position x, position y, position z, collision]
    collision_model_outputs = torch.sigmoid(collision_model_outputs)
    clamp_value = 0.02
    collision_model_outputs = torch.clamp(collision_model_outputs, min=clamp_value, max=1.-clamp_value)
    collision_model_outputs = (collision_model_outputs - clamp_value)/(1.-2.*clamp_value)
    cost_collision = collision_model_outputs

    #### distance cost
    position_model_outputs = model_outputs[:, :, :2]    # [position x, position y, position z, collision]
    position_goal = goal.get('position')
    position_goal = torch.Tensor(position_goal).unsqueeze(0).unsqueeze(0)
    dot_product = torch.sum(position_model_outputs * position_goal, dim=2)
    position_model_outputs_norm = torch.linalg.norm(position_model_outputs, dim=2)
    position_goal_norm = torch.linalg.norm(position_goal, dim=2)
    cos_theta_raw = dot_product / torch.max(position_model_outputs_norm * position_goal_norm, torch.tensor(1e-4).to(device))
    cos_theta = torch.clamp(cos_theta_raw, min=-1.+1e-4, max=1.-1e-4)
    theta = torch.acos(cos_theta)
    cost_position = (1. / np.pi) * torch.abs(theta)
    # cost_position = torch.nn.Sigmoid()(goal.get('cost_weights.position_sigmoid_scale') * (cost_position - goal.get('cost_weights.position_sigmoid_center')))
    
    #### magnitude action cost
    angular_velocity = actions[:, :, 1]
    cost_action_magnitude = torch.square(angular_velocity)
    
    #### smooth action cost 
    cost_action_smooth = torch.cat([torch.square(angular_velocity[:, 1:] - angular_velocity[:, :-1]), torch.zeros(angular_velocity.shape[0], 1).to(device)], dim=1)
    
    total_cost = goal.get('cost_weights.collision') * cost_collision + \
                    goal.get('cost_weights.position') * cost_position + \
                    goal.get('cost_weights.action_magnitude') * cost_action_magnitude + \
                    goal.get('cost_weights.action_smooth') * cost_action_smooth
                    
    return total_cost

model = combined_model().to(device)
model.load_state_dict(torch.load(PATH, map_location='cpu'))
model.eval()

# initial setting for real time plot

fig, (ax_img, ax_path) = plt.subplots(nrows=1, ncols=2)

img_array = np.random.randint(0, 100, size=(IM_HEIGHT, IM_WIDTH), dtype=np.uint8)
l_img = ax_img.imshow(img_array)
ax_img.set_title('Camera View')
ax_img.set_xticks([])
ax_img.set_yticks([])

ax_path.set_title('Possible Path')
ax_path.legend()


while world is not None:
    
    # spectator = world.get_spectator()
    # spectator.set_transform(carla.Transform(ego_vehicle.get_location() + carla.Location(z=2.5), ego_vehicle.get_transform().rotation))
    
    spectator = world.get_spectator()
    spectator.set_transform(
    carla.Transform(ego_vehicle.get_location() + carla.Location(z=10), carla.Rotation(pitch=-90)))

    # to add some noise to foward velocity and steering angle
    if step_counter == 0:
        forward_velocity_applied = np.random.normal(2, 0)
        ego_vehicle.enable_constant_velocity(carla.Vector3D(x=forward_velocity_applied,y=0,z=0))
        
        steer_applied = 0.0
        ego_vehicle.apply_control(carla.VehicleControl(steer=steer_applied))
        
        best_action_queue.put(torch.randn((horizon, 2)))       # put a random array into the queue to avoid the queue is empty
        world.tick()
        
    else:
        # Get the best action from the queue
        past_best_action = best_action_queue.get()
        past_mean = past_best_action[0]
        shifted_mean = torch.cat((past_best_action[1:], past_best_action[-1:]), dim=0)
        
        # Apply the best action to the ego vehicle
        forward_velocity_applied = past_best_action[0][0]
        steer_applied = past_best_action[0][1]
        ego_vehicle.enable_constant_velocity(carla.Vector3D(x=float(forward_velocity_applied),y=0,z=0))
        ego_vehicle.apply_control(carla.VehicleControl(steer=float(steer_applied)))
        world.tick()
        
    # Get the ego vehicle location and yaw
    location_queue.put(ego_vehicle.get_location())
    ego_transform = ego_vehicle.get_transform()
    yaw = ego_transform.rotation.yaw
    yaw_queue.put(yaw)
    
    # Get the model output using image data, yaw/position of step-0 and random action
    try:
        location_data = location_queue.get()            # float [x,y,z]
        location_data = [location_data.x, location_data.y, location_data.z]
        yaw_data = yaw_queue.get()                      # float
        yaw_data_radians = yaw_data * np.pi / 180       # convert to radians
        image_data = rgb_image_queue.get()                
    except queue.Empty:
        print("[Warning] Some sensor data has been missed")
        continue 
    
    snapshot = world.get_snapshot()
    delta_seconds = snapshot.timestamp.delta_seconds
    elapsed_seconds = snapshot.timestamp.elapsed_seconds
    # print('delta_seconds is %s' % delta_seconds)
    # print('elapsed_seconds is %s' % elapsed_seconds)
    
    # Whether arrive the final position
    goal_x = goal.get('position')[0]
    goal_y = goal.get('position')[1]
    if abs(location_data[0] - goal_x) < 2 and abs(location_data[1] - goal_y) < 2 and step_counter > 10:
        print("Arrive the final position")
        print("Current location: ", location_data)
        break
    
    if step_counter > 0 :
        # # Generate the random action
        # delta_limits = 0.5*(torch.tensor([2.2,0.5]) - torch.tensor([1.8,-0.5]))         # Not sure if this is correct!!!!
        # # eps = torch.normal(mean=0.0, std=delta_limits*sigma, size=(N, horizon, 2))
        # eps = np.random.normal(loc=0, scale=delta_limits*sigma, size=(N, horizon, 2))
        # eps = torch.from_numpy(eps).float()
        # actions = []
        # lower_limits = np.array([1.8,-0.5])                                         
        # upper_limits = np.array([2.2,0.5])                                    
        # for h in range(horizon):
        #     if h == 0:
        #         action_h = beta * (shifted_mean[h, :] + eps[:, h, :]) + (1 - beta) * past_mean
        #     else:
        #         action_h = beta * (shifted_mean[h, :] + eps[:, h, :]) + (1 - beta) * actions[-1]
        #     action_h= action_h.detach().numpy()                                   # action_h is with shape [N,2], i.e. [N, action_dim] (forward_velocity, steer), convert to numpy array
        #     action_h = np.clip(action_h, lower_limits, upper_limits)              # use np.clip to limit the action range
        #     action_h = torch.from_numpy(action_h)                                   
        #     actions.append(action_h)                                              # actions is a list with 8 elements, each element is with shape [N,2], i.e. [N, action_dim] (forward_velocity, steer) 
        # actions = torch.stack(actions, dim=1)                                     # actions is with shape [N,8,2], i.e. [N, horizon, action_dim] (forward_velocity, steer)
        # actions = actions.to(torch.float32)
        
        actions = torch.zeros([N, horizon, 2])
        steer = np.linspace(-1.0, 1.0, num=N)
        for i in range(N):
            actions[i,:,0] = torch.tensor([2.0]*horizon) 
            actions[i,:,1] = torch.tensor([steer[i]]*horizon)
        
        # Get the minmum cost index from the model output, xie bu wan le aaaaaaaaa!!!!!!!!!
        # image_path = 'data/eval/eval.png'
        # img = Image.open(image_path)
        # image_data = transforms.ToTensor()(img).to(device)
        # image_data = image_data[:-1,:,:].unsqueeze(0)                       # remove the alpha channel, and add a dimension to the front
        im_array = np.copy(np.frombuffer(image_data.raw_data, dtype=np.dtype("uint8")))
        im_array = np.reshape(im_array, (image_data.height, image_data.width, 4))
        img = im_array[:, :, :3][:, :, ::-1]
        l_img.set_data(img)
        # plt.imshow(img)
        # plt.show()
        # plt.pause(0.001)
        image_data = transforms.ToTensor()(img.copy())       # convert to tensor, copy() is to avoid the negative stride error
        location_data = torch.tensor(location_data)          # add a dimension to the front
        yaw_data_radians = torch.tensor(yaw_data_radians)
        
        # image_data_input: (BATCH_SIZE, 3, height, width)
        image_data_input = torch.randn(N, 3, IM_HEIGHT, IM_WIDTH)
        for i in range(N):
            image_data_input[i] = image_data 
        # action_input_data: (BATCH_SIZE, horizon, 2)
        # ground_truth_data: (BATCH_SIZE, horizon+1, 8)
        ground_truth = torch.randn(N, horizon+1, 8)
        for i in range(N):
            ground_truth[i, 0, 5] = yaw_data_radians
            ground_truth[i, 0, 1:4] = location_data
        
        # image_data with shape [3, 192, 256], location_data with shape [3], yaw_data_radians with shape [], actions with shape [N, 8, 2]
        model_output = model(image_data_input, actions, ground_truth, N, horizon)
        
        costs_per_timestep = cost_function(model_output, actions, goal)
        costs = torch.mean(costs_per_timestep, dim=1)         # costs is with shape [N,]
        # cost_min_index = torch.argmin(costs)
        cost_min_index = torch.topk(costs, 3, largest=False)[1].tolist()        # get the index of the k smallest costs
        
        # print("actions are: ", actions)
        # print("model outputs are: ", model_output[:,:,:2])
        # print("costs: ", costs)
        
        possible_path_min = model_output[torch.argmin(costs),:,:2].detach().numpy()       # possible_path is with shape [5,8,2], i.e. [k, horizon, position_dim] (x,y)
        possible_path_max = model_output[torch.argmax(costs),:,:2].detach().numpy()       # possible_path is with shape [5,8,2], i.e. [k, horizon, position_dim] (x,y)
        
        scores = -costs[cost_min_index]
        probs = torch.exp(gamma*(scores - torch.max(scores)))
        probs = probs / torch.sum(probs) + 1e-10
        new_mppi_mean = torch.sum(actions[cost_min_index] * probs.unsqueeze(-1).unsqueeze(-1) , dim=0)
        
        # new_mppi_mean = torch.mean(actions[cost_min_index], dim=0)           # new_mppi_mwan is with shape [8,2], i.e. [horizon, action_dim] (forward_velocity, steer)
        # print("costs are: ", costs)
        # print("model outputs are: ", model_output[:,:,:2])
        print("location data is: ", location_data[0].item(), location_data[1].item())
        print("new mppi mean steer is: ", new_mppi_mean[0,1].item())
        # print("model output is: ", model_output[cost_min_index,0,0].item(), model_output[cost_min_index,0,1].item())
        print()
        
        # Store the best action into queue
        best_action_queue.put(new_mppi_mean)                  # best_action is with shape [8,2], i.e. [horizon, action_dim] (forward_velocity, steer)
        
        if step_counter > 1e4:
            past_mean = best_action_queue.get()
        
        ax_path.clear()
        ax_path.plot(location_data[0].item(), location_data[1].item(), 'o', color='g', label='current')
        ax_path.plot(possible_path_min[:,0], possible_path_min[:,1], '*', color='r', label='min cost path')
        ax_path.plot(possible_path_max[:,0], possible_path_max[:,1], '*', color='b', label='max cost path')
        ax_path.legend()
        
        plt.draw()
        plt.pause(0.1)
        
        start_point = ego_vehicle.get_location()
        path_min_list = [start_point]
        path_max_list = [start_point]
        for i in range(horizon):
            temp_min_x = float(possible_path_min[i,0])
            temp_min_y = float(possible_path_min[i,1])
            temp_min = carla.Location(x=temp_min_x, y=temp_min_y, z=0.0)
            
            temp_max_x = float(possible_path_max[i,0])
            temp_max_y = float(possible_path_max[i,1])
            temp_max = carla.Location(x=temp_max_x, y=temp_max_y, z=0.0)
        
            world.debug.draw_line(begin = path_min_list[-1], end = temp_min, color=carla.Color(255, 0, 0), life_time=delta_seconds*2)
            # world.debug.draw_line(begin = path_max_list[-1], end = temp_max, color=carla.Color(0, 0, 255), life_time=delta_seconds*2)
            
            path_min_list.append(temp_min)
            path_max_list.append(temp_max)
        
    step_counter += 1  