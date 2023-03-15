import os
import json
import pandas as pd
import numpy as np
import shutil
from PIL import Image
import random

json_path = '/home/lshi23/carla_test/data/raw_data'
json_dir = os.listdir(json_path)
json_dir.sort()

num_frame = len(json_dir)                   # how many time steps collected
num_horizon = 8                             # how many time steps in one horizon
num_steps = num_horizon + 1                 # how many time steps in one datapoint (including current time step, i.e. step 0)
num_datapoints = int(num_frame/num_steps)   # how many datapoints in total
num_remains = num_frame % num_steps         # how many time steps left and need to be discarded

img_path = '/home/lshi23/carla_test/data/image/rgb_out'
img_dir = os.listdir(img_path)
img_dir.sort()

# Due to abandon the first five frames when resetting vehicle, the img_dir and json_dir are not consecutive, but jumping somewhere in the middle
# when ctrl C in data_collection.py, maybe #image is ONE bigger than #json
while num_frame < len(img_dir):
    os.remove(os.path.join(img_path, img_dir[-1]))
    img_dir = img_dir[:-1]
   
# assert num_frame == len(img_dir), 'number of json and image files are not equal'
assert json_dir[0][:6] == img_dir[0][:6], 'first json and image files are not in the same frame'
assert json_dir[-1][:6] == img_dir[-1][:6], 'last json and image files are not in the same frame'

# to delete the remains image and json files, also update the directory
for i in range(num_remains):
    os.remove(os.path.join(json_path, json_dir[-1]))
    os.remove(os.path.join(img_path, img_dir[-1]))
    img_dir = img_dir[:-1]
    json_dir = json_dir[:-1]

assert len(json_dir) % num_steps == 0, 'number of json & image files are not multiple of num_steps, remains are not deleted'

print('Datapoint creation process')

for i in range(num_datapoints):
    # Input: delete image num_steps*i ~ num_steps*i+horizon, i.e image for 1~8, only keep image time step 0
    for j in range(num_steps*i+1,num_steps*i+num_steps):
        os.remove(os.path.join(img_path, img_dir[j]))

final_img_dir = os.listdir(img_path)
final_img_dir.sort()

print('Number of datapoints: ', len(final_img_dir))

for i in range(num_datapoints):
    
    # initialize data_dict: action_input for time step 0~7, ground_truth for time step 0~8, image_input for time step 0; not declare key "image_input" in advance
    data_dict = {"action_input":{"linear_velocity":[], "steer":[]}, "ground_truth":{"collision":[], "location":[], "velocity":[], "yaw":[], "steer":[], "latlong":[], "reset":[]}}
    
    # convert image into array and store for time step 0
    img = Image.open(os.path.join(img_path, final_img_dir[i]))
    np_img = np.array(img).tolist()
    data_dict["image_input"] = np_img
    
    # store action input for time step 0~7
    for j in range(num_horizon):
        json_path_single = os.path.join(json_path, json_dir[num_steps*i+j])                 # action input for time step 0 ~ horizon-1
        with open(json_path_single) as f:
            json_single_data = json.load(f)
            linear_velocity = json_single_data[4]
            steer = json_single_data[6]    
            data_dict["action_input"]["linear_velocity"].append(linear_velocity)
            data_dict["action_input"]["steer"].append(steer)
    
    # store ground truth for time step 0~8
    for j in range(num_horizon+1):
        json_path_horizon = os.path.join(json_path, json_dir[num_steps*i+j])              # vector output for time step 0 ~ horizon
        with open(json_path_horizon) as f:
            json_horizon_data = json.load(f)
            data_dict["ground_truth"]["collision"].append(json_horizon_data[0])
            data_dict["ground_truth"]["location"].append(json_horizon_data[1:4])      
            data_dict["ground_truth"]["velocity"].append(json_horizon_data[4])
            data_dict["ground_truth"]["yaw"].append(json_horizon_data[5])
            data_dict["ground_truth"]["steer"].append(json_horizon_data[6]) 
            data_dict["ground_truth"]["latlong"].append(json_horizon_data[7:9])
            data_dict["ground_truth"]["reset"].append(json_horizon_data[9])              
            # 10 elements: [collision, location x y z, velocity, yaw, steer, latitude/longitude, reset]
            
    # save data_dict as json file
    if np.count_nonzero(data_dict['ground_truth']['collision']) == 0:
        json_object = json.dumps(data_dict, indent=4)  
        with open("/home/lshi23/carla_test/data/datapoints/%06df.json" % i, "w") as outfile:outfile.write(json_object)
        
    else:
        index_collision = np.nonzero(data_dict['ground_truth']['collision'])[0][0]
        for j in range(index_collision+1, num_horizon+1):
            data_dict['ground_truth']['collision'][j] = data_dict['ground_truth']['collision'][index_collision]
            data_dict['ground_truth']['location'][j] = data_dict['ground_truth']['location'][index_collision]
            data_dict['ground_truth']['velocity'][j] = data_dict['ground_truth']['velocity'][index_collision]
            data_dict['ground_truth']['yaw'][j] = data_dict['ground_truth']['yaw'][index_collision]
            data_dict['ground_truth']['steer'][j] = data_dict['ground_truth']['steer'][index_collision]
            data_dict['ground_truth']['latlong'][j] = data_dict['ground_truth']['latlong'][index_collision]
            data_dict['ground_truth']['reset'][j] = data_dict['ground_truth']['reset'][index_collision]
            
        for j in range(index_collision+1, num_horizon):
            data_dict['action_input']['linear_velocity'][j] = np.random.normal(2, 0.01)
            data_dict['action_input']['steer'][j] = np.random.normal(0, 0.03)
        
        json_object = json.dumps(data_dict, indent=4)
        with open("/home/lshi23/carla_test/data/datapoints/%06dt.json" % i, "w") as outfile:outfile.write(json_object)
    
    print ("-Dataset creation---%.0f%%----" % (100 * i/num_datapoints))
    
    
print('Datapoint creation process finished')


# split dataset into training and test set
test_split = 0.2

# randomlized dataset
datapoints_folder_path = 'data/datapoints'
datapoints_file_list = os.listdir(datapoints_folder_path)
datapoints_file_list.sort()
random.shuffle(datapoints_file_list)

# collision_t_num = 0
# collision_f_num = 0
# ct_datapoints_file_list = []
# cf_datapoints_file_list = []

# for i in range(len(datapoints_file_list)):
    # if datapoints_file_list[i][6] == 't':
        # collision_t_num += 1
        # ct_datapoints_file_list.append(datapoints_file_list[i])
    # else:
        # collision_f_num += 1
        # cf_datapoints_file_list.append(datapoints_file_list[i])
  
# balance dataset      
# num_balance = min(collision_t_num, collision_f_num)
# num_training_datapoints = int(num_balance*(1-test_split))
# num_test_datapoints = int(num_balance*test_split)
# training_datapoints_file_list = ct_datapoints_file_list[:num_training_datapoints] + cf_datapoints_file_list[:num_training_datapoints]
# test_datapoints_file_list = ct_datapoints_file_list[-num_test_datapoints:] + cf_datapoints_file_list[-num_test_datapoints:]
# random.shuffle(training_datapoints_file_list)
# random.shuffle(test_datapoints_file_list)

# unbalanced dataset
test_datapoints = int(len(datapoints_file_list)*test_split)
training_datapoints_file_list = datapoints_file_list[:-test_datapoints]
test_datapoints_file_list = datapoints_file_list[-test_datapoints:]

dataset_= {"training": training_datapoints_file_list, "test": test_datapoints_file_list}
json_object_dataset = json.dumps(dataset_, indent=4)
with open("data/dataset_split.json", "w") as outfile: outfile.write(json_object_dataset)