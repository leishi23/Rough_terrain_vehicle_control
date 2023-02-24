import os
import json
import pandas as pd
import numpy as np
import shutil

json_path = '/home/lshi23/carla_test/data/raw_data'
json_dir = os.listdir(json_path)
json_dir.sort()
test_data_num = 135                        # for test, must be multiple of 9*15, i.e. horizon*BATCH_SIZE
test_json_dir = json_dir[-test_data_num:]
json_dir = json_dir[:-test_data_num]        # remove the last 20% files

num_frame = len(json_dir)                   # how many time steps collected
num_horizon = 8                             # how many time steps in the horizon
num_steps = num_horizon + 1                 # how many time steps in one datapoint (including current time step)
num_datapoints = int(num_frame/num_steps)   # how many datapoints in total
num_remains = num_frame % num_steps         # how many time steps left and need to be discarded

test_num_frame = len(test_json_dir)
test_num_horizon = 8
test_num_steps = test_num_horizon + 1
test_num_datapoints = int(test_num_frame/test_num_steps)

img_path = '/home/lshi23/carla_test/data/image/rgb_out'
test_img_path = '/home/lshi23/carla_test/data/test_dataset/image/rgb_out'
img_dir = os.listdir(img_path)
img_dir.sort()

# when ctrl C in data_collection.py, maybe #image is ONE bigger than #json
if num_frame < len(img_dir)-test_data_num:
    os.remove(os.path.join(img_path, img_dir[-1]))
    img_dir = img_dir[:-1]
   
assert num_frame == len(img_dir)-test_data_num, 'number of json and image files are not equal'
 
# to delete the remains image and json files, also update the directory
for i in range(num_remains):
    os.remove(os.path.join(json_path, json_dir[-1]))
    os.remove(os.path.join(img_path, img_dir[-1]))
    img_dir = img_dir[:-1]
    json_dir = json_dir[:-1]

assert len(json_dir) % num_steps == 0, 'number of json/image files are not multiple of num_steps, remains are not deleted'

print('Datapoint creation process')

for i in range(num_datapoints+test_num_datapoints):
    # Input: delete image num_steps*i ~ num_steps*i+horizon, i.e image for 1~8, only keep image time step 0
    for j in range(num_steps*i+1,num_steps*i+num_steps):
        os.remove(os.path.join(img_path, img_dir[j]))

final_img_dir = os.listdir(img_path)
final_img_dir.sort()
for h in range(test_num_datapoints):
    src_path = os.path.join(img_path, final_img_dir[h-test_num_datapoints])
    dst_path = os.path.join(test_img_path)  
    shutil.move(src_path, dst_path)      

for i in range(num_datapoints):    
    # Action input for datapoint i
    action_input = np.zeros((2, num_horizon))
    for j in range(num_horizon):
        json_path_single = os.path.join(json_path, json_dir[num_steps*i+j])                 # action input for time step 0 ~ horizon-1
        with open(json_path_single) as f:
            json_single_data = json.load(f)
            linear_velocity = json_single_data[4]
            angular_velocity = json_single_data[6]
            action_input[0][j] = linear_velocity
            action_input[1][j] = angular_velocity
    df1 = pd.DataFrame(action_input)
    df1.to_csv('/home/lshi23/carla_test/data/action_input/action_input_%06d.csv' % i, index=False, header=False)
    
    # Output: the ground truth vector for datapoint i
    ground_truth = np.zeros((10,num_steps))
    for j in range(num_horizon+1):
        json_path_horizon = os.path.join(json_path, json_dir[num_steps*i+j])              # vector output for time step 0 ~ horizon
        with open(json_path_horizon) as f:
            json_horizon_data = json.load(f)
            ground_truth[0][j] = json_horizon_data[0]
            ground_truth[1][j] = json_horizon_data[1]
            ground_truth[2][j] = json_horizon_data[2]
            ground_truth[3][j] = json_horizon_data[3]
            ground_truth[4][j] = json_horizon_data[4]
            ground_truth[5][j] = json_horizon_data[5]
            ground_truth[6][j] = json_horizon_data[6]
            ground_truth[7][j] = json_horizon_data[7]
            ground_truth[8][j] = json_horizon_data[8]            
            ground_truth[9][j] = json_horizon_data[9]                         
            # 10: [collision, location 3, velocity, yaw, angular velocity, latlong 2, reset]
    df2 = pd.DataFrame(ground_truth)
    df2.to_csv('/home/lshi23/carla_test/data/ground_truth/ground_truth_%06d.csv' % i, index=False, header=False)
            
    print ("-training dataset---%.0f%%----" % (100 * i/num_datapoints))
       
for i in range(test_num_datapoints):
    
    # Action input for datapoint i
    action_input = np.zeros((2, test_num_horizon))
    for j in range(test_num_horizon):
        json_path_single = os.path.join(json_path, test_json_dir[test_num_steps*i+j])                 # action input for time step 0 ~ horizon-1
        with open(json_path_single) as f:
            json_single_data = json.load(f)
            linear_velocity = json_single_data[4]
            angular_velocity = json_single_data[6]
            action_input[0][j] = linear_velocity
            action_input[1][j] = angular_velocity
    df1 = pd.DataFrame(action_input)
    df1.to_csv('/home/lshi23/carla_test/data/test_dataset/action_input/action_input_%06d.csv' % i, index=False, header=False)
    
    # Output: the ground truth vector for datapoint i
    ground_truth = np.zeros((10,test_num_steps))
    for j in range(num_horizon+1):
        json_path_horizon = os.path.join(json_path, test_json_dir[num_steps*i+j])              # vector output for time step 0 ~ horizon
        with open(json_path_horizon) as f:
            json_horizon_data = json.load(f)
            ground_truth[0][j] = json_horizon_data[0]
            ground_truth[1][j] = json_horizon_data[1]
            ground_truth[2][j] = json_horizon_data[2]
            ground_truth[3][j] = json_horizon_data[3]
            ground_truth[4][j] = json_horizon_data[4]
            ground_truth[5][j] = json_horizon_data[5]
            ground_truth[6][j] = json_horizon_data[6]
            ground_truth[7][j] = json_horizon_data[7]
            ground_truth[8][j] = json_horizon_data[8]       
            ground_truth[9][j] = json_horizon_data[9]                              
            # 10: [collision, location 3, velocity, yaw, angular velocity, latlong 2, reset]
    df2 = pd.DataFrame(ground_truth)
    df2.to_csv('/home/lshi23/carla_test/data/test_dataset/ground_truth/ground_truth_%06d.csv' % i, index=False, header=False)
            
    print ("-test dataset---%.0f%%----" % (100 * i/test_num_datapoints))
   
print ("Done!") 