# %%
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from tqdm import tqdm
import pickle
import torch
from torch import nn, tensor
# %%
data_folder = './data/datapoints'
location = []
yaws = []
steer = []
linear_velocity = []
for f in os.listdir(data_folder):
    with open(os.path.join(data_folder, f), 'rb') as fj:
        data = json.load(fj)
    location.append(np.array(data['ground_truth']['location']))
    yaws.append(np.array(data['ground_truth']['yaw']))
    steer.append(np.array(data['action_input']['steer']))
    linear_velocity.append(np.array(data['action_input']['linear_velocity']))

location = np.array(location)
yaws = np.array(yaws)
actions = np.stack([steer, linear_velocity], axis=2)
# %%
errors = []
for pos, yaw in zip(location[:1000], yaws[:1000]):
    dis = np.arange(9) * 0.25
    pred_pos = pos[0, :2] + np.expand_dims(dis, axis=1) * np.array([np.cos(yaw[0]), np.sin(yaw[0])])
    plt.plot(pos[:,0], pos[:,1])
    plt.plot(pred_pos[:,0], pred_pos[:,1])
    errors.append(np.linalg.norm(pos[:,:2] - pred_pos, axis=1))
# %%
print(np.mean(errors))

# %%
plt.scatter(location[:,:,0].flatten(), location[:,:,1].flatten())
# %%
for action in tqdm(actions[:2000]):
    plt.plot(action[:, 0], action[:, 1], 'r*')
plt.xlabel('steer')
plt.ylabel('linear_velocity')

# %% 
