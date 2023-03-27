# %%
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from tqdm import tqdm
import pickle
import torch
from torch import nn, tensor
import random
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
random_idx = random.randint(0, location.shape[0])
count = 0
for pos, yaw in tqdm(zip(location[:random_idx], yaws[:random_idx])):
    dis = np.arange(9) * 0.25
    pred_pos = pos[0, :2] + np.expand_dims(dis, axis=1) * np.array([np.cos(yaw[0]), np.sin(yaw[0])])
    if count == random_idx - 1:
        plt.plot(pos[:,0], pos[:,1], label='ground truth')
        plt.plot(pred_pos[:,0], pred_pos[:,1], label='prediction')
        print("this error is", np.linalg.norm(pos[:,:2] - pred_pos, axis=1).mean())
    errors.append(np.linalg.norm(pos[:,:2] - pred_pos, axis=1))
    count += 1
plt.legend()
print(np.mean(errors))

# %%
plt.scatter(location[:,:,0].flatten(), location[:,:,1].flatten())
# %%
for action in tqdm(actions):
    plt.plot(action[:, 0], action[:, 1], 'r*')
plt.xlabel('steer')
plt.ylabel('linear_velocity')

# %%
cov = np.cov(actions[:, :, 0].flatten(), actions[:, :, 1].flatten())
print(cov)
print(np.linalg.eig(cov))

# %%
