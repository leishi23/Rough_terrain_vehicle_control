import numpy as np
import torch

class MPPI(object):
    
    def __init__(self, model, action_dim, N, horizon, dt, device):
        self.model = model
        self.action_dim = action_dim
        self.N = N
        self.horizon = horizon
        self.dt = dt
        self.device = device
        
    def cost(self, model_outputs, actions, goal):
        #### collision cost
        collision_model_outputs = model_outputs[:, :, 3]     # [position x, position y, position z, collision]
        clamp_value = 0.02
        collision_model_outputs = torch.clamp(collision_model_outputs, min=clamp_value, max=1.-clamp_value)
        collision_model_outputs = (collision_model_outputs - clamp_value)/(1.-2.*clamp_value)
        cost_collision = collision_model_outputs

        #### distance cost
        position_model_outputs = model_outputs[:, :, 0:3]    # [position x, position y, position z, collision]
        position_goal = goal.position
        dot_product = torch.sum(position_model_outputs * position_goal, dim=2)
        position_model_outputs_norm = torch.linalg.norm(position_model_outputs, dim=2)
        position_goal_norm = torch.linalg.norm(position_goal, dim=2)
        cos_theta = dot_product / torch.max(position_model_outputs_norm * position_goal_norm, torch.tensor(1e-6).to(self.device))
        cos_theta = torch.clamp(cos_theta, min=-1.+1e-4, max=1.-1e-4)
        theta = torch.acos(cos_theta)
        theta = (1. / np.pi) * torch.abs(theta)
        cost_position = torch.nn.Sigmoid()(goal.cost_weights.position_sigmoid_scale * (theta - goal.cost_weights.position_sigmoid_center))
        
        #### magnitude action cost
        angular_velocity = actions[:, :, 1]
        cost_action_magnitude = torch.square(angular_velocity)
        
        #### smooth action cost (TBD)
        cost_action_smooth = torch.cat([torch.square(angular_velocity[:, 1:] - angular_velocity[:, :-1]), torch.zeros(angular_velocity.shape[0], 1).to(self.device)], dim=1)
        
        total_cost = goal.cost_weights.collision * cost_collision + \
                        goal.cost_weights.position * cost_position + \
                        goal.cost_weights.action_magnitude * cost_action_magnitude + \
                        goal.cost_weights.action_smooth * cost_action_smooth
                        
        return total_cost