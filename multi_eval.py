import torch
import torch.nn as nn
import torchvision.models as tv_models

# import the skrl components to build the RL system
from isaaclab.utils.dict import print_dict
from skrl.multi_agents.torch.mappo import MAPPO, MAPPO_DEFAULT_CONFIG
from skrl.multi_agents.torch.ippo import IPPO, IPPO_DEFAULT_CONFIG
from skrl.envs.loaders.torch import load_isaaclab_env
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveLR
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed
from skrl.utils.spaces.torch import unflatten_tensorized_space

import os
import numpy as np
import cv2
import gymnasium as gym
from gymnasium.spaces import Box
from datetime import datetime


# seed for reproducibility
set_seed(42)  # e.g. `set_seed(42)` for fixed seed


# define shared model (stochastic and deterministic models) using mixins
class GaussianModel(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device):
        Model.__init__(self, observation_space, action_space, device)
        # GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)
        GaussianMixin.__init__(
            self,
            clip_actions=False,
            clip_log_std=True,
            min_log_std=-20.0,
            max_log_std=2.0,
            reduction="sum",
        )

        # self.rnn = nn.RNN(input_size=13, hidden_size=128, num_layers=1, batch_first=True)
        # self.net_container = nn.Sequential(
        #     nn.Linear(in_features=149, out_features=256),
        #     nn.PReLU(),
        #     nn.Linear(in_features=256, out_features=256),
        #     nn.PReLU(),
        # )

        self.net_container = nn.Sequential(
            nn.Linear(in_features=86, out_features=256),
            nn.PReLU(),
            nn.Linear(in_features=256, out_features=128),
            nn.PReLU(),
        )


        # self.rnn_other = nn.RNN(input_size=13, hidden_size=128, num_layers=1, batch_first=True)
        # self.net_container_other = nn.Sequential(
        #     nn.Linear(in_features=142, out_features=128),
        #     nn.PReLU(),
        # )

        self.net_container_other = nn.Sequential(
            nn.Linear(in_features=79, out_features=256),
            nn.PReLU(),
            nn.Linear(in_features=256, out_features=128),
            nn.PReLU(),
        )

        # self.net_container_concat = nn.Sequential(
        #     nn.Linear(in_features=384, out_features=256),
        #     nn.PReLU(),
        #     nn.Linear(in_features=256, out_features=128),
        #     nn.PReLU(),
        #     nn.Linear(in_features=128, out_features=64),
        #     nn.PReLU(),
        # )

        self.net_container_concat = nn.Sequential(
            nn.Linear(in_features=256, out_features=256),
            nn.PReLU(),
            nn.Linear(in_features=256, out_features=128),
            nn.PReLU(),
        )

        self.policy_action_layer = nn.Linear(in_features=128, out_features=self.num_actions)
        self.log_std_parameter = nn.Parameter(torch.full(size=(self.num_actions,), fill_value=0.0), requires_grad=True)
        
    def compute(self, inputs, role=""):
        states = unflatten_tensorized_space(self.observation_space, inputs.get("states"))
        taken_actions = unflatten_tensorized_space(self.action_space, inputs.get("taken_actions"))
        # out, _ = self.rnn(torch.cat([states['joint'], states['actions']], dim=-1))
        # output = self.net_container(torch.cat([out[:, -1, :], states['object'][:, -1, :], states['goal'][:, -1, :]], dim=-1))
        flat_joints = states['joint'].view(states['joint'].size(0), -1)
        flat_actions = states['actions'].view(states['actions'].size(0), -1)
        output = self.net_container(torch.cat([flat_joints, flat_actions, states['object'][:, -1, :], states['goal'][:, -1, :]], dim=-1))
        # out_other, _ = self.rnn_other(torch.cat([states['joint_other'], states['actions_other']], dim=-1))
        # output_other = self.net_container_other(torch.cat([out_other[:, -1, :], states['object_other'][:, -1, :]], dim=-1))
        flat_joints_other = states['joint_other'].view(states['joint_other'].size(0), -1)
        flat_actions_other = states['actions_other'].view(states['actions_other'].size(0), -1)
        output_other = self.net_container_other(torch.cat([flat_joints_other, flat_actions_other, states['object_other'][:, -1, :]], dim=-1))
        output = self.net_container_concat(torch.cat([output, output_other], dim=-1))
        mu = self.policy_action_layer(output)
        mu = nn.functional.tanh(mu)
        
        return mu, self.log_std_parameter, {}

class DeterministicModel(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions=False)

        # self.rnn = nn.RNN(input_size=61, hidden_size=512, num_layers=2, batch_first=True)
        
        # self.net_container = nn.Sequential(
        #     nn.Linear(in_features=512, out_features=256),
        #     nn.PReLU(),
        #     nn.Linear(in_features=256, out_features=256),
        #     nn.PReLU(),
        #     nn.Linear(in_features=256, out_features=128),
        #     nn.PReLU(),
        #     nn.Linear(in_features=128, out_features=64),
        #     nn.PReLU(),
        # )

        self.net_container = nn.Sequential(
            nn.Linear(in_features=305, out_features=512),
            nn.PReLU(),
            nn.Linear(in_features=512, out_features=512),
            nn.PReLU(),
            nn.Linear(in_features=512, out_features=256),
            nn.PReLU(),
            nn.Linear(in_features=256, out_features=128),
            nn.PReLU(),
        )
        

        self.value_layer = nn.Linear(in_features=128, out_features=1)

    def compute(self, inputs, role=""):
        states = unflatten_tensorized_space(self.observation_space, inputs.get("states"))
        taken_actions = unflatten_tensorized_space(self.action_space, inputs.get("taken_actions"))
        # out, _ = self.rnn(torch.cat([states['joint'], states['actions'], states['object'], states['goal'], states['joint_other'], states['actions_other'], states['object_other']], dim=-1))
        # output = self.net_container(out[:, -1, :])
        combined_seq = torch.cat([
            states['joint'], 
            states['actions'], 
            states['object'], 
            states['goal'], 
            states['joint_other'], 
            states['actions_other'], 
            states['object_other']
        ], dim=-1)
        flattened_input = combined_seq.view(combined_seq.size(0), -1)
        output = self.net_container(flattened_input)
        output = self.value_layer(output)
        
        return output, {}

# load and wrap the Isaac Lab environment
env = load_isaaclab_env(task_name="Isaac-Turtlebot3-Multi-Direct-v0")
env = wrap_env(env)

device = env.device

# instantiate a memory as rollout buffer (any memory can be used for this)
memories = {}
memories["robot_1"] = RandomMemory(memory_size=24, num_envs=env.num_envs, device=device)
memories["robot_2"] = RandomMemory(memory_size=24, num_envs=env.num_envs, device=device)

# instantiate the agent's models (function approximators).
# PPO requires 2 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#models

models = {}
models["robot_1"] = {}
models["robot_1"]["policy"] = GaussianModel(env.observation_spaces["robot_1"], env.action_spaces["robot_1"], device)
# models["robot_1"]["value"] = DeterministicModel(env.state_space("robot_1"), env.action_spaces["robot_1"], device)
models["robot_1"]["value"] = DeterministicModel(env.observation_spaces["robot_1"], env.action_spaces["robot_1"], device)
models["robot_2"] = {}
models["robot_2"]["policy"] = GaussianModel(env.observation_spaces["robot_2"], env.action_spaces["robot_2"], device)
# models["robot_2"]["value"] = DeterministicModel(env.state_space("robot_2"), env.action_spaces["robot_2"], device)
models["robot_2"]["value"] = DeterministicModel(env.observation_spaces["robot_2"], env.action_spaces["robot_2"], device)

# configure and instantiate the agent (visit its documentation to see all the options)
# https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#configuration-and-hyperparameters
# cfg = MAPPO_DEFAULT_CONFIG.copy()
cfg = IPPO_DEFAULT_CONFIG.copy()
cfg["random_timesteps"] = 0
cfg["state_preprocessor"] = None
cfg["state_preprocessor_kwargs"] = {}
cfg["value_preprocessor"] = RunningStandardScaler
cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}
# logging to TensorBoard and write checkpoints (in timesteps)

agent = IPPO(possible_agents=env.possible_agents,
            models=models,
            memories=memories,
            cfg=cfg,
            observation_spaces=env.observation_spaces,
            action_spaces=env.action_spaces,
            device=device)

# configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 700000, "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

# ---------------------------------------------------------
# comment the code above: `trainer.train()`, and...
# uncomment the following lines to evaluate a trained agent
# ---------------------------------------------------------
# from skrl.utils.huggingface import download_model_from_huggingface

# download the trained agent's checkpoint from Hugging Face Hub and load it
# path = "/home/takenami/sim2real_ros2/runs/torch/Isaac-Turtlebot3-Multi-Image-Direct-v0/25-11-23_12-26-06-973047_IPPO/checkpoints/agent_288000.pt"
# path = "/home/takenami/sim2real_ros2/runs/torch/Isaac-Turtlebot3-Multi-Image-Direct-v0/26-01-10_00-39-48-025307_IPPO/checkpoints/agent_294800.pt"
path = "/home/takenami/sim2real_ros2/runs/torch/Isaac-Turtlebot3-Multi-Image-Direct-v0/26-01-12_13-52-49-095216_IPPO/checkpoints/agent_205100.pt"
# path = "/home/takenami/sim2real_ros2/runs/torch/Isaac-Turtlebot3-Multi-Image-Direct-v0/26-01-16_15-58-56-043259_IPPO/checkpoints/agent_53900.pt"
# path = "/home/takenami/sim2real_ros2/runs/torch/Isaac-Turtlebot3-Multi-Image-Direct-v0/26-01-16_21-22-17-325664_IPPO/checkpoints/agent_164000.pt"
# path = "/home/takenami/sim2real_ros2/runs/torch/Isaac-Turtlebot3-Multi-Image-Direct-v0/26-01-16_10-55-52-743738_IPPO/checkpoints/agent_11300.pt"
agent.load(path)

# start evaluation
trainer.eval()

