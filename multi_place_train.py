import torch
import torch.nn as nn
import torchvision.models as tv_models

# import the skrl components to build the RL system
from isaaclab.utils.dict import print_dict
from skrl.multi_agents.torch.mappo import MAPPO, MAPPO_DEFAULT_CONFIG, MAPPO_SHARED
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

import numpy as np
from functools import partial

# seed for reproducibility
set_seed(42)  # e.g. `set_seed(42)` for fixed seed

class OrthogonalInitMixin:
    def apply_orthogonal_init(self, module_gains: dict):
        for module, gain in module_gains.items():
            # self.init_weights は staticmethod なのでクラスから参照可能
            module.apply(partial(self._ortho_weights, gain=gain))

    @staticmethod
    def _ortho_weights(module: nn.Module, gain: float = 1):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(module.weight, gain=gain)
            if module.bias is not None:
                module.bias.data.fill_(0.0)
        elif isinstance(module, (nn.RNN, nn.LSTM, nn.GRU)):
            for name, param in module.named_parameters():
                if 'weight_ih' in name:
                    nn.init.orthogonal_(param.data, gain=gain)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data, gain=gain)
                elif 'bias' in name:
                    param.data.fill_(0.0)


# define shared model (stochastic and deterministic models) using mixins
class GaussianModel(GaussianMixin, Model, OrthogonalInitMixin):
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

        self.net_container = nn.Sequential(
            nn.Linear(in_features=27, out_features=256),
            nn.PReLU(),
            nn.Linear(in_features=256, out_features=256),
            nn.PReLU(),
            nn.Linear(in_features=256, out_features=128),
            nn.PReLU(),
            nn.Linear(in_features=128, out_features=64),
            nn.PReLU(),
        )

        self.policy_layer = nn.Linear(in_features=64, out_features=self.num_actions)
        self.log_std_parameter = nn.Parameter(torch.full(size=(self.num_actions,), fill_value=0.0), requires_grad=True)
        
        for param in self.net_container.parameters():
            param.requires_grad = False    
        for param in self.policy_layer.parameters():
            param.requires_grad = False
        self.log_std_parameter.requires_grad = False

        self.net_container_move = nn.Sequential(
            nn.Linear(in_features=34, out_features=256),
            nn.PReLU(),
            nn.Linear(in_features=256, out_features=256),
            nn.PReLU(),
            nn.Linear(in_features=256, out_features=128),
            nn.PReLU(),
            nn.Linear(in_features=128, out_features=64),
            nn.PReLU(),
        )

        self.move_policy_layer = nn.Linear(in_features=64, out_features=self.num_actions)
        self.move_log_std_parameter = nn.Parameter(torch.full(size=(self.num_actions,), fill_value=0.0), requires_grad=True)
        
        for param in self.net_container_move.parameters():
            param.requires_grad = False    
        for param in self.move_policy_layer.parameters():
            param.requires_grad = False
        self.move_log_std_parameter.requires_grad = False

        self.net_container_place = nn.Sequential(
            nn.Linear(in_features=86, out_features=256),
            nn.PReLU(),
            nn.Linear(in_features=256, out_features=128),
            nn.PReLU(),
        )

        self.net_container_other_place = nn.Sequential(
            nn.Linear(in_features=79, out_features=256),
            nn.PReLU(),
            nn.Linear(in_features=256, out_features=128),
            nn.PReLU(),
        )

        self.net_container_concat_place = nn.Sequential(
            nn.Linear(in_features=256, out_features=256),
            nn.PReLU(),
            nn.Linear(in_features=256, out_features=128),
            nn.PReLU(),
        )

        self.place_policy_layer = nn.Linear(in_features=128, out_features=self.num_actions)
        self.place_log_std_parameter = nn.Parameter(torch.full(size=(self.num_actions,), fill_value=0.0), requires_grad=True)
        
        self.apply_orthogonal_init({
            self.net_container: np.sqrt(2),
            self.net_container_move: np.sqrt(2),
            self.net_container_place: np.sqrt(2),
            self.net_container_other_place: np.sqrt(2),
            self.net_container_concat_place: np.sqrt(2),
            self.policy_layer: 0.01,  # Policy出力は0.01
            self.move_policy_layer: 0.01,
            self.place_policy_layer: 0.01,
        })
        
    def compute(self, inputs, role=""):
        states = unflatten_tensorized_space(self.observation_space, inputs.get("states"))
        taken_actions = unflatten_tensorized_space(self.action_space, inputs.get("taken_actions"))
        task_ids = states['task_id'].squeeze(-1)
        batch_size = task_ids.shape[0]
        is_grasp = (task_ids == 0)
        is_move = (task_ids == 1)
        is_place = (task_ids == 2)
        output = torch.zeros((batch_size, self.num_actions), device=self.device)
        log_std_parameter = torch.zeros((batch_size, self.num_actions), device=self.device)

        if is_grasp.any():
            net_grasp = self.net_container(torch.cat([states['joint'][:, -1, :][is_grasp], states['actions'][:, -1, :][is_grasp], states['object'][:, -1, :][is_grasp]], dim=-1))
            mu_grasp = self.policy_layer(net_grasp)
            output[is_grasp] = mu_grasp
            log_std_parameter[is_grasp] = self.log_std_parameter

        if is_move.any():
            net_move = self.net_container_move(torch.cat([states['joint'][:, -1, :][is_move], states['actions'][:, -1, :][is_move], states['object'][:, -1, :][is_move], states['goal'][:, -1, :][is_move]], dim=-1))
            mu_move = self.move_policy_layer(net_move)
            output[is_move] = mu_move 
            log_std_parameter[is_move] = self.move_log_std_parameter

        if is_place.any():
            flat_joints = states['joint'][is_place].view(states['joint'][is_place].size(0), -1)
            flat_actions = states['actions'][is_place].view(states['actions'][is_place].size(0), -1)
            net_place = self.net_container_place(torch.cat([flat_joints, flat_actions, states['object'][:, -1, :][is_place], states['goal'][:, -1, :][is_place]], dim=-1))
            flat_joints_other = states['joint_other'][is_place].view(states['joint_other'][is_place].size(0), -1)
            flat_actions_other = states['actions_other'][is_place].view(states['actions_other'][is_place].size(0), -1)
            net_other_place = self.net_container_other_place(torch.cat([flat_joints_other, flat_actions_other, states['object_other'][:, -1, :][is_place]], dim=-1))
            net_concat_place = self.net_container_concat_place(torch.cat([net_place, net_other_place], dim=-1))
            mu_place = self.place_policy_layer(net_concat_place)

            output[is_place] = mu_place
            log_std_parameter[is_place] = self.place_log_std_parameter
        
        output = nn.functional.tanh(output)
        
        return output, log_std_parameter, {}

class DeterministicModel(DeterministicMixin, Model, OrthogonalInitMixin):
    def __init__(self, observation_space, action_space, device):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions=False)

        self.net_container = nn.Sequential(
            nn.Linear(in_features=27, out_features=256),
            nn.PReLU(),
            nn.Linear(in_features=256, out_features=256),
            nn.PReLU(),
            nn.Linear(in_features=256, out_features=128),
            nn.PReLU(),
            nn.Linear(in_features=128, out_features=64),
            nn.PReLU(),
        )
        self.value_layer = nn.Linear(in_features=64, out_features=1)

        self.net_container_move = nn.Sequential(
            nn.Linear(in_features=34, out_features=256),
            nn.PReLU(),
            nn.Linear(in_features=256, out_features=256),
            nn.PReLU(),
            nn.Linear(in_features=256, out_features=128),
            nn.PReLU(),
            nn.Linear(in_features=128, out_features=64),
            nn.PReLU(),
        )

        self.move_value_layer = nn.Linear(in_features=64, out_features=1)

        self.net_container_place = nn.Sequential(
            nn.Linear(in_features=305, out_features=512),
            nn.PReLU(),
            nn.Linear(in_features=512, out_features=512),
            nn.PReLU(),
            nn.Linear(in_features=512, out_features=256),
            nn.PReLU(),
            nn.Linear(in_features=256, out_features=128),
            nn.PReLU(),
        )

        self.place_value_layer = nn.Linear(in_features=128, out_features=1)

        self.apply_orthogonal_init({
            self.net_container: np.sqrt(2),
            self.net_container_move: np.sqrt(2),
            self.net_container_place: np.sqrt(2),
            self.value_layer: 1.0,
            self.move_value_layer: 1.0,
            self.place_value_layer: 1.0,
        })
    
    def compute(self, inputs, role=""):
        states = unflatten_tensorized_space(self.observation_space, inputs.get("states"))
        taken_actions = unflatten_tensorized_space(self.action_space, inputs.get("taken_actions"))

        task_ids = states['task_id'].squeeze(-1)
        batch_size = task_ids.shape[0]
        is_grasp = (task_ids == 0)
        is_move = (task_ids == 1)
        is_place = (task_ids == 2)
        output = torch.zeros((batch_size, 1), device=self.device)

        if is_grasp.any():
            net_grasp = self.net_container(torch.cat([states['joint'][:, -1, :][is_grasp], states['actions'][:, -1, :][is_grasp], states['object'][:, -1, :][is_grasp]], dim=-1))
            grasp_value = self.value_layer(net_grasp)
            output[is_grasp] = grasp_value

        if is_move.any():
            net_move = self.net_container_move(torch.cat([states['joint'][:, -1, :][is_move], states['actions'][:, -1, :][is_move], states['object'][:, -1, :][is_move], states['goal'][:, -1, :][is_move]], dim=-1))
            move_value = self.move_value_layer(net_move)
            output[is_move] = move_value

        if is_place.any():
            combined_seq = torch.cat([
                states['joint'][is_place], 
                states['actions'][is_place], 
                states['object'][is_place], 
                states['goal'][is_place], 
                states['joint_other'][is_place], 
                states['actions_other'][is_place], 
                states['object_other'][is_place]
            ], dim=-1)
            flattened_input = combined_seq.view(combined_seq.size(0), -1)
            net_place = self.net_container_place(flattened_input)
            place_value = self.place_value_layer(net_place)
            output[is_place] = place_value

        return output, {}

# load and wrap the Isaac Lab environment
env = load_isaaclab_env(task_name="Isaac-Turtlebot3-Multi-Place-Direct-v0")
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
cfg["rollouts"] = 24  # memory_size
cfg["learning_epochs"] = 8
cfg["mini_batches"] = 24
cfg["discount_factor"] = 0.97
cfg["lambda"] = 0.92
cfg["learning_rate"] = 5.0e-04
cfg["learning_rate_scheduler"] = KLAdaptiveLR
cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.01, "kl_factor": 2, "min_lr": 1.0e-06, "max_lr": 5.0e-04, "lr_factor": 1.1}
cfg["random_timesteps"] = 0
cfg["learning_starts"] = 0
cfg["grad_norm_clip"] = 1.0
cfg["ratio_clip"] = 0.2
cfg["value_clip"] = 0.2
cfg["clip_predicted_values"] = True
cfg["entropy_loss_scale"] = 0.005
cfg["value_loss_scale"] = 2.0
cfg["kl_threshold"] = 0.0
cfg["rewards_shaper_scale"] = 0.01
cfg["time_limit_bootstrap"] = False
cfg["state_preprocessor"] = None
cfg["state_preprocessor_kwargs"] = {}
cfg["value_preprocessor"] = RunningStandardScaler
cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}
# logging to TensorBoard and write checkpoints (in timesteps)
cfg["experiment"]["write_interval"] = 1
cfg["experiment"]["checkpoint_interval"] = 120
cfg["experiment"]["directory"] = "runs/torch/Isaac-Turtlebot3-Multi-Image-Place-Direct-v0"

cfg["experiment"]["wandb"] = True
cfg["experiment"]["wandb_kwargs"]["project"] = "multi_cube_place_project"

# agent = MAPPO(possible_agents=env.possible_agents,
#             models=models,
#             memories=memories,
#             cfg=cfg,
#             observation_spaces=env.observation_spaces,
#             action_spaces=env.action_spaces,
#             device=device,
#             shared_observation_spaces=env.state_spaces)

agent = IPPO(possible_agents=env.possible_agents,
            models=models,
            memories=memories,
            cfg=cfg,
            observation_spaces=env.observation_spaces,
            action_spaces=env.action_spaces,
            device=device)


# configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 1000000, "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

# ---------------------------------------------------------
# comment the code above: `trainer.train()`, and...
# uncomment the following lines to evaluate a trained agent
# ---------------------------------------------------------
# from skrl.utils.huggingface import download_model_from_huggingface

# download the trained agent's checkpoint from Hugging Face Hub and load it
# path = "/home/takenami/sim2real_ros2/runs/torch/Isaac-Turtlebot3-Image-Direct-v0/25-05-08_13-47-18-171591_PPO/checkpoints/best_agent.pt"
# path = "/home/takenami/sim2real_ros2/runs/torch/Isaac-Turtlebot3-Multi-Image-Direct-v0/25-06-24_13-50-14-495506_IPPO/checkpoints/best_agent.pt"
# path = "/home/takenami/sim2real_ros2/runs/torch/Isaac-Turtlebot3-Multi-Image-Direct-v0/25-06-24_18-19-07-597924_IPPO/checkpoints/agent_77000.pt"
path = "/home/takenami/sim2real_ros2/runs/torch/Isaac-Turtlebot3-Multi-Image-Direct-v0/25-11-23_01-40-05-968627_IPPO/checkpoints/agent_50000.pt"
# path = "/home/takenami/sim2real_ros2/runs/torch/Isaac-Turtlebot3-Multi-Image-Direct-v0/25-11-11_13-20-43-064217_MAPPO/checkpoints/agent_5000.pt"
# path = "/home/takenami/sim2real_ros2/runs/torch/Isaac-Turtlebot3-Multi-Image-Direct-v0/25-11-11_00-57-02-969356_MAPPO/checkpoints/agent_140000.pt"
path = "/home/takenami/sim2real_ros2/runs/torch/Isaac-Turtlebot3-Multi-Image-Direct-v0/25-11-25_23-05-11-156770_IPPO/checkpoints/agent_311000.pt"
path = "/home/takenami/sim2real_ros2/runs/torch/Isaac-Turtlebot3-Image-Place-Direct-v0/25-12-08_23-05-26-844061_PPO/checkpoints/best_agent.pt"
path = "/home/takenami/sim2real_ros2/runs/torch/Isaac-Turtlebot3-Image-Place-Direct-v0/25-12-27_18-23-49-348286_PPO/checkpoints/agent_32000.pt"
path = "/home/takenami/sim2real_ros2/runs/torch/Isaac-Turtlebot3-Image-Place-Direct-v0/26-01-18_03-04-57-147328_PPO/checkpoints/agent_88600.pt"
# path = "/home/takenami/sim2real_ros2/runs/torch/Isaac-Turtlebot3-Multi-Image-Place-Direct-v0/26-01-18_23-46-41-272596_IPPO/checkpoints/agent_22080.pt"
path = "/home/takenami/sim2real_ros2/runs/torch/Isaac-Turtlebot3-Image-Place-Direct-v0/26-01-18_17-46-44-960092_PPO/checkpoints/agent_61800.pt"
# path = "/home/takenami/sim2real_ros2/runs/torch/Isaac-Turtlebot3-Multi-Image-Place-Direct-v0/26-01-19_12-15-10-671093_IPPO/checkpoints/agent_38080.pt"
path = "/home/takenami/sim2real_ros2/runs/torch/Isaac-Turtlebot3-Image-Place-Direct-v0/26-01-20_20-02-37-056309_PPO/checkpoints/agent_41640.pt"
path = "/home/takenami/sim2real_ros2/runs/torch/Isaac-Turtlebot3-Image-Place-Direct-v0/26-01-21_02-46-41-220243_PPO/checkpoints/agent_12960.pt"
path = "/home/takenami/sim2real_ros2/runs/torch/Isaac-Turtlebot3-Multi-Image-Place-Direct-v0/26-01-21_05-42-08-638027_IPPO/checkpoints/agent_59360.pt"

agent.load(path)

# start training
trainer.train()

# start evaluation
# trainer.eval()

