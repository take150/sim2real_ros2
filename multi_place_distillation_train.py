import torch
import torch.nn as nn
import torchvision.models as tv_models
from torch.optim.lr_scheduler import LinearLR

# import the skrl components to build the RL system
from skrl.multi_agents.torch.distillation.multi_distillation_student import MultiDistillationStudent, DISTILLATION_DEFAULT_CONFIG
from skrl.multi_agents.torch.distillation.multi_distillation_teacher import MultiDistillationTeacher
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
class DeterministicStudentModel(DeterministicMixin, Model, OrthogonalInitMixin):
    def __init__(self, observation_space, action_space, device):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(
            self,
            clip_actions=False,
        )

        self.features_extractor_rgb_container = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=7, stride=4, padding=2),
            nn.PReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.PReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.PReLU(),
            nn.Flatten(),
            nn.Linear(in_features=8192,out_features=512),
            nn.PReLU(),
        )

        self.net_container = nn.Sequential(
            nn.Linear(in_features=525, out_features=256),
            nn.PReLU(),
            nn.Linear(in_features=256, out_features=128),
            nn.PReLU(),
        )

        self.net_container_other = nn.Sequential(
            nn.Linear(in_features=13, out_features=256),
            nn.PReLU(),
            nn.Linear(in_features=256, out_features=128),
            nn.PReLU(),
        )

        self.net_container_concat = nn.Sequential(
            nn.Linear(in_features=256, out_features=256),
            nn.PReLU(),
            nn.Linear(in_features=256, out_features=128),
            nn.PReLU(),
        )

        self.policy_action_layer = nn.Linear(in_features=128, out_features=self.num_actions)

        self.apply_orthogonal_init({
            self.features_extractor_rgb_container: np.sqrt(2),
            self.net_container: np.sqrt(2),
            self.net_container_other: np.sqrt(2),
            self.net_container_concat: np.sqrt(2),
            self.policy_action_layer: 0.01
        })

        
    def compute(self, inputs, role=""):
        states = unflatten_tensorized_space(self.observation_space, inputs.get("states"))
        taken_actions = unflatten_tensorized_space(self.action_space, inputs.get("taken_actions"))
        features_extractor_rgb = self.features_extractor_rgb_container(torch.permute(states['rgb'], (0, 3, 1, 2)))
        # flat_joints = states['joint'].view(states['joint'].size(0), -1)
        # flat_actions = states['actions'].view(states['actions'].size(0), -1)
        # output = self.net_container(torch.cat([features_extractor_rgb, flat_joints, flat_actions], dim=-1))
        output = self.net_container(torch.cat([features_extractor_rgb, states['joint'][:, -1, :], states['actions'][:, -1, :]], dim=-1))
        # flat_joints_other = states['joint_other'].view(states['joint_other'].size(0), -1)
        # flat_actions_other = states['actions_other'].view(states['actions_other'].size(0), -1)    
        # output_other = self.net_container_other(torch.cat([flat_joints_other, flat_actions_other], dim=-1))
        output_other = self.net_container_other(torch.cat([states['joint_other'][:, -1, :], states['actions_other'][:, -1, :]], dim=-1))
        output = self.net_container_concat(torch.cat([output, output_other], dim=-1))
        
        output = self.policy_action_layer(output)
        output = nn.functional.tanh(output)
        
        return output, {}

class DeterministicTeacherModel(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(
            self,
            clip_actions=False,
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
        
    def compute(self, inputs, role=""):
        states = unflatten_tensorized_space(self.observation_space, inputs.get("states"))
        taken_actions = unflatten_tensorized_space(self.action_space, inputs.get("taken_actions"))
        task_ids = states['task_id'].squeeze(-1)
        batch_size = task_ids.shape[0]
        is_grasp = (task_ids == 0)
        is_move = (task_ids == 1)
        is_place = (task_ids == 2)
        output = torch.zeros((batch_size, self.num_actions), device=self.device)
        # log_std_parameter = torch.zeros((batch_size, self.num_actions), device=self.device)

        if is_grasp.any():
            net_grasp = self.net_container(torch.cat([states['joint'][:, -1, :][is_grasp], states['actions'][:, -1, :][is_grasp], states['object'][:, -1, :][is_grasp]], dim=-1))
            mu_grasp = self.policy_layer(net_grasp)
            output[is_grasp] = mu_grasp
            # log_std_parameter[is_grasp] = self.log_std_parameter

        if is_move.any():
            net_move = self.net_container_move(torch.cat([states['joint'][:, -1, :][is_move], states['actions'][:, -1, :][is_move], states['object'][:, -1, :][is_move], states['goal'][:, -1, :][is_move]], dim=-1))
            mu_move = self.move_policy_layer(net_move)
            output[is_move] = mu_move 
            # log_std_parameter[is_move] = self.move_log_std_parameter

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
            # log_std_parameter[is_place] = self.place_log_std_parameter
        
        output = nn.functional.tanh(output)
        
        return output, {}

# load and wrap the Isaac Lab environment
env = load_isaaclab_env(task_name="Isaac-Turtlebot3-Multi-Place-Distillation-Direct-v0")
env = wrap_env(env)

device = env.device

# instantiate a memory as rollout buffer (any memory can be used for this)
memories = {}
memories["robot_1"] = RandomMemory(memory_size=24, num_envs=env.num_envs, device=device)
memories["robot_2"] = RandomMemory(memory_size=24, num_envs=env.num_envs, device=device)

# instantiate the agent's models (function approximators).
# PPO requires 2 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#models

student_models = {}
student_models["robot_1"] = {}
student_models["robot_1"]["policy"] = DeterministicStudentModel(env.observation_spaces["robot_1"], env.action_spaces["robot_1"], device)
student_models["robot_2"] = {}
student_models["robot_2"]["policy"] = DeterministicStudentModel(env.observation_spaces["robot_2"], env.action_spaces["robot_2"], device)

# configure and instantiate the agent (visit its documentation to see all the options)
# https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#configuration-and-hyperparameters
# cfg = MAPPO_DEFAULT_CONFIG.copy()
student_cfg = DISTILLATION_DEFAULT_CONFIG.copy()
student_cfg = DISTILLATION_DEFAULT_CONFIG.copy()
student_cfg["rollouts"] = 24  # memory_size
student_cfg["learning_epochs"] = 8
student_cfg["mini_batches"] = 96
student_cfg["learning_rate"] = 5.0e-06
# student_cfg["learning_rate"] = 0.0000030332
# student_cfg["learning_rate"] = 1.0e-06
# student_cfg["learning_rate_scheduler"] = KLAdaptiveLR
student_cfg["learning_rate_scheduler"] = LinearLR
student_cfg["learning_rate_scheduler_kwargs"] = {"start_factor": 1.0, "end_factor": 0.1, "total_iters": 1000}
student_cfg["grad_norm_clip"] = 1.0
student_cfg["state_preprocessor"] = None
student_cfg["state_preprocessor_kwargs"] = {}
# logging to TensorBoard and write checkpoints (in timesteps)
student_cfg["experiment"]["write_interval"] = 1
student_cfg["experiment"]["checkpoint_interval"] = 120
student_cfg["experiment"]["directory"] = "runs/torch/Isaac-Turtlebot3-Multi-Image-Direct-v0"

student_cfg["experiment"]["wandb"] = True
student_cfg["experiment"]["wandb_kwargs"]["project"] = "multi_cube_grasp_project"

student_agent = MultiDistillationStudent(possible_agents=env.possible_agents,
            models=student_models,
            memories=memories,
            cfg=student_cfg,
            observation_spaces=env.observation_spaces,
            action_spaces=env.action_spaces,
            device=device)

teacher_models = {}
teacher_models["robot_1"] = {}
teacher_models["robot_1"]["policy"] = DeterministicTeacherModel(env.observation_spaces["robot_1"], env.action_spaces["robot_1"], device)
teacher_models["robot_2"] = {}
teacher_models["robot_2"]["policy"] = DeterministicTeacherModel(env.observation_spaces["robot_2"], env.action_spaces["robot_2"], device)

teacher_agent = MultiDistillationTeacher(possible_agents=env.possible_agents,
            models=teacher_models,
            memories=memories,
            observation_spaces=env.observation_spaces,
            action_spaces=env.action_spaces,
            device=device)

# configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 50000, "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=student_agent)

# ---------------------------------------------------------
# comment the code above: `trainer.train()`, and...
# uncomment the following lines to evaluate a trained agent
# ---------------------------------------------------------
# from skrl.utils.huggingface import download_model_from_huggingface

# download the trained agent's checkpoint from Hugging Face Hub and load it
# path = "/home/takenami/sim2real_ros2/runs/torch/Isaac-Turtlebot3-Image-Direct-v0/25-05-08_13-47-18-171591_PPO/checkpoints/best_agent.pt"
# path = "/home/takenami/sim2real_ros2/runs/torch/Isaac-Turtlebot3-Multi-Image-Direct-v0/25-06-24_13-50-14-495506_IPPO/checkpoints/best_agent.pt"
# path = "/home/takenami/sim2real_ros2/runs/torch/Isaac-Turtlebot3-Multi-Image-Direct-v0/25-06-24_18-19-07-597924_IPPO/checkpoints/agent_77000.pt"
teacher_path = "/home/takenami/sim2real_ros2/runs/torch/Isaac-Turtlebot3-Multi-Image-Place-Direct-v0/26-01-21_12-01-57-254643_IPPO/checkpoints/agent_10080.pt"
path = "/home/takenami/sim2real_ros2/runs/torch/Isaac-Turtlebot3-Multi-Image-Direct-v0/26-01-21_13-12-06-858108_MultiDistillationStudent/checkpoints/best_agent.pt"
# path = "/home/takenami/sim2real_ros2/runs/torch/Isaac-Turtlebot3-Multi-Image-Direct-v0/25-11-27_13-34-48-552097_IPPO/checkpoints/agent_898800.pt"
# path = "/home/takenami/Desktop/agent_52000.pt"
# path = "/home/takenami/Desktop/agent_2720.pt"

teacher_agent.load(teacher_path)
student_agent.load(path)

# start training
trainer.distillation(teacher_agent=teacher_agent)

# start evaluation
# trainer.eval()

