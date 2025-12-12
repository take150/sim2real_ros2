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


# seed for reproducibility
set_seed(42)  # e.g. `set_seed(42)` for fixed seed


# define shared model (stochastic and deterministic models) using mixins
class DeterministicStudentModel(DeterministicMixin, Model):
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
            nn.LazyLinear(out_features=512),
            nn.PReLU(),
        )

        self.rnn = nn.RNN(input_size=13, hidden_size=128, num_layers=1, batch_first=True)
        self.net_container = nn.Sequential(
            nn.LazyLinear(out_features=256),
            nn.PReLU(),
            nn.LazyLinear(out_features=256),
            nn.PReLU(),
        )

        self.rnn_other = nn.RNN(input_size=13, hidden_size=128, num_layers=1, batch_first=True)
        self.net_container_other = nn.Sequential(
            nn.LazyLinear(out_features=128),
            nn.PReLU(),
        )

        self.net_container_concat = nn.Sequential(
            nn.LazyLinear(out_features=256),
            nn.PReLU(),
            nn.LazyLinear(out_features=128),
            nn.PReLU(),
            nn.LazyLinear(out_features=64),
            nn.PReLU(),
        )

        self.policy_action_layer = nn.LazyLinear(out_features=self.num_actions)
        self.log_std_parameter = nn.Parameter(torch.full(size=(self.num_actions,), fill_value=0.0), requires_grad=True)
        
    def compute(self, inputs, role=""):
        states = unflatten_tensorized_space(self.observation_space, inputs.get("states"))
        taken_actions = unflatten_tensorized_space(self.action_space, inputs.get("taken_actions"))
        features_extractor_rgb = self.features_extractor_rgb_container(torch.permute(states['rgb'], (0, 3, 1, 2)))
        out, _ = self.rnn(torch.cat([states['joint'], states['actions']], dim=-1))
        output = self.net_container(torch.cat([features_extractor_rgb, out[:, -1, :]], dim=-1))
        out_other, _ = self.rnn_other(torch.cat([states['joint_other'], states['actions_other']], dim=-1))
        output_other = self.net_container_other(out_other[:, -1, :])
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

        self.rnn = nn.RNN(input_size=13, hidden_size=128, num_layers=1, batch_first=True)
        self.net_container = nn.Sequential(
            nn.LazyLinear(out_features=256),
            nn.PReLU(),
            nn.LazyLinear(out_features=256),
            nn.PReLU(),
        )

        self.rnn_other = nn.RNN(input_size=13, hidden_size=128, num_layers=1, batch_first=True)
        self.net_container_other = nn.Sequential(
            nn.LazyLinear(out_features=128),
            nn.PReLU(),
        )

        self.net_container_concat = nn.Sequential(
            nn.LazyLinear(out_features=256),
            nn.PReLU(),
            nn.LazyLinear(out_features=128),
            nn.PReLU(),
            nn.LazyLinear(out_features=64),
            nn.PReLU(),
        )

        self.policy_action_layer = nn.LazyLinear(out_features=self.num_actions)
        self.log_std_parameter = nn.Parameter(torch.full(size=(self.num_actions,), fill_value=0.0), requires_grad=True)
        
    def compute(self, inputs, role=""):
        states = unflatten_tensorized_space(self.observation_space, inputs.get("states"))
        taken_actions = unflatten_tensorized_space(self.action_space, inputs.get("taken_actions"))
        out, _ = self.rnn(torch.cat([states['joint'], states['actions']], dim=-1))
        output = self.net_container(torch.cat([out[:, -1, :], states['object'][:, -1, :], states['goal'][:, -1, :]], dim=-1))
        out_other, _ = self.rnn_other(torch.cat([states['joint_other'], states['actions_other']], dim=-1))
        output_other = self.net_container_other(torch.cat([out_other[:, -1, :], states['object_other'][:, -1, :]], dim=-1))
        output = self.net_container_concat(torch.cat([output, output_other], dim=-1))
        output = self.policy_action_layer(output)
        output = nn.functional.tanh(output)
        
        return output, {}

class DeterministicModel(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions=False)

        self.rnn = nn.RNN(input_size=61, hidden_size=512, num_layers=2, batch_first=True)
        
        self.net_container = nn.Sequential(
            nn.LazyLinear(out_features=256),
            nn.PReLU(),
            nn.LazyLinear(out_features=256),
            nn.PReLU(),
            nn.LazyLinear(out_features=128),
            nn.PReLU(),
            nn.LazyLinear(out_features=64),
            nn.PReLU(),
        )

        self.value_layer = nn.LazyLinear(out_features=1)

    def compute(self, inputs, role=""):
        states = unflatten_tensorized_space(self.observation_space, inputs.get("states"))
        taken_actions = unflatten_tensorized_space(self.action_space, inputs.get("taken_actions"))
        out, _ = self.rnn(torch.cat([states['joint'], states['actions'], states['object'], states['goal'], states['joint_other'], states['actions_other'], states['object_other']], dim=-1))
        output = self.net_container(out[:, -1, :])
        output = self.value_layer(output)
        
        return output, {}

# load and wrap the Isaac Lab environment
env = load_isaaclab_env(task_name="Isaac-Turtlebot3-Multi-Distillation-Direct-v0")
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
student_cfg["mini_batches"] = 6
student_cfg["learning_rate"] = 1.0e-03
# student_cfg["learning_rate"] = 0.0000030332
# student_cfg["learning_rate"] = 1.0e-06
# student_cfg["learning_rate_scheduler"] = KLAdaptiveLR
student_cfg["learning_rate_scheduler"] = LinearLR
# student_cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.01, "kl_factor": 2, "min_lr": 1.0e-06, "max_lr": 5.0e-04, "lr_factor": 1.1}
student_cfg["learning_rate_scheduler_kwargs"] = {"start_factor": 1.0, "end_factor": 0.005, "total_iters": 1500}
student_cfg["grad_norm_clip"] = 1.0
student_cfg["state_preprocessor"] = None
student_cfg["state_preprocessor_kwargs"] = {}
# logging to TensorBoard and write checkpoints (in timesteps)
student_cfg["experiment"]["write_interval"] = 1
student_cfg["experiment"]["checkpoint_interval"] = 100
student_cfg["experiment"]["directory"] = "runs/torch/Isaac-Turtlebot3-Multi-Image-Direct-v0"

student_cfg["experiment"]["wandb"] = False
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
teacher_models["robot_1"]["value"] = DeterministicModel(env.observation_spaces["robot_1"], env.action_spaces["robot_1"], device)
teacher_models["robot_2"] = {}
teacher_models["robot_2"]["policy"] = DeterministicTeacherModel(env.observation_spaces["robot_2"], env.action_spaces["robot_2"], device)
teacher_models["robot_2"]["value"] = DeterministicModel(env.observation_spaces["robot_2"], env.action_spaces["robot_2"], device)

teacher_agent = MultiDistillationTeacher(possible_agents=env.possible_agents,
            models=teacher_models,
            memories=memories,
            observation_spaces=env.observation_spaces,
            action_spaces=env.action_spaces,
            device=device)

# configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 30000, "headless": True}
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
# path = "/home/takenami/sim2real_ros2/runs/torch/Isaac-Turtlebot3-Multi-Image-Direct-v0/25-06-26_13-55-07-436154_IPPO/checkpoints/agent_411000.pt"
# path = "/home/takenami/sim2real_ros2/runs/torch/Isaac-Turtlebot3-Multi-Image-Direct-v0/25-11-11_13-20-43-064217_MAPPO/checkpoints/agent_5000.pt"
path = "/home/takenami/sim2real_ros2/runs/torch/Isaac-Turtlebot3-Multi-Image-Direct-v0/25-11-27_13-34-48-552097_IPPO/checkpoints/agent_898800.pt"
teacher_agent.load(path)

# start training
trainer.distillation(teacher_agent=teacher_agent)

# start evaluation
# trainer.eval()

