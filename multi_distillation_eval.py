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
        
    def compute(self, inputs, role=""):
        states = unflatten_tensorized_space(self.observation_space, inputs.get("states"))
        image_np = (states["rgb"][0] * 255.0).cpu().numpy().astype(np.uint8)  # RGB形式
        # RGBからBGRに変換（OpenCVの表示用）
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        # 画像を拡大。ここでは2倍に拡大する例。
        scale_factor = 10.0
        image_np = cv2.resize(image_np, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)

        # ウィンドウを表示（WINDOW_NORMALでウィンドウサイズの変更を可能に）
        cv2.namedWindow('Camera Feed', cv2.WINDOW_NORMAL)

        # 画像をリアルタイムで表示
        cv2.imshow('Camera Feed', image_np)

        # 'q'キーが押されたらウィンドウを閉じる処理
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            exit()
        features_extractor_rgb = self.features_extractor_rgb_container(torch.permute(states['rgb'], (0, 3, 1, 2)))
        # out, _ = self.rnn(torch.cat([states['joint'], states['actions']], dim=-1))
        # output = self.net_container(torch.cat([features_extractor_rgb, out[:, -1, :]], dim=-1))
        # flat_joints = states['joint'].view(states['joint'].size(0), -1)
        # flat_actions = states['actions'].view(states['actions'].size(0), -1)
        # output = self.net_container(torch.cat([features_extractor_rgb, flat_joints, flat_actions], dim=-1))
        output = self.net_container(torch.cat([features_extractor_rgb, states['joint'][:, -1, :], states['actions'][:, -1, :]], dim=-1))
        # out_other, _ = self.rnn_other(torch.cat([states['joint_other'], states['actions_other']], dim=-1))
        # output_other = self.net_container_other(out_other[:, -1, :])
        # flat_joints_other = states['joint_other'].view(states['joint_other'].size(0), -1)
        # flat_actions_other = states['actions_other'].view(states['actions_other'].size(0), -1)
        # output_other = self.net_container_other(torch.cat([flat_joints_other, flat_actions_other], dim=-1))
        output_other = self.net_container_other(torch.cat([states['joint_other'][:, -1, :], states['actions_other'][:, -1, :]], dim=-1))
        output = self.net_container_concat(torch.cat([output, output_other], dim=-1))
        
        output = self.policy_action_layer(output)
        output = nn.functional.tanh(output)
        
        return output, {}

class DeterministicStudentModel_(DeterministicMixin, Model):
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
        
    def compute(self, inputs, role=""):
        states = unflatten_tensorized_space(self.observation_space, inputs.get("states"))
        image_np = (states["rgb"][0] * 255.0).cpu().numpy().astype(np.uint8)  # RGB形式
        # RGBからBGRに変換（OpenCVの表示用）
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        # 画像を拡大。ここでは2倍に拡大する例。
        scale_factor = 10.0
        image_np = cv2.resize(image_np, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)

        # ウィンドウを表示（WINDOW_NORMALでウィンドウサイズの変更を可能に）
        cv2.namedWindow('Camera Feed_', cv2.WINDOW_NORMAL)

        # 画像をリアルタイムで表示
        cv2.imshow('Camera Feed_', image_np)

        # 'q'キーが押されたらウィンドウを閉じる処理
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            exit()
        features_extractor_rgb = self.features_extractor_rgb_container(torch.permute(states['rgb'], (0, 3, 1, 2)))
        # out, _ = self.rnn(torch.cat([states['joint'], states['actions']], dim=-1))
        # output = self.net_container(torch.cat([features_extractor_rgb, out[:, -1, :]], dim=-1))
        # flat_joints = states['joint'].view(states['joint'].size(0), -1)
        # flat_actions = states['actions'].view(states['actions'].size(0), -1)
        # output = self.net_container(torch.cat([features_extractor_rgb, flat_joints, flat_actions], dim=-1))
        output = self.net_container(torch.cat([features_extractor_rgb, states['joint'][:, -1, :], states['actions'][:, -1, :]], dim=-1))
        # out_other, _ = self.rnn_other(torch.cat([states['joint_other'], states['actions_other']], dim=-1))
        # output_other = self.net_container_other(out_other[:, -1, :])
        # flat_joints_other = states['joint_other'].view(states['joint_other'].size(0), -1)
        # flat_actions_other = states['actions_other'].view(states['actions_other'].size(0), -1)
        # output_other = self.net_container_other(torch.cat([flat_joints_other, flat_actions_other], dim=-1))
        output_other = self.net_container_other(torch.cat([states['joint_other'][:, -1, :], states['actions_other'][:, -1, :]], dim=-1))
        output = self.net_container_concat(torch.cat([output, output_other], dim=-1))
        
        output = self.policy_action_layer(output)
        output = nn.functional.tanh(output)
        
        return output, {}

# load and wrap the Isaac Lab environment
# from multi_env import Turtlebot3Image
# from image_log_env import Turtlebot3Image
# env = Turtlebot3Image()
env = load_isaaclab_env(task_name="Isaac-Turtlebot3-Multi-Place-Distillation-Direct-v0")
env = wrap_env(env)
# env = wrap_env(env=env, wrapper="isaaclab-multi-agent")

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
student_models["robot_2"]["policy"] = DeterministicStudentModel_(env.observation_spaces["robot_2"], env.action_spaces["robot_2"], device)

# configure and instantiate the agent (visit its documentation to see all the options)
# https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#configuration-and-hyperparameters
# cfg = MAPPO_DEFAULT_CONFIG.copy()
student_cfg = DISTILLATION_DEFAULT_CONFIG.copy()

student_agent = MultiDistillationStudent(possible_agents=env.possible_agents,
            models=student_models,
            memories=memories,
            cfg=student_cfg,
            observation_spaces=env.observation_spaces,
            action_spaces=env.action_spaces,
            device=device)

# configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 1000000, "headless": True}
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
path = "/home/takenami/sim2real_ros2/runs/torch/Isaac-Turtlebot3-Multi-Image-Direct-v0/26-01-16_12-41-42-178899_MultiDistillationStudent/checkpoints/agent_14000.pt"
# path = "/home/takenami/sim2real_ros2/runs/torch/Isaac-Turtlebot3-Multi-Image-Direct-v0/26-01-19_21-54-33-360376_MultiDistillationStudent/checkpoints/agent_24960.pt"
path = "/home/takenami/sim2real_ros2/runs/torch/Isaac-Turtlebot3-Multi-Image-Direct-v0/26-01-21_13-12-06-858108_MultiDistillationStudent/checkpoints/agent_14400.pt"
# path = "/home/takenami/Desktop/agent_49800.pt"
student_agent.load(path)

# start training
trainer.eval()

# start evaluation
# trainer.eval()

