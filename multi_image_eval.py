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
        self.features_extractor_joints_container = nn.Sequential(
            nn.LazyLinear(out_features=64),
            nn.PReLU(),
        )
        # self.features_extractor_joints_other_container = nn.Sequential(
        #     nn.LazyLinear(out_features=64),
        #     nn.PReLU(),
        # )
        # self.net_container = nn.Sequential(
        #     nn.LazyLinear(out_features=256),
        #     nn.PReLU(),
        #     nn.LazyLinear(out_features=256),
        #     nn.PReLU(),
        #     nn.LazyLinear(out_features=128),
        #     nn.PReLU(),
        #     nn.LazyLinear(out_features=64),
        #     nn.PReLU(),
        # )

        self.net_container_1 = nn.Sequential(
            nn.LazyLinear(out_features=256),
            nn.PReLU(),
            nn.LazyLinear(out_features=256),
            nn.PReLU(),
        )

        # self.features_extractor_other_container = nn.Sequential(
        #     nn.LazyLinear(out_features=128),
        #     nn.PReLU(),
        #     nn.LazyLinear(out_features=64),
        #     nn.PReLU(),
        # )

        # self.features_gate_layer = nn.Sequential(
        #     nn.LazyLinear(256),
        #     nn.Sigmoid(),
        # )

        # self.cross_attn = nn.MultiheadAttention(
        #     embed_dim=256, num_heads=4, batch_first=True
        # )

        # self.net_container_2 = nn.Sequential(
        #     nn.LazyLinear(out_features=256),
        #     nn.PReLU(),
        #     nn.LazyLinear(out_features=128),
        #     nn.PReLU(),
        # )

        self.policy_action_layer = nn.LazyLinear(out_features=self.num_actions)
        self.log_std_parameter = nn.Parameter(torch.full(size=(self.num_actions,), fill_value=0.0), requires_grad=True)
        
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
        taken_actions = unflatten_tensorized_space(self.action_space, inputs.get("taken_actions"))
        features_extractor_rgb = self.features_extractor_rgb_container(torch.permute(states['rgb'], (0, 3, 1, 2)))
        features_extractor_joints = self.features_extractor_joints_container(torch.cat([states['joint'], states['actions']], dim=-1))
        features = self.net_container_1(torch.cat([features_extractor_rgb, features_extractor_joints], dim=1))
        # other_features = self.features_extractor_other_container(states['z_actor'])
        # output = self.features_gate_layer(torch.cat([features, states['z_actor']], dim=1))
        # output = output * states['z_actor']
        # query = features.unsqueeze(1)
        # key   = states["z_actor"].unsqueeze(1)
        # value = key
        # context, _ = self.cross_attn(query, key, value, need_weights=False)
        # context = context.squeeze(1)
        # output = self.net_container_2(torch.cat([features, context], dim=1))
        # output = self.net_container_2(torch.cat([features, output], dim=1))
        # output = self.net_container_2(torch.cat([features, other_features], dim=1))
        # features_extractor_joints_other = self.features_extractor_joints_other_container(torch.cat([states['joint_other'], states['actions_other']], dim=-1))
        # output = self.net_container_2(torch.cat([features, states['z_actor']], dim=1))
        # output = self.net_container_2(torch.cat([features, features_extractor_joints_other], dim=1))
        # output = self.net_container_2(features)
        # output = self.policy_action_layer(output)
        output = self.policy_action_layer(features)
        output = nn.functional.tanh(output)
        # return output, self.log_std_parameter, {"features": features}
        return output, self.log_std_parameter, {}

class GaussianModel_(GaussianMixin, Model):
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
        self.features_extractor_joints_container = nn.Sequential(
            nn.LazyLinear(out_features=64),
            nn.PReLU(),
        )
        # self.features_extractor_joints_other_container = nn.Sequential(
        #     nn.LazyLinear(out_features=64),
        #     nn.PReLU(),
        # )
        # self.net_container = nn.Sequential(
        #     nn.LazyLinear(out_features=256),
        #     nn.PReLU(),
        #     nn.LazyLinear(out_features=256),
        #     nn.PReLU(),
        #     nn.LazyLinear(out_features=128),
        #     nn.PReLU(),
        #     nn.LazyLinear(out_features=64),
        #     nn.PReLU(),
        # )

        self.net_container_1 = nn.Sequential(
            nn.LazyLinear(out_features=256),
            nn.PReLU(),
            nn.LazyLinear(out_features=256),
            nn.PReLU(),
        )

        # self.features_extractor_other_container = nn.Sequential(
        #     nn.LazyLinear(out_features=128),
        #     nn.PReLU(),
        #     nn.LazyLinear(out_features=64),
        #     nn.PReLU(),
        # )

        # self.features_gate_layer = nn.Sequential(
        #     nn.LazyLinear(256),
        #     nn.Sigmoid(),
        # )

        # self.cross_attn = nn.MultiheadAttention(
        #     embed_dim=256, num_heads=4, batch_first=True
        # )

        # self.net_container_2 = nn.Sequential(
        #     nn.LazyLinear(out_features=256),
        #     nn.PReLU(),
        #     nn.LazyLinear(out_features=128),
        #     nn.PReLU(),
        # )

        self.policy_action_layer = nn.LazyLinear(out_features=self.num_actions)
        self.log_std_parameter = nn.Parameter(torch.full(size=(self.num_actions,), fill_value=0.0), requires_grad=True)
        
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
        taken_actions = unflatten_tensorized_space(self.action_space, inputs.get("taken_actions"))
        features_extractor_rgb = self.features_extractor_rgb_container(torch.permute(states['rgb'], (0, 3, 1, 2)))
        features_extractor_joints = self.features_extractor_joints_container(torch.cat([states['joint'], states['actions']], dim=-1))
        features = self.net_container_1(torch.cat([features_extractor_rgb, features_extractor_joints], dim=1))
        # other_features = self.features_extractor_other_container(states['z_actor'])
        # output = self.features_gate_layer(torch.cat([features, states['z_actor']], dim=1))
        # output = output * states['z_actor']
        # query = features.unsqueeze(1)
        # key   = states["z_actor"].unsqueeze(1)
        # value = key
        # context, _ = self.cross_attn(query, key, value, need_weights=False)
        # context = context.squeeze(1)
        # output = self.net_container_2(torch.cat([features, context], dim=1))
        # output = self.net_container_2(torch.cat([features, output], dim=1))
        # output = self.net_container_2(torch.cat([features, other_features], dim=1))
        # features_extractor_joints_other = self.features_extractor_joints_other_container(torch.cat([states['joint_other'], states['actions_other']], dim=-1))
        # output = self.net_container_2(torch.cat([features, states['z_actor']], dim=1))
        # output = self.net_container_2(torch.cat([features, features_extractor_joints_other], dim=1))
        # output = self.net_container_2(features)
        # output = self.policy_action_layer(output)
        output = self.policy_action_layer(features)
        output = nn.functional.tanh(output)
        # return output, self.log_std_parameter, {"features": features}
        return output, self.log_std_parameter, {}
    

class DeterministicModel(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions=False)

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
        # output = self.net_container(torch.cat([states['actions'], states['object']], dim=-1))
        # output = self.net_container(torch.cat([states['joint'], states['actions'], states['object'], states['z_critic']], dim=-1))
        # output = self.net_container(torch.cat([states['joint'], states['actions'], states['object']], dim=-1))
        output = self.net_container(states)        
        output = self.value_layer(output)
        return output, {}

# load and wrap the Isaac Lab environment
env = load_isaaclab_env(task_name="Isaac-Turtlebot3-Multi-Image-Direct-v0")
# video_kwargs = {
#             "video_folder": os.path.join(os.getcwd(), datetime.now().strftime("%Y%m%d_%H%M%S"), "videos", "train"),
#             "step_trigger": lambda step: step % 200 == 0,
#             "video_length": 100,
#             "disable_logger": True,
#         }
# print("[INFO] Recording videos during training.")
# print_dict(video_kwargs, nesting=4)
# env = gym.wrappers.RecordVideo(env, **video_kwargs)
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
models["robot_1"]["value"] = DeterministicModel(env.state_space("robot_1"), env.action_spaces["robot_1"], device)
# models["robot_1"]["value"] = DeterministicModel(env.observation_spaces["robot_1"], env.action_spaces["robot_1"], device)
models["robot_2"] = {}
models["robot_2"]["policy"] = GaussianModel_(env.observation_spaces["robot_2"], env.action_spaces["robot_2"], device)
models["robot_2"]["value"] = DeterministicModel(env.state_space("robot_2"), env.action_spaces["robot_2"], device)
# models["robot_2"]["value"] = DeterministicModel(env.observation_spaces["robot_2"], env.action_spaces["robot_2"], device)

# configure and instantiate the agent (visit its documentation to see all the options)
# https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#configuration-and-hyperparameters
cfg = MAPPO_DEFAULT_CONFIG.copy()
# cfg = IPPO_DEFAULT_CONFIG.copy()
cfg["random_timesteps"] = 0
cfg["state_preprocessor"] = None
cfg["state_preprocessor_kwargs"] = {}
cfg["value_preprocessor"] = RunningStandardScaler
cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}
# logging to TensorBoard and write checkpoints (in timesteps)
cfg["experiment"]["write_interval"] = 1
cfg["experiment"]["checkpoint_interval"] = 1000
cfg["experiment"]["directory"] = "runs/torch/Isaac-Turtlebot3-Multi-Image-Direct-v0"

agent = MAPPO(possible_agents=env.possible_agents,
            models=models,
            memories=memories,
            cfg=cfg,
            observation_spaces=env.observation_spaces,
            action_spaces=env.action_spaces,
            device=device,
            shared_observation_spaces=env.state_spaces)

# agent = IPPO(possible_agents=env.possible_agents,
#             models=models,
#             memories=memories,
#             cfg=cfg,
#             observation_spaces=env.observation_spaces,
#             action_spaces=env.action_spaces,
#             device=device)

# configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 700000, "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

# ---------------------------------------------------------
# comment the code above: `trainer.train()`, and...
# uncomment the following lines to evaluate a trained agent
# ---------------------------------------------------------
# from skrl.utils.huggingface import download_model_from_huggingface

# download the trained agent's checkpoint from Hugging Face Hub and load it
# path = "/home/takenami/sim2real_ros2/runs/torch/Isaac-Turtlebot3-Multi-Image-Direct-v0/25-07-08_12-38-45-981554_MAPPO/checkpoints/best_agent.pt"
path = "/home/takenami/sim2real_ros2/runs/torch/Isaac-Turtlebot3-Multi-Image-Direct-v0/25-11-12_16-42-31-720567_MAPPO/checkpoints/best_agent.pt"
# path = "/home/takenami/sim2real_ros2/runs/torch/Isaac-Turtlebot3-Multi-Image-Direct-v0/25-10-08_13-52-49-845551_MAPPO/checkpoints/best_agent.pt"
# path = "/home/takenami/sim2real_ros2/runs/torch/Isaac-Turtlebot3-Multi-Image-Direct-v0/25-11-05_14-05-22-244343_MAPPO/checkpoints/agent_349000.pt"
agent.load(path)

# start evaluation
trainer.eval()

