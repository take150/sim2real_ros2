import torch
import torch.nn as nn

# import the skrl components to build the RL system
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.envs.loaders.torch import load_isaaclab_env
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveLR
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed
from skrl.utils.spaces.torch import unflatten_tensorized_space

import cv2
import numpy as np


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
        self.rnn = nn.RNN(input_size=13, hidden_size=64, num_layers=1, batch_first=True)
        self.features_extractor_joints_container = nn.Sequential(
            nn.LazyLinear(out_features=64),
            nn.PReLU(),
        )
        self.net_container = nn.Sequential(
            nn.LazyLinear(out_features=256),
            nn.PReLU(),
            nn.LazyLinear(out_features=256),
            nn.PReLU(),
        )
        self.policy_layer = nn.LazyLinear(out_features=self.num_actions)
        self.log_std_parameter = nn.Parameter(torch.full(size=(self.num_actions,), fill_value=0.0), requires_grad=True)
        
    def compute(self, inputs, role=""):
        states = unflatten_tensorized_space(self.observation_space, inputs.get("states"))
        # image_np = (states["rgb"][0] * 255.0).cpu().numpy().astype(np.uint8)  # RGB形式
        # # RGBからBGRに変換（OpenCVの表示用）
        # image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        # # 画像を拡大。ここでは2倍に拡大する例。
        # scale_factor = 10.0
        # image_np = cv2.resize(image_np, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)

        # # ウィンドウを表示（WINDOW_NORMALでウィンドウサイズの変更を可能に）
        # cv2.namedWindow('Camera Feed', cv2.WINDOW_NORMAL)

        # # 画像をリアルタイムで表示
        # cv2.imshow('Camera Feed', image_np)

        # # 'q'キーが押されたらウィンドウを閉じる処理
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     cv2.destroyAllWindows()
        #     exit()
        taken_actions = unflatten_tensorized_space(self.action_space, inputs.get("taken_actions"))
        features_extractor_rgb = self.features_extractor_rgb_container(torch.permute(states['rgb'], (0, 3, 1, 2)))
        out, _ = self.rnn(torch.cat([states['joint_list'], states['actions_list']], dim=-1))
        features_extractor_joints = self.features_extractor_joints_container(out[:, -1, :])
        # features_extractor_joints = self.features_extractor_joints_container(torch.cat([states['joint'], states['actions']], dim=-1))
        net = self.net_container(torch.cat([features_extractor_joints, features_extractor_rgb], dim=1))
        # self._shared_output = net
        output = self.policy_layer(net)
        output = nn.functional.tanh(output)
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
        output = self.net_container(torch.cat([states['joint'], states['actions'], states['object']], dim=-1))
        output = self.value_layer(output)
        return output, {}

    
class SharedModel(GaussianMixin,DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(
            self,
            clip_actions=False,
            clip_log_std=True,
            min_log_std=-20.0,
            max_log_std=2.0,
            reduction="sum",
            role="policy",
        )
        DeterministicMixin.__init__(self, clip_actions=False, role="value")

        self.features_extractor_rgb_container = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=7, stride=3, padding=2),
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
        self.net_container = nn.Sequential(
            nn.LazyLinear(out_features=256),
            nn.PReLU(),
            nn.LazyLinear(out_features=256),
            nn.PReLU(),
        )
        self.policy_layer = nn.LazyLinear(out_features=self.num_actions)
        self.log_std_parameter = nn.Parameter(torch.full(size=(self.num_actions,), fill_value=0.0), requires_grad=True)
        self.value_layer = nn.LazyLinear(out_features=1)

    def act(self, inputs, role):
        if role == "policy":
            return GaussianMixin.act(self, inputs, role)
        elif role == "value":
            return DeterministicMixin.act(self, inputs, role)
    
    def compute(self, inputs, role=""):
        if role == "policy":
            states = unflatten_tensorized_space(self.observation_space, inputs.get("states"))
            # states["rgb"] = torch.randn_like(states["rgb"])
            # カメラデータからRGB画像を取り出し、CPU上に移動し、numpy配列に変換（uint8型）
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
            features_extractor_joints = self.features_extractor_joints_container(states['joint'])
            net = self.net_container(torch.cat([features_extractor_joints, features_extractor_rgb], dim=1))
            self._shared_output = net
            output = self.policy_layer(net)
            output = nn.functional.tanh(output)
            return output, self.log_std_parameter, {}
        elif role == "value":
            if self._shared_output is None:
                states = unflatten_tensorized_space(self.observation_space, inputs.get("states"))
                taken_actions = unflatten_tensorized_space(self.action_space, inputs.get("taken_actions"))
                features_extractor_rgb = self.features_extractor_rgb_container(torch.permute(states['rgb'], (0, 3, 1, 2)))
                features_extractor_joints = self.features_extractor_joints_container(states['joint'])
                net = self.net_container(torch.cat([features_extractor_joints, features_extractor_rgb], dim=1))
                shared_output = net
            else:
                shared_output = self._shared_output
            self._shared_output = None
            output = self.value_layer(shared_output)
            return output, {}
        
# load and wrap the Isaac Lab environment
env = load_isaaclab_env(task_name="Isaac-Turtlebot3-Image-Direct-v6")

env = wrap_env(env)

device = env.device

# instantiate the agent's models (function approximators).
# PPO requires 2 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#models
models = {}
# models["policy"] = SharedModel(env.observation_space, env.action_space, device)
# models["value"] = models["policy"]  # same instance: shared model

models["policy"] = GaussianModel(env.observation_space, env.action_space, device)
models["value"] = DeterministicModel(env.observation_space, env.action_space, device)

# configure and instantiate the agent (visit its documentation to see all the options)
# https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#configuration-and-hyperparameters
cfg = PPO_DEFAULT_CONFIG.copy()
cfg["random_timesteps"] = 0
cfg["state_preprocessor"] = None
cfg["state_preprocessor_kwargs"] = {}
cfg["value_preprocessor"] = RunningStandardScaler
cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}
# logging to TensorBoard and write checkpoints (in timesteps)
cfg["experiment"]["write_interval"] = 400
cfg["experiment"]["checkpoint_interval"] = 16000
cfg["experiment"]["directory"] = "runs/torch/Isaac-Turtlebot3-Image-Direct-v0"

agent = PPO(models=models,
            memory=None,
            cfg=cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device)


# configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 96000, "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

# start training
# trainer.train()


# ---------------------------------------------------------
# comment the code above: `trainer.train()`, and...
# uncomment the following lines to evaluate a trained agent
# ---------------------------------------------------------
# from skrl.utils.huggingface import download_model_from_huggingface

# download the trained agent's checkpoint from Hugging Face Hub and load it
# path = "/home/takenami/sim2real_ros2/runs/torch/Isaac-Turtlebot3-Image-Direct-v0/25-06-20_19-08-42-465010_PPO/checkpoints/agent_416000.pt"
path = "/home/takenami/sim2real_ros2/runs/torch/Isaac-Turtlebot3-Image-Direct-v0/25-06-23_01-30-13-007985_PPO/checkpoints/agent_147000.pt"
path = "/home/takenami/sim2real_ros2/runs/torch/Isaac-Turtlebot3-Image-Direct-v0/25-06-23_10-20-24-136950_PPO/checkpoints/agent_50000.pt"
path = "/home/takenami/sim2real_ros2/runs/torch/Isaac-Turtlebot3-Image-Direct-v0/25-06-25_11-49-23-003009_PPO/checkpoints/best_agent.pt"
path = "/home/takenami/sim2real_ros2/runs/torch/Isaac-Turtlebot3-Image-Direct-v0/25-11-21_22-34-02-870951_PPO/checkpoints/best_agent.pt"
# path = "/home/takenami/sim2real_ros2/runs/torch/Isaac-Turtlebot3-Image-Direct-v0/25-07-12_14-06-27-579949_PPO/checkpoints/best_agent.pt"

agent.load(path)

# start evaluation
trainer.eval()
