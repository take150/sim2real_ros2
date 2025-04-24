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


# seed for reproducibility
set_seed(42)  # e.g. `set_seed(42)` for fixed seed

# define shared model (stochastic and deterministic models) using mixins
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

        # mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
        # std  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)
        # self.register_buffer("rgb_mean", mean)
        # self.register_buffer("rgb_std",  std)

        # self.preprocess = tv_models.ResNet34_Weights.DEFAULT.transforms()

        # self.features_extractor_resnet = nn.Sequential(
        #     *list(tv_models.resnet34(
        #         weights=tv_models.ResNet34_Weights.DEFAULT
        #     ).children())[:-2]
        # ).to(device)

        # for param in self.features_extractor_resnet.parameters():
        #     param.requires_grad_(False)

        # for m in self.features_extractor_resnet.modules():
        #     if isinstance(m, nn.BatchNorm2d):
        #         m.eval()
        #         # BN の weight/bias も更新不要に
        #         m.weight.requires_grad_(False)
        #         m.bias.requires_grad_(False)

        self.features_extractor_rgb_container = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=7, stride=3, padding=2),
            nn.PReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.PReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.PReLU(),
            # nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
            # nn.Conv2d(in_channels=3, out_channels=32, kernel_size=7, stride=3, padding=3),
            # nn.ReLU(),
            # nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=2),
            # nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=2, padding=3),
            # nn.ReLU(),
            # nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            # nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2),
            # nn.ReLU(),
            # nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1),
            # nn.ReLU(),
            # nn.Conv2d(in_channels=512, out_channels=128, kernel_size=1, stride=1, padding=0),
            # nn.ReLU(),
            nn.Flatten(),
            nn.LazyLinear(out_features=512),
            nn.PReLU(),
            # nn.LazyLinear(out_features=512),
            # nn.ReLU(),
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
            taken_actions = unflatten_tensorized_space(self.action_space, inputs.get("taken_actions"))
            # with torch.no_grad():
            #     rgb = states["rgb"].permute(0,3,1,2).to(self.device)
            #     rgb = self.preprocess(rgb)
            #     rgb = self.features_extractor_resnet(rgb)
            # rgb = states["rgb"].permute(0,3,1,2).float().to(self.device) / 255.0
            # rgb = (rgb - self.rgb_mean) / self.rgb_std
            features_extractor_rgb = self.features_extractor_rgb_container(torch.permute(states['rgb'], (0, 3, 1, 2)))
            # features_extractor_rgb = self.features_extractor_rgb_container(states["rgb"])
            features_extractor_joints = self.features_extractor_joints_container(states['joint'])
            net = self.net_container(torch.cat([features_extractor_joints, features_extractor_rgb], dim=1))
            # features_extractor_rgb = self.features_extractor_rgb_container(torch.permute(states['rgb'], (0, 3, 1, 2)))
            # net = self.net_container(torch.cat([states['joint'], features_extractor_rgb], dim=1))
            self._shared_output = net
            output = self.policy_layer(net)
            output = nn.functional.tanh(output)
            return output, self.log_std_parameter, {}
        elif role == "value":
            if self._shared_output is None:
                states = unflatten_tensorized_space(self.observation_space, inputs.get("states"))
                taken_actions = unflatten_tensorized_space(self.action_space, inputs.get("taken_actions"))
                # with torch.no_grad():
                #     rgb = states["rgb"].permute(0,3,1,2).to(self.device)
                #     rgb = self.preprocess(rgb)
                #     rgb = self.features_extractor_resnet(rgb)
                # rgb = states["rgb"].permute(0,3,1,2).float().to(self.device) / 255.0
                # rgb = (rgb - self.rgb_mean) / self.rgb_std
                features_extractor_rgb = self.features_extractor_rgb_container(torch.permute(states['rgb'], (0, 3, 1, 2)))
                # features_extractor_rgb = self.features_extractor_rgb_container(states["rgb"])
                features_extractor_joints = self.features_extractor_joints_container(states['joint'])
                net = self.net_container(torch.cat([features_extractor_joints, features_extractor_rgb], dim=1))
                # features_extractor_rgb = self.features_extractor_rgb_container(torch.permute(states['rgb'], (0, 3, 1, 2)))
                # net = self.net_container(torch.cat([states['joint'], features_extractor_rgb], dim=1))
                shared_output = net
            else:
                shared_output = self._shared_output
            self._shared_output = None
            output = self.value_layer(shared_output)
            return output, {}
        
# load and wrap the Isaac Lab environment
env = load_isaaclab_env(task_name="Isaac-Turtlebot3-Image-Direct-v4")

env = wrap_env(env)

device = env.device

# instantiate the agent's models (function approximators).
# PPO requires 2 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#models
models = {}
models["policy"] = SharedModel(env.observation_space, env.action_space, device)
models["value"] = models["policy"]  # same instance: shared model

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
# path = "/home/takenami/sim2real_ros2/runs/torch/Isaac-Turtlebot3-Image-Direct-v0/red_normal/checkpoints/best_agent.pt"
# path = "/home/takenami/sim2real_ros2/runs/torch/Isaac-Turtlebot3-Image-Direct-v0/3colors_normal/checkpoints/best_agent.pt"
# path = "/home/takenami/sim2real_ros2/runs/torch/Isaac-Turtlebot3-Image-Direct-v0/red_background_rotation_conv_32732_64321_128321_linear_512_64_256_256_v0/checkpoints/best_agent.pt"
# path = "/home/takenami/sim2real_ros2/runs/torch/Isaac-Turtlebot3-Image-Direct-v0/25-04-22_09-13-27-572855_PPO/checkpoints/agent_415000.pt"
path = "/home/takenami/sim2real_ros2/runs/torch/Isaac-Turtlebot3-Image-Direct-v0/25-04-22_22-58-15-205327_PPO/checkpoints/agent_330000.pt"

agent.load(path)

# start evaluation
trainer.eval()
