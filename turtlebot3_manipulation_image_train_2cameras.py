import torch
import torch.nn as nn
import torchvision.models as tv_models

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
        )

        # self.features_extractor_rgb_container_2 = nn.Sequential(
        #     nn.Conv2d(in_channels=3, out_channels=32, kernel_size=7, stride=3, padding=2),
        #     nn.PReLU(),
        #     nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
        #     nn.PReLU(),
        #     nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
        #     nn.PReLU(),
        # )

        self.features_cross_attention_container = nn.MultiheadAttention(
            embed_dim=128,
            num_heads=8,
            batch_first=True,
        )

        self.features_concat_rgb_container = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(out_features=1024),
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
            taken_actions = unflatten_tensorized_space(self.action_space, inputs.get("taken_actions"))
            # with torch.no_grad():
            #     rgb = states["rgb"].permute(0,3,1,2).to(self.device)
            #     rgb = self.preprocess(rgb)
            #     rgb = self.features_extractor_resnet(rgb)
            # rgb = states["rgb"].permute(0,3,1,2).float().to(self.device) / 255.0
            # rgb = (rgb - self.rgb_mean) / self.rgb_std
            features_extractor_rgb_1 = self.features_extractor_rgb_container(torch.permute(states['rgb_1'], (0, 3, 1, 2)))
            features_extractor_rgb_2 = self.features_extractor_rgb_container(torch.permute(states['rgb_2'], (0, 3, 1, 2)))
            # features_extractor_rgb = self.features_extractor_rgb_container(states["rgb"])
            tokens_1 = features_extractor_rgb_1.flatten(2).transpose(1, 2)
            tokens_2 = features_extractor_rgb_2.flatten(2).transpose(1, 2)
            features_cross_attention_1, _ = self.features_cross_attention_container(tokens_1, tokens_2, tokens_2)
            features_cross_attention_2, _ = self.features_cross_attention_container(tokens_2, tokens_1, tokens_1)
            
            features_concat_rgb = self.features_concat_rgb_container(torch.cat([features_cross_attention_1, features_cross_attention_2], dim=-1))
            features_extractor_joints = self.features_extractor_joints_container(states['joint'])
            net = self.net_container(torch.cat([features_extractor_joints, features_concat_rgb], dim=1))
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
                features_extractor_rgb_1 = self.features_extractor_rgb_container(torch.permute(states['rgb_1'], (0, 3, 1, 2)))
                features_extractor_rgb_2 = self.features_extractor_rgb_container(torch.permute(states['rgb_2'], (0, 3, 1, 2)))
                # features_extractor_rgb = self.features_extractor_rgb_container(states["rgb"])
                tokens_1 = features_extractor_rgb_1.flatten(2).transpose(1, 2)
                tokens_2 = features_extractor_rgb_2.flatten(2).transpose(1, 2)
                features_cross_attention_1, _ = self.features_cross_attention_container(tokens_1, tokens_2, tokens_2)
                features_cross_attention_2, _ = self.features_cross_attention_container(tokens_2, tokens_1, tokens_1)
                
                features_concat_rgb = self.features_concat_rgb_container(torch.cat([features_cross_attention_1, features_cross_attention_2], dim=-1))
                features_extractor_joints = self.features_extractor_joints_container(states['joint'])
                net = self.net_container(torch.cat([features_extractor_joints, features_concat_rgb], dim=1))
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

# instantiate a memory as rollout buffer (any memory can be used for this)
memory = RandomMemory(memory_size=24, num_envs=env.num_envs, device=device)

# instantiate the agent's models (function approximators).
# PPO requires 2 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#models
models = {}
models["policy"] = SharedModel(env.observation_space, env.action_space, device)
models["value"] = models["policy"]  # same instance: shared model

# configure and instantiate the agent (visit its documentation to see all the options)
# https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#configuration-and-hyperparameters
cfg = PPO_DEFAULT_CONFIG.copy()
cfg["rollouts"] = 24  # memory_size
cfg["learning_epochs"] = 8
cfg["mini_batches"] = 12
cfg["discount_factor"] = 0.99
cfg["lambda"] = 0.95
cfg["learning_rate"] = 1.0e-04
cfg["learning_rate_scheduler"] = KLAdaptiveLR
cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.01, "kl_factor": 2, "min_lr": 5.0e-06, "max_lr": 5.0e-03, "lr_factor": 1.5}
cfg["random_timesteps"] = 0
cfg["learning_starts"] = 0
cfg["grad_norm_clip"] = 1.0
cfg["ratio_clip"] = 0.2
cfg["value_clip"] = 0.2
cfg["clip_predicted_values"] = True
cfg["entropy_loss_scale"] = 0.0
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
cfg["experiment"]["checkpoint_interval"] = 1000
cfg["experiment"]["directory"] = "runs/torch/Isaac-Turtlebot3-Image-Direct-v0"

cfg["experiment"]["wandb"] = True
cfg["experiment"]["wandb_kwargs"]["project"] = "cube_grasp_project"
cfg["experiment"]["wandb_kwargs"]["tags"] = ["512_envs", "image_conv_32732_64321_128321_linear_512", "joint_64", "fusion_256_256", "random_backgrounds_20_hdr", "random_grounds_r0.1to0.2_g0.1to0.2_b0.1to0.2", "random_cubes_r0.1to0.2_g0.1to0.2_b0.1to0.2", "2_cameras"]

agent = PPO(models=models,
            memory=memory,
            cfg=cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device)


# configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 331200, "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

# ---------------------------------------------------------
# comment the code above: `trainer.train()`, and...
# uncomment the following lines to evaluate a trained agent
# ---------------------------------------------------------
# from skrl.utils.huggingface import download_model_from_huggingface

# download the trained agent's checkpoint from Hugging Face Hub and load it
# path = "/home/takenami/sim2real_ros2/runs/torch/Isaac-Turtlebot3-Image-Direct-v0/25-04-21_13-10-24-218304_PPO/checkpoints/best_agent.pt"
# path = "/home/takenami/sim2real_ros2/runs/torch/Isaac-Turtlebot3-Image-Direct-v0/red_backgrounds/checkpoints/agent_144000.pt"
# agent.load(path)

# start training
trainer.train()

# start evaluation
# trainer.eval()

