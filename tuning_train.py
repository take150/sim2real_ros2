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
            nn.Linear(in_features=13, out_features=256),
            nn.PReLU(),
            nn.Linear(in_features=256, out_features=128),
            nn.PReLU(),
            nn.Linear(in_features=128, out_features=64),
            nn.PReLU(),
        )

        self.policy_layer = nn.Linear(in_features=64, out_features=self.num_actions)
        # self.log_std_parameter = nn.LazyLinear(out_features=self.num_actions)
        self.log_std_parameter = nn.Parameter(torch.full(size=(self.num_actions,), fill_value=0.0), requires_grad=True)
        
        self.apply_orthogonal_init({
            self.net_container: np.sqrt(2),
            self.policy_layer: 0.01,
        })
    
    def compute(self, inputs, role=""):
        states = unflatten_tensorized_space(self.observation_space, inputs.get("states"))
        taken_actions = unflatten_tensorized_space(self.action_space, inputs.get("taken_actions"))
        output = self.net_container(torch.cat([states['joint'], states['actions']], dim=-1))
        # output = self.net_container(states['joint'])
        mu = self.policy_layer(output)
        mu = nn.functional.tanh(mu)
        # log_std_parameter = self.log_std_parameter(output)
        
        # return mu, log_std_parameter, {}
        return mu, self.log_std_parameter, {}

class DeterministicModel(DeterministicMixin, Model, OrthogonalInitMixin):
    def __init__(self, observation_space, action_space, device):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions=False)

        self.net_container = nn.Sequential(
            nn.Linear(in_features=13, out_features=256),
            nn.PReLU(),
            nn.Linear(in_features=256, out_features=128),
            nn.PReLU(),
            nn.Linear(in_features=128, out_features=64),
            nn.PReLU(),
        )

        self.value_layer = nn.Linear(in_features=64, out_features=1)

        self.apply_orthogonal_init({
            self.net_container: np.sqrt(2),
            self.value_layer: 1.0,
        })

    def compute(self, inputs, role=""):
        states = unflatten_tensorized_space(self.observation_space, inputs.get("states"))
        taken_actions = unflatten_tensorized_space(self.action_space, inputs.get("taken_actions"))
        output = self.net_container(torch.cat([states['joint'], states['actions']], dim=-1))
        # output = self.net_container(states['joint'])
        output = self.value_layer(output)
        return output, {}

# load and wrap the Isaac Lab environment
env = load_isaaclab_env(task_name="Isaac-Turtlebot3-Tuning-Direct-v0")
env = wrap_env(env)

device = env.device

# instantiate a memory as rollout buffer (any memory can be used for this)
memory = RandomMemory(memory_size=24, num_envs=env.num_envs, device=device)

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
cfg["rollouts"] = 24  # memory_size
cfg["learning_epochs"] = 8
cfg["mini_batches"] = 6
cfg["discount_factor"] = 0.97
cfg["lambda"] = 0.92
cfg["learning_rate"] = 5.0e-04
# cfg["learning_rate"] = 0.0000030332
# cfg["learning_rate"] = 1.0e-06
cfg["learning_rate_scheduler"] = KLAdaptiveLR
cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.01, "kl_factor": 2, "min_lr": 1.0e-06, "max_lr": 5.0e-04, "lr_factor": 1.1}
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
cfg["experiment"]["checkpoint_interval"] = 100
cfg["experiment"]["directory"] = "runs/torch/Isaac-Turtlebot3-Tuning-Direct-v0"

cfg["experiment"]["wandb"] = True
cfg["experiment"]["wandb_kwargs"]["project"] = "tuning_project"

agent = PPO(models=models,
            memory=memory,
            cfg=cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device)


# configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 200000, "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

# ---------------------------------------------------------
# comment the code above: `trainer.train()`, and...
# uncomment the following lines to evaluate a trained agent
# ---------------------------------------------------------
# from skrl.utils.huggingface import download_model_from_huggingface

# download the trained agent's checkpoint from Hugging Face Hub and load it
# path = "/home/takenami/sim2real_ros2/runs/torch/Isaac-Turtlebot3-Image-Direct-v0/25-05-08_13-47-18-171591_PPO/checkpoints/best_agent.pt"
# path = "/home/takenami/sim2real_ros2/runs/torch/Isaac-Turtlebot3-Image-Direct-v0/25-06-23_01-30-13-007985_PPO/checkpoints/agent_147000.pt"
# path = "/home/takenami/sim2real_ros2/runs/torch/Isaac-Turtlebot3-Image-Direct-v0/25-06-25_11-49-23-003009_PPO/checkpoints/best_agent.pt"
# path = "/home/takenami/sim2real_ros2/runs/torch/Isaac-Turtlebot3-Tuning-Direct-v0/25-12-17_20-33-27-781869_PPO/checkpoints/best_agent.pt"
# agent.load(path)

# start training
trainer.train()

# start evaluation
# trainer.eval()
