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

        self.rnn = nn.RNN(input_size=13, hidden_size=128, num_layers=1, batch_first=True)
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

        self.policy_layer = nn.LazyLinear(out_features=self.num_actions)
        self.log_std_parameter = nn.Parameter(torch.full(size=(self.num_actions,), fill_value=0.0), requires_grad=True)
        
    def compute(self, inputs, role=""):
        states = unflatten_tensorized_space(self.observation_space, inputs.get("states"))
        taken_actions = unflatten_tensorized_space(self.action_space, inputs.get("taken_actions"))
        out, _ = self.rnn(torch.cat([states['joint'], states['actions']], dim=-1))
        output = self.net_container(torch.cat([out[:, -1, :], states['object'][:, -1, :]], dim=-1))
        output = self.policy_layer(output)
        output = nn.functional.tanh(output)
        
        return output, self.log_std_parameter, {}

class DeterministicModel(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions=False)

        self.rnn = nn.RNN(input_size=27, hidden_size=256, num_layers=2, batch_first=True)

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
        out, _ = self.rnn(torch.cat([states['joint'], states['actions'], states['object']], dim=-1))
        output = self.net_container(out[:, -1, :])
        output = self.value_layer(output)
        return output, {}

# load and wrap the Isaac Lab environment
env = load_isaaclab_env(task_name="Isaac-Turtlebot3-Single-Direct-v0")
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
cfg["experiment"]["checkpoint_interval"] = 1000
cfg["experiment"]["directory"] = "runs/torch/Isaac-Turtlebot3-Image-Direct-v0"

cfg["experiment"]["wandb"] = True
cfg["experiment"]["wandb_kwargs"]["project"] = "cube_grasp_project"
cfg["experiment"]["wandb_kwargs"]["tags"] = ["512_envs", "random_backgrounds_20_hdr", "random_grounds_r0.1to0.2_g0.1to0.2_b0.1to0.2", "random_cubes_r0.1to0.2_g0.1to0.2_b0.1to0.2"]

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
# agent.load(path)

# start training
trainer.train()

# start evaluation
# trainer.eval()

