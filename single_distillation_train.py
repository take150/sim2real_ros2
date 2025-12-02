import torch
import torch.nn as nn
import torchvision.models as tv_models
from torch.optim.lr_scheduler import LinearLR

# import the skrl components to build the RL system
from skrl.agents.torch.distillation.distillation_student import DistillationStudent, DISTILLATION_DEFAULT_CONFIG
from skrl.agents.torch.distillation.distillation_teacher import DistillationTeacher
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
            nn.LazyLinear(out_features=128),
            nn.PReLU(),
            nn.LazyLinear(out_features=64),
            nn.PReLU(),
        )

        self.policy_layer = nn.LazyLinear(out_features=self.num_actions)
        
    def compute(self, inputs, role=""):
        states = unflatten_tensorized_space(self.observation_space, inputs.get("states"))
        features_extractor_rgb = self.features_extractor_rgb_container(torch.permute(states['rgb'], (0, 3, 1, 2)))        
        out, _ = self.rnn(torch.cat([states['joint'], states['actions']], dim=-1))
        output = self.net_container(torch.cat([features_extractor_rgb, out[:, -1, :]], dim=-1))
        output = self.policy_layer(output)
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
        
        return output, {}


# load and wrap the Isaac Lab environment
env = load_isaaclab_env(task_name="Isaac-Turtlebot3-Single-Distillation-Direct-v0")
env = wrap_env(env)

device = env.device

# instantiate a memory as rollout buffer (any memory can be used for this)
memory = RandomMemory(memory_size=24, num_envs=env.num_envs, device=device)

# instantiate the agent's models (function approximators).
# PPO requires 2 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#models
student_models = {}
student_models["policy"] = DeterministicStudentModel(env.observation_space, env.action_space, device)

# configure and instantiate the agent (visit its documentation to see all the options)
# https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#configuration-and-hyperparameters
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
student_cfg["learning_rate_scheduler_kwargs"] = {"start_factor": 1.0, "end_factor": 0.005, "total_iters": 2000}
student_cfg["grad_norm_clip"] = 1.0
student_cfg["state_preprocessor"] = None
student_cfg["state_preprocessor_kwargs"] = {}
# logging to TensorBoard and write checkpoints (in timesteps)
student_cfg["experiment"]["write_interval"] = 1
student_cfg["experiment"]["checkpoint_interval"] = 100
student_cfg["experiment"]["directory"] = "runs/torch/Isaac-Turtlebot3-Image-Direct-v0"

student_cfg["experiment"]["wandb"] = True
student_cfg["experiment"]["wandb_kwargs"]["project"] = "cube_grasp_project"
student_cfg["experiment"]["wandb_kwargs"]["tags"] = ["512_envs", "random_backgrounds_20_hdr", "random_grounds_r0.1to0.2_g0.1to0.2_b0.1to0.2", "random_cubes_r0.1to0.2_g0.1to0.2_b0.1to0.2"]

student_agent = DistillationStudent(models=student_models,
            memory=memory,
            cfg=student_cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device)


teacher_models = {}
teacher_models["policy"] = DeterministicTeacherModel(env.observation_space, env.action_space, device)

# configure and instantiate the agent (visit its documentation to see all the options)
# https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#configuration-and-hyperparameters

teacher_agent = DistillationTeacher(models=teacher_models,
            memory=memory,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device)

# configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 200000, "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=student_agent)

# ---------------------------------------------------------
# comment the code above: `trainer.train()`, and...
# uncomment the following lines to evaluate a trained agent
# ---------------------------------------------------------
# from skrl.utils.huggingface import download_model_from_huggingface

# download the trained agent's checkpoint from Hugging Face Hub and load it
# path = "/home/takenami/sim2real_ros2/runs/torch/Isaac-Turtlebot3-Image-Direct-v0/25-05-08_13-47-18-171591_PPO/checkpoints/best_agent.pt"
# path = "/home/takenami/sim2real_ros2/runs/torch/Isaac-Turtlebot3-Image-Direct-v0/25-06-23_01-30-13-007985_PPO/checkpoints/agent_147000.pt"
path = "/home/takenami/sim2real_ros2/runs/torch/Isaac-Turtlebot3-Image-Direct-v0/25-11-29_12-51-34-735903_PPO/checkpoints/agent_200000.pt"
teacher_agent.load(path)

# start training
trainer.distillation(teacher_agent=teacher_agent)

# start evaluation
# trainer.eval()

