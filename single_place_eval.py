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


# define shared model (stochastic and deterministic models) using mixins
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

        self.rnn = nn.RNN(input_size=13, hidden_size=128, num_layers=1, batch_first=True)
        self.net_container = nn.Sequential(
            nn.Linear(in_features=142, out_features=256),
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
        
        self.rnn_move = nn.RNN(input_size=13, hidden_size=128, num_layers=1, batch_first=True)
        self.net_container_move = nn.Sequential(
            nn.Linear(in_features=156, out_features=256),
            nn.PReLU(),
            # nn.SiLU(),
            nn.Linear(in_features=256, out_features=256),
            nn.PReLU(),
            # nn.SiLU(),
            nn.Linear(in_features=256, out_features=128),
            nn.PReLU(),
            # nn.SiLU(),
            nn.Linear(in_features=128, out_features=64),
            nn.PReLU(),
            # nn.SiLU(),
        )

        self.move_policy_layer = nn.Linear(in_features=64, out_features=self.num_actions)
        # self.move_log_std_layer = nn.Linear(in_features=64, out_features=self.num_actions)
        self.move_log_std_parameter = nn.Parameter(torch.full(size=(self.num_actions,), fill_value=0.0), requires_grad=True)
        
        self.rnn_place = nn.RNN(input_size=13, hidden_size=128, num_layers=1, batch_first=True)
        self.net_container_place = nn.Sequential(
            nn.Linear(in_features=156, out_features=256),
            nn.PReLU(),
            # nn.SiLU(),
            nn.Linear(in_features=256, out_features=256),
            nn.PReLU(),
            # nn.SiLU(),
            nn.Linear(in_features=256, out_features=128),
            nn.PReLU(),
            # nn.SiLU(),
            nn.Linear(in_features=128, out_features=64),
            nn.PReLU(),
            # nn.SiLU(),
        )

        self.place_policy_layer = nn.Linear(in_features=64, out_features=self.num_actions)
        self.place_log_std_parameter = nn.Parameter(torch.full(size=(self.num_actions,), fill_value=0.0), requires_grad=True)
        
        # self.apply_orthogonal_init({
        #     self.net_container: np.sqrt(2),
        #     self.net_container_place: np.sqrt(2),
        #     self.rnn: np.sqrt(2),
        #     self.rnn_place: np.sqrt(2),
        #     self.policy_layer: 0.01,  # Policy出力は0.01
        #     self.place_policy_layer: 0.01,
        # })

    def compute(self, inputs, role=""):
        states = unflatten_tensorized_space(self.observation_space, inputs.get("states"))
        taken_actions = unflatten_tensorized_space(self.action_space, inputs.get("taken_actions"))
        
        task_ids = states['task_id'].squeeze(-1)
        batch_size = task_ids.shape[0]
        is_grasp = (task_ids == 0)
        is_move = (task_ids == 1)
        is_place = (task_ids == 2)
        output = torch.zeros((batch_size, self.num_actions), device=self.device)
        log_std_parameter = torch.zeros((batch_size, self.num_actions), device=self.device)

        if is_grasp.any():
            rnn_grasp, _ = self.rnn(torch.cat([states['joint'][is_grasp], states['actions'][is_grasp]], dim=-1))
            net_grasp = self.net_container(torch.cat([rnn_grasp[:, -1, :], states['object'][:, -1, :][is_grasp]], dim=-1))
            mu_grasp = self.policy_layer(net_grasp)
            output[is_grasp] = mu_grasp
            log_std_parameter[is_grasp] = self.log_std_parameter

        if is_move.any():
            rnn_move, _ = self.rnn_move(torch.cat([states['joint'][is_move], states['actions'][is_move]], dim=-1))
            net_move = self.net_container_move(torch.cat([rnn_move[:, -1, :], states['object'][:, -1, :][is_move], states['goal'][:, -1, :][is_move]], dim=-1))
            mu_move = self.move_policy_layer(net_move)
            output[is_move] = mu_move 
            # log_std_parameter[is_move] = self.move_log_std_layer(net_move)
            log_std_parameter[is_move] = self.move_log_std_parameter

        if is_place.any():
            rnn_place, _ = self.rnn_place(torch.cat([states['joint'][is_place], states['actions'][is_place]], dim=-1))
            net_place = self.net_container_place(torch.cat([rnn_place[:, -1, :], states['object'][:, -1, :][is_place], states['goal'][:, -1, :][is_place]], dim=-1))
            mu_place = self.place_policy_layer(net_place)
            output[is_place] = mu_place 
            log_std_parameter[is_place] = self.place_log_std_parameter
        
        output = nn.functional.tanh(output)
        
        return output, log_std_parameter, {}
    

class DeterministicModel(DeterministicMixin, Model, OrthogonalInitMixin):
    def __init__(self, observation_space, action_space, device):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions=False)

        self.rnn = nn.RNN(input_size=27, hidden_size=256, num_layers=2, batch_first=True)
        self.net_container = nn.Sequential(
            nn.Linear(in_features=256, out_features=256),
            nn.PReLU(),
            nn.Linear(in_features=256, out_features=256),
            nn.PReLU(),
            nn.Linear(in_features=256, out_features=128),
            nn.PReLU(),
            nn.Linear(in_features=128, out_features=64),
            nn.PReLU(),
        )

        self.value_layer = nn.Linear(in_features=64, out_features=1)

        self.rnn_move = nn.RNN(input_size=41, hidden_size=256, num_layers=2, batch_first=True)
        self.net_container_move = nn.Sequential(
            nn.Linear(in_features=256, out_features=256),
            nn.PReLU(),
            # nn.SiLU(),
            nn.Linear(in_features=256, out_features=256),
            nn.PReLU(),
            # nn.SiLU(),
            nn.Linear(in_features=256, out_features=128),
            nn.PReLU(),
            # nn.SiLU(),
            nn.Linear(in_features=128, out_features=64),
            nn.PReLU(),
            # nn.SiLU(),
        )
        self.move_value_layer = nn.Linear(in_features=64, out_features=1)

        self.rnn_place = nn.RNN(input_size=41, hidden_size=256, num_layers=2, batch_first=True)
        self.net_container_place = nn.Sequential(
            nn.Linear(in_features=256, out_features=256),
            nn.PReLU(),
            # nn.SiLU(),
            nn.Linear(in_features=256, out_features=256),
            nn.PReLU(),
            # nn.SiLU(),
            nn.Linear(in_features=256, out_features=128),
            nn.PReLU(),
            # nn.SiLU(),
            nn.Linear(in_features=128, out_features=64),
            nn.PReLU(),
            # nn.SiLU(),
        )

        self.place_value_layer = nn.Linear(in_features=64, out_features=1)

        # self.apply_orthogonal_init({
        #     self.net_container: np.sqrt(2),
        #     self.net_container_place: np.sqrt(2),
        #     self.rnn: np.sqrt(2),
        #     self.rnn_place: np.sqrt(2),
        #     self.value_layer: 1.0,    # Value出力は1.0
        #     self.place_value_layer: 1.0,
        # })

    def compute(self, inputs, role=""):
        states = unflatten_tensorized_space(self.observation_space, inputs.get("states"))
        taken_actions = unflatten_tensorized_space(self.action_space, inputs.get("taken_actions"))

        task_ids = states['task_id'].squeeze(-1)
        batch_size = task_ids.shape[0]
        is_grasp = (task_ids == 0)
        is_move = (task_ids == 1)
        is_place = (task_ids == 2)
        output = torch.zeros((batch_size, 1), device=self.device)

        if is_grasp.any():
            rnn_grasp, _ = self.rnn(torch.cat([states['joint'][is_grasp], states['actions'][is_grasp], states['object'][is_grasp]], dim=-1))
            net_grasp = self.net_container(rnn_grasp[:, -1, :])
            grasp_value = self.value_layer(net_grasp)
            output[is_grasp] = grasp_value
        
        if is_move.any():
            rnn_move, _ = self.rnn_move(torch.cat([states['joint'][is_move], states['actions'][is_move], states['object'][is_move], states['goal'][is_move]], dim=-1))
            net_move = self.net_container_move(rnn_move[:, -1, :])
            move_value = self.move_value_layer(net_move)
            output[is_move] = move_value

        if is_place.any():
            rnn_place, _ = self.rnn_place(torch.cat([states['joint'][is_place], states['actions'][is_place], states['object'][is_place], states['goal'][is_place]], dim=-1))
            net_place = self.net_container_place(rnn_place[:, -1, :])
            place_value = self.place_value_layer(net_place)
            output[is_place] = place_value
        
        return output, {}

# load and wrap the Isaac Lab environment
env = load_isaaclab_env(task_name="Isaac-Turtlebot3-Single-Place-Direct-v0")
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
# models["value"] = DeterministicModel(env.observation_space, env.action_space, device)

# configure and instantiate the agent (visit its documentation to see all the options)
# https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#configuration-and-hyperparameters
agent = PPO(models=models,
            memory=memory,
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
path = "/home/takenami/sim2real_ros2/runs/torch/Isaac-Turtlebot3-Image-Place-Direct-v0/25-12-09_10-12-55-204247_PPO/checkpoints/agent_144600.pt"
# path = "/home/takenami/sim2real_ros2/runs/torch/Isaac-Turtlebot3-Image-Place-Direct-v0/25-12-08_23-05-26-844061_PPO/checkpoints/agent_305400.pt"
# path = "/home/takenami/sim2real_ros2/runs/torch/Isaac-Turtlebot3-Image-Direct-v0/25-11-29_12-51-34-735903_PPO/checkpoints/agent_200000.pt"
agent.load(path)

# start training
trainer.eval()

# start evaluation
# trainer.eval()

