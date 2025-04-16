import time
import numpy as np
import torch
import gymnasium as gym

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles
import sensor_msgs.msg
import geometry_msgs.msg
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


class Turtlebot3Reach(gym.Env):
    def __init__(self, control_space="joint"):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.control_space = control_space  # joint or cartesian

        # spaces
        self.observation_space = gym.spaces.Box(low=-10, high=10, shape=(7,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        
        self.target_joints = ['joint1', 'joint2', 'joint3', 'joint4']
        self.start = time.perf_counter()

        # initialize the ROS node
        rclpy.init()
        self.node = Node(self.__class__.__name__)

        import threading
        threading.Thread(target=self._spin).start()

        # create publishers

        self.pub_command_joint = self.node.create_publisher(JointTrajectory, '/arm_controller/joint_trajectory', 10)
        self.trajectory_joint_msg = JointTrajectory()
        self.trajectory_joint_msg.joint_names = ['joint1', 'joint2', 'joint3', 'joint4']
        self.point_joint = JointTrajectoryPoint()
        self.point_joint.time_from_start.nanosec = 50000000
        self.reset_point_joint = JointTrajectoryPoint()
        self.reset_point_joint.time_from_start.sec = 3

        # keep compatibility with libiiwa Python API
        self.robot_state = {"joint_position": torch.zeros((4,))}

        # create subscribers
        self.node.create_subscription(msg_type=sensor_msgs.msg.JointState,
                                      topic='/joint_states',
                                      callback=self._callback_joint_states,
                                      qos_profile=QoSPresetProfiles.SYSTEM_DEFAULT.value)

        print("Robot connected")

        self.motion = None
        self.motion_thread = None

        self.dt = 0.05
        self.action_scale = 1.0
        self.dof_vel_scale = 0.1
        self.max_episode_length = 100
        self.robot_dof_speed_scales = 0.5
        self.target_pos = torch.tensor([0.2, 0.0, 0.25], device=self.device)
        self.robot_default_dof_pos = torch.tensor([0.0, 0.0, 0.0, 0.0], device=self.device)
        self.robot_dof_lower_limits = torch.tensor([-2.8274, -1.7907, -0.9425, -1.7907], device=self.device)
        self.robot_dof_upper_limits = torch.tensor([2.8274, 1.5708, 1.3823, 2.0420], device=self.device)

        self.progress_buf = 1
        self.obs_buf = torch.zeros((7,), dtype=torch.float32, device=self.device)

    def _spin(self):
        rclpy.spin(self.node)

    def _callback_joint_states(self, msg):
        joint_map = dict(zip(msg.name, msg.position))
        joint_position = [joint_map[joint] for joint in self.target_joints]
        self.robot_state["joint_position"] = torch.tensor(joint_position, device=self.device)

    def _get_observation_reward_done(self):
        # observation

        joint_pos = self.robot_state["joint_position"]
        joint_pos = joint_pos.to(self.device)

        self.obs_buf = torch.cat((joint_pos, self.target_pos), dim=-1)

        # reward
        distance = self.target_pos
        reward = -distance

        # done
        done = self.progress_buf >= self.max_episode_length - 1

        # print("Distance:", distance)
        if done:
            print("Target or Maximum episode length reached")
            time.sleep(1)

        return self.obs_buf, reward, done

    def reset(self):
        print("Resetting...")

        # go to 1) safe position, 2) random position
        self.reset_point_joint.positions = self.robot_default_dof_pos.tolist()
        self.trajectory_joint_msg.points = [self.reset_point_joint]
        self.pub_command_joint.publish(self.trajectory_joint_msg)
        time.sleep(5)

        # get target position from prompt
        self.target_pos = self.generate_random_xyz(
            x_range=(0.15, 0.25), y_range=(-0.1, 0.1), z_range=(0.1, 0.3)
        )

        print("goal", self.target_pos)

        self.progress_buf = 0
        observation, reward, done = self._get_observation_reward_done()

        return observation, {}

    def generate_random_xyz(self, x_range, y_range, z_range):
        x = x_range[0] + (x_range[1] - x_range[0]) * torch.rand(1, device=self.device)
        y = y_range[0] + (y_range[1] - y_range[0]) * torch.rand(1, device=self.device)
        z = z_range[0] + (z_range[1] - z_range[0]) * torch.rand(1, device=self.device)
        
        return torch.cat([x, y, z])

    def step(self, action):
        self.progress_buf += 1

        # control space
        # joint

        while time.perf_counter() - self.start < self.dt:
            pass
        
        # print(1, self.robot_state["joint_position"])
        # print(2, action)
        joint_positions = self.robot_state["joint_position"] + self.robot_dof_speed_scales * self.dt * torch.tensor(action, device=self.device)
        # joint_positions = self.robot_state["joint_position"] + self.robot_dof_speed_scales * self.dt * torch.tensor([0.0, -1.0, 0.0, 0.0], device=self.device)
        self.point_joint.positions = joint_positions.tolist()
        self.trajectory_joint_msg.points = [self.point_joint]
        # print(self.trajectory_joint_msg.points)
        self.pub_command_joint.publish(self.trajectory_joint_msg)
        # self.pub_command_joint.publish(msg)

        # the use of time.sleep is for simplicity. It does not guarantee control at a specific frequency        
        self.start = time.perf_counter()

        observation, reward, terminated = self._get_observation_reward_done()

        return observation, reward, terminated, False, {}

    def render(self, *args, **kwargs):
        pass

    def close(self):
        # shutdown the node
        self.node.destroy_node()
        rclpy.shutdown()
