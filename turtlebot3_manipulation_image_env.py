import time
import numpy as np
import torch
import gymnasium as gym

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles
import sensor_msgs.msg
import geometry_msgs.msg


class Turtlebot3Image(gym.Env):
    def __init__(self, control_space="joint"):

        self.control_space = control_space  # joint or cartesian

        # spaces
        self.observation_space = gym.spaces.Box(low=-1000, high=1000, shape=(18,), dtype=np.float32)
        if self.control_space == "joint":
            self.action_space = gym.spaces.Box(low=-1, high=1, shape=(7,), dtype=np.float32)
        else:
            raise ValueError("Invalid control space:", self.control_space)

        # initialize the ROS node
        rclpy.init()
        self.node = Node(self.__class__.__name__)

        import threading
        threading.Thread(target=self._spin).start()

        # create publishers

        self.pub_command_joint = self.node.create_publisher(sensor_msgs.msg.JointState, '/robot/joint', QoSPresetProfiles.SYSTEM_DEFAULT.value)
        
        # keep compatibility with libiiwa Python API
        self.robot_state = {"joint_position": np.zeros((7,)),
                            "joint_velocity": np.zeros((7,)),
                            "cartesian_position": np.zeros((3,))}

        # create subscribers
        self.node.create_subscription(msg_type=sensor_msgs.msg.JointState,
                                      topic='/robot/joint_states',
                                      callback=self._callback_joint_states,
                                      qos_profile=QoSPresetProfiles.SYSTEM_DEFAULT.value)
        self.node.create_subscription(msg_type=geometry_msgs.msg.Pose,
                                      topic='/robot/end_effector_pose',
                                      callback=self._callback_end_effector_pose,
                                      qos_profile=QoSPresetProfiles.SYSTEM_DEFAULT.value)
        self.node.create_subscription(msg_type=geometry_msgs.msg.Pose,
                                      topic='/robot/camera_image',
                                      callback=self._callback_rgb,
                                      qos_profile=QoSPresetProfiles.SYSTEM_DEFAULT.value)

        print("Robot connected")

        self.motion = None
        self.motion_thread = None

        self.dt = 1 / 60.0
        self.action_scale = 2.5
        self.dof_vel_scale = 0.1
        self.max_episode_length = 100
        self.robot_dof_speed_scales = 1
        self.target_pos = np.array([0.0, 0.0, 0.0])
        self.robot_default_dof_pos = np.radians([0, 0, 0, 0, 0, 0, 0])
        self.robot_dof_lower_limits = np.array([-2.9671, -2.0944, -2.9671, -2.0944, -2.9671, -2.0944, -3.0543])
        self.robot_dof_upper_limits = np.array([ 2.9671,  2.0944,  2.9671,  2.0944,  2.9671,  2.0944,  3.0543])

        self.progress_buf = 1
        self.obs_buf = np.zeros((18,), dtype=np.float32)

    def _spin(self):
        rclpy.spin(self.node)

    def _callback_joint_states(self, msg):
        self.robot_state["joint_position"] = np.array(msg.position)
        self.robot_state["joint_velocity"] = np.array(msg.velocity)

    def _callback_end_effector_pose(self, msg):
        position = msg.position
        self.robot_state["cartesian_position"] = np.array([position.x, position.y, position.z])

    def _callback_rgb(self, msg):
        rgb = msg

    def _get_observation_reward_done(self):
        # observation
        end_effector_pos = self.robot_state["cartesian_position"]

        rgb = rgb / 255.0

        joint_pos = self.robot_state["joint_position"]
        joint_vel = self.robot_state["joint_velocity"]

        self.obs_buf = {
            "joint": torch.cat(
                (
                    joint_pos,
                    joint_vel,
                ),
                dim=-1,
            ),
            "rgb": rgb,
        }

        # reward
        distance = np.linalg.norm(end_effector_pos - self.target_pos)
        reward = -distance

        # done
        done = self.progress_buf >= self.max_episode_length - 1

        print("Distance:", distance)
        if done:
            print("Target or Maximum episode length reached")
            time.sleep(1)

        return self.obs_buf, reward, done

    def reset(self):
        print("Resetting...")

        # go to 1) safe position, 2) random position
        msg = sensor_msgs.msg.JointState()
        msg.position = self.robot_default_dof_pos.tolist()
        self.pub_command_joint.publish(msg)
        time.sleep(3)
        msg.position = (self.robot_default_dof_pos + 0.25 * (np.random.rand(7) - 0.5)).tolist()
        self.pub_command_joint.publish(msg)
        time.sleep(1)

        # get target position from prompt
        while True:
            try:
                print("Enter target position (X, Y, Z) in meters")
                raw = input("or press [Enter] key for a random target position: ")
                if raw:
                    self.target_pos = np.array([float(p) for p in raw.replace(' ', '').split(',')])
                else:
                    noise = (2 * np.random.rand(3) - 1) * np.array([0.1, 0.2, 0.2])
                    self.target_pos = np.array([0.6, 0.0, 0.4]) + noise
                print("Target position:", self.target_pos)
                break
            except ValueError:
                print("Invalid input. Try something like: 0.65, 0.0, 0.4")

        input("Press [Enter] to continue")

        self.progress_buf = 0
        observation, reward, done = self._get_observation_reward_done()

        return observation, {}

    def step(self, action):
        self.progress_buf += 1

        # control space
        # joint
        if self.control_space == "joint":
            joint_positions = self.robot_state["joint_position"] + (self.robot_dof_speed_scales * self.dt * action * self.action_scale)
            msg = sensor_msgs.msg.JointState()
            msg.position = joint_positions.tolist()
            self.pub_command_joint.publish(msg)

        # the use of time.sleep is for simplicity. It does not guarantee control at a specific frequency
        time.sleep(1 / 60.0)

        observation, reward, terminated = self._get_observation_reward_done()

        return observation, reward, terminated, False, {}

    def render(self, *args, **kwargs):
        pass

    def close(self):
        # shutdown the node
        self.node.destroy_node()
        rclpy.shutdown()
