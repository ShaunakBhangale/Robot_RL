import pybullet as p
import pybullet_data
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class KukaReachEnv(gym.Env):
    def __init__(self, render=False):
        super().__init__()
        self.render_mode = render
        self.step_limit = 1000
        self.current_step = 0

        if render:
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        self.num_joints = 7

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.num_joints,), dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=-np.pi, high=np.pi, shape=(self.num_joints + 3,), dtype=np.float32
        )

        self.target_pos = None
        self.robot = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)

        p.loadURDF("plane.urdf")
        self.robot = p.loadURDF("kuka_iiwa/model.urdf", useFixedBase=True)

        self.target_pos = np.array([
            np.random.uniform(0.4, 0.7),
            np.random.uniform(-0.3, 0.3),
            np.random.uniform(0.1, 0.5)
        ])

        visual = p.createVisualShape(p.GEOM_SPHERE, radius=0.05, rgbaColor=[1, 0, 0, 1])
        p.createMultiBody(baseVisualShapeIndex=visual, basePosition=self.target_pos)

        self.current_step = 0
        return self._get_obs(), {}

    def _get_obs(self):
        joint_angles = []
        for i in range(self.num_joints):
            pos, _, _, _ = p.getJointState(self.robot, i)
            joint_angles.append(pos)
        return np.array(joint_angles + list(self.target_pos), dtype=np.float32)

    def step(self, action):
        for i in range(self.num_joints):
            p.setJointMotorControl2(
                self.robot, i,
                controlMode=p.VELOCITY_CONTROL,
                targetVelocity=float(action[i])
            )

        p.stepSimulation()
        self.current_step += 1

        ee_state = p.getLinkState(self.robot, 6)
        ee_pos = np.array(ee_state[0])

        distance = np.linalg.norm(ee_pos - self.target_pos)
        reward = -distance

        success = distance < 0.05
        if success:
            reward += 10.0

        terminated = bool(success)
        truncated = self.current_step >= self.step_limit

        return self._get_obs(), reward, terminated, truncated, {}

    def close(self):
        p.disconnect()