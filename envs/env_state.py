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
       
        obs_dim = self.num_joints * 2 + 3 + 3  
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )   

        self.target_pos = None
        self.robot = None
        self.target_visual_id = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        if self.robot is None:
            p.resetSimulation()
            p.setGravity(0, 0, -9.8)
            p.loadURDF("plane.urdf")
            self.robot = p.loadURDF("kuka_iiwa/model.urdf", useFixedBase=True)
        else:
            for i in range(self.num_joints):
                p.resetJointState(self.robot, i, 0)
            if self.target_visual_id is not None:
                p.removeBody(self.target_visual_id)
            
        p.setGravity(0, 0, -9.8)

        self.target_pos = np.array([
            np.random.uniform(0.4, 0.7),
            np.random.uniform(-0.3, 0.3),
            np.random.uniform(0.1, 0.5)
        ])

        visual = p.createVisualShape(p.GEOM_SPHERE, radius=0.05, rgbaColor=[1, 0, 0, 1])
        self.target_visual_id = p.createMultiBody(
            baseVisualShapeIndex=visual,
            basePosition=self.target_pos
        )

        self.current_step = 0
        ee_state = p.getLinkState(self.robot, 6)
        ee_pos = np.array(ee_state[4])
        return self._get_obs(ee_pos), {}

    def _get_obs(self, ee_pos):
        joint_angles = []
        joint_velocities = []
        for i in range(self.num_joints):
            pos, vel, _, _ = p.getJointState(self.robot, i)
            joint_angles.append(pos)
            joint_velocities.append(vel)

        error = self.target_pos - ee_pos

        return np.array(
            joint_angles +
            joint_velocities +
            list(error) +
            list(self.target_pos),
            dtype=np.float32
        )

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
        ee_pos = np.array(ee_state[4])

        distance = np.linalg.norm(ee_pos - self.target_pos)
        action_penalty = 0.01 * np.linalg.norm(action)
        reward = -distance - action_penalty

        success = distance < 0.05
        if success:
            reward += 10.0
        if distance < 0.02:
            reward += 5.0

        terminated = bool(success)
        truncated = self.current_step >= self.step_limit

        return self._get_obs(ee_pos), reward, terminated, truncated, {}

    def close(self):
        p.disconnect()