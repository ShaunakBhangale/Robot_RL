import pybullet as p
import pybullet_data
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import cv2

class KukaReachEnvVision(gym.Env):
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
        self.detected_target = None

    def _get_target_from_camera(self):
        if self.robot is None or self.target_pos is None:
            return np.zeros(3, dtype=np.float32)

        width, height = 320, 240
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=[2.5, 0, 1.5],
            cameraTargetPosition=[0.5, 0, 0.3],
            cameraUpVector=[0, 0, 1]
        )
        projection_matrix = p.computeProjectionMatrixFOV(60, width/height, 0.1, 10)

        _, _, rgb, _, _ = p.getCameraImage(width, height, view_matrix, projection_matrix)
        frame = np.array(rgb, dtype=np.uint8).reshape(height, width, 4)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        lower_red = np.array([0, 120, 70])
        upper_red = np.array([10, 255, 255])
        mask = cv2.inRange(hsv, lower_red, upper_red)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 50:
                M = cv2.moments(largest)
                if M["m00"] > 0:
                    cx = M["m10"] / M["m00"]
                    cy = M["m01"] / M["m00"]
                    x = 0.4 + (cx / width) * 0.3
                    y = -0.3 + (cy / height) * 0.6
                    z = 0.3
                    return np.array([x, y, z], dtype=np.float32)

        return self.target_pos.astype(np.float32)

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

        self.detected_target = self._get_target_from_camera()
        self.current_step = 0
        return self._get_obs(), {}

    def _get_obs(self):
        joint_angles = []
        for i in range(self.num_joints):
            pos, _, _, _ = p.getJointState(self.robot, i)
            joint_angles.append(pos)
        return np.array(joint_angles + list(self.detected_target), dtype=np.float32)

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