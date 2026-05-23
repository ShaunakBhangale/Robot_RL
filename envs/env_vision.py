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

        obs_dim = self.num_joints * 2 + 3 + 3
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self.target_pos = None
        self.robot = None
        self.detected_target = None

    def _get_target_from_camera(self): 
        if self.robot is None or self.target_pos is None:
            return np.zeros(3, dtype=np.float32)

        #M_int and M_ext 
        width, height = 320, 240
        near, far = 0.1, 10.0
        fov = 60
        aspect = width / height

        eye = [2.5, 0, 1.5]
        target = [0.5, 0, 0.3]
        up = [0, 0, 1]

        #M_ext view matrixs that takes us from world to camera coordinates you get [R t]
        #                                                                          [0 1]
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=eye,
            cameraTargetPosition=target,
            cameraUpVector=up
        )
        #K is a 3x3, and then 3D homegenous to 2D homogenous is 3x4 (K|0), but this is 4x4 3D homogenous --> 3D homogenous
        projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)
        #essentially M_int but goes from camera corods to clip space not straight to pixels

        _, _, rgb, depth, _ = p.getCameraImage(width, height, view_matrix, projection_matrix)

        #Convert RGB for OpenCV
        frame = np.array(rgb, dtype=np.uint8).reshape(height, width, 4)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

        #HSV blob detection for red target
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        lower_red = np.array([0, 120, 70])
        upper_red = np.array([10, 255, 255])
        mask = cv2.inRange(hsv, lower_red, upper_red)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return self.target_pos.astype(np.float32)

        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) <= 50:
            return self.target_pos.astype(np.float32)

        M = cv2.moments(largest)
        if M["m00"] <= 0:
            return self.target_pos.astype(np.float32)

        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"] #finding x and y pixel coordinates of the target in the image frame, 
        #which is the centroid of the largest red contour detected by OpenCV. 
        #This will be used for back-projection to get the 3D coordinates of the target in the world frame.

        #Get depth at the centroid pixel
        depth_buffer = np.array(depth).reshape(height, width)
        cx_int = int(round(cx)) #needs to be int to look up in depth buffer 
        cy_int = int(round(cy))
        #Clamp to image bounds so that if its right on the edge it wont crash
        cx_int = max(0, min(width - 1, cx_int))
        cy_int = max(0, min(height - 1, cy_int))
        depth_normalized = depth_buffer[cy_int, cx_int]

     
        if depth_normalized >= 1.0:
            return self.target_pos.astype(np.float32)

        #Reshape matrices to 4x4
        proj_mat = np.array(projection_matrix).reshape(4, 4, order='F')
        view_mat = np.array(view_matrix).reshape(4, 4, order='F')

        #perspective projection techniques are implicitly applied here, but openGL (pybullet) doesnt give z_c directly
        #instead just gives us a nonlinear depth value that we have to back-project to get the actual 3D coordinates
        ndc_x = (2.0 * cx) / width - 1.0 #these take us from pixel to NDC, wasnt needed in sree's because his M_int 
        #took us straight from camera frame to pixel
        ndc_y = 1.0 - (2.0 * cy) / height
        ndc_z = 2.0 * depth_normalized - 1.0

        
        ndc_point = np.array([ndc_x, ndc_y, ndc_z, 1.0])

        proj_view = proj_mat @ view_mat #just M_int * M_ext to get the fully transformation 
        inv_proj_view = np.linalg.inv(proj_view) 
        world_point_hom = inv_proj_view @ ndc_point #1x4 homogenous coordinates in world space, 
        #still need to dehomogenize by dividing by w
        #=[x·w, y·w, z·w, w]

        if abs(world_point_hom[3]) < 1e-8:
            return self.target_pos.astype(np.float32)
        #Dividing by w to dehomogenize the coordinates and get the actual 3D position in world space. 
        world_point = world_point_hom[:3] / world_point_hom[3]

        return world_point.astype(np.float32)

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
        joint_velocities = []
        for i in range(self.num_joints):
            pos, vel, _, _ = p.getJointState(self.robot, i)
            joint_angles.append(pos)
            joint_velocities.append(vel)

        
        ee_state = p.getLinkState(self.robot, 6)
        ee_pos = np.array(ee_state[4])

        #error vector using DETECTED target (not ground truth)
        #this is what the agent actually sees and has to learn to minimize
        error = self.detected_target - ee_pos

        return np.array(
            joint_angles +
            joint_velocities +
            list(error) +
            list(self.detected_target),
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
        action_penalty = 0.01 * np.linalg.norm(action) #esentially to make the agent learn to be more efficient and 
                    #not just slam the joints at max velocity so this is really for sim2real
        reward = -distance - action_penalty

        success = distance < 0.05
        if success:
            reward += 10.0
        if distance < 0.02:
            reward += 5.0

        terminated = bool(success)
        truncated = self.current_step >= self.step_limit

        return self._get_obs(), reward, terminated, truncated, {}

    def close(self):
        p.disconnect()