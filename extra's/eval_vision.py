import pybullet as p
import pybullet_data
import numpy as np
import cv2
import time
from envs import env_vision
from stable_baselines3 import SAC

# Load trained model
model = SAC.load("kuka_reach_sac")

# Create environment with rendering
env = env_vision.KukaReachEnv(render=True)
obs, _ = env.reset()

# Camera setup
width, height = 640, 480
view_matrix = p.computeViewMatrix(
    cameraEyePosition=[2.5, 0, 1.5],
    cameraTargetPosition=[0.5, 0, 0.3],
    cameraUpVector=[0, 0, 1]
)
projection_matrix = p.computeProjectionMatrixFOV(60, width/height, 0.1, 10)

print("Running trained agent with vision pipeline... Press Q to quit")

episode = 0
done = False

while episode < 5:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated

    # Capture camera frame
    _, _, rgb, _, _ = p.getCameraImage(width, height, view_matrix, projection_matrix)
    frame = np.array(rgb, dtype=np.uint8).reshape(height, width, 4)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

    # Detect red target
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])
    mask = cv2.inRange(hsv, lower_red, upper_red)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if cv2.contourArea(cnt) > 100:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Target", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Vision Pipeline", frame)

    if done:
        episode += 1
        obs, _ = env.reset()

    time.sleep(1/60)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
env.close()