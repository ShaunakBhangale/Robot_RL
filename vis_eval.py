from envs.env_vision import KukaReachEnvVision
from stable_baselines3 import SAC
import numpy as np
import pybullet as p

env = KukaReachEnvVision(render=False)
model = SAC.load("kuka_reach_sac_vision_obs_fixed")

errors = []
successes = []

for ep in range(100):
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    ee_state = p.getLinkState(env.unwrapped.robot, 6)
    ee_pos = np.array(ee_state[4])
    target = env.unwrapped.target_pos
    dist = np.linalg.norm(ee_pos - target)
    errors.append(dist)
    successes.append(dist < 0.05)

print(f"Mean error: {np.mean(errors):.4f}m")
print(f"Success rate: {np.mean(successes)*100:.1f}%")
env.close()