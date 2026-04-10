from envs.env_state import KukaReachEnv as StateEnv
from envs.env_vision import KukaReachEnvVision as VisionEnv
from stable_baselines3 import SAC
import time

MODE = "state"

if MODE == "state":
    env = StateEnv(render=True)
    model = SAC.load("kuka_reach_sac_state")
else:
    env = VisionEnv(render=True)
    model = SAC.load("kuka_reach_sac_vision")

obs, _ = env.reset()
total_rewards = []

for ep in range(5):
    obs, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated
        time.sleep(1/60)

    total_rewards.append(total_reward)
    print(f"Episode {ep+1} reward: {total_reward:.2f}")

print(f"\nAverage reward: {sum(total_rewards)/len(total_rewards):.2f}")
env.close()