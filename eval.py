import argparse
from envs.env_state import KukaReachEnv as StateEnv
from envs.env_vision import KukaReachEnvVision as VisionEnv
from stable_baselines3 import SAC
import time

parser = argparse.ArgumentParser(description="Evaluate trained SAC policy on Kuka reach task")
parser.add_argument("--mode", choices=["state", "vision"], default="state", help="Environment mode: state or vision")
parser.add_argument("--model", type=str, default=None, help="Model path to load (optional, uses default if not provided)")
parser.add_argument("--episodes", type=int, default=5, help="Number of evaluation episodes")

args = parser.parse_args()

if args.mode == "state":
    env = StateEnv(render=True)
    model_path = args.model or "kuka_reach_sac_state"
else:
    env = VisionEnv(render=True)
    model_path = args.model or "kuka_reach_sac_vision_obs_fixed"

model = SAC.load(model_path)
print(f"Loaded model: {model_path}")
print(f"Mode: {args.mode}")

total_rewards = []

for ep in range(args.episodes):
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