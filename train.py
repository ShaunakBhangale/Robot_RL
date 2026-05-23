import argparse
from stable_baselines3 import SAC
import torch

parser = argparse.ArgumentParser(description="Train SAC on Kuka reach task")

parser.add_argument("--mode", choices=["state", "vision"], default="state", help="Environment mode: state or vision")
parser.add_argument("--timesteps", type=int, default=500_000, help="Total training timesteps")
parser.add_argument("--save", type=str, default=None, help="Model save name (optional, auto-generated if not provided)")

args = parser.parse_args()

if args.mode == "state":
    from envs.env_state import KukaReachEnv
    env = KukaReachEnv(render=False)
    save_name = args.save or "kuka_reach_sac_state"

else:
    from envs.env_vision import KukaReachEnvVision as KukaReachEnv
    env = KukaReachEnv(render=False)
    save_name = args.save or "kuka_reach_sac_vision"

print(f"Training mode: {args.mode}")
print(f"CUDA available: {torch.cuda.is_available()}")

model = SAC("MlpPolicy", env, verbose=1, tensorboard_log="./logs/", device="cpu")
print(f"Training on device: {model.device}")

model.learn(total_timesteps=args.timesteps)
model.save(save_name)

print(f"Training complete! Model saved to {save_name}")
env.close()