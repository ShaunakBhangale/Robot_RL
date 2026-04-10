from envs.env_state import KukaReachEnv
#from envs.env_vision import KukaReachEnvVision as KukaReachEnv
from stable_baselines3 import SAC
import torch

print(torch.cuda.is_available())

env = KukaReachEnv(render=False)

model = SAC("MlpPolicy", env, verbose=1, tensorboard_log="./logs/")

print(model.device)

model.learn(total_timesteps=500_000)

model.save("kuka_reach_sac_state")

print("Training complete!")
env.close()