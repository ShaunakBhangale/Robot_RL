from env_vision import KukaReachEnvVision
import numpy as np

env = KukaReachEnvVision(render=False)

print("Testing back-projection accuracy across 20 random targets...\n")
print(f"{'Episode':<10}{'True target':<35}{'Detected target':<35}{'Error (m)':<10}")
print("-" * 90)

errors = []

for ep in range(20):
    env.reset()
    true_pos = env.target_pos
    detected_pos = env.detected_target
    error = np.linalg.norm(true_pos - detected_pos)
    errors.append(error)
    print(f"{ep+1:<10}{str(true_pos.round(3)):<35}{str(detected_pos.round(3)):<35}{error:.4f}")

print("-" * 90)
print(f"\nMean error: {np.mean(errors):.4f}m")
print(f"Max error: {np.max(errors):.4f}m")
print(f"Min error: {np.min(errors):.4f}m")

env.close()