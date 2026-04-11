# Robotic Arm RL Simulation

A reinforcement learning pipeline for training a 7-DOF robotic arm to reach randomized 3D targets in simulation, with a comparison against a classical IK baseline.

## Overview
This project trains a SAC (Soft Actor-Critic) policy to control a Kuka IIWA arm using joint velocity commands. Two RL environments are implemented and compared against PyBullet's built-in IK solver:

- **State-based:** Agent observes exact target coordinates
- **Vision-based:** Agent observes camera-estimated target position via OpenCV color detection
- **IK Baseline:** PyBullet's numerical IK solver for classical comparison

## Results

| Method | Avg End-Effector Distance | Success Rate (7cm threshold) |
|--------|--------------------------|------------------------------|
| SAC (state-based) | ~0.19m | TODO |
| SAC (vision-based) | ~0.32m | TODO |
| IK Baseline | 0.057m | 100% |

### Analysis
The IK baseline achieves significantly lower end-effector error than either SAC policy, demonstrating that for structured reach tasks with known kinematics, classical control outperforms learned policies on raw precision. However, the 5.7cm error reflects a limitation of the solver itself — PyBullet uses a basic damped least-squares Jacobian method, which trades precision for numerical stability. A more sophisticated solver (or closed-loop control on real hardware with encoder feedback) would achieve sub-centimeter accuracy.

SAC's value lies not in precision but in generalization. A learned policy can adapt to sensor noise, model uncertainty, and task variations that would require significant re-engineering of a classical IK pipeline. Introducing camera-based target localization (vision-based env) reduced SAC performance by ~40%, quantifying the impact of perception noise on policy learning — a core challenge when moving from privileged state information to realistic sensor-based observations.

## File Structure

```
Robot_RL
  envs
    env_state.py        
    env_vision.py       
  train.py              
  eval.py               
  ik_baseline.py        
```
​

## Setup

​'''
python -m venv venv
venv\Scripts\activate
pip install pybullet stable-baselines3[extra] gymnasium opencv-python tensorboard
​'''
## Training

​```
python train.py
​```

Change the import in `train.py` to switch between `env_state` and `env_vision`.

## Evaluation

​```
python eval.py
​```

Change `MODE = "state"` or `MODE = "vision"` in `eval.py` to switch between models.

## IK Baseline

​```
python ik_baseline.py
​```

## Monitoring Training

​```
venv\Scripts\tensorboard.exe --logdir=logs
​```

Then open http://localhost:6006

## Stack
- **PyBullet** — Physics simulation
- **Stable-Baselines3** — SAC implementation
- **Gymnasium** — RL environment interface
- **OpenCV** — Camera-based target localization
- **TensorBoard** — Training visualization

## Future Work
- Replace PyBullet IK with custom Jacobian pseudoinverse solver
- Vision-based RL using CnnPolicy (agent learns from pixels)
- Variable frequency camera calls during episode (active perception)
- Domain randomization for sim-to-real transfer
- Extend to pick-and-place tasks
