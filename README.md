# Robotic Arm RL Simulation

A reinforcement learning pipeline for training a 7-DOF robotic arm to reach randomized 3D targets in simulation, with a comparison against a classical IK baseline.

## Overview

This project trains a SAC (Soft Actor-Critic) policy to control a Kuka IIWA arm using joint velocity commands. Two RL environments are implemented and compared against PyBullet's built-in IK solver:

- **State-based:** Agent observes joint angles, joint velocities, error vector, and ground-truth target coordinates
- **Vision-based:** Agent observes joint angles, joint velocities, error vector, and camera-estimated target position via OpenCV color detection + depth-buffer back-projection
- **IK Baseline:** PyBullet's numerical IK solver for classical comparison

## Results

| Method | Mean Error | Success Rate (5cm threshold) |
|--------|-----------|------------------------------|
| SAC state v1 (10D obs, no velocities) | ~0.19m | low |
| SAC vision v1 (broken 2D→3D mapping) | ~0.32m | low |
| IK baseline (default 20 iterations) | ~0.042m | 85% |
| SAC state v2 (20D obs) | 0.049m | 100% |
| SAC vision v2 (fixed perception) | 0.050m | 99% |
| IK baseline (100 iterations, tuned) | ~0.0001m | 100% |

### Observation Space (20D)

- Joint angles q1..q7 (rad)
- Joint velocities dq1..dq7 (rad/s)
- Error vector ex, ey, ez = target minus ee_pos (m)
- Target position tx, ty, tz (m)

### Vision Pipeline

1. Render 320x240 RGB + depth from fixed external camera (eye-to-hand setup)
2. HSV thresholding + contour detection to find red target centroid (cx, cy)
3. Look up depth at (cx, cy) from PyBullet depth buffer
4. Back-project pixel + depth to world coordinates via inv(projection x view) matrix

### Action Space

7D continuous velocity commands in [-1, 1] rad/s, one per joint.

### Reward

- reward = -distance - 0.01 * norm(action)
- +10.0 if distance < 0.05m (success bonus)
- +5.0 if distance < 0.02m (precision bonus)

## File Structure

    Robot_RL/
    ├── envs/
    │   ├── env_state.py      # State-based Gymnasium env
    │   └── env_vision.py     # Vision-based Gymnasium env with depth back-projection
    ├── train.py              # SAC training with argparse
    ├── eval.py               # Policy evaluation with GUI rendering
    └── ik_baseline.py        # Classical IK comparison baseline

## Setup

    python -m venv venv
    venv\Scripts\activate
    pip install pybullet stable-baselines3[extra] gymnasium opencv-python tensorboard

## Training

    # Train state-based policy
    python train.py --mode state

    # Train vision-based policy
    python train.py --mode vision

    # Custom timesteps or save name
    python train.py --mode state --timesteps 1000000 --save my_model

## Evaluation

    # Evaluate state policy (opens GUI)
    python eval.py --mode state

    # Evaluate vision policy
    python eval.py --mode vision

    # Custom model and episodes
    python eval.py --mode vision --model kuka_reach_sac_vision_obs_fixed --episodes 20

## IK Baseline

    python ik_baseline.py

## Monitoring Training

    venv\Scripts\tensorboard.exe --logdir=logs

Then open http://localhost:6006

## Stack

- PyBullet — physics simulation and rendering
- Stable-Baselines3 — SAC implementation
- Gymnasium — RL environment interface
- OpenCV — HSV thresholding, contour detection, image moments
- NumPy — matrix math for depth back-projection
- TensorBoard — training visualization

## Known Limitations

- Kuka IIWA (7-DOF) used as stand-in for a custom 6-DOF arm — policy does not transfer directly to real hardware
- Depth-buffer back-projection is sim-only; real deployment requires ArUco markers + solvePnP
- Single-seed results; multiple seeds with mean ± std would be more statistically rigorous
- Velocity control capped at ±1 rad/s; real Kuka operates at higher velocities

## Future Work

- Deploy on real 6-DOF arm with ROS2, MoveIt2, and ArUco-based perception
- End-to-end pixel-to-action learning with CnnPolicy
- Domain randomization for sim-to-real transfer
- Extend to pick-and-place and contact-rich tasks
- IK baseline with custom Jacobian pseudoinverse solver
