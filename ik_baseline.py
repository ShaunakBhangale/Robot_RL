import pybullet as p
import pybullet_data
import numpy as np
import time

def run_ik_baseline(num_episodes=100, render=False):
    if render:
        client = p.connect(p.GUI)
    else:
        client = p.connect(p.DIRECT)
    
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    results = []
    
    for ep in range(num_episodes):
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        p.loadURDF("plane.urdf")
        robot = p.loadURDF("kuka_iiwa/model.urdf", useFixedBase=True)
        target_pos = [
            np.random.uniform(0.4, 0.65),
            np.random.uniform(-0.2, 0.2),
            np.random.uniform(0.2, 0.45)
        ]
        visual = p.createVisualShape(p.GEOM_SPHERE, radius=0.05, rgbaColor=[1, 0, 0, 1])
        p.createMultiBody(baseVisualShapeIndex=visual, basePosition=target_pos)
        num_joints = 7
        joint_poses = p.calculateInverseKinematics(
            robot, 6, target_pos,
            maxNumIterations=100, 
            residualThreshold=1e-4
        )

        for i in range(num_joints):
            p.setJointMotorControl2(
                robot, i,
                controlMode=p.POSITION_CONTROL,
                targetPosition=joint_poses[i],
                force=500)
 
        for _ in range(1000):
            p.stepSimulation()
            if render:
                time.sleep(1/240)

        ee_state = p.getLinkState(robot, 6)
        ee_pos = np.array(ee_state[4]) 
        target = np.array(target_pos)
        
        distance = np.linalg.norm(ee_pos - target)
        success = distance < 0.05
        results.append({
            "episode": ep + 1,
            "distance": distance,
            "success": success
        })
        
        print(f"Episode {ep+1}: distance={distance:.4f}m, success={success}")
    
    avg_distance = np.mean([r["distance"] for r in results])
    success_rate = np.mean([r["success"] for r in results]) * 100
    
    print(f"\n--- IK Baseline Results ---")
    print(f"Average distance: {avg_distance:.4f}m")
    print(f"Success rate: {success_rate:.1f}%")
    
    p.disconnect()
    return avg_distance, success_rate

if __name__ == "__main__":
    run_ik_baseline(num_episodes=100, render=False)