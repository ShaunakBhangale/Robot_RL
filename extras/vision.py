import pybullet as p
import pybullet_data
import numpy as np
import cv2

# Connect to PyBullet
physics_client = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)

# Load scene
p.loadURDF("plane.urdf")
robot = p.loadURDF("kuka_iiwa/model.urdf", useFixedBase=True)

# Place red cube target
target_pos = [0.5, 0.1, 0.1]
visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.05, 0.05, 0.05], rgbaColor=[1, 0, 0, 1])
collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.05, 0.05, 0.05])
target = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=collision,
                           baseVisualShapeIndex=visual, basePosition=target_pos)

width, height = 640, 480
fov = 60
aspect = width / height

view_matrix = p.computeViewMatrix(
    cameraEyePosition=[2.5, 0, 1.5],
    cameraTargetPosition=[0.5, 0, 0.3],
    cameraUpVector=[0, 0, 1]
)
projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, 0.1, 10)

print("Starting vision pipeline... Press Q to quit")

while True:
    p.stepSimulation()

    # Capture camera frame
    _, _, rgb, _, _ = p.getCameraImage(width, height, view_matrix, projection_matrix)

    # Convert to OpenCV format
    frame = np.array(rgb, dtype=np.uint8).reshape(height, width, 4)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

    # Detect red object using color masking
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])
    mask = cv2.inRange(hsv, lower_red, upper_red)

    # Find contours of red region
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if cv2.contourArea(cnt) > 100:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Target", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Vision Pipeline", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
p.disconnect()