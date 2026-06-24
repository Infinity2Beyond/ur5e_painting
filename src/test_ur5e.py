import mujoco
import mujoco.viewer
import numpy as np
import time

# Load the UR5e model
model = mujoco.MjModel.from_xml_path(
    "mujoco_menagerie/universal_robots_ur5e/scene.xml"
)
data = mujoco.MjData(model)

print("=== UR5e Model Info ===")
print(f"Number of joints (DoF): {model.nv}")
print(f"Number of actuators:    {model.nu}")
print(f"Joint names:")
for i in range(model.njnt):
    print(f"  Joint {i}: {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)}")

# Launch interactive viewer
print("\nLaunching viewer... (close the window to exit)")
with mujoco.viewer.launch_passive(model, data) as viewer:
    start_time = time.time()
    while viewer.is_running():
        t = time.time() - start_time

        # Simple sinusoidal motion on joint 0 (shoulder pan)
        data.ctrl[0] = 0.5 * np.sin(0.5 * t)   # shoulder_pan_joint
        data.ctrl[1] = -1.0 + 0.3 * np.sin(0.3 * t)  # shoulder_lift_joint

        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.002)  # ~500 Hz simulation