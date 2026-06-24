import mujoco
import rtde_receive
import time
import mujoco.viewer
import numpy as np

if __name__ == '__main__':

    robot_ip = "172.17.0.2"
    rtde_rec = rtde_receive.RTDEReceiveInterface(robot_ip)
    pose = rtde_rec.getActualQ()
    print("Actual joint positions:", pose)


    # Load the UR5e model
    model = mujoco.MjModel.from_xml_path(
        "mujoco_menagerie/universal_robots_ur5e/scene.xml"
    )
    data = mujoco.MjData(model)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        start_time = time.time()
        while viewer.is_running():
            t = time.time() - start_time

            data.qpos = np.array(rtde_rec.getActualQ())  # Update joint positions from the robot

            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.002)  # ~500 Hz simulation