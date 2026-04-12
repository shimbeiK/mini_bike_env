'''
必要なこと
 ービジュアライズの確認
 ー観測値の確認
 ー制御入力の確認
 ー物理環境の調整（timestepsとか。必要？）
 入力の角度上限がないのか、観測が正しいか、観測値にノイズや遅れは生じているか
'''
import mujoco
import mujoco.viewer
import numpy as np
import time
from scipy.spatial.transform import Rotation as R

# XMLファイルを読み込み
# MODEL_PATH = "two_wheel_robot/sim_env/bike.xml"
# MODEL_PATH = "mjcf2/scene.xml"
MODEL_PATH = "models/HBP_V2_mjcf/scene.xml"
init_angle = 0  # フォークの初期角度設定(deg)
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

counter = 0
a = 0
num = 0.021
# ビューアを起動
with mujoco.viewer.launch_passive(model, data) as viewer:
    print("Viewer started. Press Ctrl+C to exit.")
    # data.qpos[3:7] = [1, 0, 0, 0]
    # 2. Define the Euler rotation (roll, pitch, yaw)
    # converting 45 degrees pitch to quaternion
    mujoco.mju_euler2Quat(data.qpos[3:7], np.array([np.deg2rad(-5), 0, 0]), "xyz")    
    # data.ctrl[0] = np.deg2rad(180)   # 後輪トルク
    mujoco.mj_forward(model, data)
    l_wheel_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "back_tire_pitch")
    f_wheel_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "front_tire_pitch")

    while viewer.is_running():
        i=0
        time.sleep(0.001)

        # data.ctrl[1] = -0.084   # 後輪トルク
        data.ctrl[1] = -0.025   # 後輪トルク
        # print(data.qvel[model.jnt_dofadr[f_wheel_id]])
        drive_vel = data.qvel[model.jnt_dofadr[l_wheel_id]]
        print(drive_vel)

        # data.ctrl[2] = -0.021 
        # 2. 1 ステップ進める
        mujoco.mj_step(model, data)
            
        # print(data.qvel[model.jnt_dofadr[l_wheel_id]] * 0.031)

        viewer.sync()    