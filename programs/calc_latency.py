import mujoco
import mujoco.viewer
import numpy as np
import time
from scipy.spatial.transform import Rotation as R

# XMLファイルを読み込み
MODEL_PATH = "models/HBP_V2_mjcf/scene.xml"
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

# --- 遅延測定用の状態管理変数 ---
latency_test_state = 1      # 1: 静止待ち, 2: 出力開始, 3: 応答待ち, 0: 終了/通常
latency_start_time = 0.0
last_measured_latency_ms = 0.0

# テスト用の設定値
STEP_TORQUE = -0.002575         # テスト入力トルク (Nm)
VEL_THRESHOLD = np.rad2deg(1)        # 応答とみなす速度の閾値 (rad/s)

# --- 制御周期（10ms）の設定 ---
CONTROL_PERIOD_MS = 10.0
# XMLからシミュレーションの1ステップの時間を取得（デフォルトは 2.0 ms）
sim_timestep_ms = model.opt.timestep * 1000.0  
# 10ms進めるために必要なステップ数を計算（デフォルトなら 5ステップ）
steps_per_control = int(CONTROL_PERIOD_MS / sim_timestep_ms) 

with mujoco.viewer.launch_passive(model, data) as viewer:
    print(f"Physics Timestep: {sim_timestep_ms} ms")
    print(f"Executing {steps_per_control} steps per control loop to achieve {CONTROL_PERIOD_MS} ms control period.")
    print("Viewer started. Latency test will begin automatically.")
    
    # 初期姿勢の設定 (ピッチ角 -5度)
    mujoco.mju_euler2Quat(data.qpos[3:7], np.array([np.deg2rad(-90), 0, 0]), "xyz")    
    mujoco.mj_forward(model, data)
    
    l_wheel_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "back_tire_pitch")

    while viewer.is_running():
        # 現在の後輪速度を取得
        drive_vel = data.qvel[model.jnt_dofadr[l_wheel_id]]

        # --- 遅延測定ステートマシン ---
        if latency_test_state == 1:
            # [静止待ち] トルクを0にして速度が収まるのを待つ
            data.ctrl[1] = 0.0
            if abs(drive_vel) < 0.001:
                latency_test_state = 2
                print("Status: Stopped. Applying step input...")

        elif latency_test_state == 2:
            # [テスト開始] トルクを入力し、シミュレーション内時間を記録
            data.ctrl[1] = STEP_TORQUE
            latency_start_time = data.time  # pythonのtime()ではなく、MuJoCoの物理時間を使う
            latency_test_state = 3

        elif latency_test_state == 3:
            # [応答待ち] 速度が閾値を超えるまで待つ
            data.ctrl[1] = STEP_TORQUE
            
            if abs(drive_vel) > VEL_THRESHOLD:
                # 動いた！シミュレーション時間での遅延を計算
                latency_sec = data.time - latency_start_time
                last_measured_latency_ms = latency_sec * 1000.0
                print(f"Simulated Delay: {latency_sec:.6f} s ({last_measured_latency_ms:.2f} ms)")
                latency_test_state = 0  # テスト終了
                # data.ctrl[1] = 0.0      # 測定後は停止

            elif data.time - latency_start_time > 1.0:
                # シミュレーション内で1秒経っても動かなければタイムアウト
                print("Timeout! No response detected.")
                latency_test_state = 0
                data.ctrl[1] = 0.0
        
        else:
            # [通常制御] テスト完了後はトルク0で待機
            data.ctrl[1] = 0.0

        # --- 実機に近い動作の再現（ゼロ次ホールド） ---
        # 決定した制御入力 (data.ctrl) を保持したまま、10ms分物理演算を進める
        for _ in range(steps_per_control):
            mujoco.mj_step(model, data)
        
        # 描画の同期（time.sleepは現実の描画速度を落とすためだけで、物理時間には影響しません）
        time.sleep(0.01)
        viewer.sync()
