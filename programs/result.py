import mujoco
import mujoco.viewer
import time, socket
import numpy as np
from stable_baselines3 import PPO, SAC
from gymnasium import wrappers
from envs.env_nonsteer_v25 import StandingEnv
# from bike_env_v3_NonSteer_withrealnoise import StandingEnv
# from bike_env_v3_turn import StandingEnv
# from bike_env_v3_integrate import StandingEnv
teleplotAddr = ("127.0.0.1",47269)
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def sendTelemetry(name, value):
    now = time.time() * 1000
    msg = name+":"+str(now)+":"+str(value)+"|g"
    sock.sendto(msg.encode(), teleplotAddr)
# --- 1. Setup the Environment for Evaluation ---
# We use ONE environment with render_mode="human"
MODEL_PATH = "models/HBP_V2_5_mjcf/scene.xml"
# MODEL_PATH = "mjcf2/scene.xml"
m = mujoco.MjModel.from_xml_path(MODEL_PATH)
d = mujoco.MjData(m)

# Important: Set render_mode="human" here to open the window
# env = StandingEnv(m, d, render_mode="human")
env = StandingEnv(xml_path=MODEL_PATH, render_mode="human")
# env = VecNormalize.load(stats_load_path, env)
# # 【重要】推論時は統計情報を更新せず、報酬の正規化も行わない設定にする
# env.training = False
# env.norm_reward = False
l_wheel_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, "tire_back_pitch")
# prev_angular_vel = env.data.qvel[env.model.jnt_dofadr[l_wheel_id]]
# angular_vel = env.data.sensor("imu_gyro").data.copy()[0]+np.random.normal(0, 0.01)  # ジャイロのx軸の角速度にノイズを加える

# --- 2. Load the Trained Model ---
model = PPO.load("results/stop_withCon_v3/best_model", env=env)
# model = SAC.load("results/sac_results/20260310-1256_eval/best_model", env=env)
# --- 3. Run the Simulation Loop ---
obs, _ = env.reset()
print("Running trained model... Press Ctrl+C to stop.")
# print(env.TARGET_VEL)
# env.target_steer_angle = 0.0
counter = 0
prev_odometry = 0
action_num = 0
range_max = np.deg2rad(60)
exclude_val = np.deg2rad(10.0) # 除外したい角度

# 10 〜 range_max の範囲で乱数を生成し、ランダムに 1 か -1 を掛ける
sign = np.random.choice([-1, 1])

while True:
    counter+=1
    action, _ = model.predict(obs, deterministic=True)
    prev_angular_vel = env.data.qvel[env.model.jnt_dofadr[l_wheel_id]]
    obs, reward, terminated, truncated, info = env.step(action)
    action_num += np.rad2deg(action[0]*np.deg2rad(80))
    # print(np.rad2deg(obs[0]), obs[1], obs[2], action[0], env.total_odometry)

    # sendTelemetry("f", front_out)
    sendTelemetry("b", action[0]*460)
    sendTelemetry("rollvel", np.rad2deg(obs[1]))
    sendTelemetry("tire spd", obs[2])
    sendTelemetry("pitch", np.rad2deg(obs[0]))
    prev_odometry = env.total_odometry
    env.render()
    
    # Slow down slightly to match real-time (optional, depends on your PC speed)
    time.sleep(0.01)

    if terminated or truncated:
        print("ouch!")
        obs, _ = env.reset()
        # print(env.TARGET_VEL, np.rad2deg(env.target_steer_angle))
        time.sleep(2)
        counter = 0
        action_num = 0
