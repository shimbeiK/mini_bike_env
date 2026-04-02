'''
consider with constraint of observation and action space. and input/output noise.
'''

import gymnasium as gym
from gymnasium import spaces
import mujoco, json, time
import numpy as np
import os
from scipy.spatial.transform import Rotation as R
from cfgs.cfg_nonsteer_v25 import PythonConfig
from collections import deque

class StandingEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    # def __init__(self, model, data, render_mode=None, max_step=parameters["max_step"]):
    def __init__(self, xml_path, render_mode=None):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        # self.model = model
        # self.data = data
        self.step_count = 1
        self.l_wheel_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "back_tire_pitch")
        self.drive_vel = 0.0
        self.wheel_pos = self.data.qvel[self.model.jnt_dofadr[self.l_wheel_id]]
        self.total_odometry = 0.0
        self.prev_odometry = 0.0
        self.alpha = 0.99
        self.prev_action = 0.0

        self.env_cfg, obs_cfg, self.reward_cfg, self.cmd_cfg = PythonConfig.get_cfgs()

        self.ANGLE_THRESHOLD = self.env_cfg["termination_if_roll_greater_than"]  # radians
        self.POSX_THRESHOLD = self.env_cfg["termination_if_posX_greater_than"]  # meters
        self.POSY_THRESHOLD = self.env_cfg["termination_if_posY_greater_than"]  # meters
        self.MAX_STEP = self.env_cfg["termination_if_step_count_greater_than"]
        self.frame_skip = self.env_cfg["frame_skip"]
        self.DT = self.env_cfg["frame_skip"] * self.model.opt.timestep

        # --- ドメインランダマイゼーション用の基準値保存 ---
        # 質量 (Mass)
        self.nominal_mass = self.model.body_mass.copy()
        # 摩擦係数 (Friction: [スライディング, トーション, ローリング])
        self.nominal_friction = self.model.geom_friction.copy()
        # 関節のダンピング (Damping)
        self.nominal_damping = self.model.dof_damping.copy()
        # アクチュエータのゲイン (Gain: モータ出力)
        self.nominal_actuator_gain = self.model.actuator_gainprm.copy()
        # 重心位置 (Center of Mass: 各ボディのローカル座標系での位置)
        self.nominal_ipos = self.model.body_ipos.copy()
        self.torque_gain = 0.0

        # self.latency_step = 3
        # self.control_queue = deque([np.zeros(2)] * (self.latency_step + 1), maxlen=self.latency_step+1)
        self.latency_step = 0
        self.encoder_queue = deque([np.zeros(3)] * (self.latency_step + 1), maxlen=self.latency_step+1)

        # 1. render_mode を保存する
        self.render_mode = render_mode

        # 2. レンダラーとビューワーは初期値 None (使う時に作成する "Lazy initialization")
        self.viewer = None
        self.renderer = None        
        self.MAX_TORQUE = self.env_cfg["drive_torque_scale"]  # 最大トルク
        self.MAX_STEER = self.env_cfg["steering_angle_scale"]
        self.NOISE_ANGLE = self.env_cfg["noise_angle"]  # 初期傾きのノイズ幅（±deg）
        self.action_space = spaces.Box(-1.0, 1.0, dtype=np.float32)
        
        self.ep_rew_survival = 0.0
        self.ep_rew_upright = 0.0
        self.ep_rew_odometry = 0.0
        
        # 観測空間：位置、速度、角度、角速度
        high = np.array([self.ANGLE_THRESHOLD, np.finfo(np.float32).max, 
                         np.finfo(np.float32).max, np.finfo(np.float32).max,
                         np.finfo(np.float32).max, self.MAX_STEER, 
                         np.finfo(np.float32).max, self.MAX_STEER, np.finfo(np.float32).max], 
                         dtype=np.float32) # 角度、角速度、角加速度、前回のアクション、前々回のアクション、オドメトリ
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

    def _randomize_domain(self):
        """物理パラメータをランダム化する"""
        
        # 1. 質量のランダム化 (±10%の変動)
        mass_noise = np.random.uniform(0.95, 1.05, size=self.nominal_mass.shape)
        self.model.body_mass[:] = self.nominal_mass * mass_noise

        # 2. タイヤと床の摩擦係数のランダム化 (±20%の変動)
        # ※バイクの横滑りやグリップに直結するため重要です
        friction_noise = np.random.uniform(0.6, 1.3, size=self.nominal_friction.shape)
        self.model.geom_friction[:] = self.nominal_friction * friction_noise

        # 5. 重心位置(CoM)のランダム化 (±5mm の変動)
        # X, Y, Z軸それぞれに対して、-0.005m 〜 +0.005m のズレを生じさせる
        ipos_noise = np.random.uniform(-0.003, 0.003, size=self.nominal_ipos[0].shape)
        
        # worldbody (id=0) の重心は変更しない方が安全なため、id=1以降のボディにのみ適用する
        # print("Applying domain randomization:ipos_noise =", self.model.body_ipos)
        # self.model.body_ipos[1] = self.nominal_ipos[1] + ipos_noise
        # print("Applying domain randomization:ipos_noise =", self.model.body_ipos)
        # 変更をMuJoCoの内部モデルに反映させるための再計算
        mujoco.mj_setConst(self.model, self.data)

    # センサから観測する。位置はMujoco環境から得る
    def _get_obs(self):
        rotmat = self.data.xmat[1].reshape(3, 3)
        rot = R.from_matrix(rotmat)
        angle = rot.as_euler('xyz', degrees=False)
        
        angular_vel = self.data.sensor("body_gyro").data[0]
        drive_vel = self.data.qvel[self.model.jnt_dofadr[self.l_wheel_id]]
        # steer_pos = self.data.joint("tire_holder_yaw").qpos[0]
        steer_pos = 0
        self.obs = steer_pos
        # --- FIX: Ensure the actions are flat numbers (scalars), not arrays ---
        if(self.env_cfg["real_syncro_noise"] == True):
            # angular_vel += np.random.normal(0, np.deg2rad(0.01))
            if(abs(angular_vel)>2):
                angular_vel = np.sign(angular_vel)*2
            # drive_vel += np.random.normal(0, 1e-1)
            accel = self.data.sensor("body_accel").data
            gyro = self.data.sensor("body_gyro").data           
            accel_roll = np.arctan2(accel[1], accel[2]) + np.random.normal(0, 0.00015)
            gyro_roll = gyro[0] + np.random.normal(0, 0.0008)
            # print("accel2:", accel[0], accel[1], accel[2], np.arctan2(accel[1], accel[2]), np.arctan2(accel[2], accel[1]))
            # print("accel2:", gyro[0], gyro[1], gyro[2], np.arctan2(accel[1], accel[2]), np.arctan2(accel[2], accel[1]))
            # self.filtered_roll = self.alpha * (self.filtered_roll + gyro_roll * self.DT) + (1 - self.alpha) * accel_roll
            self.filtered_roll = angle[0] + np.random.normal(0, np.deg2rad(1))
        else:
            self.filtered_roll = angle[0]

        self.encoder_queue.append(np.array([self.filtered_roll, angular_vel, drive_vel]))
        oldest_data = self.encoder_queue[0]
        self.filtered_roll = oldest_data[0]
        angular_vel = oldest_data[1]  
        drive_vel = oldest_data[2]    
        self.total_odometry += drive_vel * 0.034 * self.DT

        diff_angle = (self.target_steer_angle - steer_pos) / self.MAX_STEER
        return np.array([self.filtered_roll, angular_vel, drive_vel, 
                         0, 0, 0, 0, 0, 0], dtype=np.float32)

    # バイクの傾きと位置の変化から報酬を決定
    def _reward(self, obs, action):
        # imu, angular_vel, angular_acc, action_back, body_pos_x, body_pos_y = obs
        imu, _, _, _, _, _, _, _, _,  = obs
        # normalized_steer = (abs(self.control_queue[0][0] - self.control_queue[-1][0])) / self.MAX_STEER
        # normalized_torque = (abs(self.control_queue[0][1] - self.control_queue[-1][1])) / self.MAX_TORQUE

        r_survival = self.reward_cfg["survival_bonus"]  # 生存ボーナス（時間経過に対する報酬）
        r_upright = self.reward_cfg["upright_posture"] * (np.deg2rad(45) - abs(imu)) / np.deg2rad(45)
        # r_steering = self.reward_cfg["steering_penalty"] * normalized_steer / self.env_cfg["steering_angle_scale"]
        # r_odometry = self.reward_cfg["odometry_penalty"] * (abs(self.prev_odometry) - abs(self.total_odometry)) * 200
        r_odometry = self.reward_cfg["odometry_penalty"] * abs(self.total_odometry)
        # r_torque = self.reward_cfg["penalty_torque_unstable"] * normalized_torque + self.reward_cfg["penalty_torque_abs"] * abs(action[0])/self.MAX_TORQUE
        self.prev_odometry = self.total_odometry
        self.prev_action = action

        reward = r_survival + r_upright + r_odometry 
        # + r_steering + r_torque

        reward_info = {
            "r_survival": r_survival,
            "r_upright": r_upright,
            "r_odometry": r_odometry,
            # "r_torque": r_torque,
        }

        return reward, reward_info

    def step(self, action):
        # print(action)
        bias= 0.0021
        # action_angle = action[0] * self.MAX_STEER
        # action_angle = np.deg2rad(-60)
        action_torque = action[0] * self.MAX_TORQUE + self.torque_gain
        if self.env_cfg["real_syncro_noise"] == True:
            # action_angle += np.random.normal(0, 0.005) * self.MAX_STEER  # ステアリングにノイズを加える
            action_torque += np.random.normal(0, 0.0233 * 0.002)  # トルクにノイズを加える
            if action_torque > bias:
                action_torque = action_torque-bias
            elif action_torque < -bias:
                action_torque = action_torque+bias
            else:
                action_torque = 0.0   
            # self.control_queue.append(np.array([action_angle, action_torque]))
            # action_angle = self.control_queue[0][0]
            # action_torque = self.control_queue[0][1]

        # self.data.ctrl[0] = np.deg2rad(-60)
        self.data.ctrl[0] = action_torque
        # if(self.step_count>200):
        #     self.data.ctrl[1] = 0.021
        # else:
        #     self.data.ctrl[1] = -0.021

        # self.data.ctrl[0] = 0
        # self.data.ctrl[1] = 0
        for i in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
            # time.sleep(0.002)
                
        obs = self._get_obs()
        reward, rew_info = self._reward(obs, action[0])

        terminated = bool(abs(obs[0]) > self.ANGLE_THRESHOLD)
        truncated = bool(self.step_count >= self.MAX_STEP)
        
        # 各報酬をエピソード合計に加算
        self.ep_rew_survival += rew_info["r_survival"]
        self.ep_rew_upright += rew_info["r_upright"]
        self.ep_rew_odometry += rew_info["r_odometry"]
        # self.ep_rew_torque += rew_info["r_torque"]

        info = {}
        if terminated or truncated:
            info["ep_rew_survival"] = self.ep_rew_survival
            info["ep_rew_upright"] = self.ep_rew_upright
            info["ep_rew_odometry"] = self.ep_rew_odometry        
            # info["ep_rew_torque"] = self.ep_rew_torque        
        
        self.step_count += 1
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # configファイル等でオンオフを切り替えられるようにしておくと便利です
        if self.env_cfg["domain_randomization"] == True:
            self._randomize_domain()
            self.torque_gain = np.random.uniform(-0.01, 0.01) * self.MAX_TORQUE
            # self.latency_step = int(np.random.choice(range(1, 3)))
            # self.control_queue = deque([np.zeros(2)] * (self.latency_step + 1), maxlen=self.latency_step+1)
            self.latency_step = int(np.random.choice(range(0, 2)))
            self.encoder_queue = deque([np.zeros(3)] * (self.latency_step + 1), maxlen=self.latency_step+1)
        mujoco.mj_resetData(self.model, self.data)
        self.TARGET_VEL = 0.0
        self.data.qpos[8] = self.env_cfg["initial_steer_deg"]
        angle = self.env_cfg["initial_tilt_deg"]
        if self.env_cfg["init_tilt_noise"] == True:
            angle += np.random.normal(0, np.deg2rad(self.NOISE_ANGLE))
        # self.data.ctrl[0] = self.env_cfg["initial_steer_deg"]
        self.data.ctrl[0] = self.env_cfg["initial_torque"] * self.MAX_TORQUE
        self.data.qpos[3:7] = [1, 0, 0, 0]
        mujoco.mju_euler2Quat(self.data.qpos[3:7], np.array([angle, 0, 0]), "xyz")    

        self.step_count = 1
        self.filtered_roll = 0.0
        self.target_steer_angle = 0.0
        self.wheel_pos = self.data.qvel[self.model.jnt_dofadr[self.l_wheel_id]]
        self.drive_vel = 0.0
        self.total_odometry = 0.0
        self.prev_odometry = 0.0
        self.prev_action = 0.0
        self.ep_rew_survival = 0.0
        self.ep_rew_upright = 0.0
        self.ep_rew_odometry = 0.0
        self.ep_rew_torque = 0.0

        # for _ in range(15):
        mujoco.mj_step(self.model, self.data)
        return self._get_obs(), {}

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.sync()

        # B. rgb_array モード: 画像データ (numpy array) を返す
        elif self.render_mode == "rgb_array":
            if self.renderer is None:
                # オフスクリーンレンダラーを作成
                self.renderer = mujoco.Renderer(self.model, height=480, width=640)
            
            # レンダラーに現在の物理状態を反映してピクセルを読み込む
            self.renderer.update_scene(self.data)
            return self.renderer.render()
        
    def close(self):
        # ウィンドウやリソースの解放
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

if __name__ == "__main__":
    print("実行するファイルが違うで。")