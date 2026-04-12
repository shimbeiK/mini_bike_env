from stable_baselines3 import PPO
# from bike_env_v3_NonSteer_withrealnoise import StandingEnv
# from bike_env_v3 import StandingEnv
from envs.env_stay_syncro import StandingEnv
import mujoco, time, os, pickle, shutil
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from cfgs.cfg_stay import PythonConfig

train_cfg_dict = PythonConfig.get_train_cfg()

class RenderCallback(BaseCallback):
    def __init__(self, env, verbose=0):
        super().__init__(verbose)
        self.env = env

    def _on_step(self) -> bool:
        # 毎ステップ呼び出される
        self.env.render()
        return True

class TensorboardCallback(BaseCallback):
    """
    環境のinfoから個別の報酬コンポーネントを読み取り、TensorBoardに記録するコールバック
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # 複数の環境が並列で動いているので、それぞれのinfoを確認する
        for info in self.locals.get("infos", []):
            # Gymnasiumの仕様上、終了ステップのinfoは "final_info" に格納される
            real_info = info.get("final_info", info)
            
            # エピソード終了時に記録された報酬があればTensorBoardに送る
            if "ep_rew_upright" in real_info:
                self.logger.record("custom_rewards/upright_ep", real_info["ep_rew_upright"])
                # self.logger.record("custom_rewards/torque_ep", real_info["ep_rew_torque"])
                self.logger.record("custom_rewards/odometry_ep", real_info["ep_rew_odometry"])
                # self.logger.record("custom_rewards/vel_ep", real_info["ep_rew_vel"])

        return True

# --- カリキュラム更新用のコールバック ---
class CurriculumCallback(BaseCallback):
    def __init__(self, curriculum_max_steps: int, verbose=0):
        super().__init__(verbose)
        self.curriculum_max_steps = curriculum_max_steps

    def _on_step(self) -> bool:
        # SB3が管理している正確な現在のトータルステップ数
        current_step = self.num_timesteps 
        
        # 進行度を計算 (0.0 から最大 1.0 まで)
        factor = min(1.0, current_step / self.curriculum_max_steps)
        
        # 全ての並列環境に対して、計算した factor をセットするよう指示
        self.training_env.env_method("set_curriculum_factor", factor)
        
        # デバッグ用（10万ステップごとに現在値を出力）
        if current_step % 100000 == 0:
            print(f"[{current_step} steps] Curriculum Factor updated to: {factor:.3f}")
            
        return True
     
MODEL_PATH = "models/HBP_V2_mjcf/scene.xml"
# MODEL_PATH = "mjcf2/scene.xml"
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)
# log_dir = "tboard_logs/kourin21"
save_model_dir = "results/stop_withCon_v3/"
env_cfg, obs_cfg, reward_cfg, cmd_cfg = PythonConfig.get_cfgs()
train_cfg = PythonConfig.get_train_cfg()
# すべての設定を一つの辞書にまとめる
full_config = {
    "env_cfg": env_cfg,
    "obs_cfg": obs_cfg,
    "reward_cfg": reward_cfg,
    "cmd_cfg": cmd_cfg
}
# 3. 過去のログデータのクリーンアップ
def clean_log_dir(log_dir):
    if os.path.exists(log_dir): # もし同じ名前のログフォルダがすでに存在していたら
        shutil.rmtree(log_dir)  # 過去のデータが混ざらないようにフォルダごと完全に削除する
    os.makedirs(log_dir, exist_ok=True) # 新しくログ保存用の空フォルダを作成する


if __name__ == "__main__":
    # clean_log_dir(log_dir) # ログディレクトリのクリーンアップを実行
    # clean_log_dir(save_model_dir) # ベストモデル保存用のサブフォルダもクリーンアップ

    # 4. 学習設定のバックアップ保存
    # pickle.dump(
    #     [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg], # 保存する設定データのリスト
    #     open(f"{log_dir}/cfgs.pkl", "wb"), # "wb"（バイナリ書き込みモード）でファイルを開いて保存する
    # )
    env = make_vec_env(
        StandingEnv, 
        n_envs=16, 
        vec_env_cls=SubprocVecEnv,
        env_kwargs={"xml_path": MODEL_PATH, 
        "render_mode": None})

    eval_env = make_vec_env(
        StandingEnv, 
        n_envs=1, 
        env_kwargs={"xml_path": MODEL_PATH, 
        "render_mode": None})

    render_callback = RenderCallback(env)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"results/stop_withCon_v3/",
        eval_freq=5000,
        deterministic=True,
        render=False
    )

    tb_callback = TensorboardCallback()
    curriculum_callback = CurriculumCallback(curriculum_max_steps=7_000_000)

    ppo_model = PPO(**train_cfg_dict, env=env)
# 設定内容を文字列にして TensorBoard に送る
    import json
    config_str = json.dumps(full_config, indent=4)
    # model.get_logger().record("tboard_logs/kourin25/hyperparameters", config_str)
    # ppo_model = PPO.load("results/stop_withCon_v3/kourin21_con", env=env, )
    time.sleep(0.1)
    # ppo_model.learn(total_timesteps=10000000, callback=[eval_callback, tb_callback], reset_num_timesteps=False)
    ppo_model.learn(total_timesteps=10000000, callback=[eval_callback, tb_callback, curriculum_callback])

    ppo_model.save(f"results/stop_withCon_v3/kourin21_con.zip")