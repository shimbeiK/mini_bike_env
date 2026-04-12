import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

def identify_motor_parameters(t, current, omega, Kt):
    """
    過渡状態の時系列データからモータパラメータ(J, D, Tf)を同定する

    Parameters:
    t       : 時間の1次元配列 (秒)
    current : 電流の1次元配列 (A)
    omega   : モータの回転速度の1次元配列 (rad/s)
    Kt      : トルク定数 (Nm/A)

    Returns:
    J_est   : 慣性モーメント (kg*m^2) -> MuJoCoの armature に相当
    D_est   : 粘性摩擦係数
    Tf_est  : クーロン摩擦
    """
    dt = t[1] - t[0]

    # 1. 速度データの平滑化 (ノイズ対策)
    # 窓枠サイズ(window_length)はデータ数やサンプリングレートに合わせて調整してください（奇数である必要があります）
    window_length = min(51, len(omega) // 2 * 2 - 1)
    if window_length < 3: window_length = 3
    omega_smoothed = savgol_filter(omega, window_length=window_length, polyorder=3)

    # 2. 角加速度の計算 (平滑化した速度を微分)
    alpha = np.gradient(omega_smoothed, dt)

    # 3. 最小二乗法のための行列作成
    # X = [角加速度, 角速度, 符号(角速度)]
    X = np.column_stack((alpha, omega_smoothed, np.sign(omega_smoothed)))
    
    # Y = 発生トルク (Kt * 電流)
    Y = Kt * current

    # 最小二乗法で theta = [J, D, Tf] を解く
    # X * theta = Y  =>  theta = (X^T * X)^-1 * X^T * Y
    theta, residuals, rank, s = np.linalg.lstsq(X, Y, rcond=None)

    J_est, D_est, Tf_est = theta
    
    return J_est, D_est, Tf_est, omega_smoothed

# ==========================================
# テスト実行用のメイン処理
# ==========================================
if __name__ == "__main__":
    # --- 1. テスト用の擬似データ生成（実データがある場合はここを置き換える） ---
    dt = 0.002  # サンプリング周期 (500Hz)
    t = np.arange(0, 2.0, dt)

    # 同定したい「真の」パラメータ
    J_true = 0.0045   # armature (kg*m^2)
    D_true = 0.012    # 粘性摩擦
    Tf_true = 0.025   # クーロン摩擦
    Kt_known = 0.11   # トルク定数 (既知とする)

    # 入力電流 (0.2秒から1.0秒までステップ状に電流を印加、その後フリーランダウン)
    current_log = np.zeros_like(t)
    current_log[(t >= 0.2) & (t <= 1.0)] = 2.5 

    # オイラー法でモータの速度をシミュレーション
    omega_log = np.zeros_like(t)
    for i in range(1, len(t)):
        tau_m = Kt_known * current_log[i-1]
        
        # 簡易的な摩擦モデル（速度ゼロ付近の静止摩擦は省略）
        if abs(omega_log[i-1]) > 1e-3:
            friction = D_true * omega_log[i-1] + Tf_true * np.sign(omega_log[i-1])
        else:
            friction = tau_m if abs(tau_m) < Tf_true else Tf_true * np.sign(tau_m)

        alpha_true = (tau_m - friction) / J_true
        omega_log[i] = omega_log[i-1] + alpha_true * dt

    # 実機データを模倣するため、速度データにガウシアンノイズを付加
    np.random.seed(42)
    omega_noisy = omega_log + np.random.normal(0, 0.5, len(t))

    # --- 2. パラメータ同定の実行 ---
    J_est, D_est, Tf_est, omega_smooth = identify_motor_parameters(t, current_log, omega_noisy, Kt_known)

    print("--- 同定結果 ---")
    print(f"慣性モーメント J (armature) : {J_est:.6f} [kg*m^2]  (真値: {J_true})")
    print(f"粘性摩擦係数 D               : {D_est:.6f}          (真値: {D_true})")
    print(f"クーロン摩擦 Tf              : {Tf_est:.6f}          (真値: {Tf_true})")

    # --- 3. 結果の可視化 ---
    plt.figure(figsize=(10, 6))
    
    plt.subplot(2, 1, 1)
    plt.plot(t, omega_noisy, label="Noisy Measurement Data", alpha=0.5)
    plt.plot(t, omega_smooth, label="Smoothed Data (Savitzky-Golay)", color='red', linewidth=2)
    plt.ylabel("Velocity [rad/s]")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(t, current_log, label="Input Current", color='green')
    plt.xlabel("Time [s]")
    plt.ylabel("Current [A]")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()