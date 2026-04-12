#include <Wire.h>
#include <M5Unified.h>
#include <MadgwickAHRS.h>
#include "parameters.h"
#include <PS4Controller.h>
#include <deque>

// --- 定数定義 ---
#define HISTORY_LENGTH 15
#define OBS_DIM 3
#define INPUT_SIZE (HISTORY_LENGTH * OBS_DIM) // 45次元

// Pythonの latency_step に対応 (0の場合は 1, 1以上の場合はその値を設定)
#define LATENCY_SIZE 1 

// --- グローバル変数 ---
Madgwick filter;
unsigned long microsPre;
float filtered_gy = 0.0;
float alpha = 0.70; // Python: self.alpha = 0.7

// 観測バッファ
float obs_history[INPUT_SIZE]; 
float encoder_queue[LATENCY_SIZE][OBS_DIM];

// 推論用バッファ
float input[INPUT_SIZE];
float layer1[64];
float layer2[64];
float output[2];

// モーター設定
#define flont_motor_id 0x65
#define back_motor_id 0x64
#define REG_ENABLE 0x00
#define REG_MODE 0x01
#define REG_CURRENT 0xB0
#define Speed_Readback 0x60
#define current_mode 0x03
int current_max = 460;
int zero_position = 55;

// UI・状態管理
bool buttonPressed = false;
bool emergency_button = false;
#define EMERGENCY_BUTTON_X 50
#define EMERGENCY_BUTTON_Y 200
#define EMERGENCY_CIRCLE 35

// --- ユーティリティ関数 ---
int sign(float val) { return (0.0 < val) - (val < 0.0); }

// --- 観測履歴更新 (Pythonの _get_obs を再現) ---
void update_obs_history(float roll_rad, float gyro_rad_filtered, float wheel_spd) {
    // 1. encoder_queue (レイテンシ用) の更新
    for (int i = 0; i < LATENCY_SIZE - 1; i++) {
        for (int j = 0; j < OBS_DIM; j++) {
            encoder_queue[i][j] = encoder_queue[i + 1][j];
        }
    }
    encoder_queue[LATENCY_SIZE - 1][0] = roll_rad;
    encoder_queue[LATENCY_SIZE - 1][1] = gyro_rad_filtered;
    encoder_queue[LATENCY_SIZE - 1][2] = wheel_spd;

    // 2. キューの先頭（最も古いデータ）を取得
    float actor_roll = encoder_queue[0][0];
    float actor_gyro = encoder_queue[0][1];
    float actor_drive_vel = encoder_queue[0][2];

    // 3. obs_history (15ステップ履歴) をずらして追加
    for (int i = 0; i < (HISTORY_LENGTH - 1) * OBS_DIM; i++) {
        obs_history[i] = obs_history[i + OBS_DIM];
    }
    obs_history[INPUT_SIZE - 3] = actor_roll;
    obs_history[INPUT_SIZE - 2] = actor_gyro;
    obs_history[INPUT_SIZE - 1] = actor_drive_vel;
}

// --- 推論処理 ---
void run_inference() {
    for (int i = 0; i < 64; i++) {
        float sum = b1[i];
        for (int j = 0; j < INPUT_SIZE; j++) sum += W1[i * INPUT_SIZE + j] * input[j];
        layer1[i] = tanhf(sum);
    }
    for (int i = 0; i < 64; i++) {
        float sum = b2[i];
        for (int j = 0; j < 64; j++) sum += W2[i * 64 + j] * layer1[j];
        layer2[i] = tanhf(sum);
    }
    for (int i = 0; i < 2; i++) {
        float sum = b3[i];
        for (int j = 0; j < 64; j++) sum += W3[i * 64 + j] * layer2[j];
        output[i] = constrain(sum, -1.0f, 1.0f);
    }
}

// --- モーター制御関数 ---
void set_motor_enable(uint8_t addr, bool en) {
    Wire.beginTransmission(addr);
    Wire.write(REG_ENABLE);
    Wire.write(en ? 0x01 : 0x00);
    Wire.endTransmission();
}

void set_control_mode(uint8_t addr, uint8_t mode) {
    Wire.beginTransmission(addr);
    Wire.write(REG_MODE);
    Wire.write(mode);
    Wire.endTransmission();
}

void set_current(int8_t addr, int32_t current) {
    int32_t val = current * 100;
    Wire.beginTransmission(addr);
    Wire.write(REG_CURRENT);
    Wire.write(val & 0xFF); Wire.write((val >> 8) & 0xFF);
    Wire.write((val >> 16) & 0xFF); Wire.write((val >> 24) & 0xFF);
    Wire.endTransmission();
}

float read_speed(uint8_t addr) {
    Wire.beginTransmission(addr);
    Wire.write(Speed_Readback);
    Wire.endTransmission();
    Wire.requestFrom(addr, (uint8_t)4);
    if (Wire.available() >= 4) {
        long r = (long)Wire.read() | (long)Wire.read() << 8 | (long)Wire.read() << 16 | (long)Wire.read() << 24;
        return (r / 100.0f) * (2.0f * M_PI / 60.0f);
    }
    return 0;
}

// --- 初期設定 ---
void setup() {
    auto cfg = M5.config();
    M5.begin(cfg);
    M5.Imu.begin();
    Wire.begin();
    filter.begin(25);
    PS4.begin("08:F9:E0:F5:E7:D6");

    set_motor_enable(flont_motor_id, true);
    set_motor_enable(back_motor_id, true);
    set_control_mode(flont_motor_id, current_mode);
    set_control_mode(back_motor_id, current_mode);

    // バッファ初期化
    memset(obs_history, 0, sizeof(obs_history));
    memset(encoder_queue, 0, sizeof(encoder_queue));

    M5.Display.fillScreen(TFT_BLACK);
    microsPre = micros();
}

// --- メインループ ---
void loop() {
    M5.update();

    // 1. IMU 取得 & Python式フィルタ適用
    float ax, ay, az, gx, gy, gz;
    M5.Imu.getAccel(&ax, &ay, &az);
    M5.Imu.getGyro(&gx, &gy, &gz);

    // Python: angular_vel = gyro[0] (実機の軸に合わせて gx か gy を選択)
    float gyro_rad = gx * (M_PI / 180.0f); 
    
    // Python: self.filtered_gy = -(alpha * angular_vel + (1-alpha) * self.filtered_gy)
    filtered_gy = (alpha * gyro_rad + (1.0f - alpha) * filtered_gy);
    float final_gyro_input = -filtered_gy; 

    // リミッター
    if (final_gyro_input > 2.0f) final_gyro_input = 2.0f;
    if (final_gyro_input < -2.0f) final_gyro_input = -2.0f;

    // 2. Roll 取得
    unsigned long now = micros();
    float dt = (float)(now - microsPre) / 1000000.0f;
    microsPre = now;
    if (dt > 0) filter.begin(1.0f / dt);
    filter.updateIMU(gx, gy, gz, ax, ay, az);
    float current_roll = -filter.getRoll() * (M_PI / 180.0f); // Radian

    // 3. エンコーダ取得
    float back_wheel_speed = read_speed(back_motor_id);

    // 4. 観測更新 & 推論
    update_obs_history(current_roll, final_gyro_input, back_wheel_speed);
    memcpy(input, obs_history, sizeof(input));
    run_inference();

    // 5. 出力計算
    float back_out = output[1] * current_max; // Python: action[1] がトルク

    // UI更新 (簡易)
    if (M5.Touch.getCount() > 0) {
        auto t = M5.Touch.getDetail();
        if (!buttonPressed && dist(t.x, t.y, EMERGENCY_BUTTON_X, EMERGENCY_BUTTON_Y) < EMERGENCY_CIRCLE) {
            emergency_button = !emergency_button;
            buttonPressed = true;
        }
    } else { buttonPressed = false; }

    // 6. モーター出力
    if (emergency_button) {
        set_current(flont_motor_id, back_out); // 同期
        set_current(back_motor_id, -back_out);
    } else {
        set_current(flont_motor_id, 0);
        set_current(back_motor_id, 0);
    }

    delay(2); // 制御周期調整
}

float dist(int x1, int y1, int x2, int y2) {
    return sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2));
}