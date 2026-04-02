#include <Wire.h>
#include <M5Unified.h>
#include <MadgwickAHRS.h>
#include "parameters.h"
#include <PS4Controller.h>

// sign関数の表現
int sign(float val) {
    return (0.0 < val) - (val < 0.0);
}

// Madgwick filterのセットアップ
Madgwick filter;
unsigned long microsPerReading, microsPre;
float accelScale, gyroScale;

float ax,ay,az;
float gx,gy,gz;
float roll,pitch,yaw;
unsigned long microsNow;

//  imu角速度のローパスフィルタ
float filtered_gy = 0.0;
float alpha = 0.70;

#define BUTTON_X 15
#define BUTTON_Y 15
#define BUTTON_W 70
#define BUTTON_H 50
#define BUTTON_INC_X 110
#define BUTTON_INC_Y 90

#define EMERGENCY_BUTTON_X 50
#define EMERGENCY_BUTTON_Y 200
#define EMERGENCY_CIRCLE 35

#define OUTPUT_X 130
#define OUTPUT_Y 190
bool buttonPressed = false;
bool emergency_button = false;

// roller485のid
#define flont_motor_id 0x65
#define back_motor_id 0x64

// レジスタアドレス
#define REG_ENABLE         0x00      // モーター有効化レジスタ
#define REG_MODE           0x01      // 動作モード設定レジスタ
#define REG_SPEED        0x40      // 速度設定レジスタ
#define REG_POSITION        0x80      // 位置設定レジスタ
#define REG_CURRENT        0xB0      // 電流設定レジスタ
#define Speed_Readback     0x60      // エンコーダ(rpm)読み取りレジスタ
#define Position_Readback  0x90      // エンコーダ(位置)読み取りレジスタ

#define speed_mode   0x01      //動作モード：速度制御
#define position_mode   0x02      //動作モード：位置制御
#define current_mode    0x03      //動作モード：電流制御

// 電流指定用変数
int current_min = 41;
int current_max = 460;

// オドメトリ用変数
float odom_y = 0.0;
float theta_total = 0.0;

// コントローラ
float target_speed_max = 1.0;
float target_angel_max = M_PI / 3;
float stick_left_y = 0.0; 
float stick_right_x = 0.0; 
float wheel_radius = 0.0155;

// PPOの推論用
float input[9];   
float layer1[64];
float layer2[64];
float output[2];  
// 関数プロトタイプ
void emergency_button_draw();
void display_touch();
void run_inference();
// roller485用関数
void set_motor_enable(uint8_t address, bool enable);
void set_control_mode(uint8_t address, uint8_t mode) ;
void set_current(int8_t address, int32_t current);
void set_speed(int8_t address, int32_t current);
void set_position(int8_t address, int32_t current);
float read_speed(uint8_t address);
float read_position(uint8_t address);
int zero_position = 55;
float front_out = zero_position;

float inc_k_roll = 0.05;
float inc_k_rollvel = 0.05; 
float inc_k_tirespd = 0.05;

float k_roll = 1.0;
float k_rollvel = 1.0;     
float k_tirespd = 1.0;

void setup() {
    auto cfg = M5.config();

    M5.begin(cfg);
    M5.Imu.begin();
    Serial.begin(115200);
    Wire.begin(); 
    filter.begin(25);

    M5.Display.setTextSize(2);
    M5.Display.setTextColor(TFT_WHITE, TFT_BLACK);
    M5.Display.fillScreen(TFT_BLACK);
    PS4.begin("08:F9:E0:F5:E7:D6");

    Serial.println("--- Roller Motor Control Start ---");

    set_motor_enable(flont_motor_id, true);
    set_motor_enable(back_motor_id, true);

    set_control_mode(flont_motor_id, position_mode);
    set_control_mode(back_motor_id, current_mode);

    microsPerReading = 1000000 / 25;
    microsPre = micros();

        M5.Display.setTextSize(2);
    M5.Display.setTextColor(TFT_WHITE, TFT_BLACK);
    M5.Display.fillScreen(TFT_BLACK);

    // ボタン描画
    for(int i = 0; i < 3; i++) {
        for(int j = 0; j < 2; j++){
            int x = BUTTON_X + BUTTON_INC_X * i;
            int y = BUTTON_Y + BUTTON_INC_Y * j;
            M5.Display.fillRoundRect(x, y, BUTTON_W, BUTTON_H, 10, TFT_DARKGRAY);

            int triY1 = y + (j==0 ? 15 : BUTTON_H - 15);
            int triY2 = y + (j==0 ? BUTTON_H - 18 : 18);
            M5.Display.fillTriangle(x + BUTTON_W / 2 , triY1, x + 20, triY2, x + BUTTON_W - 20, triY2, TFT_BLACK);
        }
    }
}

void set_pid_num(){
        M5.update();

    // PID 値表示
    float pid_vals[3] = {k_roll, k_rollvel, k_tirespd};
    const char* pid_names[3] = {"k_rl", "k_vel", "k_tire"};

    for(int i = 0; i < 3; i++) {
        int x_val = BUTTON_X + BUTTON_INC_X * i - (i==1 ? 10 : 0);
        int y_val = BUTTON_Y + BUTTON_H + (BUTTON_INC_Y - BUTTON_H) / 2;
        int x_name = BUTTON_X + BUTTON_INC_X * i + BUTTON_W / 2 - 10;
        int y_name = y_val -15;

        M5.Display.setCursor(x_val, y_val);
        M5.Display.printf(i==1 ? "%.6f" : "%.4f", pid_vals[i]);

        M5.Display.setCursor(x_name, y_name);
        M5.Display.print(pid_names[i]);
    }

    static bool old_emergency_button = !emergency_button;  
    if (emergency_button != old_emergency_button) {
        if (emergency_button == true) {
            M5.Display.setTextColor(TFT_WHITE, TFT_GREEN);
            M5.Display.fillCircle(EMERGENCY_BUTTON_X, EMERGENCY_BUTTON_Y, EMERGENCY_CIRCLE, TFT_GREEN);
            M5.Display.drawCenterString("ON", EMERGENCY_BUTTON_X,EMERGENCY_BUTTON_Y - 5);
        } else {
            M5.Display.setTextColor(TFT_WHITE, TFT_RED);
            M5.Display.fillCircle(EMERGENCY_BUTTON_X, EMERGENCY_BUTTON_Y,EMERGENCY_CIRCLE, TFT_RED);
            M5.Display.drawCenterString("OFF", EMERGENCY_BUTTON_X,EMERGENCY_BUTTON_Y - 5);
        }
        old_emergency_button = emergency_button;  
        M5.Display.setTextColor(TFT_WHITE, TFT_BLACK);
    }

    if (M5.Touch.getCount() > 0) {
        auto t = M5.Touch.getDetail();
        if(!buttonPressed) {
            buttonPressed = true;
            for(int i = 0; i < 3; i++){
                for(int j = 0; j < 2; j++){
                    int x = BUTTON_X + BUTTON_INC_X * i;
                    int y = BUTTON_Y + BUTTON_INC_Y * j;
                    if(t.x > x && t.x < x + BUTTON_W && t.y > y && t.y < y + BUTTON_H){
                        float inc = (i==0 ? inc_k_roll : i==1 ? inc_k_rollvel : inc_k_tirespd);
                        if(j == 0) { if(i==0) k_roll += inc; else if(i==1) k_rollvel += inc; else if(i==2) k_tirespd += inc;}
                        else { if(i==0) k_roll -= inc; else if(i==1) k_rollvel -= inc; else if(i==2) k_tirespd -= inc; }
                    }
                }
            }
            if(t.x > EMERGENCY_BUTTON_X - EMERGENCY_CIRCLE && t.x < EMERGENCY_BUTTON_X + EMERGENCY_CIRCLE &&
               t.y > EMERGENCY_BUTTON_Y - EMERGENCY_CIRCLE && t.y < EMERGENCY_BUTTON_Y + EMERGENCY_CIRCLE){
                emergency_button = !emergency_button;
            }
        }
    } else {
        buttonPressed = false;
    }
}

void loop() {
    M5.update();

    // 画面描画
    emergency_button_draw();
    // display_touch();
    set_pid_num();

    // IMUの値取得
    M5.Imu.getAccel(&ax, &ay, &az);
    M5.Imu.getGyro(&gx, &gy, &gz);
    float gy_rad = gy * (M_PI / 180.0f);
    filtered_gy = -(alpha * gy + (1-alpha) * filtered_gy);
    if(abs(filtered_gy*M_PI/180)>2){
        filtered_gy = sign(filtered_gy)*2;
    }

    // Madgwickfilter(pitch取得)
    microsNow = micros();
    float dt = (float)(microsNow - microsPre) / 1000000.0f;
    if (dt > 0) {
        filter.begin(1.0f / dt); 
    }    

    filter.updateIMU(gx, gy, gz, ax, ay, az);
    pitch = -filter.getPitch();
    microsPre = microsNow;

    // エンコーダ値取得
    float flont_wheel_position = (read_position(flont_motor_id)*10 + zero_position) * (2 * M_PI / 360);
    float back_wheel_speed = read_speed(back_motor_id);

    // コントローラ値取得
    if (PS4.isConnected()) {
        if (PS4.LStickY()) {
            int stick_value = PS4.LStickY();
            stick_left_y = (float)stick_value / 130; //正規化 
            if(-0.1 < stick_left_y && stick_left_y < 0.1){
                stick_left_y = 0;
            }
        }
        if (PS4.RStickX()) {
            int stick_value = PS4.RStickX();
            stick_right_x = (float)stick_value / 130; //正規化 
            if(-0.1 < stick_right_x && stick_right_x < 0.1){
                stick_right_x = 0;
            }
        }
    }
    float target_angle = stick_right_x * target_angel_max;

    // オドメトリ
    float wheel_distance = 0.034;
    float delta_theta = dt * wheel_radius * back_wheel_speed * tan(flont_wheel_position) / wheel_distance;
    theta_total += delta_theta;
    float delta_x = wheel_radius * back_wheel_speed * cos(flont_wheel_position) * dt * cos(theta_total + delta_theta / 2);
    float delta_y = sign(flont_wheel_position) * sqrt(sq(wheel_radius * back_wheel_speed * dt) - sq(delta_x)) * cos(theta_total);
    odom_y += delta_y;

    // ニューラルネットワークの入力に代入
    // input[0] = 0.08236867;
    // input[1] = -0.6337656;
    // input[2] = 1.6897228;
    input[0] = pitch*M_PI/180 * k_roll;
    input[1] = filtered_gy*M_PI/180 * k_rollvel;
    input[2] = back_wheel_speed;
    // if(stick_right_x == 0){
    //     input[3] = -(stick_left_y * target_speed_max / wheel_radius);  // 停止or直進時
    // }else{
    //     input[3] = -(0.2 + 0.3*(1-abs(target_angle) / (10 * M_PI / 180))); // 旋回時
    // }
    // input[4] = input[3] - input[2];
    // // input[5] = odom_y;
    // input[5] = 0;
    // input[6] = flont_wheel_position;
    // input[7] = stick_right_x * target_angel_max;
    // input[8] = (input[7] - input[6]) / target_angel_max;
    input[3] = 0;  // 停止or直進時
    input[4] = 0;
    // input[5] = odom_y;
    input[5] = 0;
    input[6] = 0;
    input[7] = 0;
    input[8] = 0;
    run_inference();


    float front_out = -60+zero_position;
    // float front_out = output[0]*80 + zero_position;
    // front_out += 15;
    // float back_out = current_max;
    float back_out = output[1]*current_max*k_tirespd;
    M5.Display.fillRect(OUTPUT_X, OUTPUT_Y, 150, 80, TFT_BLACK);
    M5.Display.setCursor(OUTPUT_X, OUTPUT_Y);
    // 12行すべてを収めるため、文字サイズを少し小さくする
    M5.Display.setTextSize(1.5); 

    Serial.print(">pitch:");
    Serial.println(pitch);
    Serial.print(">f:");
    Serial.println(front_out);
    Serial.print(">b:");
    Serial.println(back_out);
    Serial.print(">rollvel:");
    Serial.println(input[1]*180/M_PI);
    Serial.print(">tire spd:");
    Serial.println(input[2]);

    int start_x = 120;
    int start_y = 10; // 画面の上の方（Y=10）から描画スタート
    int step_y = 18;  // 1行ごとの縦の幅

    // M5.Display.setCursor(start_x, start_y); M5.Display.printf("pitch : %.3f", pitch); start_y += step_y;
    // M5.Display.setCursor(start_x, start_y); M5.Display.printf("f : %.3f", front_out); start_y += step_y;
    // M5.Display.setCursor(start_x, start_y); M5.Display.printf("b : %.3f", back_out); start_y += step_y;
    // M5.Display.setCursor(start_x, start_y); M5.Display.printf("imu : %.3f", input[0]); start_y += step_y;
    // M5.Display.setCursor(start_x, start_y); M5.Display.printf("rollvel : %.3f", input[1]); start_y += step_y;
    // M5.Display.setCursor(start_x, start_y); M5.Display.printf("tire_spd : %.3f", input[2]); start_y += step_y;
    // M5.Display.setCursor(start_x, start_y); M5.Display.printf("tgt_vel : %.3f", input[3]); start_y += step_y;
    // M5.Display.setCursor(start_x, start_y); M5.Display.printf("diff_vel : %.3f", input[4]); start_y += step_y;
    // M5.Display.setCursor(start_x, start_y); M5.Display.printf("y : %.3f", input[5]); start_y += step_y;
    // M5.Display.setCursor(start_x, start_y); M5.Display.printf("steer : %.3f", input[6]); start_y += step_y;
    // M5.Display.setCursor(start_x, start_y); M5.Display.printf("tgt_ang : %.3f", input[7]); start_y += step_y;
    // M5.Display.setCursor(start_x, start_y); M5.Display.printf("diff_ang: %.3f", input[8]); 

    // 他の描画（緊急停止ボタンのON/OFFなど）に影響が出ないよう、文字サイズを元に戻す
    M5.Display.setTextSize(2);

    // モータのオンオフ
    if (emergency_button == false){
        front_out = -5.0;
        back_out = 0.0;
    }

    // モータ制御
    set_position(flont_motor_id, front_out);
    set_current(back_motor_id, back_out);

    delay(1.5); // 制御周期を10msにするため
}

// 緊急停止
void emergency_button_draw(){
    static bool old_emergency_button = !emergency_button;  
    if (emergency_button != old_emergency_button) {
        if (emergency_button == true) {
            M5.Display.setTextColor(TFT_WHITE, TFT_GREEN);
            M5.Display.fillCircle(EMERGENCY_BUTTON_X, EMERGENCY_BUTTON_Y, EMERGENCY_CIRCLE, TFT_GREEN);
            M5.Display.drawCenterString("ON", EMERGENCY_BUTTON_X,EMERGENCY_BUTTON_Y - 5);
        } else {
            M5.Display.setTextColor(TFT_WHITE, TFT_RED);
            M5.Display.fillCircle(EMERGENCY_BUTTON_X, EMERGENCY_BUTTON_Y,EMERGENCY_CIRCLE, TFT_RED);
            M5.Display.drawCenterString("OFF", EMERGENCY_BUTTON_X,EMERGENCY_BUTTON_Y - 5);
        }
        old_emergency_button = emergency_button;  
        M5.Display.setTextColor(TFT_WHITE, TFT_BLACK);
    }

}

// 画面タッチ認識
void display_touch() {
    if (M5.Touch.getCount() > 0) {
        auto t = M5.Touch.getDetail();
        if(!buttonPressed) {
            buttonPressed = true;
            if(t.x > EMERGENCY_BUTTON_X - EMERGENCY_CIRCLE && t.x < EMERGENCY_BUTTON_X + EMERGENCY_CIRCLE &&
               t.y > EMERGENCY_BUTTON_Y - EMERGENCY_CIRCLE && t.y < EMERGENCY_BUTTON_Y + EMERGENCY_CIRCLE){
                emergency_button = !emergency_button;
            }
        }
    } else {
        buttonPressed = false;
    }
}

// roller485のセットアップ
void set_motor_enable(uint8_t address, bool enable) {
    Wire.beginTransmission(address);
    Wire.write(REG_ENABLE);
    Wire.write(enable ? 0x01 : 0x00);
    Wire.endTransmission();
}

// roller485の動作モード設定
void set_control_mode(uint8_t address, uint8_t mode) {
    Wire.beginTransmission(address);
    Wire.write(REG_MODE);
    Wire.write(mode);
    Wire.endTransmission();
}

// roller485の電流制御
void set_current(int8_t address, int32_t current) {
    int32_t current_value = current * 100;

    Wire.beginTransmission(address);
    Wire.write(REG_CURRENT);

    // 32ビットの値をリトルエンディアンで4バイトに分割して送信
    Wire.write(current_value & 0xFF);
    Wire.write((current_value >> 8) & 0xFF);
    Wire.write((current_value >> 16) & 0xFF);
    Wire.write((current_value >> 24) & 0xFF);

    Wire.endTransmission();
}

// roller485の速度制御
void set_speed(int8_t address, int32_t current) {
    float speed_rpm = current * (60 / 2 * M_PI);
    int32_t current_value = speed_rpm * 100;

    Wire.beginTransmission(address);
    Wire.write(REG_SPEED);

    // 32ビットの値をリトルエンディアンで4バイトに分割して送信
    Wire.write(current_value & 0xFF);
    Wire.write((current_value >> 8) & 0xFF);
    Wire.write((current_value >> 16) & 0xFF);
    Wire.write((current_value >> 24) & 0xFF);

    Wire.endTransmission();
}

// 位置制御
void set_position(int8_t address, int32_t current) {
    int32_t current_value = current * 100;

    Wire.beginTransmission(address);
    Wire.write(REG_POSITION);

    // 32ビットの値をリトルエンディアンで4バイトに分割して送信
    Wire.write(current_value & 0xFF);
    Wire.write((current_value >> 8) & 0xFF);
    Wire.write((current_value >> 16) & 0xFF);
    Wire.write((current_value >> 24) & 0xFF);

    Wire.endTransmission();
}

// roller485の角速度取得
float read_speed(uint8_t address){
    Wire.beginTransmission(address);
    Wire.write(Speed_Readback);
    Wire.endTransmission();

    Wire.requestFrom(address, (uint8_t)4);

    if(Wire.available() >= 4){
        byte byte0 = Wire.read();
        byte byte1 = Wire.read();
        byte byte2 = Wire.read();
        byte byte3 = Wire.read();

        long rpm_value = (long)byte3 << 24 | (long)byte2 << 16 | (long)byte1 << 8 | (long)byte0;
        rpm_value = rpm_value / 100; // データシートより
        float speed_rad = rpm_value * (2.0f * M_PI / 60.0f);
        return speed_rad;
    }

    return 0;
}

// roller485の角度取得
float read_position(uint8_t address){
    Wire.beginTransmission(address);
    Wire.write(Position_Readback);
    Wire.endTransmission();

    Wire.requestFrom(address, (uint8_t)4);
    
    if(Wire.available() >= 4){
        byte byte0 = Wire.read();
        byte byte1 = Wire.read();
        byte byte2 = Wire.read();
        byte byte3 = Wire.read();

        long angle_deg = (long)byte3 << 24 | (long)byte2 << 16 | (long)byte1 << 8 | (long)byte0;
        angle_deg = angle_deg / 100; // データシートより
        float angle_rad = angle_deg * (2.0f * M_PI / 60.0f);
        return angle_rad;
    }

    return 0;
}

// AIの計算実行 (推論)
void run_inference() {
    // --- 第1層: input(5) -> layer1(64) ---
    for (int i = 0; i < 64; i++) {
        float sum = b1[i];
        for (int j = 0; j < 9; j++) {
            sum += W1[i * 9 + j] * input[j];
        }
        layer1[i] = tanhf(sum); 
    }

    // --- 第2層: layer1(64) -> layer2(64) ---
    for (int i = 0; i < 64; i++) {
        float sum = b2[i];
        for (int j = 0; j < 64; j++) {
            sum += W2[i * 64 + j] * layer1[j];
        }
        layer2[i] = tanhf(sum);
    }

    // --- 第3層: layer2(64) -> output(2) ---
    for (int i = 0; i < 2; i++) {
        float sum = b3[i];
        for (int j = 0; j < 64; j++) {
            sum += W3[i * 64 + j] * layer2[j];
        }
        // output[i] = tanhf(sum);
        output[i] = constrain(sum, -1.0f, 1.0f);
    }
}