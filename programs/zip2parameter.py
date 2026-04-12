import torch
from stable_baselines3 import PPO
import numpy as np
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
# model_path = os.path.join(script_dir, "..", "results/betters/stop_better/best_model")
model_path = os.path.join(script_dir, "..", "results/stop_withCon_v3_0403/best_model")

# モデルの読み込み
model = PPO.load(model_path)
policy = model.policy.mlp_extractor.policy_net
action_net = model.policy.action_net

def to_cpp_array(name, tensor):
    data = tensor.detach().cpu().numpy()
    flat_data = data.flatten()
    array_str = ", ".join([f"{x:.10f}f" for x in flat_data])
    return f"const float {name}[] = {{{array_str}}};\n"

with open("./real_env/src/parameters.h", "w") as f:
    f.write("#ifndef PARAMETERS_H\n#define PARAMETERS_H\n\n")
    
    # 第1層 (mlp_extractor.policy_net[0])
    f.write(to_cpp_array("W1", policy[0].weight))
    f.write(to_cpp_array("b1", policy[0].bias))
    
    # 第2層 (mlp_extractor.policy_net[2])
    f.write(to_cpp_array("W2", policy[2].weight))
    f.write(to_cpp_array("b2", policy[2].bias))
    
    # 第3層 (action_net) ← これが最後の Gemm に相当します
    f.write(to_cpp_array("W3", action_net.weight))
    f.write(to_cpp_array("b3", action_net.bias))
    
    f.write("\n#endif")

print("parameters.h を生成しました。これをM5Stackのプロジェクトに追加してください。")
