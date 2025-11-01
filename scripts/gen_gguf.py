import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
import subprocess
import sys

# --- 1. 路径配置 (您的路径都是正确的) ---

BASE_MODEL_ID = "hfl/chinese-alpaca-2-1.3b"
LORA_ADAPTER_PATH = "/root/robocon-llm/outputs/chinese-alpaca-2-1.3b-lora-finetuned-robocon"
MERGED_MODEL_DIR = "/root/robocon-llm/merged-model-alpaca-final"
LLAMA_CPP_DIR = "/root/llama.cpp"

# --- 新的两步GGUF转换配置 ---
INTERMEDIATE_GGUF_TYPE = "f16"  # 步骤2: 转换时的中间格式
FINAL_GGUF_TYPE = "q4_k_m"   # 步骤3: 最终想要的量化格式

GGUF_NAME_INTERMEDIATE = f"r2-robot.{INTERMEDIATE_GGUF_TYPE}.gguf"
GGUF_NAME_FINAL = f"r2-robot.{FINAL_GGUF_TYPE}.gguf"

# -----------------------------------------------

def main():
    
    # --- 步骤 1: 合并 LoRA 权重 (或跳过) ---
    print(f"--- 步骤 1: 检查权重合并 ---")
    
    final_gguf_path = os.path.join(MERGED_MODEL_DIR, GGUF_NAME_FINAL)
    intermediate_gguf_path = os.path.join(MERGED_MODEL_DIR, GGUF_NAME_INTERMEDIATE)

    if os.path.exists(final_gguf_path):
        print(f"🎉 最终 GGUF 文件 {final_gguf_path} 已存在.")
        print("所有步骤已完成, 退出脚本.")
        return

    if not os.path.exists(MERGED_MODEL_DIR):
        print(f"加载基础模型: {BASE_MODEL_ID}")
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
        print(f"加载LoRA适配器: {LORA_ADAPTER_PATH}")
        model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH)
        print("正在合并权重...")
        model = model.merge_and_unload()
        
        print("正在修正基础模型的 generation_config.json...")
        if hasattr(model, "generation_config") and model.generation_config.do_sample is False:
            model.generation_config.temperature = None
            model.generation_config.top_p = None
        print("Config 修正完毕.")

        print(f"保存合并后的完整模型到: {MERGED_MODEL_DIR}")
        os.makedirs(MERGED_MODEL_DIR, exist_ok=True)
        model.save_pretrained(MERGED_MODEL_DIR)
        tokenizer.save_pretrained(MERGED_MODEL_DIR)
        print(f"--- 步骤 1: 权重合并完成 ---")
    else:
        print(f"检测到已合并的模型目录: {MERGED_MODEL_DIR}, 跳过合并.")


    # --- 步骤 2: 转换为 f16 GGUF ---
    print(f"\n--- 步骤 2: 转换为 {INTERMEDIATE_GGUF_TYPE} GGUF ---")

    convert_script = os.path.join(LLAMA_CPP_DIR, "convert_hf_to_gguf.py")
    
    if not os.path.exists(intermediate_gguf_path):
        command = [
            sys.executable,
            convert_script,
            MERGED_MODEL_DIR,
            "--outfile",
            intermediate_gguf_path,
            "--outtype",
            INTERMEDIATE_GGUF_TYPE
        ]

        print(f"执行命令: {' '.join(command)}")
        try:
            subprocess.run(command, check=True)
            print(f"--- 步骤 2: 转换为 {INTERMEDIATE_GGUF_TYPE} GGUF 成功! ---")
        except subprocess.CalledProcessError as e:
            print(f"GGUF 转换失败. 错误信息: {e}")
            return
    else:
        print(f"检测到已存在的 {INTERMEDIATE_GGUF_TYPE} GGUF 文件, 跳过转换.")
        

    # --- 步骤 3: 量化为 q4_k_m ---
    print(f"\n--- 步骤 3: 量化为 {FINAL_GGUF_TYPE} ---")

    #
    # ===================================================================
    # !! 关键修正(v5) !!
    # 修正了 'quantize' 为 'llama-quantize'
    # ===================================================================
    quantize_exe = os.path.join(LLAMA_CPP_DIR, "build/bin/llama-quantize")
    
    if not os.path.exists(quantize_exe):
        print(f"错误: 无法在 {quantize_exe} 找到 'llama-quantize' 可执行文件.")
        print(f"请确保您已经在 {LLAMA_CPP_DIR}/build 目录中成功运行了 'cmake --build .' 命令!")
        return
        
    command = [
        quantize_exe,
        intermediate_gguf_path,  # 输入文件
        final_gguf_path,         # 输出文件
        FINAL_GGUF_TYPE          # 目标类型
    ]

    print(f"执行命令: {' '.join(command)}")
    try:
        subprocess.run(command, check=True)
        print(f"--- 步骤 3: 量化为 {FINAL_GGUF_TYPE} 成功! ---")
        print(f"\n🎉 最终模型文件已生成在: {final_gguf_path}")
        print("您现在可以将其复制到 Ollama 机器上并创建 Modelfile 了.")

    except subprocess.CalledProcessError as e:
        print(f"GGUF 量化失败. 错误信息: {e}")
    except FileNotFoundError:
        print(f"错误: 无法执行 {quantize_exe}. 权限是否正确?")

if __name__ == "__main__":
    main()