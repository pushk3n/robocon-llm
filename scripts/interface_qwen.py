import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import PeftModel
import os

# --- 路径和模型配置 ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))

MODEL_ID = "Qwen/Qwen2.5-1.5B"
LORA_ADAPTER_PATH = os.path.join(project_root, "outputs", "qwen2.5-1.5b-lora-finetuned")
MAP_XML_PATH = os.path.join(script_dir, "map.xml")

# --- 1. 加载地图上下文数据 ---
try:
    with open(MAP_XML_PATH, 'r', encoding='utf-8') as f:
        map_xml_content = f.read()
    print(f"成功加载地图文件: {MAP_XML_PATH}")
except FileNotFoundError:
    print(f"错误: 未在脚本目录中找到 'map.xml' 文件。请确保文件存在。")
    exit()

# --- 2. 加载 Tokenizer 并进行关键配置 ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
# **关键修复: 确保 pad_token 设置与训练时完全一致**
tokenizer.pad_token = tokenizer.eos_token

# --- 3. 4-bit 量化配置 ---
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# --- 4. 加载基础模型和 LoRA 适配器 ---
print(f"正在加载基础模型: {MODEL_ID} (4-bit 量化)")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

print(f"正在加载 LoRA 适配器: {LORA_ADAPTER_PATH}")
model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH)

# --- 5. 合并 LoRA 权重到基础模型 ---
print("正在合并 LoRA 权重到基础模型...")
model = model.merge_and_unload()
model.config.use_cache = True
model.eval()

# --- 6. 初始化对话历史 ---
print("\n" + "="*50)
print("模型加载完成。正在初始化地图上下文...")
print("="*50 + "\n")

system_prompt = "你是2026年robocon机器人比赛的R2机器人的决策大脑。你将根据初始提供的地图XML数据来回答所有问题。"

# 将地图数据作为第一个“用户”输入，并添加一个初始的“助手”回复
conversation_history = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": f"这是本次比赛的地图数据，请加载并理解它：\n\n{map_xml_content}"},
    {"role": "assistant", "content": "地图数据已加载。我已准备好根据此地图数据回答问题。请问有什么可以帮助你的？"}
]

# 打印初始的助手回复，让用户知道可以开始提问
print(f"AI 助手 (Assistant): {conversation_history[-1]['content']}\n")


# --- 7. 交互式对话循环 ---
while True:
    user_input = input("用户 (User): ")
    if user_input.lower() in ["exit", "quit"]:
        print("退出对话模式。")
        break

    # 在循环内部，我们将用户的新输入添加到历史记录中
    conversation_history.append({"role": "user", "content": user_input})

    # 将完整的对话历史传递给模型
    chat_text = tokenizer.apply_chat_template(
        conversation_history,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    inputs = tokenizer([chat_text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=2048,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    response_text = tokenizer.decode(outputs[0, inputs['input_ids'].shape[-1]:], skip_special_tokens=True).strip()
    print(f"AI 助手 (Assistant): {response_text}\n")

    # 将模型的回复也添加到历史记录中，为下一轮对话做准备
    conversation_history.append({"role": "assistant", "content": response_text})