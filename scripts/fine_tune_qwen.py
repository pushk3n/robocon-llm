import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig
from trl import SFTTrainer
import os

# --- 1. 配置模型和数据集路径 ---

# 获取当前脚本文件的绝对路径
script_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录 (scripts目录的上两级)
project_root = os.path.dirname(os.path.dirname(script_dir))

# 你的模型ID
MODEL_ID = "hfl/chinese-llama-2-1.3b"

# 使用绝对路径构建数据集和输出目录的路径
DATASET_PATH = os.path.join(project_root, "repo", "data", "processed", "training_dataset.jsonl")
OUTPUT_DIR = os.path.join(project_root, "outputs", "qwen2.5-1.5b-lora-finetuned")

# --- 2. 数据处理 ---

# 验证文件路径是否存在
if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"数据集文件未找到，请确认路径是否正确: {DATASET_PATH}")

print(f"正在从 {DATASET_PATH} 加载数据集...")
dataset = load_dataset('json', data_files=DATASET_PATH, split='train')

# 定义一个函数来格式化你的 "instruction", "input", "output"
# 转换为 Qwen2 的 ChatML 格式
def format_dataset_to_chatml(example):
    return {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"{example['instruction']}\n{example['input']}"},
            {"role": "assistant", "content": example['output']}
        ]
    }

# 应用格式化
processed_dataset = dataset.map(format_dataset_to_chatml, remove_columns=dataset.column_names)

# --- 3. 加载 Tokenizer 和模型 ---

print(f"正在加载模型: {MODEL_ID}")
# 加载 Tokenizer
# 保持 force_download=True 确保小文件没有损坏
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, force_download=True)
tokenizer.pad_token = tokenizer.eos_token

# 4-bit 量化配置 (QLoRA)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16, # 推荐使用 bfloat16
    bnb_4bit_use_double_quant=True,
)

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

model.config.use_cache = False
model.config.pretraining_tp = 1

# --- 4. LoRA 配置 ---
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
)

# --- 5. 训练参数 ---
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,       # <--- 关键修改：减小到 1 以适应 8GB 显存
    gradient_accumulation_steps=16,      # <--- 关键修改：增大到 16 以保持有效 Batch Size (1*16)
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_steps=50,
    optim="paged_adamw_8bit",
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    report_to="none",
)

# --- 6. 初始化 Trainer ---
# 移除了所有不兼容的参数 (dataset_format, max_seq_length, tokenizer)
trainer = SFTTrainer(
    model=model,
    train_dataset=processed_dataset,
    peft_config=peft_config,
    args=training_args,
)

# --- 7. 开始训练 ---
print("开始训练...")
trainer.train()

# --- 8. 保存最终的模型 ---
print("训练完成，保存模型...")
trainer.save_model(OUTPUT_DIR)
print(f"模型已保存到 {OUTPUT_DIR}")