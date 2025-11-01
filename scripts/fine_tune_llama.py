import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig  # 在 trl 0.8.6+ 中这是正确的
import os
# ====================================================================
# [!! 强制离线模式 !!]
# 告诉 Hugging Face 库，不进行任何网络请求，只使用本地缓存。
# ====================================================================
os.environ['TRANSFORMERS_OFFLINE'] = '1'


# --- 1. 路径 ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))

MODEL_ID = "hfl/chinese-alpaca-2-1.3b"
DATASET_PATH = os.path.join(project_root, "repo", "data", "processed", "training_dataset_v2.jsonl")
OUTPUT_DIR = os.path.join(project_root, "outputs", "chinese-alpaca-2-1.3b-lora-finetuned-robocon")

# --- 2. 数据 ---
print(f"正在加载数据集: {DATASET_PATH}")
dataset = load_dataset('json', data_files=DATASET_PATH, split='train')

def formatting_func(example):
    instruction = example["instruction"]
    input_text = example["input"]
    output = example["output"]

    if input_text and input_text.strip():
        text = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
    else:
        text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
    
    return {"text": text}

print("正在处理数据集...")
processed_dataset = dataset.map(formatting_func, remove_columns=list(dataset.features))

# --- 3. 模型 ---
print(f"正在加载模型: {MODEL_ID}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=False)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 修复 padding_side 警告
tokenizer.padding_side = 'right' 

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    use_cache=False, 
)
model.config.pretraining_tp = 1
print("模型加载完成。")

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

# --- 5. SFT 训练参数 (使用 SFTConfig) ---
# API 规则: dataset_text_field 和 packing 在 SFTConfig 中
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_steps=50,
    optim="paged_adamw_8bit",
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    report_to="none",
    
    # [!! 在 SFTConfig 中定义 !!]
    dataset_text_field="text",
    packing=True,
)

# --- 6. Trainer ---
# API 规则: tokenizer 和 max_seq_length 在 SFTTrainer 中
trainer = SFTTrainer(
    model=model,
    train_dataset=processed_dataset,
    peft_config=peft_config,
    args=training_args,         # 传入 SFTConfig
    
    # [!! 在 SFTTrainer 中定义 !!]
    tokenizer=tokenizer,
    max_seq_length=2048, 
)

# --- 7. 训练 ---
print("开始使用 chinese-alpaca-2-1.3b 模型进行训练...")
trainer.train()

# --- 8. 保存 ---
print("训练完成，保存模型...")
trainer.save_model(OUTPUT_DIR)
print(f"模型已保存到 {OUTPUT_DIR}")

