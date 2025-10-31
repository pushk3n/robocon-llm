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
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))

# **关键变更：使用正确的 chinese-alpaca-2-1.3b 模型ID**
MODEL_ID = "hfl/chinese-alpaca-2-1.3b"
DATASET_PATH = os.path.join(project_root, "repo", "data", "processed", "training_dataset.jsonl")
# **关键变更：使用匹配的输出目录名**
OUTPUT_DIR = os.path.join(project_root, "outputs", "chinese-alpaca-2-1.3b-lora-finetuned-robocon")

# --- 2. 数据处理 ---
print(f"正在从 {DATASET_PATH} 加载数据集...")
dataset = load_dataset('json', data_files=DATASET_PATH, split='train')

# **关键变更：使用Alpaca的标准指令模板，而不是ChatML**
def formatting_func(example):
    instruction = example["instruction"]
    # 将XML数据作为Input部分
    input_text = f"地图数据:\n{example['input']}"
    output = example["output"]
    # 遵循Alpaca的官方格式，构建单一的text字段
    text = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
    return {"text": text}

print("正在格式化数据集以适配Alpaca模型...")
processed_dataset = dataset.map(formatting_func, remove_columns=list(dataset.features))

# --- 3. 加载 Tokenizer 和模型 ---
print(f"正在加载模型: {MODEL_ID}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
# **关键实践：确保pad_token被设置，Llama模型同样需要**
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

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
    trust_remote_code=True,
    use_cache=False,  # 训练时禁用缓存以兼容梯度检查点
)
model.config.pretraining_tp = 1

# --- 4. LoRA 配置 (这些模块适用于Llama架构，是正确的) ---
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
)

# --- 5. 训练参数 (已为8GB显存优化，保持不变) ---
training_args = TrainingArguments(
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
)

# --- 6. 初始化 Trainer (使用正确的参数) ---
trainer = SFTTrainer(
    model=model,
    train_dataset=processed_dataset,
    peft_config=peft_config,
    dataset_text_field="text",  # **关键变更：告诉Trainer使用我们创建的'text'列**
    max_seq_length=2048,        # **关键实践：明确指定最大长度以容纳XML**
    tokenizer=tokenizer,        # **关键实践：明确传递tokenizer**
    args=training_args,
    packing=True,               # 开启packing可以提高小模型在长上下文上的训练效率
)

# --- 7. 开始训练 ---
print("开始使用 chinese-alpaca-2-1.3b 模型进行训练...")
trainer.train()

# --- 8. 保存最终的模型 ---
print("训练完成，保存模型...")
trainer.save_model(OUTPUT_DIR)
print(f"模型已保存到 {OUTPUT_DIR}")