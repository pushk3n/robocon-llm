import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
import os

# --- 1. 路径 ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))

MODEL_ID = "hfl/chinese-alpaca-2-1.3b"
DATASET_PATH = os.path.join(project_root, "repo", "data", "processed", "training_dataset.jsonl")
OUTPUT_DIR = os.path.join(project_root, "outputs", "chinese-alpaca-2-1.3b-lora-finetuned-robocon")

# --- 2. 数据 ---
dataset = load_dataset('json', data_files=DATASET_PATH, split='train')

def formatting_func(example):
    instruction = example["instruction"]
    input_text = f"地图数据:\n{example['input']}"
    output = example["output"]
    text = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
    return {"text": text}

processed_dataset = dataset.map(formatting_func, remove_columns=list(dataset.features))

# --- 3. 模型 ---
print(f"正在加载模型: {MODEL_ID}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=False)
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
    use_cache=False,
    use_safetensors=False,
)
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

# --- 5. SFT 训练参数 ---
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
    max_seq_length=2048,
)

# --- 6. Trainer ---
trainer = SFTTrainer(
    model=model,
    train_dataset=processed_dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    tokenizer=tokenizer,
    args=training_args,
    packing=True,
)

# --- 7. 训练 ---
print("开始使用 chinese-alpaca-2-1.3b 模型进行训练...")
trainer.train()

# --- 8. 保存 ---
print("训练完成，保存模型...")
trainer.save_model(OUTPUT_DIR)
print(f"模型已保存到 {OUTPUT_DIR}")
