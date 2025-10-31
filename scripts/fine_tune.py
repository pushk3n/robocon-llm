import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
import os

# --- 1. 核心路径配置 (用户定义的 "宏") ---
# 脚本的父目录 (项目根目录, 即 /root/robocon-llm)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

# 模型的Hugging Face仓库ID (您指定的正确模型)
MODEL_REPO_ID = "Qwen/Qwen2.5-1.5B"

# 您希望将模型保存到的本地路径
# 路径将是: /root/robocon-llm/models/Qwen/Qwen2.5-1.5B
MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, "models", MODEL_REPO_ID)

# 输入数据集
DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "training_dataset.jsonl")

# 总输出目录 (用于日志和检查点)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", MODEL_REPO_ID)

# 最终 LoRA 适配器保存目录
ADAPTER_DIR = os.path.join(PROJECT_ROOT, "outputs", "lora_adapter", MODEL_REPO_ID)

# --- 2. 训练参数配置 ---
MAX_SEQ_LENGTH = 2048
DTYPE = None 
LOAD_IN_4BIT = True

# --- 3. 准备并加载模型 (核心逻辑) ---
# 检查模型是否已经下载到本地
if not os.path.exists(os.path.join(MODEL_SAVE_PATH, "config.json")):
    print(f"未在 '{MODEL_SAVE_PATH}' 找到本地模型。")
    print(f"正在从 Hugging Face ({MODEL_REPO_ID}) 下载模型...")
    print("这只会在第一次运行时发生。")
    
    # 从Hugging Face下载
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_REPO_ID,
        max_seq_length = MAX_SEQ_LENGTH,
        dtype = DTYPE,
        load_in_4bit = LOAD_IN_4BIT,
    )
    
    # 保存到您的本地路径以备将来使用
    print(f"正在将模型保存到 '{MODEL_SAVE_PATH}' 供下次使用...")
    model.save_pretrained(MODEL_SAVE_PATH)
    tokenizer.save_pretrained(MODEL_SAVE_PATH)
    print("模型已成功保存到本地。")

else:
    # 如果模型已存在于本地，则直接加载
    print(f"在 '{MODEL_SAVE_PATH}' 找到本地模型。")
    print("正在从本地路径加载...")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_SAVE_PATH, # 关键：从本地路径加载
        max_seq_length = MAX_SEQ_LENGTH,
        dtype = DTYPE,
        load_in_4bit = LOAD_IN_4BIT,
    )
print("模型加载完成。")

# --- 4. 配置 LoRA (PEFT) ---
print("正在配置 LoRA 适配器...")
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, 
    lora_alpha = 32,
    # Qwen 1.5, 2.0, 2.5 的目标模块通常是相同的
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_dropout = 0.05,
    bias = "none",
    use_gradient_checkpointing = True,
    random_state = 3407,
)

# --- 5. 数据格式化 ---
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples.get("output", [])
    texts = []
    for instruction, input_data, output in zip(instructions, inputs, outputs):
        text = alpaca_prompt.format(instruction, input_data, output) + tokenizer.eos_token
        texts.append(text)
    return { "text" : texts, }

print(f"正在加载数据集: {DATASET_PATH}")
dataset = load_dataset("json", data_files={"train": DATASET_PATH}, split="train")
dataset = dataset.map(formatting_prompts_func, batched=True)

# --- 6. 配置训练器 (SFTTrainer) ---
print("正在配置 SFTTrainer...")
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = MAX_SEQ_LENGTH,
    dataset_num_proc = 2,
    packing = False, 
    
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        num_train_epochs = 3,
        learning_rate = 2e-4,
        optim = "adamw_8bit",
        lr_scheduler_type = "linear",
        warmup_steps = 5,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        save_steps = 50,
        output_dir = OUTPUT_DIR,
        seed = 3407,
    ),
)

# --- 7. 开始训练 ---
print("======================")
print("     开始LoRA微调     ")
print("======================")
trainer_stats = trainer.train()

# --- 8. 保存最终的 LoRA 适配器 ---
print("训练完成！正在保存最终的 LoRA 适配器...")
model.save_pretrained(ADAPTER_DIR)

print(f"LoRA 适配器已成功保存到: {ADAPTER_DIR}")
