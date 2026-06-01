try:
    import kick_zscaler
except ImportError:
    pass

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, PreTrainedTokenizerBase
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer, SFTConfig

model_name = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"

# 4 位量化配置 -- QLoRA 中的 Q
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                    # 用 4 位精度加载模型
    bnb_4bit_quant_type="nf4",            # 量化类型
    bnb_4bit_compute_dtype="float16",     # 计算数据类型
    bnb_4bit_use_double_quant=True        # 应用嵌套量化
)

# 在 GPU 上加载要训练的模型，如果 GPU 支持, device_map="auto" 会加载到 GPU
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",

    # 普通 SFT 可以忽略此设置
    quantization_config=bnb_config,
)

model.config.use_cache = False
model.config.pretraining_tp = 1

# 加载 Llama 分词器
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

tokenizer.save_pretrained("tinyllama-1.1b-4bit")
model.save_pretrained("tinyllama-1.1b-4bit")

template_token = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

def format_prompt(example):
    """利用TinyLlama使用的<|user|>模板格式化提示词"""

    chat = example["messages"]
    prompt = template_token.apply_chat_template(chat, tokenize=False)
    return {"text": prompt}

dataset = (
    load_dataset("HuggingFaceH4/ultrachat_200k", split='test_sft')
    .select(range(3_000))
    .map(format_prompt)
    .select_columns(["text"])
)


peft_config = LoraConfig(
    lora_alpha=128,   # LoRA 缩放
    lora_dropout=0.1, # LoRA 层的 dropout
    r=64,             # Rank
    bias="none",
    task_type="CAUSAL_LM",
    target_modules = ["k_proj", "gate_proj", "v_proj", "up_proj", "q_proj", "o_proj", "down_proj"] # 目标层
)

# 准备用于训练的模型
model = prepare_model_for_kbit_training(model)  # model 是前面用 4 位量化后的模型
model = get_peft_model(model, peft_config)

output_dir = "tinyllama-1.1b-4bit-fine-tuned"

training_arguments = SFTConfig(
    output_dir=output_dir,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    optim="paged_adamw_32bit",
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    num_train_epochs=1,
    logging_steps=10,
    fp16=True,
    gradient_checkpointing=True,
    dataset_text_field="text",
    max_length=512,
)

trainer = SFTTrainer(
    model=model,  # type: ignore[arg-type]
    train_dataset=dataset,
    processing_class=tokenizer,
    args=training_arguments,
)

trainer.train()

trainer.model.save_pretrained(output_dir)  # type: ignore[union-attr]
tokenizer.save_pretrained(output_dir)  # type: ignore[union-attr]