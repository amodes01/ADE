# %%capture
# !pip install pip3-autoremove
# !pip-autoremove torch torchvision torchaudio -y
# !pip install torch torchvision torchaudio xformers --index-url https://download.pytorch.org/whl/cu121
# !pip install unsloth

from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
import json
from trl import SFTTrainer
from transformers import TrainingArguments


max_seq_length = 4096 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

# Get a ready Mistral pre trained model (Mistral 7b)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/mistral-7b-bnb-4bit", # Choose ANY! eg teknium/OpenHermes-2.5-Mistral-7B
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    use_gradient_checkpointing = True,
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)



# alpaca prompt used to train mistral, suitable for our fine-tuning
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token  # Must add EOS_TOKEN

def formatting_prompts_func(examples):
    instructions =  examples["instruction"]
    inputs =        examples["input"]
    outputs =       examples["output"]

    texts = []
    for instruction, input_text, output in zip(instructions, inputs, outputs):
        if isinstance(output, dict) or isinstance(output, list):
            formatted_output = json.dumps(output, indent=2, ensure_ascii=False)  # Pretty-print JSON (we don't want to analyze it as a json but as a String)
        else:
            formatted_output = str(output)  # Store as a string if not JSON

        # Format the example using the Alpaca-style prompt
        text = alpaca_prompt.format(instruction, input_text, formatted_output) + EOS_TOKEN
        texts.append(text)

    return {"text": texts}

# Custom dataset path
jsonl_path = "./attitude_fine-tune.jsonl"

# Load dataset
dataset = load_dataset("json", data_files=jsonl_path, split="train")

# Apply formatting function to dataset
dataset = dataset.map(formatting_prompts_func, batched=True)




trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60,
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none", # Use this for WandB etc
    ),
)

# The Actual Training
trainer_stats = trainer.train()
# This will take a while



model_save_path = "./lora_model"
model.save_pretrained(model_save_path)      # Save model
tokenizer.save_pretrained(model_save_path)  # Save tokenizer too

