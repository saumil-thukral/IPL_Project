import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model

def train():
    # 1. Configuration
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    new_model_name = "ipl_analyst_adapter"
    print(f"🚀 Starting training with base model: {model_name}")

    # 2. Quantization (4-bit loading)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # 3. Load Model
    # device_map={"": 0} forces the model onto the GPU immediately
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map={"": 0},
        use_cache=False
    )
    
    # Enable gradient checkpointing and prepare for LoRA
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    # 4. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # Important for Causal LM

    # 5. Load & Format Dataset
    dataset = load_dataset("json", data_files="ipl_training_data.json", split="train")
    print(f"📚 Loaded {len(dataset)} examples.")

    # Define the formatting function
    def format_prompts(batch):
        # The column in your JSON is 'text', so we just tokenize it directly
        # If you needed to combine columns, you would do it here.
        # We append EOS token to help the model know when to stop.
        texts = [example + tokenizer.eos_token for example in batch['text']]
        
        # Tokenize
        tokenized = tokenizer(
            texts, 
            padding="max_length", 
            truncation=True, 
            max_length=512
        )
        # For Causal LM, labels are the same as input_ids
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    # Apply formatting
    tokenized_dataset = dataset.map(format_prompts, batched=True)

    # 6. LoRA Configuration
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 7. Training Arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        logging_steps=1,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
        report_to="none"
    )

    # 8. Trainer (Standard Hugging Face Trainer)
    trainer = Trainer(
        model=model,
        train_dataset=tokenized_dataset,
        args=training_args,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )

    # 9. Train
    print("🏋️‍♂️ Training started...")
    trainer.train()
    print("✅ Training finished.")

    # 10. Save Adapter
    trainer.model.save_pretrained(new_model_name)
    print(f"💾 Saved adapter to {new_model_name}")

if __name__ == "__main__":
    train()
