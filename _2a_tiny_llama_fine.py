#!/usr/bin/env python3
"""
Fixed version for 8GB GPUs.
Parameter-efficient LoRA tuning with small sequence length, gradient checkpointing,
and Eager attention (FlashAttention2 disabled for Windows compatibility).
"""

import torch, fitz, os
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model


PDF_PATH = "./data/input/CPSC_254_SYL.pdf"
OUTPUT_DIR = "./tiny_llama_fine_model"
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

device = "cuda" if torch.cuda.is_available() else "cpu"


# -------------------------------------------------------------------------
# PDF ingestion
# -------------------------------------------------------------------------
def load_pdf_text(path):
    doc = fitz.open(path)
    pages = [page.get_text().strip() for page in doc]
    return "\n".join(pages)


# -------------------------------------------------------------------------
# Build dataset
# -------------------------------------------------------------------------
def build_dataset(syllabus_text):
    '''Given the syllabus text, build a Dataset of QA pairs with context.'''
    base = {
        "instruction": "???", # TODO: What instruction will you give the model to obtain concise answers using the context?"
        "context": syllabus_text,
    }

    # TODO: (Optional) Expand or modify these QA pairs as needed -- How does it influence model performance?
    qa_pairs = [
        ("What is the course about?", "The syllabus describes the course's purpose."),
        ("How is the grade determined?", "The syllabus lists grading components."),
        ("What materials are required?", "The syllabus specifies required materials."),
        ("What policies should students follow?", "The syllabus outlines course policies."),
        ("What important dates are included?", "The syllabus includes important dates.")
    ]

    data = []
    for q, a in qa_pairs:
        sample = dict(base)
        sample["question"] = q
        sample["answer"] = a
        data.append(sample)

    return Dataset.from_list(data)


def format_example(example):
    text = (
        f"Instruction: {example['instruction']}\n"
        f"Context: {example['context']}\n"
        f"Question: {example['question']}\n"
        f"Answer: {example['answer']}"
    )
    return {"text": text}


# -------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading syllabus...")
    syllabus = load_pdf_text(PDF_PATH)

    print("Building dataset...")
    dataset = build_dataset(syllabus).map(format_example)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",   # Windows safe
        device_map={"": device}
    )

    print("Enabling gradient checkpointing...")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    print("Applying LoRA...")
    lora_config = LoraConfig(
        r=???,
        lora_alpha=???,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=???,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    print("Tokenizing...")
    # TODO: Fill in the blanks
    tokenized = dataset.map(
        lambda x: tokenizer(
            x["text"],
            truncation=True,
            max_length=???,
        ),
        batched=True,
        remove_columns=dataset.column_names
    )

    # FIXED COLLATOR (causal LM needs labels)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False    # causal LM
    )

    # TODO: Fill in the blanks
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=???,
        gradient_accumulation_steps=???,
        bf16=(device == "cuda"),
        learning_rate=???,
        num_train_epochs=???,
        logging_steps=???,
        save_strategy="epoch",
        report_to="none",
        optim="paged_adamw_8bit",
    )

    print("Training...")
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=tokenized,
        data_collator=data_collator,
        args=training_args
    )

    trainer.train()

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("Training complete! Model saved.")


if __name__ == "__main__":
    main()
