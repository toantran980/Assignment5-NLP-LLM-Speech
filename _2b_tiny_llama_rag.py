#!/usr/bin/env python3
"""
_2b_tiny_llama_rag.py
RAG-enhanced LoRA fine-tuning of TinyLlama using MiniLM embeddings
so that retrieval matches the STT RAG system (_3a_stt_rag.py).

Outputs model + RAG artifacts to: ./tiny_llama_rag_model
"""

import os
import torch
import fitz
import numpy as np
from datasets import Dataset

from sentence_transformers import SentenceTransformer

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

from peft import LoraConfig, get_peft_model


# -----------------------------------
# CONFIG
# -----------------------------------
PDF_PATH = "./data/input/CPSC_254_SYL.pdf"
OUTPUT_DIR = os.path.abspath("./tiny_llama_rag_model")
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

device = "cuda" if torch.cuda.is_available() else "cpu"

print(">>> TRAINING SCRIPT RUNNING FROM:", os.getcwd())
print(">>> OUTPUT_DIR =", OUTPUT_DIR)


# -----------------------------------
# Load + Chunk PDF
# -----------------------------------
def load_pdf_chunks(path, chunk_size=400):
    doc = fitz.open(path)
    full = "\n".join([page.get_text().strip() for page in doc])
    words = full.split()
    chunks = [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks


# -----------------------------------
# Embeddings (MiniLM-L6)
# -----------------------------------
def compute_embeddings(chunks):
    """Compute normalized MiniLM embeddings compatible with STT RAG."""
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    embs = embedder.encode(
        chunks,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    return embs, embedder


def retrieve_context(query, chunks, embedder, embs, top_k=3):
    """Dot-product semantic search identical to _3a_stt_rag.py logic."""
    q_emb = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
    sims = np.dot(embs, q_emb)
    idxs = sims.argsort()[-top_k:][::-1]
    return "\n\n".join([chunks[i] for i in idxs])


# -----------------------------------
# Build Training Dataset
# -----------------------------------
def build_dataset(chunks, embs, embedder):
    '''
    Generate the dataset using retrieval to provide context for each QA pair.
    Hint: See the structure for the _2a_tiny_llama_fine.py build_dataset function.
    What is QA-training? If you don't use it, does the model still perform well?
    '''
    dataset = []
    for chunk in chunks:
        ctx = retrieve_context(chunk, chunks, embedder, embs)
        dataset.append({
            "context": ctx, 
            "question": chunk,
            "instruction": "Answer the question concisely using the provided context.",
            "answer": chunk 
        })
    
    return Dataset.from_list(dataset)


def format_example(example):
    """Use exact prompt format used by _3a_stt_rag.py during inference."""
    text = (
        "### System:\n"
        "You are a concise assistant. Use ONLY the provided context to answer briefly.\n\n"
        "### Context:\n"
        f"{example['context']}\n\n"
        "### Instruction:\n"
        f"{example['question']}\n\n"
        "### Response:\n"
        f"{example['answer']}"
    )
    return {"text": text}


# -----------------------------------
# Save RAG Artifacts
# -----------------------------------
def save_rag_artifacts(chunks, embs):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.save(os.path.join(OUTPUT_DIR, "chunks.npy"), np.array(chunks, dtype=object))
    np.save(os.path.join(OUTPUT_DIR, "embeddings.npy"), embs)
    print("\n>>> SAVED RAG ARTIFACTS (MiniLM-based)\n")


# -----------------------------------
# MAIN TRAINING
# -----------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading PDF...")
    chunks = load_pdf_chunks(PDF_PATH)

    print("Computing MiniLM embeddings...")
    embs, embedder = compute_embeddings(chunks)

    print("Saving RAG artifacts...")
    save_rag_artifacts(chunks, embs)

    print("Building dataset...")
    dataset = build_dataset(chunks, embs, embedder)
    dataset = dataset.map(format_example)

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map={"": device}
    )

    # LoRA config
    # TODO: Fill in the blanks
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)

    # Tokenize dataset
    tokenized = dataset.map(
        lambda x: tokenizer(
            x["text"], truncation=True, max_length=2048, padding="max_length"
        ),
        batched=True
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        fp16=(device == "cuda"),
        learning_rate=2e-5,
        num_train_epochs=3,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
    )

    print("\n>>> Training model...\n")
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

    print("\n>>> DONE. Model + RAG artifacts saved in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
