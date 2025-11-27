#!/usr/bin/env python3
"""
_2c_tiny_llama_test.py
Safe test script for TinyLlama fine-tuned and RAG models.

Retrieval now matches STT RAG (_3a_stt_rag.py):
- MiniLM-L6-v2 embeddings
- Normalized vectors
- Dot-product search
"""

import torch
import fitz
import numpy as np

from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForCausalLM


FINE_MODEL_DIR = "./tiny_llama_fine_model"
RAG_MODEL_DIR = "./tiny_llama_rag_model"
PDF_PATH = "./data/input/CPSC_254_SYL.pdf"

device = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------
# Chunk PDF
# ------------------------------
def load_pdf_chunks(path, chunk_size=350):
    doc = fitz.open(path)
    full = "\n".join([page.get_text().strip() for page in doc])
    words = full.split()
    chunks = [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks


# ------------------------------
# Retrieval identical to STT-RAG
# ------------------------------
def retrieve_chunk(query, embedder, chunks):
    chunk_embs = embedder.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)
    q_emb = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]

    sims = np.dot(chunk_embs, q_emb)
    best_idx = sims.argmax()

    return chunks[best_idx]


# ------------------------------
# Prompt formatting
# ------------------------------
def build_prompt(instruction, context_block, question):
    ''' Return the prompt string given the instruction, context, and question. '''
    return (
        "### System:\n"
        "You are a concise assistant. Use ONLY the provided context to answer briefly.\n\n"
        "### Context:\n"
        f"{context_block}\n\n"
        "### Instruction:\n"
        f"{question}\n\n"
        "### Response:\n"
    )


# ------------------------------
# Generate
# ------------------------------
def generate(model, tokenizer, prompt):
    ''' Generate an answer from the model given the prompt. '''
    raise NotImplementedError("Fill in the generate function logic here.")


# ------------------------------
# MAIN
# ------------------------------
def main():
    instruction = "???" # TODO: What instruction will you give the model to obtain concise answers using the context?

    print("Loading PDF...")
    chunks = load_pdf_chunks(PDF_PATH)

    print("Loading sentence transformer...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2", device=device)

    # DON'T CHANGE THESE QUESTIONS
    # You can add more for your own testing if you like, but these will be used for grading.
    questions = [
        "Who is the instructor for the course?",
        "How many credits/units is the course?",
        "What is the instructor's email?",
        "What is the grading policy for this course?",
    ]

    print("Loading models...")

    tokenizer_f = AutoTokenizer.from_pretrained(FINE_MODEL_DIR)
    model_f = AutoModelForCausalLM.from_pretrained(FINE_MODEL_DIR, device_map={"": device})

    tokenizer_r = AutoTokenizer.from_pretrained(RAG_MODEL_DIR)
    model_r = AutoModelForCausalLM.from_pretrained(RAG_MODEL_DIR, device_map={"": device})

    print("\n=== Evaluating Models (STT-RAG aligned) ===\n")

    for q in questions:
        ctx_f = retrieve_chunk(q, embedder, chunks)
        ctx_r = retrieve_chunk(q, embedder, chunks)

        prompt_f = build_prompt(instruction, ctx_f, q)
        prompt_r = build_prompt(instruction, ctx_r, q)

        ans_f = generate(model_f, tokenizer_f, prompt_f)
        ans_r = generate(model_r, tokenizer_r, prompt_r)

        sim = float(util.cos_sim(
            embedder.encode(ans_f, convert_to_tensor=True),
            embedder.encode(ans_r, convert_to_tensor=True)
        ))

        print(f"QUESTION: {q}")
        print(f"Fine-tuned Answer: {ans_f}")
        print(f"RAG Answer:       {ans_r}")
        print(f"Answer Similarity: {sim:.4f}")
        print("-" * 60)

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
