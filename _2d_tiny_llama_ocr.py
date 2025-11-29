#!/usr/bin/env python3
"""
tiny_llama_ocr.py

1) Converts `shopping_summary.csv` -> `extracted_receipt_contents.txt`
2) Loads ground truth `true_receipt_contents.txt`
3) Builds a retrieval index (chunks.npy + embeddings.npy) from the ground truth (RAG index)
   and saves it into `--out_dir` (defaults to ./tiny_llama_rag_model)
4) For each extracted OCR entry, retrieves the best-matching ground-truth chunk and writes
   results to `--rag_out` (defaults to rag_receipt_contents.txt)
5) Prints results for you to screenshot and performs a simple baseline-vs-rag match count.

Requirements:
  pip install sentence-transformers numpy tqdm

Note: This script takes a *retrieval-first* approach (RAG-style) using semantic embeddings.
      It does not require any LLM weights to run. If you later want to combine the retrieval
      results with a TinyLlama generator, the saved ./tiny_llama_rag_model/chunks.npy and
      embeddings.npy are compatible with the test harness you showed earlier.

Author: assistant (for your assignment)
"""
import argparse
import os
import csv
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import re
import textwrap

def csv_to_extracted_text(csv_path, extracted_txt_path):
    """
    Read shopping_summary.csv with columns store,item,amount
    and convert to a plain text file where each line is: STORE | ITEM | AMOUNT
    """
    lines = []
    with open(csv_path, newline='', encoding='utf-8', errors='replace') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        # If header looks like 'store,item,amount' skip; otherwise treat as first row
        for row in reader:
            if not row:
                continue
            # normalize row length
            if len(row) < 3:
                # pad
                row = (row + ['']*3)[:3]
            store, item, amount = row[0].strip(), row[1].strip(), row[2].strip()
            # Skip empty rows
            if not (store or item or amount):
                continue
            # Clean up common OCR artifacts
            store = re.sub(r'\s+', ' ', store)
            item  = re.sub(r'\s+', ' ', item)
            amount = amount.replace('"','').replace("'", "").strip()
            lines.append(f"STORE: {store} | ITEM: {item} | AMOUNT: {amount}")
    # write to file
    with open(extracted_txt_path, "w", encoding="utf-8") as out:
        for l in lines:
            out.write(l + "\n")
    return lines

def load_true_chunks(true_txt_path):
    """
    Load the true receipt contents file and split it into reasonable 'chunks'
    (one chunk per non-empty paragraph/line). Returns list of chunk strings.
    """
    true_txt_path = os.path.expanduser(true_txt_path)
    with open(true_txt_path, "r", encoding="utf-8", errors="replace") as f:
        true_text = f.read().strip()
    chunks = re.split(r'(?<!\n)\n(?!\n)', true_text)
    chunks = [c.strip() for c in chunks if c.strip()]
    return chunks


def build_and_save_embeddings(chunks, model_name, out_dir):
    """
    Embed the chunks with a SentenceTransformer model and save chunks.npy and embeddings.npy
    """
    chunks = [c.strip() for c in chunks if c.strip()]
    embedder = SentenceTransformer(model_name)
    embs = embedder.encode(chunks, convert_to_numpy=True)
    # normalize embeddings
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs = embs / norms
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "chunks.npy"), np.array(chunks, dtype=object))
    np.save(os.path.join(out_dir, "embeddings.npy"), embs)
    print("\n>>> SAVED RAG ARTIFACTS (MiniLM-based)\n")
    return embs

def semantic_search_single(query, embedder, chunks, embeddings, top_k=1):
    """
    Returns list of (idx, chunk, score) for top_k matches to query
    """
    q_emb = embedder.encode([query], convert_to_numpy=True)[0]
    # normalize query embedding
    q_norm = np.linalg.norm(q_emb)
    if q_norm > 0:
        q_emb = q_emb / q_norm
    sims = np.dot(embeddings, q_emb)
    idxs = sims.argsort()[-top_k:][::-1]
    return [(idx, chunks[idx], sims[idx]) for idx in idxs]

def simple_baseline_exact_match(extracted_line, true_text_lower):
    """
    Baseline: check if the item substring (item field) exists verbatim in the true text.
    We'll extract the ITEM: ... portion from the extracted line.
    """
    m = re.search(r'ITEM:\s*(.*?)\s*\|', extracted_line)
    if not m:
        # try end-of-line
        m2 = re.search(r'ITEM:\s*(.*)$', extracted_line)
        item = m2.group(1).strip() if m2 else extracted_line.strip()
    else:
        item = m.group(1).strip()
    item_norm = re.sub(r'[^0-9a-zA-Z]+', ' ', item).strip().lower()
    if not item_norm:
        return False, item
    return (item_norm in true_text_lower), item

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="./data/input/shopping_summary.csv", help="Input OCR CSV (store,item,amount)")
    parser.add_argument("--true_txt", default="./data/input/true_receipt_contents.txt", help="Human-transcribed ground truth text")
    parser.add_argument("--extracted_txt", default="./data/output/extracted_receipt_contents.txt", help="Path to write extracted plain text")
    parser.add_argument("--out_dir", default="./tiny_llama_receipt_model", help="Directory to save RAG index (chunks.npy, embeddings.npy)")
    parser.add_argument("--rag_out", default="./data/output/rag_receipt_contents.txt", help="Output RAG-mapped results")
    parser.add_argument("--embed_model", default="all-MiniLM-L6-v2", help="SentenceTransformer model for embeddings")
    parser.add_argument("--top_k", type=int, default=1, help="How many retrieval results per query")
    parser.add_argument("--sim_threshold", type=float, default=0.60, help="Similarity threshold for confident match (0..1)")
    args = parser.parse_args()

    # Step 1 -> convert CSV to extracted text
    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"CSV file not found: {args.csv}")
    print("[STEP 1] Converting CSV -> extracted text...")
    extracted_lines = csv_to_extracted_text(args.csv, args.extracted_txt)
    print(f"[STEP 1] Wrote {len(extracted_lines)} lines to {args.extracted_txt}\n")

    # Step 2 -> load true chunks
    if not os.path.exists(args.true_txt):
        raise FileNotFoundError(f"True text file not found: {args.true_txt}")
    print("[STEP 2] Loading ground-truth receipt text and splitting into chunks...")
    chunks = load_true_chunks(args.true_txt)
    print(f"[STEP 2] Got {len(chunks)} ground-truth chunks.\n")

    # Combine true text lowered for simple baseline checks
    with open(args.true_txt, "r", encoding="utf-8", errors="replace") as f:
        true_text_lower = f.read().lower()

    # Step 3 -> build embeddings and save (RAG index)
    embeddings_path = os.path.join(args.out_dir, "embeddings.npy")
    chunks_path = os.path.join(args.out_dir, "chunks.npy")
    if os.path.exists(embeddings_path) and os.path.exists(chunks_path):
        print("[STEP 3] Found existing embeddings and chunks; loading them...")
        chunks = list(np.load(chunks_path, allow_pickle=True))
        embeddings = np.load(embeddings_path)
        print(f"[STEP 3] Loaded {len(chunks)} chunks and embeddings.")
        embedder = SentenceTransformer(args.embed_model)  # needed for queries
    else:
        print("[STEP 3] Computing embeddings for ground-truth chunks (this may take a moment)...")
        embedder = SentenceTransformer(args.embed_model)
        embeddings = build_and_save_embeddings(chunks, args.embed_model, args.out_dir)

    # Step 4 -> For each extracted OCR line, retrieve best-matching ground-truth chunk
    print("\n[STEP 4] Running retrieval for each extracted OCR line...")
    rag_results = []
    baseline_matches = 0
    retrieval_confident = 0

    for i, line in enumerate(tqdm(extracted_lines, desc="queries")):
        # Baseline exact substring match test
        baseline_ok, item_str = simple_baseline_exact_match(line, true_text_lower)
        if baseline_ok:
            baseline_matches += 1

        # Form a query combining store + item (better context)
        q = line
        hits = semantic_search_single(q, embedder, np.array(chunks, dtype=object), embeddings, top_k=args.top_k)
        best_hit = hits[0] if hits else (None, "", 0.0)
        idx, matched_chunk, score = best_hit
        confident = score >= args.sim_threshold
        if confident:
            retrieval_confident += 1

        rag_results.append({
            "extracted_line": line,
            "item_field": item_str,
            "baseline_exact_match": baseline_ok,
            "matched_idx": int(idx) if idx is not None else None,
            "matched_chunk": matched_chunk,
            "similarity": score,
            "confident": confident
        })

    # Step 5 -> write rag_receipt_contents.txt and a simple purchased-items list
    print(f"\n[STEP 5] Writing RAG results to {args.rag_out} ...")
    with open(args.rag_out, "w", encoding="utf-8") as out:
        out.write("extracted_line\titem_field\tbaseline_exact_match\tmatched_idx\tsimilarity\tconfident\tmatched_chunk\n")
        for r in rag_results:
            out.write(
                f"{r['extracted_line']}\t{r['item_field']}\t{r['baseline_exact_match']}\t"
                f"{r['matched_idx']}\t{r['similarity']:.4f}\t{r['confident']}\t{r['matched_chunk']}\n"
            )

    # Also write a simplified purchased-items file (best-guess items)
    purchased_out = os.path.splitext(args.rag_out)[0] + "_purchased_items.txt"
    with open(purchased_out, "w", encoding="utf-8") as p:
        p.write("extracted_line\tbest_matched_true_text\tscore\n")
        for r in rag_results:
            p.write(f"{r['extracted_line']}\t{r['matched_chunk']}\t{r['similarity']:.4f}\n")

    # Summary report printed to console for quick screenshot
    total = len(rag_results)
    print("\nSUMMARY:")
    print(f" - Total extracted OCR lines: {total}")
    print(f" - Baseline exact-substring matches (before retrieval): {baseline_matches}/{total}")
    print(f" - Retrieval confident matches (score >= {args.sim_threshold}): {retrieval_confident}/{total}")
    print(f" - RAG results written to: {args.rag_out}")
    print(f" - Purchased-items guess file: {purchased_out}")
    print(f" - RAG index files (chunks + embeddings) in: {args.out_dir}")
    print("\nExample top results (first 10):\n")
    for r in rag_results[:10]:
        print("-" * 80)
        print("Extracted: ", r['extracted_line'])
        print("Item field:", r['item_field'])
        print("Baseline exact match:", r['baseline_exact_match'])
        print("Best match (score {:.4f}):".format(r['similarity']))
        print(textwrap.fill(r['matched_chunk'], width=100))
    print("\nDONE.")

if __name__ == "__main__":
    main()
