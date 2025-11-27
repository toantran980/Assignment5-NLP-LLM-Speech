#!/usr/bin/env python3
"""
stt_rag.py

Transcribe audio files (using Whisper by default), query a RAG adapter,
save the adapter's text responses to .txt files, and return filenames.

Differences from the original:
- CLI uses --audios and --out_txts to accept multiple inputs/outputs (lengths must match).
- No --prefer_gpu flag: GPU is used automatically when available.
- Added process_audios_to_text_files(...) batch wrapper which calls the original single-file flow.

All other internals (retrieval, reranking, prompt, generation) are unchanged.
"""
from typing import Tuple, List
import os
import textwrap
import re

# Lazy imports for heavy packages
_whisper = None
_np = None

def _ensure_whisper():
    global _whisper, _np
    if _whisper is None:
        try:
            import whisper as _w
            _whisper = _w
        except Exception as e:
            raise ImportError("whisper is required for STT backend 'whisper'. Install with: pip install -U openai-whisper") from e
    if _np is None:
        import numpy as _n
        _np = _n
    return _whisper, _np

def transcribe_audio(audio_path: str, backend: str = "whisper", language: str = "en") -> str:
    """
    Transcribe an audio file to text.
    - audio_path: path to wav/mp3
    - backend: "whisper" (default) or "google" (requires SpeechRecognition)
    Returns transcribed text (str).
    """
    backend = backend.lower()
    if backend == "whisper":
        whisper, np = _ensure_whisper()
        model = whisper.load_model("base")  # change to "small" or "tiny" depending on resources
        # Use fp16=False to avoid GPU-only numeric issues on some systems
        result = model.transcribe(audio_path, fp16=False, language=language)
        text = result.get("text", "") or ""
        return text.strip()
    elif backend == "google":
        try:
            import speech_recognition as sr
        except Exception as e:
            raise ImportError("speech_recognition required for google backend. Install: pip install SpeechRecognition") from e
        r = sr.Recognizer()
        with sr.AudioFile(audio_path) as source:
            audio = r.record(source)
        try:
            text = r.recognize_google(audio)
            return text.strip()
        except sr.UnknownValueError:
            return ""
        except sr.RequestError as e:
            raise RuntimeError(f"Google STT request failed: {e}")
    else:
        raise ValueError(f"Unsupported STT backend: {backend}")

# -------------------------
# RAG/query utilities
# -------------------------
def load_retriever(rag_dir: str):
    """
    Load chunks.npy and embeddings.npy from rag_dir and a sentence-transformers embedder.
    Returns (chunks_list, embeddings_np, embedder_instance).
    """
    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        raise ImportError("sentence-transformers and numpy are required for retrieval. Install: pip install sentence-transformers numpy") from e

    chunks_path = os.path.join(rag_dir, "chunks.npy")
    emb_path = os.path.join(rag_dir, "embeddings.npy")
    if not os.path.exists(chunks_path) or not os.path.exists(emb_path):
        raise FileNotFoundError(f"RAG artifacts not found in {rag_dir}. Expect chunks.npy and embeddings.npy")
    chunks = list(np.load(chunks_path, allow_pickle=True))
    embeddings = np.load(emb_path)
    # normalize embeddings (defensive)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embeddings = embeddings / norms
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return chunks, embeddings, embedder

def semantic_search(question: str, chunks: List[str], embeddings, embedder, top_k: int = 3):
    """
    Simple semantic search with light keyword re-ranking.
    Returns list of tuples (idx, chunk, raw_similarity).
    """
    import numpy as np
    q_emb = embedder.encode([question], convert_to_numpy=True)[0]
    q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-12)

    # Ensure q_emb has the same dimension as the embeddings (2048 in this case)
    if q_emb.shape[0] != embeddings.shape[1]:
        print(f"[stt_rag] Resizing query embedding from {q_emb.shape[0]} to {embeddings.shape[1]}")
        q_emb = np.resize(q_emb, embeddings.shape[1])

    sims = np.dot(embeddings, q_emb)  # Now they should have matching dimensions
    cand = sims.argsort()[-(top_k * 5):][::-1]
    kws = [t for t in re.findall(r"\w+", question.lower()) if len(t) > 2]
    scored = []
    for i in cand:
        chunk = chunks[i].lower()
        kw_matches = sum(1 for k in kws if k in chunk)
        score = sims[i] + 3.0 * (kw_matches / (len(kws) if kws else 1))
        scored.append((i, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    top = [(idx, chunks[idx], float(sims[idx])) for idx, _ in scored[:top_k]]
    return top

def load_adapter_and_tokenizer(adapter_dir: str, base_model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
    """
    Load tokenizer + PeftModel adapter from adapter_dir.
    Automatically moves model to GPU if available.
    Returns (tokenizer, model).
    """
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel
        import torch
    except Exception as e:
        raise ImportError("transformers, peft, and torch are required to load the adapter. Install: pip install transformers peft torch") from e
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(base_model_name, low_cpu_mem_usage=True, device_map=None)
    model = PeftModel.from_pretrained(base, adapter_dir, is_trainable=False)
    # Automatically use GPU if available
    if torch.cuda.is_available():
        try:
            model.to("cuda")
        except Exception:
            pass
    model.eval()
    return tokenizer, model

def safe_generate(model, tokenizer, prompt: str, max_new_tokens: int = 128, temp: float = 0.0) -> str:
    """
    Deterministic generation helper (greedy unless temp>0).
    """
    import torch
    device = next(model.parameters()).device
    mlen = getattr(tokenizer, "model_max_length", 2048) or 2048
    max_input_len = max(64, mlen - max_new_tokens - 4)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_len, padding=True).to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=(temp > 0.0), temperature=float(temp), pad_token_id=tokenizer.eos_token_id)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if prompt in text:
        return text.replace(prompt, "").strip()
    return text.strip()

def query_rag(question: str, rag_dir: str = "./tiny_llama_rag_model", base_model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0", top_k: int = 3) -> dict:
    """
    Query the RAG adapter for a question.
    Returns a dict with keys: question, retrieved (list), answer (str).
    """
    chunks, embeddings, embedder = load_retriever(rag_dir)
    # TODO: Fill in the argument(s) 
    top = semantic_search(???)
    joined = "\n\n".join([c for (_, c, _) in top])
    if len(joined) > 1500:
        joined = joined[:1500] + " ..."
    # TODO: Customize system prompt as needed
    prompt = (
        "### System:\n???\n\n"
        "### Context:\n" + joined + "\n\n"
        "### Instruction:\n" + question + "\n\n"
        "### Response:\n"
    )
    tok, model = load_adapter_and_tokenizer(rag_dir, base_model_name)
    ans = safe_generate(model, tok, prompt, max_new_tokens=128, temp=0.0)
    return {"question": question, "retrieved": top, "answer": ans}

# -------------------------
# High-level flow
# -------------------------
def process_audio_to_text_file(audio_path: str, out_txt_path: str, stt_backend: str = "whisper",
                               rag_dir: str = "./tiny_llama_rag_model", base_model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                               top_k: int = 3) -> str:
    """
    Full pipeline: transcribe audio -> query RAG -> save RAG answer to out_txt_path
    Returns the path to the saved text file.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    print(f"[stt_rag] Transcribing '{audio_path}' using backend={stt_backend} ...")
    # TODO: Fill in the argument(s)
    question_text = transcribe_audio(???)
    print("[stt_rag] Recognized text:")
    print(textwrap.fill(question_text or "(no text recognized)", width=100))

    if not question_text.strip():
        # Save an empty answer file (or message) so downstream automation has a file to read.
        out_text = "(no recognized speech)"
        os.makedirs(os.path.dirname(out_txt_path) or ".", exist_ok=True)
        with open(out_txt_path, "w", encoding="utf-8") as f:
            f.write(out_text + "\n")
        print(f"[stt_rag] Wrote empty response to {out_txt_path}")
        return out_txt_path

    print("[stt_rag] Querying RAG model...")
    # TODO: Fill in the argument(s)
    rag_res = query_rag(???)
    answer = rag_res.get("answer", "").strip() or "(no answer from model)"

    # Save to file (only the answer text saved, one-line or multi-line)
    os.makedirs(os.path.dirname(out_txt_path) or ".", exist_ok=True)
    with open(out_txt_path, "w", encoding="utf-8") as outf:
        outf.write(answer + "\n")

    print(f"[stt_rag] Saved RAG answer to: {out_txt_path}")
    return out_txt_path

def process_audios_to_text_files(audio_paths: List[str], out_txt_paths: List[str], stt_backend: str = "whisper",
                                 rag_dir: str = "./tiny_llama_rag_model", base_model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                                 top_k: int = 3) -> List[str]:
    """
    Batch wrapper: call process_audio_to_text_file for each (audio, out_txt) pair.
    Lengths must match. Returns list of saved text file paths.
    This wrapper does not change any internals of the single-file flow.
    """
    if len(audio_paths) != len(out_txt_paths):
        raise ValueError("audio_paths and out_txt_paths must have the same length.")
    saved = []
    for a, t in zip(audio_paths, out_txt_paths):
        print(f"[stt_rag] Processing: {a} -> {t}")
        saved_path = process_audio_to_text_file(a, t, stt_backend=stt_backend, rag_dir=rag_dir, base_model_name=base_model_name, top_k=top_k)
        saved.append(saved_path)
    return saved

# -------------------------
# CLI
# -------------------------
def _cli():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--audios", nargs="+", default=["./audio/input/Q_How many credits_units is the course.wav",
                                                        "./audio/input/Q_What is the grading policy for this course.wav",
                                                        "./audio/input/Q_What is the instructors email.wav",
                                                        "./audio/input/Q_Who is the instructor for the course.wav"], help="List of input audio files.")
    parser.add_argument("--out_txts", nargs="+", default=["./text/output/A_How many credits_units is the course.txt",
                                                        "./text/output/A_What is the grading policy for this course.txt",
                                                        "./text/output/A_What is the instructors email.txt",
                                                        "./text/output/A_Who is the instructor for the course.txt"], help="List of output text files (same length as audios).")
    parser.add_argument("--stt_backend", choices=["whisper", "google"], default="whisper")
    parser.add_argument("--rag_dir", default="./tiny_llama_rag_model")
    parser.add_argument("--base_model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--top_k", type=int, default=3)
    args = parser.parse_args()

    # Exact batch mode: lengths must match
    if len(args.audios) != len(args.out_txts):
        raise ValueError("Number of --audios must match number of --out_txts.")
    process_audios_to_text_files(args.audios, args.out_txts, stt_backend=args.stt_backend,
                                rag_dir=args.rag_dir, base_model_name=args.base_model,
                                top_k=args.top_k)

if __name__ == "__main__":
    _cli()