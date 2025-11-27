#!/usr/bin/env python3
"""
driver_stt_tts_rag.py

Batch driver that:
  - Receives lists of audio files and matching output text files
  - Optionally receives matching output WAV file paths (or derives them)
  - Calls stt_rag.process_audios_to_text_files(...) and then tts_rag.tts_from_text_files(...)
  - Raises exceptions on errors so test harnesses can detect failures
"""
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audios", nargs="+", default=["./audio/input/Q_How many credits_units is the course.wav",
                                                        "./audio/input/Q_What is the grading policy for this course.wav",
                                                        "./audio/input/Q_What is the instructors email.wav",
                                                        "./audio/input/Q_Who is the instructor for the course.wav"], help="List of input audio files.")
    parser.add_argument("--out_txts", nargs="+", default=["./text/output/A_How many credits_units is the course.txt",
                                                        "./text/output/A_What is the grading policy for this course.txt",
                                                        "./text/output/A_What is the instructors email.txt",
                                                        "./text/output/A_Who is the instructor for the course.txt"], help="List of output text files (same length as audios).")
    parser.add_argument("--out_wavs", nargs="*", default=["./audio/output/A_What is the grading policy for this course.wav",
                                                     "./audio/output/A_Who is the instructor for the course.wav",
                                                     "./audio/output/A_What is the instructors email.wav",
                                                     "./audio/output/A_Who is the instructor for the course.wav"], help="Optional list of output WAV files (same length). If omitted, derived from out_txts by replacing .txt -> .wav")
    parser.add_argument("--stt_backend", choices=["whisper", "google"], default="whisper")
    parser.add_argument("--rag_dir", default="./tiny_llama_rag_model")
    parser.add_argument("--base_model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--voice_index", type=int, default=None)
    parser.add_argument("--rate", type=int, default=150)
    args = parser.parse_args()

    audios = args.audios
    out_txts = args.out_txts
    out_wavs = args.out_wavs

    # Validate lengths
    if len(audios) != len(out_txts):
        raise ValueError("Number of --audios must match number of --out_txts.")

    if out_wavs is not None and len(out_wavs) != len(audios):
        raise ValueError("If provided, number of --out_wavs must match number of --audios.")

    # Derive out_wavs if not provided
    if out_wavs is None:
        derived = []
        for t in out_txts:
            base, ext = os.path.splitext(t)
            if ext.lower() == ".txt":
                derived.append(base + ".wav")
            else:
                derived.append(t + ".wav")
        out_wavs = derived

    # Import modules (let ImportError propagate)
    import _3a_stt_rag as stt_rag
    import _3b_tts_rag as tts_rag

    # Ensure all input audio files exist
    for a in audios:
        if not os.path.exists(a):
            raise FileNotFoundError(f"Audio file not found: {a}")

    # 1) Batch STT+RAG -> produce text files
    print(f"[driver] Running STT+RAG for {len(audios)} audio file(s)...")
    saved_txts = stt_rag.process_audios_to_text_files(
        audio_paths=audios,
        out_txt_paths=out_txts,
        stt_backend=args.stt_backend,
        rag_dir=args.rag_dir,
        base_model_name=args.base_model,
        top_k=args.top_k
    )

    # Validate text files created
    for t in saved_txts:
        if not os.path.exists(t):
            raise RuntimeError(f"STT/RAG did not create expected file: {t}")

    # 2) Batch TTS -> produce WAV files
    print(f"[driver] Running TTS for {len(saved_txts)} text file(s)...")
    saved_wavs = tts_rag.tts_from_text_files(saved_txts, out_wavs, voice_index=args.voice_index, rate=args.rate)

    # Validate WAVs created
    for w in saved_wavs:
        if not os.path.exists(w):
            raise RuntimeError(f"TTS did not create expected WAV: {w}")

    print(f"[driver] Completed processing {len(audios)} files.")
    for a,t,w in zip(audios, saved_txts, saved_wavs):
        print(f"  {a} -> {t} -> {w}")

if __name__ == "__main__":
    # Let exceptions propagate for your unit-test harness to catch
    main()
