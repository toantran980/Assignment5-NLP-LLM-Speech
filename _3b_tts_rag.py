#!/usr/bin/env python3
"""
tts_rag.py

Batch-capable TTS utilities using pyttsx3 (offline).

Exported functions:
- tts_save(text, out_path, voice_index=None, rate=150) -> out_path
- tts_from_text_file(text_file, out_wav, voice_index=None, rate=150) -> out_wav
- tts_from_text_files(text_files, out_wavs, voice_index=None, rate=150) -> list[out_wavs]

CLI example (batch):
python tts_rag.py --files a1.txt a2.txt --outs r1.wav r2.wav
"""
import os
from typing import List

def tts_save(text: str, out_path: str, voice_index: int = None, rate: int = 150) -> str:
    """
    Save `text` to `out_path` as a WAV using pyttsx3.
    """
    try:
        import pyttsx3
    except Exception as e:
        raise ImportError("pyttsx3 required: pip install pyttsx3") from e

    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    if voice_index is not None and 0 <= voice_index < len(voices):
        engine.setProperty('voice', voices[voice_index].id)
    engine.setProperty('rate', rate)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    engine.save_to_file(text, out_path)
    engine.runAndWait()
    return out_path

def tts_from_text_file(text_file: str, out_wav: str, voice_index: int = None, rate: int = 150) -> str:
    """
    Read a text file and synthesize it to out_wav.
    """
    if not os.path.exists(text_file):
        raise FileNotFoundError(f"Text file not found: {text_file}")
    with open(text_file, "r", encoding="utf-8") as f:
        text = f.read().strip()
    if not text:
        text = "(no text)"
    # TODO: Fill in the missing argument(s)
    return tts_save(text, out_wav, voice_index, rate)

def tts_from_text_files(text_files: List[str], out_wavs: List[str], voice_index: int = None, rate: int = 150) -> List[str]:
    """
    Batch: synthesize each text_file to corresponding out_wav (lengths must match).
    Returns list of saved WAV paths.
    """
    if len(text_files) != len(out_wavs):
        raise ValueError("text_files and out_wavs must have equal length.")
    saved = []
    for tf, ow in zip(text_files, out_wavs):
        print(f"[tts_rag] Synthesizing: {tf} -> {ow}")
        # TODO: Fill in the missing argument(s)
        outp = tts_from_text_file(tf, ow, voice_index, rate)
        saved.append(outp)
    return saved

# Optional helper to play a WAV (not necessary for batch automation)
def play_wav(wav_path: str):
    try:
        import simpleaudio as sa
    except Exception as e:
        raise ImportError("simpleaudio required for playback: pip install simpleaudio") from e
    if not os.path.exists(wav_path):
        raise FileNotFoundError(wav_path)
    wave_obj = sa.WaveObject.from_wave_file(wav_path)
    play_obj = wave_obj.play()
    play_obj.wait_done()

# -------------------------
# CLI
# -------------------------
def _cli():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", nargs="+", default=["./text/output/A_What is the grading policy for this course.txt",
                                                        "./text/output/A_Who is the instructor for the course.txt",
                                                        "./text/output/A_What is the instructors email.txt",
                                                        "./text/output/A_Who is the instructor for the course.txt"], help="List of input text files.")
    parser.add_argument("--outs", nargs="+", default=["./audio/output/A_What is the grading policy for this course.wav",
                                                     "./audio/output/A_Who is the instructor for the course.wav",
                                                     "./audio/output/A_What is the instructors email.wav",
                                                     "./audio/output/A_Who is the instructor for the course.wav"], help="List of output WAV files (same length as files).")
    parser.add_argument("--voice_index", type=int, default=None)
    parser.add_argument("--rate", type=int, default=150)
    args = parser.parse_args()

    tts_from_text_files(args.files, args.outs, voice_index=args.voice_index, rate=args.rate)
    print(f"[tts_rag] Saved {len(args.files)} WAV file(s).")

if __name__ == "__main__":
    _cli()
