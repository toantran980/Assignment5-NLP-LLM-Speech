#!/usr/bin/env python3
"""
Unit tests for _3a_stt_rag.py

These tests patch heavy dependencies (whisper, transformers, peft, torch)
and provide deterministic fakes so tests run quickly and reliably.

Save next to your _3a_stt_rag.py and run:
    python test__3a_stt_rag.py
or
    pytest -q
"""

import unittest
import tempfile
import os
import numpy as np
import importlib
from unittest.mock import patch

MODULE_NAME = "_3a_stt_rag"
module = importlib.import_module(MODULE_NAME)


class SimpleFakeEmbedder:
    """
    Deterministic fake embedder:
    - When encode(list_of_chunks, convert_to_numpy=True) is called it returns an identity-like matrix
      so chunk i has embedding e_i = one-hot(i).
    - When encode([query], convert_to_numpy=True) is called it returns a vector with 1.0 in the
      position of the chunk that best matches the query by keyword priority:
        1) 'email'  -> prefer chunks containing 'email'
        2) 'credit'|'credits'|'unit'|'units' -> prefer chunks containing 'credit' or 'unit'
        3) 'instructor' -> prefer chunks containing 'instructor'
      Fallback: tiny value at index 0.
    This gives deterministic behavior aligned with test expectations.
    """
    def __init__(self):
        self.chunks = None

    def encode(self, inputs, convert_to_numpy=False):
        # If inputs looks like the chunks list
        if isinstance(inputs, (list, tuple)) and len(inputs) > 1:
            self.chunks = list(inputs)
            n = len(self.chunks)
            mat = np.eye(n, dtype=float)
            return mat if convert_to_numpy else mat.tolist()

        # single query
        q = inputs[0] if isinstance(inputs, (list, tuple)) else inputs
        q_lower = (q or "").lower()
        if not self.chunks:
            return np.array([[0.0]]) if convert_to_numpy else [[0.0]]
        vec = np.zeros(len(self.chunks), dtype=float)

        # priority matching
        if "email" in q_lower:
            for i, c in enumerate(self.chunks):
                if "email" in c.lower():
                    vec[i] = 1.0
                    return (np.array([vec]) if convert_to_numpy else [vec.tolist()])
        # credits/units
        if any(tok in q_lower for tok in ("credit", "credits", "unit", "units")):
            for i, c in enumerate(self.chunks):
                if "credit" in c.lower() or "credits" in c.lower() or "unit" in c.lower() or "units" in c.lower():
                    vec[i] = 1.0
                    return (np.array([vec]) if convert_to_numpy else [vec.tolist()])
        # instructor
        if "instructor" in q_lower:
            for i, c in enumerate(self.chunks):
                if "instructor" in c.lower():
                    vec[i] = 1.0
                    return (np.array([vec]) if convert_to_numpy else [vec.tolist()])

        # fallback: very small presence at index 0
        vec[0] = 1e-6
        return (np.array([vec]) if convert_to_numpy else [vec.tolist()])


class FakeTokenizer:
    def __init__(self):
        self.eos_token_id = 0
        self.model_max_length = 512

    def __call__(self, prompt, return_tensors="pt", truncation=True, max_length=None, padding=True):
        # return a fake mapping; safe_generate tests patch behavior as needed
        return {"input_ids": None, "attention_mask": None}

    def decode(self, out, skip_special_tokens=True):
        return out


class FakeModel:
    def __init__(self):
        self._params = [1]

    def parameters(self):
        return iter([self])

    def generate(self, **kwargs):
        return None


class TestSTTRAG(unittest.TestCase):

    def test_semantic_search_returns_expected_top(self):
        # Build chunks and identity embeddings
        chunks = ["Instructor: Dr. A", "Credits: 4", "Instructor email: dr.a@example.com"]
        embedder = SimpleFakeEmbedder()
        embeddings = embedder.encode(chunks, convert_to_numpy=True)  # identity matrix (3x3)

        # Query that mentions "email" should prefer chunk index 2 now
        results = module.semantic_search("What is the instructor's email?", chunks, embeddings, embedder, top_k=1)
        self.assertIsInstance(results, list)
        self.assertGreaterEqual(len(results), 1)
        idx, chunk, score = results[0]
        # idx may be a numpy int, convert to int for comparison
        self.assertEqual(int(idx), 2)
        self.assertIn("email", chunk.lower())

        # Query about credits -> index 1
        r2 = module.semantic_search("How many credits is the course?", chunks, embeddings, embedder, top_k=1)
        self.assertEqual(int(r2[0][0]), 1)

        # Query about instructor only -> index 0
        r3 = module.semantic_search("Who is the instructor for the course?", chunks, embeddings, embedder, top_k=1)
        self.assertEqual(int(r3[0][0]), 0)

    def test_query_rag_constructs_prompt_and_returns_answer(self):
        # Patch load_retriever to return deterministic chunks/embeddings/embedder
        chunks = ["A", "B", "C"]
        embeddings = np.eye(3, dtype=float)
        fake_embedder = SimpleFakeEmbedder()
        fake_embedder.encode(chunks, convert_to_numpy=True)  # set internal chunks

        def fake_load_retriever(rag_dir):
            return chunks, embeddings, fake_embedder

        # Patch semantic_search to return a known top list
        fake_top = [(1, "B content", 0.95)]

        def fake_semantic_search(question, chunks_arg, embeddings_arg, embedder_arg, top_k=3):
            # assert that values passed are the ones from load_retriever
            self.assertEqual(chunks_arg, chunks)
            self.assertTrue(np.array_equal(embeddings_arg, embeddings))
            return fake_top

        captured = {}

        def fake_load_adapter_and_tokenizer(adapter_dir, base_model_name="x"):
            return FakeTokenizer(), FakeModel()

        def fake_safe_generate(model, tokenizer, prompt, max_new_tokens=128, temp=0.0):
            captured['prompt'] = prompt
            return "FAKE_ANSWER"

        with patch.object(module, "load_retriever", fake_load_retriever), \
             patch.object(module, "semantic_search", fake_semantic_search), \
             patch.object(module, "load_adapter_and_tokenizer", fake_load_adapter_and_tokenizer), \
             patch.object(module, "safe_generate", fake_safe_generate):
            res = module.query_rag("Who is instructor?", rag_dir="./some_dir", base_model_name="base", top_k=1)
            self.assertIsInstance(res, dict)
            self.assertEqual(res["question"], "Who is instructor?")
            self.assertEqual(res["retrieved"], fake_top)
            self.assertEqual(res["answer"], "FAKE_ANSWER")
            self.assertIn("B content", captured['prompt'])
            self.assertIn("Who is instructor?", captured['prompt'])
            self.assertIn("### System:", captured['prompt'])

    def test_process_audio_to_text_file_writes_answer_and_empty_case(self):
        # create temporary audio file and output path
        tmpdir = tempfile.mkdtemp()
        try:
            audio_path = os.path.join(tmpdir, "a.wav")
            with open(audio_path, "wb") as f:
                f.write(b"")  # empty file; transcribe_audio is patched anyway
            out_txt = os.path.join(tmpdir, "out.txt")

            # Case 1: transcription returns text -> query_rag returns an answer
            def fake_transcribe(audio_p, backend="whisper", language="en"):
                self.assertEqual(audio_p, audio_path)
                return "Who is the instructor for the course?"

            def fake_query_rag(question_text, rag_dir="./x", base_model_name="y", top_k=3):
                self.assertIn("instructor", question_text.lower())
                return {"question": question_text, "retrieved": [], "answer": "Dr. A"}

            with patch.object(module, "transcribe_audio", fake_transcribe), \
                 patch.object(module, "query_rag", fake_query_rag):
                saved = module.process_audio_to_text_file(audio_path, out_txt, stt_backend="whisper", rag_dir="./rag", top_k=1)
                self.assertEqual(saved, out_txt)
                self.assertTrue(os.path.exists(out_txt))
                with open(out_txt, "r", encoding="utf-8") as f:
                    txt = f.read().strip()
                self.assertEqual(txt, "Dr. A")

            # Case 2: transcription returns empty -> writes "(no recognized speech)"
            out_txt2 = os.path.join(tmpdir, "out2.txt")

            def fake_transcribe_empty(audio_p, backend="whisper", language="en"):
                return ""

            with patch.object(module, "transcribe_audio", fake_transcribe_empty):
                saved2 = module.process_audio_to_text_file(audio_path, out_txt2, stt_backend="whisper", rag_dir="./rag", top_k=1)
                self.assertEqual(saved2, out_txt2)
                with open(out_txt2, "r", encoding="utf-8") as f:
                    txt2 = f.read().strip()
                self.assertEqual(txt2, "(no recognized speech)")

        finally:
            try:
                # cleanup
                for p in [audio_path, out_txt, out_txt2 if 'out_txt2' in locals() else None]:
                    if p and os.path.exists(p):
                        os.remove(p)
                if os.path.isdir(tmpdir):
                    os.rmdir(tmpdir)
            except Exception:
                pass

    def test_process_audios_to_text_files_batch_wrapper(self):
        # patch process_audio_to_text_file so we don't need real audio/transcription
        calls = []

        def fake_single(a, t, stt_backend="whisper", rag_dir="./r", base_model_name="b", top_k=3):
            calls.append((a, t))
            os.makedirs(os.path.dirname(t) or ".", exist_ok=True)
            with open(t, "w", encoding="utf-8") as f:
                f.write("OK\n")
            return t

        with patch.object(module, "process_audio_to_text_file", fake_single):
            audio_paths = ["a1.wav", "a2.wav"]
            out_txts = [os.path.join(tempfile.gettempdir(), "o1.txt"), os.path.join(tempfile.gettempdir(), "o2.txt")]
            saved = module.process_audios_to_text_files(audio_paths, out_txts, stt_backend="whisper", rag_dir="./r", top_k=1)
            self.assertEqual(saved, out_txts)
            self.assertEqual(len(calls), 2)
            for p in out_txts:
                if os.path.exists(p):
                    os.remove(p)


if __name__ == "__main__":
    unittest.main()
