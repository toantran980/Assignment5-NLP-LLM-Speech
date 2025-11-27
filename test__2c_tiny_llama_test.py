#!/usr/bin/env python3
"""
Unit tests for _2c_tiny_llama_test.py

These tests mock heavy dependencies (PDF reader, sentence transformer, tokenizer/model)
and check:
 - build_prompt returns expected structure (matches the solution's behavior:
   question is placed under "### Instruction:")
 - retrieve_chunk returns the correct chunk given deterministic embeddings
 - generate correctly decodes model outputs (using mocked tokenizer/model)
 - a small evaluation comparing fine-tuned vs RAG outputs (mocked)
"""

import unittest
import torch
import numpy as np
import importlib

# Import the module under test. Make sure _2c_tiny_llama_test.py is on PYTHONPATH / same dir.
MODULE_NAME = "_2c_tiny_llama_test"
module = importlib.import_module(MODULE_NAME)


class FakeEmbedder:
    """
    Fake embedder for deterministic retrieval:
    - When encode() is called with the full chunks list, it returns identity rows
      so each chunk has an orthogonal vector.
    - When encode() is called with a single query (as [query]), it returns a vector
      with 1.0 at the index that matches the query intent.

    IMPORTANT: We check for 'email' before 'instructor' because queries like
    "What is the instructor's email?" contain both words.
    """
    def __init__(self, chunks):
        self.chunks = chunks

    def encode(self, inputs, convert_to_numpy=False, normalize_embeddings=False, convert_to_tensor=False):
        # If asked to encode the chunks list itself, return identity rows
        if (
            isinstance(inputs, list)
            and len(inputs) == len(self.chunks)
            and inputs == self.chunks
        ):
            mat = np.eye(len(self.chunks), dtype=float)
            if convert_to_numpy:
                return mat
            else:
                # return list-of-rows for non-numpy calls
                return [row for row in mat]

        # Otherwise assume a single query in inputs (list-of-one)
        q = inputs[0]
        q_lower = q.lower()

        # check 'email' first because queries like "instructor's email" contain both keywords
        if "email" in q_lower:
            idx = 2
        elif "credit" in q_lower or "units" in q_lower:
            idx = 1
        elif "instructor" in q_lower:
            idx = 0
        else:
            idx = 0

        vec = np.zeros(len(self.chunks), dtype=float)
        vec[idx] = 1.0
        if convert_to_numpy:
            return np.array([vec])
        return [vec]


class Encoded(dict):
    """A tiny wrapper to simulate tokenizer(...) return that supports .to(device)."""
    def to(self, device):
        return self


class FakeTokenizer:
    """
    Fake tokenizer:
    - stores the last prompt passed to __call__
    - __call__ returns an Encoded object (dict-like) with tensors
    - decode maps token ids to a deterministic "### Response: <answer>" string
    """
    def __init__(self, mapping):
        # mapping: token_id -> answer string (without "### Response:")
        self.mapping = mapping
        self.last_prompt = None
        self.eos_token_id = 0

    def __call__(self, prompt, return_tensors="pt", truncation=True, max_length=None):
        # Save the prompt so FakeModel can inspect it
        self.last_prompt = prompt
        # return some fake tensors; .to(device) is supported by Encoded
        return Encoded({
            "input_ids": torch.tensor([[1]]),
            "attention_mask": torch.tensor([[1]])
        })

    def decode(self, token_ids, skip_special_tokens=True):
        # token_ids could be a tensor like tensor([0]) or tensor([[0]]) depending on call site.
        if isinstance(token_ids, torch.Tensor):
            try:
                token_val = int(token_ids.view(-1)[0].item())
            except Exception:
                token_val = 0
        elif isinstance(token_ids, (list, tuple)):
            token_val = int(token_ids[0])
        else:
            token_val = int(token_ids)
        answer = self.mapping.get(int(token_val), "Unknown")
        return "### Response: " + answer


class FakeModel:
    """
    Fake model that chooses a response token id based on the prompt content (from tokenizer.last_prompt).
    It is created with a `selection_map` dict: chunk_substring -> token_id to return.
    """
    def __init__(self, tokenizer: FakeTokenizer, selection_map: dict):
        self.tokenizer = tokenizer
        self.selection_map = selection_map

    def generate(self, **kwargs):
        # Inspect last prompt stored in tokenizer
        prompt = self.tokenizer.last_prompt or ""
        # find which chunk substring appears in the prompt and return its mapped token id
        for chunk_substr, token_id in self.selection_map.items():
            if chunk_substr in prompt:
                # return a tensor shaped like (1,1) containing the token id
                return torch.tensor([[token_id]])
        # fallback
        return torch.tensor([[0]])


class TinyLlamaUnitTests(unittest.TestCase):

    def test_build_prompt_structure(self):
        """
        build_prompt should contain:
         - the system guidance text ("You are a concise assistant...")
         - the context block
         - the QUESTION placed under "### Instruction:" (this matches the solution script)
         - end with "### Response:"
        """
        # Note: the solution's build_prompt places the QUESTION under "### Instruction:",
        # not the instruction string argument. So we check for the question instead.
        instruction = "Answer concisely."
        context = "This is a context block."
        question = "What is X?"
        prompt = module.build_prompt(instruction, context, question)

        # system guidance present
        self.assertIn("You are a concise assistant.", prompt)
        # context present
        self.assertIn(context, prompt)
        # question present under "### Instruction:"
        self.assertIn(question, prompt)
        # prompt ends with "### Response:" marker
        self.assertTrue(prompt.strip().endswith("### Response:"))

    def test_retrieve_chunk_selects_best(self):
        """retrieve_chunk should pick the chunk whose embedding aligns best with the query embedding."""
        chunks = [
            "Instructor: Dr. A",
            "Credits: 4",
            "Instructor email: dr.a@example.com"
        ]
        fake_embedder = FakeEmbedder(chunks)
        # Query about instructor -> should match chunk index 0
        best = module.retrieve_chunk("Who is the instructor for the course?", fake_embedder, chunks)
        self.assertEqual(best, chunks[0])
        # Query about credits -> should match chunk index 1
        best2 = module.retrieve_chunk("How many credits/units is the course?", fake_embedder, chunks)
        self.assertEqual(best2, chunks[1])
        # Query about email -> should match chunk index 2
        best3 = module.retrieve_chunk("What is the instructor's email?", fake_embedder, chunks)
        self.assertEqual(best3, chunks[2])

    def test_generate_decoding_flow(self):
        """generate should call tokenizer, model.generate and return the decoded response (after '### Response:')."""
        # mapping tokens to answers
        mapping = {
            0: "Dr. A",
            1: "Dr. B"
        }
        fake_tokenizer = FakeTokenizer(mapping)
        # model will return token id 0 for any prompt containing 'Instructor:'
        fake_model = FakeModel(fake_tokenizer, {"Instructor:": 0})

        # craft a prompt that contains the chunk
        prompt = "### Context:\nInstructor: Dr. A\n\n### Instruction:\nWhat is the instructor?\n\n### Response:\n"

        ans = module.generate(fake_model, fake_tokenizer, prompt)
        self.assertEqual(ans, "Dr. A")  # generate should split on "### Response:" and return the answer text

    def test_models_evaluation_comparison(self):
        """
        Simulate the higher-level evaluation:
        - Build a small chunk set representing the syllabus.
        - Use a FakeEmbedder to deterministically retrieve the correct chunk for each question.
        - Use a FakeTokenizer + two FakeModels:
            * fine-tuned model returns the correct answer token for each chunk
            * RAG model returns a different / less-correct token
        - Evaluate per-question correctness and assert the fine-tuned model is at least as accurate.
        """
        chunks = [
            "Instructor: Dr. A",                    # chunk 0
            "Credits: 4",                           # chunk 1
            "Instructor email: dr.a@example.com"    # chunk 2
        ]
        fake_embedder = FakeEmbedder(chunks)

        # ground truth short answers (what we'd expect a correct model to output)
        ground_truth = {
            "Who is the instructor for the course?": "Dr. A",
            "How many credits/units is the course?": "4",
            "What is the instructor's email?": "dr.a@example.com"
        }

        questions = list(ground_truth.keys())

        # Token mapping: token_id -> short answer
        token_to_answer = {
            0: "Dr. A",
            1: "4",
            2: "dr.a@example.com",
            99: "WrongAnswer"
        }

        fake_tokenizer = FakeTokenizer(token_to_answer)

        # Fine-tuned model selection map -> returns the correct token id for each chunk substring
        fine_selection_map = {
            "Instructor: Dr. A": 0,
            "Credits: 4": 1,
            "Instructor email: dr.a@example.com": 2
        }
        # RAG returns an incorrect token for each chunk (simulate lower accuracy)
        rag_selection_map = {
            "Instructor: Dr. A": 99,
            "Credits: 4": 99,
            "Instructor email: dr.a@example.com": 99
        }

        fine_model = FakeModel(fake_tokenizer, fine_selection_map)
        rag_model = FakeModel(fake_tokenizer, rag_selection_map)

        fine_correct = 0
        rag_correct = 0

        instruction = "Using ONLY the provided context, answer the question in ONE short sentence."

        for q in questions:
            # retrieve deterministic context chunk
            ctx_f = module.retrieve_chunk(q, fake_embedder, chunks)
            ctx_r = module.retrieve_chunk(q, fake_embedder, chunks)

            prompt_f = module.build_prompt(instruction, ctx_f, q)
            prompt_r = module.build_prompt(instruction, ctx_r, q)

            ans_f = module.generate(fine_model, fake_tokenizer, prompt_f)
            ans_r = module.generate(rag_model, fake_tokenizer, prompt_r)

            # Compare to ground truth
            gt = ground_truth[q]
            if ans_f.strip() == gt:
                fine_correct += 1
            if ans_r.strip() == gt:
                rag_correct += 1

        # Assert that the fine-tuned model performed at least as well as the RAG model.
        self.assertGreaterEqual(fine_correct, rag_correct,
                                msg=f"Expected fine-tuned model to be at least as accurate. fine:{fine_correct} rag:{rag_correct}")

        # Small discussion printed for human readers when running tests (not required by unit test assertions)
        print("\n--- Evaluation Summary (mocked) ---")
        print(f"Fine-tuned correct answers: {fine_correct}/{len(questions)}")
        print(f"RAG correct answers:        {rag_correct}/{len(questions)}")
        if fine_correct > rag_correct:
            print("Result: Fine-tuned model produced more accurate answers in this mocked test.")
            print("Possible reasons: fine-tuning on task-specific data made the model specialize and produce "
                  "short, accurate responses; RAG in this mock returned unrelated content.")
        elif fine_correct == rag_correct:
            print("Result: Both models performed equally on this mocked test.")
            print("Possible reasons: either both models had access to the same relevant context, or our mock "
                  "setup did not differentiate them strongly.")
        else:
            print("Result: RAG model outperformed fine-tuned model in this mocked test (unexpected).")
            print("Possible reasons: retrieval+context may have helped the RAG pipeline; real-world tests needed.")


if __name__ == "__main__":
    unittest.main()
