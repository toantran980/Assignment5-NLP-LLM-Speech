#!/usr/bin/env python3
"""
Unit tests for _2d_tiny_llama_ocr.py

These tests focus on the masked parts of the skeleton:
 - load_true_chunks
 - build_and_save_embeddings (and the saved files)
 - semantic_search_single

They patch SentenceTransformer with a small deterministic fake to avoid downloads
and to produce consistent embeddings for testing.
"""

import unittest
import tempfile
import os
import shutil
import numpy as np
import importlib
from unittest.mock import patch

MODULE_NAME = "_2d_tiny_llama_ocr"
module = importlib.import_module(MODULE_NAME)


class FakeSentenceTransformer:
    """
    Deterministic fake SentenceTransformer.
    - When encode() is called with a list longer than 1 (assumed to be 'chunks'),
      it returns an identity-like matrix (n x n).
    - When encode() is called with a single-item list [query], it returns a vector
      that has 1.0 at the index of a chunk that contains the query substring (exact
      substring match), otherwise returns small random-like but deterministic vector.
    The fake stores the last-chunks list so it can map queries -> indices.
    """
    def __init__(self, model_name=None):
        self.model_name = model_name
        self._chunks = None

    def encode(self, inputs, convert_to_numpy=False, show_progress_bar=False):
        # inputs is expected to be a list of strings
        if isinstance(inputs, (list, tuple)) and len(inputs) > 1:
            # treat as chunk embedding request
            n = len(inputs)
            # store chunks
            self._chunks = list(inputs)
            # produce identity matrix (n x n) so each chunk has orthogonal embedding
            mat = np.eye(n, dtype=float)
            return mat if convert_to_numpy else mat.tolist()
        else:
            # treat as single query
            q = inputs[0] if isinstance(inputs, (list, tuple)) else inputs
            q_lower = q.lower() if isinstance(q, str) else str(q).lower()
            if not self._chunks:
                # fallback: return zeros of dim 1
                vec = np.zeros(1, dtype=float)
                return np.array([vec]) if convert_to_numpy else [vec.tolist()]
            n = len(self._chunks)
            vec = np.zeros(n, dtype=float)
            # find chunk containing the query substring
            matched = False
            for i, c in enumerate(self._chunks):
                if q_lower.strip() == c.strip().lower() or q_lower.strip() in c.strip().lower():
                    vec[i] = 1.0
                    matched = True
                    break
            if not matched:
                # deterministic fallback: set first element small positive
                vec[0] = 1e-6
            return np.array([vec]) if convert_to_numpy else [vec.tolist()]


class TinyLlamaOCRTests(unittest.TestCase):

    def setUp(self):
        # create a temp dir for files and outputs
        self.tmpdir = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(self.tmpdir, ignore_errors=True))

    def test_csv_to_extracted_text_and_load_true_chunks(self):
        # create a small CSV
        csv_path = os.path.join(self.tmpdir, "shopping_summary.csv")
        extracted_txt = os.path.join(self.tmpdir, "extracted_receipt_contents.txt")
        rows = [
            ["store", "item", "amount"],
            ["WAL*MART", "BANANAS", "0.87"],
            ["TRADER JOE'S", "TOMATOES", "1.59"],
            ["", "", ""],  # empty row should be skipped
            ["WAL*MART", "TOTAL", "$5.11"],
        ]
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            import csv
            writer = csv.writer(f)
            writer.writerows(rows)

        # call the function under test
        extracted_lines = module.csv_to_extracted_text(csv_path, extracted_txt)

        # check that extracted_lines was returned and file exists
        self.assertTrue(os.path.exists(extracted_txt))
        self.assertIsInstance(extracted_lines, list)
        # must contain the formatted lines and the prefix "STORE:"
        self.assertTrue(all(l.startswith("STORE: ") for l in extracted_lines))
        # make sure there are exactly 3 non-empty rows => 3 lines returned
        self.assertEqual(len(extracted_lines), 3)

        # Now create a true_receipt_contents.txt with multiple paragraphs and duplicates
        true_txt = os.path.join(self.tmpdir, "true_receipt_contents.txt")
        content = (
            "WAL*MART\nOPEN 24 HOURS\n\nBANANAS 0.87\nTOTAL 5.11\n\n"
            "TRADER JOE'S\nTOMATOES 1.59\nBANANAS 0.87\n"  # BANANAS repeated to test dedup
        )
        with open(true_txt, "w", encoding="utf-8") as f:
            f.write(content)

        # call load_true_chunks (should split, normalize, deduplicate preserving order)
        chunks = module.load_true_chunks(true_txt)
        # expected to contain lines like "WAL*MART", "OPEN 24 HOURS", "BANANAS 0.87", "TOTAL 5.11", "TRADER JOE'S", "TOMATOES 1.59"
        expected_subs = ["WAL*MART", "OPEN 24 HOURS", "BANANAS 0.87", "TOTAL 5.11", "TRADER JOE'S", "TOMATOES 1.59"]
        for sub in expected_subs:
            # at least one chunk should contain the substring (case-insensitive)
            self.assertTrue(any(sub.lower() in c.lower() for c in chunks), msg=f"Missing expected chunk containing: {sub}")

        # Check dedup: BANANAS 0.87 should appear only once (case-insensitive)
        lower_chunks = [c.lower() for c in chunks]
        self.assertEqual(lower_chunks.count("bananas 0.87"), 1)

    def test_build_and_save_embeddings_and_semantic_search(self):
        # small chunks to embed
        chunks = ["BANANAS 0.87", "TOMATOES 1.59", "TOTAL 5.11"]
        out_dir = os.path.join(self.tmpdir, "out_rag")
        # patch SentenceTransformer in the module to our fake
        with patch.object(module, "SentenceTransformer", FakeSentenceTransformer):
            # call build_and_save_embeddings
            embeddings = module.build_and_save_embeddings(chunks, model_name="fake-model", out_dir=out_dir)

            # files should exist
            chunks_path = os.path.join(out_dir, "chunks.npy")
            embeddings_path = os.path.join(out_dir, "embeddings.npy")
            self.assertTrue(os.path.exists(chunks_path))
            self.assertTrue(os.path.exists(embeddings_path))

            # load them back and verify shapes
            saved_chunks = list(np.load(chunks_path, allow_pickle=True))
            saved_embeddings = np.load(embeddings_path)
            self.assertEqual(len(saved_chunks), len(chunks))
            self.assertEqual(saved_embeddings.shape[0], len(chunks))

            # check the embeddings returned by function equal the saved embeddings
            np.testing.assert_allclose(embeddings, saved_embeddings, atol=1e-8)

            # check embeddings are normalized (norms near 1)
            norms = np.linalg.norm(saved_embeddings, axis=1)
            for n in norms:
                # identity rows have norm 1.0; allow small numerical tolerance
                self.assertAlmostEqual(float(n), 1.0, places=6)

            # Now test semantic_search_single with the same fake embedder
            # Create an embedder and force it to remember the chunks by calling encode
            embedder = FakeSentenceTransformer("fake-model")
            embedder.encode(chunks, convert_to_numpy=True)  # sets internal chunks state

            # Query that exactly matches second chunk should return idx=1
            query = "TOMATOES"
            results = module.semantic_search_single(query, embedder, np.array(saved_chunks, dtype=object), saved_embeddings, top_k=1)
            self.assertIsInstance(results, list)
            self.assertGreaterEqual(len(results), 1)
            idx, matched_chunk, score = results[0]
            self.assertEqual(idx, 1)
            self.assertIn("TOMATOES", matched_chunk.upper() if isinstance(matched_chunk, str) else matched_chunk)
            # score should be a float and between -1 and 1
            self.assertIsInstance(score, float)
            self.assertGreaterEqual(score, -1.0)
            self.assertLessEqual(score, 1.0)

            # Also test top_k > 1 returns the correct number of results in descending order
            results2 = module.semantic_search_single("BANANAS", embedder, np.array(saved_chunks, dtype=object), saved_embeddings, top_k=2)
            self.assertEqual(len(results2), 2)
            # First result should be BANANAS
            self.assertEqual(results2[0][0], 0)

    def tearDown(self):
        # cleanup handled by addCleanup; nothing else needed
        pass


if __name__ == "__main__":
    unittest.main()
