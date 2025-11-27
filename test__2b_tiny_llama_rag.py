# test__2b_tiny_llama_rag.py
import os
import sys
import io
import tempfile
import shutil
import unittest
import importlib
from unittest import mock
import numpy as np

HERE = os.path.abspath(os.path.dirname(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

MODULE_NAME = "_2b_tiny_llama_rag"

try:
    mod = importlib.import_module(MODULE_NAME)
except Exception as e:
    raise ImportError(f"Cannot import module '{MODULE_NAME}': {e}")

# -----------------
# Dummy helpers
# -----------------
class DummyEmbedder:
    """
    When encode called with a list of chunks, return an identity-like matrix
    shaped (n_chunks, n_chunks). On later encode([query]) calls we return a
    query vector that can be used to retrieve a desired index based on query content.
    """
    def __init__(self):
        self.n = None

    def encode(self, inputs, convert_to_numpy=True, normalize_embeddings=True):
        # inputs is list of strings
        if isinstance(inputs, (list, tuple)) and len(inputs) > 1:
            # create identity-like embeddings for each chunk
            self.n = len(inputs)
            mat = np.eye(self.n, dtype=np.float32)
            # normalize (already unit) and return
            return mat
        else:
            # single-query: produce vector that picks an index based on content
            if self.n is None:
                # fallback small vector
                return np.array([[1.0]], dtype=np.float32)
            q = inputs[0]
            # choose index based on substring "pick_i_<k>"
            pick = 0
            import re
            m = re.search(r"pick_i_(\d+)", q)
            if m:
                idx = int(m.group(1))
                if 0 <= idx < self.n:
                    pick = idx
            vec = np.zeros((1, self.n), dtype=np.float32)
            vec[0, pick] = 1.0
            return vec

class DummyTokenizer:
    def __init__(self):
        self.pad_token = None
        self.eos_token = "<eos>"
        self.saved_to = None
        self.calls = []

    def __call__(self, texts, truncation=True, max_length=None, padding=None):
        # record call
        self.calls.append({"texts_len": len(texts) if isinstance(texts, list) else 1,
                           "max_length": max_length, "padding": padding})
        # return structure similar to huggingface tokenizer batched output
        if isinstance(texts, list):
            batch_size = len(texts)
            input_ids = [[1, 2, 3][: (max_length or 3)]] * batch_size
            attention_mask = [[1] * len(input_ids[0])] * batch_size
            return {"input_ids": input_ids, "attention_mask": attention_mask}
        else:
            input_ids = [1,2,3][: (max_length or 3)]
            return {"input_ids": input_ids, "attention_mask": [1]*len(input_ids)}

    def save_pretrained(self, path):
        # emulate saving
        self.saved_to = path

class DummyModel:
    def __init__(self):
        self.saved_to = None
    def save_pretrained(self, path):
        self.saved_to = path

class DummyTrainer:
    def __init__(self, *args, **kwargs):
        self.trained = False
        self._args = args
        self._kwargs = kwargs
    def train(self):
        self.trained = True
        return {"status": "ok"}
    def save_model(self, outdir):
        # attempt to call model.save_pretrained if provided in kwargs
        model = self._kwargs.get("model", None)
        if not model:
            # try to find model object in args
            for a in self._args:
                if hasattr(a, "save_pretrained"):
                    model = a
                    break
        if model and hasattr(model, "save_pretrained"):
            model.save_pretrained(outdir)

# -----------------
# Tests
# -----------------
class TinyLlamaRagTests(unittest.TestCase):
    def test_load_pdf_chunks_splits_by_words(self):
        # patch fitz.open to return pages with get_text
        fake_pages = [
            mock.MagicMock(get_text=mock.MagicMock(return_value="word1 word2 word3")),
            mock.MagicMock(get_text=mock.MagicMock(return_value="word4 word5"))
        ]
        with mock.patch.object(mod, "fitz") as fake_fitz:
            fake_fitz.open.return_value = fake_pages
            chunks = mod.load_pdf_chunks("dummy.pdf", chunk_size=2)
            # chunk_size=2 should create chunks with 2 words each
            self.assertIsInstance(chunks, list)
            self.assertGreaterEqual(len(chunks), 2)
            # ensure words present
            joined = " ".join(chunks)
            self.assertIn("word1", joined)
            self.assertIn("word5", joined)

    def test_compute_embeddings_and_retrieve_context(self):
        # patch SentenceTransformer to our DummyEmbedder
        with mock.patch.object(mod, "SentenceTransformer", return_value=DummyEmbedder()):
            chunks = ["alpha beta", "gamma delta", "epsilon zeta"]
            embs, embedder = mod.compute_embeddings(chunks)
            # embs should be numpy array with shape (3,3)
            self.assertIsInstance(embs, np.ndarray)
            self.assertEqual(embs.shape[0], len(chunks))
            # now test retrieve_context: craft query that picks index 1 (pick_i_1)
            query = "pick_i_1"  # our DummyEmbedder will set q_emb to select index 1
            ctx = mod.retrieve_context(query, chunks, embedder, embs, top_k=2)
            # returned context should contain chunk index 1 as the top result
            self.assertIn("gamma delta", ctx)

    def test_build_dataset_uses_retrieval_and_returns_dataset(self):
        # patch retrieve_context to return deterministic short context
        def fake_retrieve(q, chunks, embedder, embs, top_k=3):
            return "CTX:" + q
        with mock.patch.object(mod, "retrieve_context", new=fake_retrieve):
            chunks = ["c1", "c2", "c3"]
            dummy_embs = np.eye(3, dtype=np.float32)
            dummy_embedder = DummyEmbedder()
            ds = mod.build_dataset(chunks, dummy_embs, dummy_embedder)
            # dataset expected to have entries matching the questions list
            self.assertTrue(len(ds) >= 1)
            ex = ds[0]
            for k in ("context", "question", "instruction", "answer"):
                self.assertIn(k, ex)
            # context should be the fake one returned
            self.assertTrue(ex["context"].startswith("CTX:"))

    def test_format_example_matches_expected_prompt(self):
        example = {"context":"CTX", "question":"What?", "answer":"Ans"}
        out = mod.format_example(example)
        self.assertIsInstance(out, dict)
        text = out["text"]
        # check the prompt sections exist
        self.assertIn("### System:", text)
        self.assertIn("### Context:", text)
        self.assertIn("### Instruction:", text)
        self.assertIn("### Response:", text)
        self.assertIn("Ans", text)

    def test_save_rag_artifacts_writes_files(self):
        tmpd = tempfile.mkdtemp(prefix="rag_test_")
        try:
            # override OUTPUT_DIR temporarily
            with mock.patch.object(mod, "OUTPUT_DIR", tmpd):
                chunks = ["x y", "a b"]
                embs = np.eye(2, dtype=np.float32)
                mod.save_rag_artifacts(chunks, embs)
                # check files exist
                chunks_path = os.path.join(tmpd, "chunks.npy")
                embs_path = os.path.join(tmpd, "embeddings.npy")
                self.assertTrue(os.path.isfile(chunks_path))
                self.assertTrue(os.path.isfile(embs_path))
                # load and verify
                loaded_chunks = np.load(chunks_path, allow_pickle=True)
                loaded_embs = np.load(embs_path)
                self.assertEqual(len(loaded_chunks), 2)
                self.assertEqual(loaded_embs.shape, embs.shape)
        finally:
            shutil.rmtree(tmpd)

    def test_main_runs_with_patched_components(self):
        # patch heavy components to avoid network / GPU usage
        dummy_embedder = DummyEmbedder()
        def fake_compute_embeddings(chunks):
            # return embeddings and embedder that knows chunk count
            embs = np.eye(len(chunks), dtype=np.float32)
            return embs, dummy_embedder

        tmpd = tempfile.mkdtemp(prefix="rag_main_")
        try:
            with mock.patch.object(mod, "load_pdf_chunks", return_value=["ch0", "ch1", "ch2"]), \
                 mock.patch.object(mod, "compute_embeddings", new=fake_compute_embeddings), \
                 mock.patch.object(mod, "save_rag_artifacts") as mock_save_rag, \
                 mock.patch.object(mod, "build_dataset") as mock_build_ds, \
                 mock.patch.object(mod, "format_example", side_effect=lambda x: {"text": "T:" + x["question"]}), \
                 mock.patch.object(mod, "AutoTokenizer") as mock_tokenizer_cls, \
                 mock.patch.object(mod, "AutoModelForCausalLM") as mock_model_cls, \
                 mock.patch.object(mod, "get_peft_model", side_effect=lambda m, cfg: m) as mock_get_peft, \
                 mock.patch.object(mod, "Trainer") as mock_trainer_cls, \
                 mock.patch.object(mod, "DataCollatorForLanguageModeling") as mock_collator_cls:

                # prepare dummy tokenizer/model/trainer objects
                dummy_tokenizer = DummyTokenizer()
                dummy_model = DummyModel()
                mock_tokenizer_cls.from_pretrained.return_value = dummy_tokenizer
                mock_model_cls.from_pretrained.return_value = dummy_model

                # make Trainer(...) construct a DummyTrainer
                def trainer_factory(*args, **kwargs):
                    return DummyTrainer(*args, **kwargs)
                mock_trainer_cls.side_effect = trainer_factory
                mock_collator_cls.return_value = lambda *a, **k: None

                # Build dataset stub: return a 'datasets' like object supporting map (we will keep simple)
                from datasets import Dataset
                entries = [
                    {"context": "ctx0", "question": "q0", "instruction":"x", "answer":"a"},
                    {"context": "ctx1", "question": "q1", "instruction":"x", "answer":"a"}
                ]
                real_ds = Dataset.from_list(entries)
                mock_build_ds.return_value = real_ds

                # Override OUTPUT_DIR to temp dir
                with mock.patch.object(mod, "OUTPUT_DIR", tmpd):
                    # run main (suppress printing)
                    with mock.patch("builtins.print"):
                        mod.main()

                # ensure save_rag_artifacts was called at least once
                self.assertTrue(mock_save_rag.called)

                # Ensure tokenizer.save_pretrained was attempted (dummy saved_to attr)
                self.assertEqual(dummy_tokenizer.saved_to, tmpd)

                # Ensure model.save_pretrained was called via trainer.save_model
                self.assertEqual(dummy_model.saved_to, tmpd)

                # Check that chunk/emb files exist under tmpd (save_rag_artifacts was patched, but main still calls save after patch)
                # Note: since we patched save_rag_artifacts, files won't be created by it; but we can ensure Trainer was called.
                self.assertTrue(mock_trainer_cls.called)
        finally:
            shutil.rmtree(tmpd)

if __name__ == "__main__":
    unittest.main(verbosity=2)
