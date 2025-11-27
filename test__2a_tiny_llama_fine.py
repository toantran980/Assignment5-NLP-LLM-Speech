# test__2a_tiny_llama_fine.py
import os
import io
import sys
import unittest
import importlib
from unittest import mock

# Ensure current dir is importable
HERE = os.path.abspath(os.path.dirname(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

MODULE_NAME = "_2a_tiny_llama_fine"

# Try importing the solution module up front so tests fail fast with a clear message if it's missing.
try:
    mod = importlib.import_module(MODULE_NAME)
except Exception as e:
    raise ImportError(
        f"Cannot import module '{MODULE_NAME}'. Make sure '{MODULE_NAME}.py' is in the same directory as this test and is importable. Original error: {e}"
    )


# --- Dummy helper classes used to patch heavy behaviors ---
class DummyTokenizer:
    def __init__(self):
        self.pad_token = None
        self.eos_token = "<eos>"
        self.saved_to = None
        self.last_call_kwargs = None

    def __call__(self, texts, truncation=True, max_length=None, **kwargs):
        # record last call (works for both batch and single)
        self.last_call_kwargs = {"truncation": truncation, "max_length": max_length, **kwargs}
        if isinstance(texts, list):
            # emulate batched return
            input_ids = [[1, 2, 3][: (max_length or 3)]] * len(texts)
            attention_mask = [[1] * len(input_ids[0])] * len(texts)
            return {"input_ids": input_ids, "attention_mask": attention_mask}
        else:
            return {"input_ids": [1, 2, 3][: (max_length or 3)], "attention_mask": [1] * (max_length or 3)}

    def save_pretrained(self, path):
        self.saved_to = path


class DummyModel:
    def __init__(self):
        self.saved_to = None
        self.gc_enabled = False

    def gradient_checkpointing_enable(self, **kwargs):
        self.gc_enabled = True

    def save_pretrained(self, path):
        self.saved_to = path


class DummyTrainer:
    def __init__(self, *args, **kwargs):
        self.trained = False
        # store args for inspection if needed
        self._args = args
        self._kwargs = kwargs

    def train(self):
        self.trained = True
        return {"status": "ok"}

    def save_model(self, outdir):
        # emulate trainer save by calling model.save_pretrained if model present in kwargs
        model = None
        if "model" in self._kwargs:
            model = self._kwargs["model"]
        elif len(self._args) > 0:
            # sometimes trainer args in positional form; attempt to find model-like object
            for a in self._args:
                if hasattr(a, "save_pretrained"):
                    model = a
                    break
        if model is not None:
            model.save_pretrained(outdir)


class GetPeftCapture:
    def __init__(self):
        self.called = False
        self.last_cfg = None
        self.last_model = None

    def __call__(self, model, lora_config):
        self.called = True
        # try to capture a few expected attributes (works with dataclass or simple object)
        cfg = {}
        for k in ("r", "lora_alpha", "target_modules", "lora_dropout", "bias", "task_type"):
            cfg[k] = getattr(lora_config, k, None) if hasattr(lora_config, k) else (lora_config.get(k) if isinstance(lora_config, dict) else None)
        self.last_cfg = cfg
        self.last_model = model
        # tag model to indicate LoRA was applied
        setattr(model, "_lora_applied", True)
        return model


# --- Test cases ---
class TinyLlamaFineTests(unittest.TestCase):
    def test_load_pdf_text_concatenates_pages(self):
        # patch fitz.open to return iterable of page objects with get_text()
        fake_pages = [
            mock.MagicMock(get_text=mock.MagicMock(return_value="Page one.\n")),
            mock.MagicMock(get_text=mock.MagicMock(return_value="Page two.\n"))
        ]
        with mock.patch.object(mod, "fitz") as fake_fitz:
            fake_fitz.open.return_value = fake_pages
            combined = mod.load_pdf_text("dummy.pdf")
            self.assertIn("Page one.", combined)
            self.assertIn("Page two.", combined)
            self.assertTrue(combined.startswith("Page one"))

    def test_build_dataset_structure_and_content(self):
        syllabus = "This is a fake syllabus for unit testing."
        ds = mod.build_dataset(syllabus)
        # dataset should support len and indexing (datasets.Dataset)
        self.assertTrue(len(ds) >= 1, "Expected at least one QA pair")
        # check keys in an example
        example = ds[0]
        for key in ("instruction", "context", "question", "answer"):
            self.assertIn(key, example)
        # context should match input
        self.assertEqual(example["context"], syllabus)

    def test_format_example_includes_required_sections(self):
        example = {
            "instruction": "Answer concisely using context only.",
            "context": "SOME CONTEXT",
            "question": "What is X?",
            "answer": "X is Y."
        }
        out = mod.format_example(example)
        self.assertIsInstance(out, dict)
        text = out.get("text", "")
        self.assertIn("Instruction:", text)
        self.assertIn("Context:", text)
        self.assertIn("Question:", text)
        self.assertIn("Answer:", text)

    def test_main_runs_with_patched_components(self):
        """
        Patch heavy dependencies: AutoTokenizer, AutoModelForCausalLM, get_peft_model, Trainer,
        and DataCollatorForLanguageModeling. Ensure main() runs and calls key steps.
        """
        # Prepare dummy objects
        dummy_tokenizer = DummyTokenizer()
        dummy_model = DummyModel()
        peft_capture = GetPeftCapture()

        # patch factories and classes inside the module by name
        tokenizer_path = f"{MODULE_NAME}.AutoTokenizer"
        model_path = f"{MODULE_NAME}.AutoModelForCausalLM"
        get_peft_path = f"{MODULE_NAME}.get_peft_model"
        trainer_path = f"{MODULE_NAME}.Trainer"
        collator_path = f"{MODULE_NAME}.DataCollatorForLanguageModeling"

        # apply patches
        with mock.patch(tokenizer_path) as mock_tokenizer_cls, \
             mock.patch(model_path) as mock_model_cls, \
             mock.patch(get_peft_path, new=peft_capture), \
             mock.patch(trainer_path) as mock_trainer_cls, \
             mock.patch(collator_path) as mock_collator:

            # configure patched constructors
            mock_tokenizer_cls.from_pretrained.return_value = dummy_tokenizer
            mock_model_cls.from_pretrained.return_value = dummy_model

            # Make Trainer(...) -> DummyTrainer. We set side_effect so calling Trainer(...) returns DummyTrainer instance.
            def trainer_side_effect(*args, **kwargs):
                # we pass model in kwargs so DummyTrainer.save_model can call model.save_pretrained
                return DummyTrainer(*args, **kwargs)

            mock_trainer_cls.side_effect = trainer_side_effect
            # Data collator: return a simple callable (not used further)
            mock_collator.return_value = lambda *a, **k: None

            # Patch OUTPUT_DIR to temp location and load_pdf_text to avoid reading disk
            tmp_out = os.path.join(HERE, "tmp_tiny_llama_fine_out")
            with mock.patch.object(mod, "OUTPUT_DIR", tmp_out), \
                 mock.patch.object(mod, "load_pdf_text", return_value="Fake syllabus text for testing"):
                # Suppress prints
                with mock.patch("builtins.print"):
                    mod.main()

            # Now assert get_peft_model (peft_capture) was called
            self.assertTrue(peft_capture.called, "get_peft_model should have been called to apply LoRA")

            # Check that the LoRA config captured at least contains the expected keys (may be None)
            for k in ("r", "lora_alpha", "target_modules", "lora_dropout", "bias", "task_type"):
                self.assertIn(k, peft_capture.last_cfg)

            # Ensure tokenizer was called during tokenization and had a positive max_length
            self.assertIsNotNone(dummy_tokenizer.last_call_kwargs, "Tokenizer should have been called by tokenization")
            ml = dummy_tokenizer.last_call_kwargs.get("max_length")
            self.assertIsNotNone(ml, "Tokenizer should be called with max_length")
            self.assertIsInstance(ml, int)
            self.assertGreater(ml, 0)

            # Check that tokenizer.save_pretrained was called (DummyTokenizer.saved_to set)
            self.assertEqual(dummy_tokenizer.saved_to, tmp_out, "Tokenizer.save_pretrained should be called with OUTPUT_DIR")

            # Check model.save_pretrained called via trainer.save_model
            self.assertEqual(dummy_model.saved_to, tmp_out, "Model.save_pretrained should be called with OUTPUT_DIR")

        # cleanup tmp directory if created
        try:
            if os.path.isdir(tmp_out):
                import shutil
                shutil.rmtree(tmp_out)
        except Exception:
            pass


if __name__ == "__main__":
    unittest.main(verbosity=2)
