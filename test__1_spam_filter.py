# test__1_spam_filter_unittest.py
import os
import io
import sys
import shutil
import unittest
import contextlib
from unittest import mock

# Ensure the current directory is importable (student's _1_spam_filter.py should be in this directory)
HERE = os.path.abspath(os.path.dirname(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

DATA_DIR = os.path.join(HERE, "data", "input")
CSV_PATH = os.path.join(DATA_DIR, "L06_NLP_emails.csv")


class SpamFilterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # REQUIRE the dataset already exists at ./data/input/L06_NLP_emails.csv
        if not os.path.exists(CSV_PATH):
            raise RuntimeError(f"Required dataset not found at {CSV_PATH}. Please ensure the file exists before running tests.")

        # Import student's module (do not run main on import)
        import importlib
        if "_1_spam_filter" in sys.modules:
            importlib.reload(sys.modules["_1_spam_filter"])
        cls._1_spam_filter = importlib.import_module("_1_spam_filter")

        # Run main once and capture stdout for reuse in tests that need the actual accuracy
        cls._stdout_buf = io.StringIO()
        with contextlib.redirect_stdout(cls._stdout_buf):
            # call main which will read ./data/input/L06_NLP_emails.csv and train model
            cls._1_spam_filter.main()
        cls._captured = cls._stdout_buf.getvalue()

    @classmethod
    def tearDownClass(cls):
        # Do NOT remove the dataset. We assume it is managed by the test environment.
        pass

    def test_preprocess_basic_behavior(self):
        """Preprocess should lowercase, remove punctuation/digits, remove stopwords, and stem/lemmatize."""
        preprocess = getattr(self._1_spam_filter, "preprocess", None)
        self.assertIsNotNone(preprocess, "preprocess function not found in _1_spam_filter module")

        sample = "  Hello!!! This is    TESTING, running tests 123. Visit: http://example.com "
        out = preprocess(sample)

        # check lowercased
        self.assertEqual(out, out.lower(), "preprocess output should be lowercase")

        # ensure no digits remain
        self.assertNotRegex(out, r"\d", "preprocess output should not contain digits")

        # no URLs or punctuation characters
        self.assertNotRegex(out, r"[^\w\s]", "preprocess output should not contain punctuation")

        # stopwords like 'this' and 'is' should be removed (if STOPWORDS defined)
        stopwords = getattr(self._1_spam_filter, "STOPWORDS", None)
        if stopwords is not None:
            for sw in ["this", "is", "the", "and"]:
                self.assertNotIn(sw, out.split(), f"stopword '{sw}' should be removed by preprocess")

        # tokens 'running' should be stemmed/lemmatized to something shorter like 'run' or similar;
        # ensure the word 'running' does not appear as-is (indicating stemming/lemmatization not applied).
        self.assertNotIn("running", out)

    def test_preprocess_non_string_returns_empty(self):
        preprocess = getattr(self._1_spam_filter, "preprocess", None)
        self.assertIsNotNone(preprocess)
        self.assertEqual(preprocess(None), "", "preprocess(None) should return an empty string")

    def test_countvectorizer_used_and_vocab_populated(self):
        """
        Patch CountVectorizer in the student's module to intercept construction parameters
        and examine the returned vectorizer's vocabulary after the pipeline runs.
        """
        # We'll patch _1_spam_filter.CountVectorizer to a wrapper that creates a real CountVectorizer
        RealCV = self._1_spam_filter.CountVectorizer

        called = {"flag": False, "kwargs": None, "instance": None}

        def wrapper(*args, **kwargs):
            called["flag"] = True
            called["kwargs"] = kwargs
            inst = RealCV(*args, **kwargs)
            called["instance"] = inst
            return inst

        # run main with patch (this will retrain — dataset is assumed available)
        with mock.patch("_1_spam_filter.CountVectorizer", new=wrapper):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                # call main again to ensure patched CountVectorizer is used
                self._1_spam_filter.main()
            captured = buf.getvalue()

        self.assertTrue(called["flag"], "CountVectorizer was not called in _1_spam_filter.main()")
        # Check expected kwarg 'max_features' present and was likely used
        kw = called["kwargs"] or {}
        self.assertIn("max_features", kw, "CountVectorizer should be constructed with max_features argument")
        # Ensure vocabulary was learned and non-empty
        inst = called["instance"]
        vocab = getattr(inst, "vocabulary_", None)
        self.assertIsNotNone(vocab, "CountVectorizer instance should have vocabulary_ populated after fit")
        self.assertGreater(len(vocab), 0, "Vocabulary should contain at least one token")

    def test_training_pipeline_prints_and_accuracy_above_threshold(self):
        """
        Ensure training ran (we expect 'Training complete.' printed)
        and that the accuracy printed is >= 90% on the provided dataset.
        """
        out = self._captured
        self.assertIn("Training complete", out, "_1_spam_filter.main() should print 'Training complete.'")

        # Find the accuracy line printed like: "Spam Filter Accuracy: 97.00%"
        acc_val = None
        for line in out.splitlines():
            if "Spam Filter Accuracy" in line:
                # parse the percentage
                import re
                m = re.search(r"([\d\.]+)\s*%", line)
                if m:
                    acc_val = float(m.group(1))
                else:
                    # Sometimes it might be printed as decimal; attempt float parse
                    m2 = re.search(r"Spam Filter Accuracy:\s*([\d\.]+)", line)
                    if m2:
                        acc_val = float(m2.group(1)) * 100.0
                break

        self.assertIsNotNone(acc_val, "Could not find 'Spam Filter Accuracy' printed by _1_spam_filter.main()")
        # require >= 90%
        self.assertGreaterEqual(acc_val, 90.0, f"Expected accuracy >= 90.0%, got {acc_val:.2f}%")

    def test_model_fit_called_when_training(self):
        """
        Basic smoke test: patch MLPClassifier.fit to observe it was called.
        We patch to a thin wrapper that calls the real MLPClassifier to ensure training still happens.
        """
        RealMLP = self._1_spam_filter.MLPClassifier

        fit_called = {"flag": False}

        class MLPWrapper(RealMLP):
            def fit(self, X, y):
                fit_called["flag"] = True
                return super().fit(X, y)

        with mock.patch("_1_spam_filter.MLPClassifier", new=MLPWrapper):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                self._1_spam_filter.main()
            # after run, fit_called should be True
            self.assertTrue(fit_called["flag"], "MLPClassifier.fit should have been called during training")


if __name__ == "__main__":
    unittest.main()
