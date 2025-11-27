#!/usr/bin/env python3
"""
Unit tests for _3b_tts_rag.py

These tests patch external audio libraries (pyttsx3, simpleaudio) to avoid
creating real audio or playing sound. They focus on the behaviors required
by the solution: correct delegation and error handling.
"""

import unittest
import tempfile
import os
import sys
import types
import importlib

MODULE_NAME = "_3b_tts_rag"
module = importlib.import_module(MODULE_NAME)


class FakeEngine:
    def __init__(self):
        # voices: objects with .id attribute
        self._voices = [types.SimpleNamespace(id="voice0"), types.SimpleNamespace(id="voice1")]
        self.props = {}
        self.saved = []  # list of tuples (text, out_path)
        self.ran = False

    def getProperty(self, name):
        if name == "voices":
            return self._voices
        return self.props.get(name)

    def setProperty(self, name, value):
        self.props[name] = value

    def save_to_file(self, text, out_path):
        # just record the request
        self.saved.append((text, out_path))

    def runAndWait(self):
        # mark that runAndWait has been called
        self.ran = True


class TestTTSRAG(unittest.TestCase):

    def test_tts_save_invokes_pyttsx3_engine(self):
        # Prepare fake pyttsx3 module in sys.modules before calling tts_save
        fake_engine = FakeEngine()
        fake_pyttsx3 = types.SimpleNamespace(init=lambda: fake_engine)
        sys.modules['pyttsx3'] = fake_pyttsx3

        td = None
        out_wav = None
        try:
            # create temp output path
            td = tempfile.mkdtemp()
            out_wav = os.path.join(td, "out.wav")
            text = "Hello world from unit test."

            # call tts_save (it will import our fake pyttsx3)
            ret = module.tts_save(text, out_wav, voice_index=1, rate=220)
            self.assertEqual(ret, out_wav)

            # engine should have recorded the save request
            self.assertTrue(fake_engine.saved, "Engine did not record save_to_file call")
            saved_text, saved_path = fake_engine.saved[-1]
            self.assertEqual(saved_text, text)
            self.assertEqual(saved_path, out_wav)

            # engine properties set: voice id should be set to voice1.id
            self.assertIn('voice', fake_engine.props)
            self.assertEqual(fake_engine.props['voice'], fake_engine._voices[1].id)
            # rate should be set
            self.assertIn('rate', fake_engine.props)
            self.assertEqual(fake_engine.props['rate'], 220)
            # runAndWait should have been called
            self.assertTrue(fake_engine.ran)
        finally:
            # cleanup fake module and any created file/dir
            sys.modules.pop('pyttsx3', None)
            try:
                if out_wav and os.path.exists(out_wav):
                    os.remove(out_wav)
                if td and os.path.isdir(td):
                    os.rmdir(td)
            except Exception:
                pass

    def test_tts_from_text_file_reads_and_delegates(self):
        # create a temp text file with content
        td = tempfile.mkdtemp()
        txt_path = os.path.join(td, "f.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("This is a short response.\n")

        # patch module.tts_save to capture args
        called = {}

        def fake_tts_save(text, out_path, voice_index=None, rate=150):
            called['text'] = text
            called['out_path'] = out_path
            called['voice_index'] = voice_index
            called['rate'] = rate
            return out_path

        orig_tts_save = module.tts_save
        module.tts_save = fake_tts_save
        out_wav = os.path.join(td, "out.wav")
        try:
            ret = module.tts_from_text_file(txt_path, out_wav, voice_index=0, rate=160)
            self.assertEqual(ret, out_wav)
            # ensure delegation and values
            self.assertEqual(called['text'], "This is a short response.")
            self.assertEqual(called['out_path'], out_wav)
            self.assertEqual(called['voice_index'], 0)
            self.assertEqual(called['rate'], 160)
        finally:
            module.tts_save = orig_tts_save
            try:
                if os.path.exists(txt_path):
                    os.remove(txt_path)
                if os.path.exists(out_wav):
                    os.remove(out_wav)
                if os.path.isdir(td):
                    os.rmdir(td)
            except Exception:
                pass

    def test_tts_from_text_file_empty_uses_no_text_marker(self):
        # create an empty text file
        td = tempfile.mkdtemp()
        txt_path = os.path.join(td, "empty.txt")
        open(txt_path, "w", encoding="utf-8").close()

        # patch tts_save to capture the text argument
        captured = {}

        def fake_tts_save(text, out_path, voice_index=None, rate=150):
            captured['text'] = text
            return out_path

        orig = module.tts_save
        module.tts_save = fake_tts_save
        out_wav = os.path.join(td, "out_empty.wav")
        try:
            ret = module.tts_from_text_file(txt_path, out_wav)
            self.assertEqual(ret, out_wav)
            # empty file should cause "(no text)"
            self.assertEqual(captured.get('text'), "(no text)")
        finally:
            module.tts_save = orig
            try:
                if os.path.exists(txt_path):
                    os.remove(txt_path)
                if os.path.exists(out_wav):
                    os.remove(out_wav)
                if os.path.isdir(td):
                    os.rmdir(td)
            except Exception:
                pass

    def test_tts_from_text_files_batch_and_length_mismatch(self):
        # mismatch should raise
        with self.assertRaises(ValueError):
            module.tts_from_text_files(["a.txt"], ["o1.wav", "o2.wav"])

        # normal batch: patch tts_from_text_file to record calls
        calls = []

        def fake_from_text_file(tf, ow, voice_index=None, rate=150):
            calls.append((tf, ow, voice_index, rate))
            return ow

        orig = module.tts_from_text_file
        module.tts_from_text_file = fake_from_text_file
        try:
            # create dummy files so function doesn't raise file missing (we patched tts_from_text_file anyway)
            tf1_dir = tempfile.mkdtemp()
            tf2_dir = tempfile.mkdtemp()
            tf1 = os.path.join(tf1_dir, "t1.txt")
            tf2 = os.path.join(tf2_dir, "t2.txt")
            for p in (tf1, tf2):
                with open(p, "w", encoding="utf-8") as f:
                    f.write("x")
            out1 = os.path.join(tempfile.gettempdir(), "o1.wav")
            out2 = os.path.join(tempfile.gettempdir(), "o2.wav")
            outs = [out1, out2]
            saved = module.tts_from_text_files([tf1, tf2], outs, voice_index=0, rate=123)
            self.assertEqual(saved, outs)
            self.assertEqual(len(calls), 2)
            self.assertEqual(calls[0][2], 0)
            self.assertEqual(calls[0][3], 123)
        finally:
            module.tts_from_text_file = orig
            # cleanup created dummy text files and dirs
            for p in (tf1, tf2):
                try:
                    if os.path.exists(p):
                        os.remove(p)
                    d = os.path.dirname(p)
                    if os.path.isdir(d):
                        os.rmdir(d)
                except Exception:
                    pass
            for p in (out1, out2):
                if os.path.exists(p):
                    os.remove(p)

    def test_play_wav_behavior(self):
        # Inject a fake simpleaudio module to avoid requiring native wheels.
        class FakePlayObj:
            def wait_done(self):
                self.done = True

        class FakeWaveObj:
            def __init__(self, path):
                self.path = path
                self.play_called = False

            def play(self):
                self.play_called = True
                return FakePlayObj()

            @classmethod
            def from_wave_file(cls, path):
                return cls(path)

        fake_simpleaudio = types.SimpleNamespace(WaveObject=FakeWaveObj)
        sys.modules['simpleaudio'] = fake_simpleaudio

        # Now missing file should raise FileNotFoundError (import succeeded)
        missing = os.path.join(tempfile.gettempdir(), "definitely_not_a_real_file_12345.wav")
        if os.path.exists(missing):
            os.remove(missing)
        try:
            with self.assertRaises(FileNotFoundError):
                module.play_wav(missing)

            # create a fake wav file (content doesn't need to be valid; our fake loader ignores it)
            td = tempfile.mkdtemp()
            wav = os.path.join(td, "f.wav")
            with open(wav, "wb") as f:
                f.write(b"RIFF....WAVE")
            # This should not raise
            module.play_wav(wav)
        finally:
            # remove our fake simpleaudio module and files
            sys.modules.pop('simpleaudio', None)
            try:
                if os.path.exists(wav):
                    os.remove(wav)
                if os.path.isdir(td):
                    os.rmdir(td)
            except Exception:
                pass


if __name__ == "__main__":
    unittest.main()
