"""End-to-end smoke test: tiny model, a few training steps, a few generated notes.

Run from the repository root:  python -m unittest discover -s tests -v
Uses the two smallest MIDI files and no token cache; runs on CPU or GPU.
"""
import glob
import os
import shutil
import sys
import tempfile
import time
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("KERAS_BACKEND", "torch")

import MidiGeneration  # noqa: E402
import MidiModelTraining  # noqa: E402
import render_audio  # noqa: E402


class RenderTest(unittest.TestCase):
    def test_tokens_to_mp3_and_mp4(self):
        tmp = tempfile.mkdtemp()
        try:
            midi = os.path.join(tmp, "tiny.mid")
            MidiGeneration.tokens_to_midi(["C4", "E4", "G4", "0.4.7", "C5"], midi)
            render_audio.main([midi, "--video", "--tail", "0.5"])
            for ext in (".mp3", ".mp4"):
                self.assertGreater(os.path.getsize(os.path.join(tmp, "tiny" + ext)), 1000, ext)
            audio = render_audio.render(render_audio.read_notes(midi), render_audio.default_soundfont(), tail=0.5)
            self.assertAlmostEqual(len(audio) / 44100, 5 * 0.25 + 0.5, places=2)  # 120 bpm eighth notes
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class SmokeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.midi_dir = os.path.join(self.tmp, "midi")
        os.makedirs(self.midi_dir)
        files = sorted(glob.glob(os.path.join(ROOT, "midi_files", "*.mid")), key=os.path.getsize)
        for f in files[:2]:
            shutil.copy(f, self.midi_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_train_then_generate(self):
        start = time.time()
        model_dir = os.path.join(self.tmp, "model")
        MidiModelTraining.train(MidiModelTraining.parse_args([
            "--midi-dir", self.midi_dir, "--cache", "", "--out-dir", model_dir,
            "--units", "16", "--layers", "1", "--seq-len", "8", "--batch-size", "16",
            "--epochs", "2", "--steps-per-epoch", "5", "--val-split", "0.5", "--verbose", "0",
        ]))
        for name in ("model.weights.h5", "vocab.json", "config.json", "history.csv", "loss_curve.png"):
            self.assertTrue(os.path.exists(os.path.join(model_dir, name)), name)

        out = os.path.join(self.tmp, "out.mid")
        stats = MidiGeneration.main([
            "--model-dir", model_dir, "--midi-dir", self.midi_dir, "--cache", "",
            "--length", "12", "--seed", "3", "--output", out, "--piano-roll",
        ])
        self.assertTrue(os.path.exists(out))
        self.assertTrue(os.path.exists(os.path.splitext(out)[0] + ".png"))
        self.assertEqual(stats["tokens"], 12)
        self.assertGreaterEqual(stats["midi_note_events"], 12)
        self.assertLess(time.time() - start, 60)


if __name__ == "__main__":
    unittest.main()
