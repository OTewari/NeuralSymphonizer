"""Smoke tests: event-token round trip, tiny model train + generate, audio rendering.

Run from the repository root:  python -m unittest discover -s tests -v
Uses the two smallest MIDI files; runs on CPU or GPU in a few seconds.
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
import music_tokens  # noqa: E402
import render_audio  # noqa: E402

SMALLEST = sorted(glob.glob(os.path.join(ROOT, "midi_files", "*.mid")), key=os.path.getsize)[:2]


class TokenTest(unittest.TestCase):
    def test_round_trip_and_transpose(self):
        tokens = music_tokens.encode_midi(SMALLEST[0])
        tmp = tempfile.mkdtemp()
        try:
            path = os.path.join(tmp, "rt.mid")
            music_tokens.decode(tokens).write(path)
            self.assertEqual(music_tokens.encode_midi(path), tokens)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        vocab = music_tokens.VOCAB
        up = music_tokens.transpose_table(vocab, 2)
        self.assertEqual(vocab[up[vocab.index("P60")]], "P62")
        self.assertEqual(up[vocab.index("P108")], -1)
        self.assertEqual(vocab[up[vocab.index("D12")]], "D12")


class SamplingTest(unittest.TestCase):
    def test_top_k_and_top_p_restrict_the_choice(self):
        import numpy
        probs = numpy.array([0.5, 0.3, 0.15, 0.05])
        rng = numpy.random.default_rng(0)
        self.assertEqual({MidiGeneration.sample(probs, 1.0, rng, top_k=1) for _ in range(50)}, {0})
        self.assertEqual({MidiGeneration.sample(probs, 1.0, rng, top_k=2) for _ in range(300)}, {0, 1})
        self.assertEqual({MidiGeneration.sample(probs, 1.0, rng, top_p=0.75) for _ in range(300)}, {0, 1})
        self.assertEqual({MidiGeneration.sample(probs, 1.0, rng, top_p=0.85) for _ in range(500)}, {0, 1, 2})
        self.assertEqual(MidiGeneration.sample(probs, 0, rng), 0)

    def test_grammar_mask(self):
        import numpy
        vocab = music_tokens.VOCAB
        masks = MidiGeneration.grammar_masks(vocab)
        rng = numpy.random.default_rng(0)
        uniform = numpy.full(len(vocab), 1 / len(vocab))
        after_p = {vocab[MidiGeneration.sample(uniform, 1.0, rng, allowed=masks["P"])][0] for _ in range(200)}
        after_d = {vocab[MidiGeneration.sample(uniform, 1.0, rng, allowed=masks["D"])][0] for _ in range(200)}
        self.assertEqual(after_p, {"D"})
        self.assertEqual(after_d, {"S", "P"})


class RenderTest(unittest.TestCase):
    def test_tokens_to_mp3_and_mp4(self):
        tmp = tempfile.mkdtemp()
        try:
            midi = os.path.join(tmp, "tiny.mid")
            music_tokens.decode(["P60", "D12", "S12", "P64", "D12", "S12", "P67", "D24"], bpm=120).write(midi)
            render_audio.main([midi, "--video", "--tail", "0.5"])
            for ext in (".mp3", ".mp4"):
                self.assertGreater(os.path.getsize(os.path.join(tmp, "tiny" + ext)), 1000, ext)
            audio = render_audio.render(render_audio.read_notes(midi), render_audio.default_soundfont(), tail=0.5)
            self.assertAlmostEqual(len(audio) / 44100, 0.25 + 0.25 + 0.5 + 0.5, places=2)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TrainGenerateTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.midi_dir = os.path.join(self.tmp, "midi")
        os.makedirs(self.midi_dir)
        for f in SMALLEST:
            shutil.copy(f, self.midi_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_train_then_generate(self):
        start = time.time()
        model_dir = os.path.join(self.tmp, "model")
        MidiModelTraining.train(MidiModelTraining.parse_args([
            "--midi-dir", self.midi_dir, "--out-dir", model_dir,
            "--units", "16", "--layers", "1", "--seq-len", "16", "--batch-size", "16",
            "--epochs", "1", "--val-split", "0.5", "--verbose", "0",
        ]))
        for name in ("model.weights.h5", "vocab.json", "config.json", "history.csv", "loss_curve.png"):
            self.assertTrue(os.path.exists(os.path.join(model_dir, name)), name)

        out = os.path.join(self.tmp, "out.mid")
        stats = MidiGeneration.main([
            "--model-dir", model_dir, "--midi-dir", self.midi_dir, "--prompt-len", "16",
            "--notes", "12", "--seed", "3", "--output", out, "--piano-roll",
        ])
        self.assertTrue(os.path.exists(out))
        self.assertTrue(os.path.exists(os.path.splitext(out)[0] + ".png"))
        self.assertEqual(stats["notes"], 12)
        self.assertLess(time.time() - start, 60)


if __name__ == "__main__":
    unittest.main()
