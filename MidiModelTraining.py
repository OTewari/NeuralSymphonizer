"""Train an LSTM to predict the next event token in classical piano MIDI files.

Pipeline (stage 2, see PROGRESSION.md):
  1. Read every note of every MIDI file in --midi-dir with pretty_midi and turn it into
     event tokens (music_tokens.py): time shifts, exact pitches with octave, note lengths.
  2. Hold out a fraction of whole pieces for validation (same seeded split as stage 1).
  3. Cut each piece into windows of --seq-len tokens. The model predicts the next token
     at every position of the window, not only the last one.
  4. Training windows are transposed by a random number of semitones (--transpose,
     default -6..+5, i.e. all 12 keys) each time they are used; validation is not.
  5. Embedding -> N stacked LSTM layers -> Dense(256, relu) -> softmax over the vocabulary.

Outputs in --out-dir (default models/): best weights by validation loss, a weights-only
export used by MidiGeneration.py, vocab.json, config.json, a per-epoch CSV log and a
loss-curve PNG.
"""
import argparse
import glob
import json
import math
import os
import random
import time

os.environ.setdefault("KERAS_BACKEND", "torch")

import keras  # noqa: E402
import numpy  # noqa: E402

import music_tokens  # noqa: E402


# ---------------------------------------------------------------- data prep
def load_pieces(midi_dir="midi_files", max_files=None):
    """Return {file name: event tokens} for every .mid file (about 5 s for the 95 files)."""
    files = sorted(glob.glob(os.path.join(midi_dir, "*.mid")))
    if max_files:
        files = files[:max_files]
    if not files:
        raise SystemExit(f"No .mid files found in {midi_dir!r}")
    return {os.path.basename(f): music_tokens.encode_midi(f) for f in files}


MAESTRO_ROOT = os.path.join("data", "maestro", "maestro-v3.0.0")


def load_maestro(root=MAESTRO_ROOT, splits=("train", "validation")):
    """Return {split: {midi_filename: token ids}} for MAESTRO v3.0.0 (see get_maestro.py).

    Tokenising all 1,276 files takes about 2.5 minutes, so the result is cached in
    <root>/tokens_cache.npz (inside the git-ignored download folder).
    """
    import csv

    cache = os.path.join(root, "tokens_cache.npz")
    if os.path.exists(cache):
        data = numpy.load(cache, allow_pickle=False)
        names, split_of, offsets, flat = data["names"], data["splits"], data["offsets"], data["tokens"]
    else:
        index = {t: i for i, t in enumerate(music_tokens.VOCAB)}
        with open(os.path.join(root, "maestro-v3.0.0.csv"), encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        seqs = [numpy.array([index[t] for t in music_tokens.encode_midi(os.path.join(root, r["midi_filename"]))],
                            dtype="int16") for r in rows]
        names = numpy.array([r["midi_filename"] for r in rows])
        split_of = numpy.array([r["split"] for r in rows])
        offsets = numpy.cumsum([0] + [len(s) for s in seqs])
        flat = numpy.concatenate(seqs)
        numpy.savez(cache, names=names, splits=split_of, offsets=offsets, tokens=flat)
    out = {s: {} for s in splits}
    for i, (name, split) in enumerate(zip(names, split_of)):
        if str(split) in out:
            out[str(split)][str(name)] = flat[offsets[i]:offsets[i + 1]].astype("int32")
    return out


def split_pieces(pieces, val_fraction, seed):
    """Split piece names into train/validation lists (whole pieces, seeded shuffle)."""
    names = sorted(pieces)
    random.Random(seed).shuffle(names)
    n_val = int(round(len(names) * val_fraction))
    if val_fraction > 0 and len(names) > 1:
        n_val = max(1, n_val)
    return names[n_val:], names[:n_val]


class Windows(keras.utils.PyDataset):
    """Batches of (tokens, next tokens) windows, optionally transposed at random.

    Each piece yields the same number of non-overlapping windows every epoch; for
    training the windows start at a new random offset each epoch and are shuffled.
    """

    def __init__(self, sequences, vocab, seq_len, batch_size, train, transpose=(0, 0), seed=0, **kwargs):
        super().__init__(**kwargs)
        self.seqs = [numpy.asarray(s, dtype="int32") for s in sequences if len(s) > seq_len + 1]
        self.seq_len, self.batch_size, self.train = seq_len, batch_size, train
        self.rng = numpy.random.default_rng(seed)
        pitch = numpy.array([music_tokens.pitch_of(t) or -1 for t in vocab])
        self.pitch = [pitch[s] for s in self.seqs]  # -1 where the token is not a pitch
        self.shifts = list(range(transpose[0], transpose[1] + 1))
        self.tables = {k: numpy.array(music_tokens.transpose_table(vocab, k)) for k in self.shifts}
        self.counts = [(len(s) - 1) // seq_len for s in self.seqs]
        self.on_epoch_end()

    def on_epoch_end(self):
        self.index = []
        for i, (s, n) in enumerate(zip(self.seqs, self.counts)):
            slack = len(s) - 1 - n * self.seq_len
            offset = int(self.rng.integers(0, slack + 1)) if self.train else 0
            self.index += [(i, offset + k * self.seq_len) for k in range(n)]
        if self.train:
            self.rng.shuffle(self.index)

    def __len__(self):
        return math.ceil(len(self.index) / self.batch_size)

    def __getitem__(self, b):
        rows = self.index[b * self.batch_size:(b + 1) * self.batch_size]
        x = numpy.empty((len(rows), self.seq_len), dtype="int32")
        y = numpy.empty((len(rows), self.seq_len), dtype="int32")
        for r, (i, start) in enumerate(rows):
            window = self.seqs[i][start:start + self.seq_len + 1]
            if self.train and len(self.shifts) > 1:
                p = self.pitch[i][start:start + self.seq_len + 1]
                p = p[p >= 0]
                lo, hi = music_tokens.PITCH_MIN - p.min(), music_tokens.PITCH_MAX - p.max()
                allowed = [k for k in self.shifts if lo <= k <= hi] or [0]
                window = self.tables[int(self.rng.choice(allowed))][window]
            x[r], y[r] = window[:-1], window[1:]
        return x, y


# ---------------------------------------------------------------- model
def create_lstm(vocab_size, units=512, layers=3, embedding_dim=64, dropout=0.2):
    """Sequence-to-sequence LSTM; accepts windows of any length."""
    model = keras.Sequential([keras.Input(shape=(None,), dtype="int32"),
                              keras.layers.Embedding(vocab_size, embedding_dim)])
    for _ in range(layers):
        model.add(keras.layers.LSTM(units, return_sequences=True))
        model.add(keras.layers.Dropout(dropout))
    model.add(keras.layers.Dense(256, activation="relu"))
    model.add(keras.layers.Dropout(dropout))
    model.add(keras.layers.Dense(vocab_size, activation="softmax"))
    model.compile(loss="sparse_categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])
    return model


def build_from_config(model_dir):
    with open(os.path.join(model_dir, "config.json"), encoding="utf-8") as fh:
        config = json.load(fh)
    model = create_lstm(config["vocab_size"], config["units"], config["layers"],
                        config["embedding_dim"], config["dropout"])
    return model, config


def export_inference_weights(out_dir):
    """Re-save the best checkpoint without the optimizer state (roughly half the size).

    The training checkpoint (best.weights.h5) also stores RMSprop's per-weight
    accumulators; MidiGeneration.py only needs the layer weights (model.weights.h5).
    """
    model, _ = build_from_config(out_dir)
    model.load_weights(os.path.join(out_dir, "best.weights.h5"), skip_mismatch=True)
    path = os.path.join(out_dir, "model.weights.h5")
    model.save_weights(path)
    print(f"Inference weights -> {path} ({os.path.getsize(path) / 1e6:.1f} MB)")


def plot_history(csv_path, png_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = numpy.atleast_1d(numpy.genfromtxt(csv_path, delimiter=",", names=True))
    epochs = data["epoch"] + 1
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, data["loss"], label="training loss")
    if "pm_val_loss" in data.dtype.names:
        ax.plot(epochs, data["pm_val_loss"], label="piano-midi.de held-out loss")
    if "val_loss" in data.dtype.names:
        ax.plot(epochs, data["val_loss"], label="validation loss")
        best = int(numpy.argmin(data["val_loss"]))
        ax.axvline(epochs[best], color="grey", linestyle=":", label=f"best val (epoch {int(epochs[best])})")
    ax.set_xlabel("epoch")
    ax.set_ylabel("cross-entropy loss per token")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(png_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------- training
class ExtraValidation(keras.callbacks.Callback):
    """Evaluate a second validation set each epoch and add it to the logs."""

    def __init__(self, data, prefix):
        super().__init__()
        self.data, self.prefix = data, prefix

    def on_epoch_end(self, epoch, logs=None):
        loss, acc = self.model.evaluate(self.data, verbose=0)
        if logs is not None:
            logs[self.prefix + "loss"], logs[self.prefix + "accuracy"] = loss, acc


def train(args):
    keras.utils.set_random_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    pieces = load_pieces(args.midi_dir, args.max_files)
    vocab = music_tokens.VOCAB
    note_to_int = {t: i for i, t in enumerate(vocab)}
    train_names, val_names = split_pieces(pieces, args.val_split, args.seed)
    ids = {n: [note_to_int[t] for t in toks] for n, toks in pieces.items()}

    transpose = (args.transpose_min, args.transpose_max)
    extra_val = None
    if args.dataset == "maestro":
        # Train and select checkpoints on MAESTRO; also score the piano-midi.de held-out
        # pieces every epoch ("pm_val_*") so results compare with earlier stages.
        maestro = load_maestro(args.maestro_root)
        train_seqs, val_seqs = list(maestro["train"].values()), list(maestro["validation"].values())
        extra_val = Windows([ids[n] for n in val_names], vocab, args.seq_len, args.batch_size, False)
        source = f"{len(train_seqs)} MAESTRO train performances"
    else:
        train_seqs, val_seqs = [ids[n] for n in train_names], [ids[n] for n in val_names]
        source = f"{len(train_names)} pieces"
    train_data = Windows(train_seqs, vocab, args.seq_len, args.batch_size, True, transpose, args.seed)
    val_data = Windows(val_seqs, vocab, args.seq_len, args.batch_size, False)
    n_train_tokens = sum(len(s) for s in train_seqs)
    print(f"vocab {len(vocab)} | train {n_train_tokens} tokens, {len(train_data.index)} windows "
          f"from {source} | val {len(val_data.index)} windows from {len(val_seqs)} pieces "
          f"| transpose {transpose}")

    config = {"stage": args.stage, "dataset": args.dataset, "seq_len": args.seq_len, "units": args.units,
              "layers": args.layers, "embedding_dim": args.embedding_dim, "dropout": args.dropout,
              "seed": args.seed, "vocab_size": len(vocab), "transpose": list(transpose),
              "batch_size": args.batch_size, "train_tokens": n_train_tokens,
              "train_pieces": train_names if args.dataset == "piano-midi" else f"MAESTRO v3.0.0 train split ({len(train_seqs)} files)",
              "val_pieces": val_names,
              "checkpoint_validation": "piano-midi.de held-out pieces" if args.dataset == "piano-midi"
              else f"MAESTRO v3.0.0 validation split ({len(val_seqs)} files)"}
    with open(os.path.join(args.out_dir, "vocab.json"), "w", encoding="utf-8") as fh:
        json.dump(vocab, fh)
    with open(os.path.join(args.out_dir, "config.json"), "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)

    model = create_lstm(len(vocab), args.units, args.layers, args.embedding_dim, args.dropout)
    model.summary()

    has_val = len(val_data) > 0
    monitor = "val_loss" if has_val else "loss"
    weights_path = os.path.join(args.out_dir, "best.weights.h5")
    csv_path = os.path.join(args.out_dir, "history.csv")
    callbacks = [ExtraValidation(extra_val, "pm_val_")] if extra_val is not None else []
    callbacks += [
        keras.callbacks.ModelCheckpoint(weights_path, monitor=monitor, save_best_only=True,
                                        save_weights_only=True, mode="min"),
        keras.callbacks.CSVLogger(csv_path),
    ]
    if args.patience:
        callbacks.append(keras.callbacks.EarlyStopping(monitor=monitor, patience=args.patience))

    start = time.time()
    model.fit(train_data, epochs=args.epochs, validation_data=val_data if has_val else None,
              steps_per_epoch=args.steps_per_epoch, callbacks=callbacks, verbose=args.verbose)
    minutes = (time.time() - start) / 60
    print(f"Training done in {minutes:.1f} min; best weights -> {weights_path}")
    plot_history(csv_path, os.path.join(args.out_dir, "loss_curve.png"))
    export_inference_weights(args.out_dir)
    return model


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=["piano-midi", "maestro"], default="piano-midi",
                   help="training data: the piano-midi.de files in --midi-dir, or MAESTRO (get_maestro.py)")
    p.add_argument("--maestro-root", default=MAESTRO_ROOT)
    p.add_argument("--stage", type=int, default=2, help="stage number recorded in config.json")
    p.add_argument("--midi-dir", default="midi_files")
    p.add_argument("--out-dir", default="models")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seq-len", type=int, default=256, help="tokens per training window")
    p.add_argument("--units", type=int, default=512, help="LSTM units per layer")
    p.add_argument("--layers", type=int, default=3, help="number of stacked LSTM layers")
    p.add_argument("--embedding-dim", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--val-split", type=float, default=0.1, help="fraction of pieces held out")
    p.add_argument("--transpose-min", type=int, default=-6, help="lowest random transposition (semitones)")
    p.add_argument("--transpose-max", type=int, default=5, help="highest random transposition; 0 0 = off")
    p.add_argument("--patience", type=int, default=0, help="early-stopping patience in epochs (0 = off)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-files", type=int, default=None, help="use only the first N MIDI files")
    p.add_argument("--steps-per-epoch", type=int, default=None)
    p.add_argument("--verbose", type=int, default=2, help="Keras fit verbosity (0, 1 or 2)")
    return p.parse_args(argv)


if __name__ == "__main__":
    train(parse_args())
