"""Train an LSTM to predict the next note/chord token in classical piano MIDI files.

Pipeline:
  1. Parse every MIDI file in --midi-dir with music21 into a list of tokens per piece.
     A single note becomes its pitch name ("E5"); a chord becomes the dot-joined
     pitch classes of its normal order ("4.7.11"). Results are cached in data/notes.json.
  2. Hold out a fraction of whole pieces for validation, so validation windows never
     overlap training windows.
  3. Slide a window of --seq-len tokens over each piece; the target is the next token.
  4. Train Embedding -> N stacked LSTM layers -> Dense(256, relu) -> softmax over the vocabulary.

Outputs in --out-dir (default models/): best weights by validation loss, the vocabulary
and model config needed by MidiGeneration.py, a per-epoch CSV log and a loss-curve PNG.
"""
import argparse
import glob
import json
import os
import random
import time

os.environ.setdefault("KERAS_BACKEND", "torch")

import keras  # noqa: E402
import music21  # noqa: E402
import numpy  # noqa: E402

CACHE_PATH = os.path.join("data", "notes.json")


# ---------------------------------------------------------------- data prep
def parse_midi(path):
    """Return the token list for one MIDI file."""
    midi = music21.converter.parse(path)
    # Use the first instrument part if the file has several, otherwise all notes.
    instruments = music21.instrument.partitionByInstrument(midi)
    if instruments:
        to_parse = instruments.parts[0].recurse()
    else:
        to_parse = midi.flatten().notes

    tokens = []
    for element in to_parse:
        if isinstance(element, music21.note.Note):
            tokens.append(str(element.pitch))
        elif isinstance(element, music21.chord.Chord):
            tokens.append(".".join(str(n) for n in element.normalOrder))
    return tokens


def load_pieces(midi_dir="midi_files", cache_path=CACHE_PATH, rebuild=False, max_files=None):
    """Return {file name: token list}, building the JSON cache on first run.

    The cache is rebuilt automatically when the set of MIDI files changes.
    """
    files = sorted(glob.glob(os.path.join(midi_dir, "*.mid")))
    if max_files:
        files = files[:max_files]
    names = [os.path.basename(f) for f in files]
    if not files:
        raise SystemExit(f"No .mid files found in {midi_dir!r}")

    if cache_path and not rebuild and os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as fh:
            cached = json.load(fh)
        if sorted(cached) == names:
            print(f"Loaded token cache {cache_path} ({len(cached)} pieces)")
            return cached
        print("MIDI files changed since the cache was built; rebuilding it")

    pieces = {}
    start = time.time()
    for i, path in enumerate(files, 1):
        pieces[os.path.basename(path)] = parse_midi(path)
        print(f"  parsed {i}/{len(files)} {os.path.basename(path)}", flush=True)
    print(f"Parsed {len(files)} files in {time.time() - start:.0f}s")
    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as fh:
            json.dump(pieces, fh)
    return pieces


def split_pieces(pieces, val_fraction, seed):
    """Split piece names into train/validation lists (whole pieces, seeded shuffle)."""
    names = sorted(pieces)
    random.Random(seed).shuffle(names)
    n_val = int(round(len(names) * val_fraction))
    if val_fraction > 0 and len(names) > 1:
        n_val = max(1, n_val)
    return names[n_val:], names[:n_val]


def make_sequences(pieces, names, note_to_int, seq_len):
    """Build (inputs, targets): every window of seq_len tokens inside one piece -> next token."""
    inputs, targets = [], []
    for name in names:
        ids = [note_to_int[t] for t in pieces[name]]
        for i in range(len(ids) - seq_len):
            inputs.append(ids[i:i + seq_len])
            targets.append(ids[i + seq_len])
    return numpy.array(inputs, dtype="int32").reshape(-1, seq_len), numpy.array(targets, dtype="int32")


# ---------------------------------------------------------------- model
def create_lstm(vocab_size, seq_len, units=512, layers=3, embedding_dim=64, dropout=0.2):
    model = keras.Sequential([keras.Input(shape=(seq_len,), dtype="int32"),
                              keras.layers.Embedding(vocab_size, embedding_dim)])
    for i in range(layers):
        model.add(keras.layers.LSTM(units, return_sequences=i < layers - 1))
        model.add(keras.layers.Dropout(dropout))
    model.add(keras.layers.Dense(256, activation="relu"))
    model.add(keras.layers.Dropout(dropout))
    model.add(keras.layers.Dense(vocab_size, activation="softmax"))
    model.compile(loss="sparse_categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])
    return model


def plot_history(csv_path, png_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = numpy.genfromtxt(csv_path, delimiter=",", names=True)
    data = numpy.atleast_1d(data)
    epochs = data["epoch"] + 1
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, data["loss"], label="training loss")
    if "val_loss" in data.dtype.names:
        ax.plot(epochs, data["val_loss"], label="validation loss")
        best = int(numpy.argmin(data["val_loss"]))
        ax.axvline(epochs[best], color="grey", linestyle=":", label=f"best val (epoch {int(epochs[best])})")
    ax.set_xlabel("epoch")
    ax.set_ylabel("cross-entropy loss")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(png_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------- training
def train(args):
    keras.utils.set_random_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    pieces = load_pieces(args.midi_dir, args.cache, args.rebuild_cache, args.max_files)
    vocab = sorted({t for tokens in pieces.values() for t in tokens})
    note_to_int = {t: i for i, t in enumerate(vocab)}
    train_names, val_names = split_pieces(pieces, args.val_split, args.seed)

    x_train, y_train = make_sequences(pieces, train_names, note_to_int, args.seq_len)
    x_val, y_val = make_sequences(pieces, val_names, note_to_int, args.seq_len)
    if args.max_sequences:
        x_train, y_train = x_train[:args.max_sequences], y_train[:args.max_sequences]
        x_val, y_val = x_val[:args.max_sequences], y_val[:args.max_sequences]
    print(f"vocab {len(vocab)} tokens | train {len(x_train)} windows from {len(train_names)} pieces"
          f" | val {len(x_val)} windows from {len(val_names)} pieces")

    config = {"seq_len": args.seq_len, "units": args.units, "layers": args.layers,
              "embedding_dim": args.embedding_dim, "dropout": args.dropout, "seed": args.seed,
              "vocab_size": len(vocab), "train_pieces": train_names, "val_pieces": val_names}
    with open(os.path.join(args.out_dir, "vocab.json"), "w", encoding="utf-8") as fh:
        json.dump(vocab, fh)
    with open(os.path.join(args.out_dir, "config.json"), "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)

    model = create_lstm(len(vocab), args.seq_len, args.units, args.layers, args.embedding_dim, args.dropout)
    model.summary()

    has_val = len(x_val) > 0
    monitor = "val_loss" if has_val else "loss"
    weights_path = os.path.join(args.out_dir, "best.weights.h5")
    csv_path = os.path.join(args.out_dir, "history.csv")
    callbacks = [
        keras.callbacks.ModelCheckpoint(weights_path, monitor=monitor, save_best_only=True,
                                        save_weights_only=True, mode="min"),
        keras.callbacks.CSVLogger(csv_path),
    ]
    if args.patience:
        callbacks.append(keras.callbacks.EarlyStopping(monitor=monitor, patience=args.patience))

    start = time.time()
    model.fit(x_train, y_train, epochs=args.epochs, batch_size=args.batch_size,
              validation_data=(x_val, y_val) if has_val else None,
              steps_per_epoch=args.steps_per_epoch, callbacks=callbacks, verbose=args.verbose)
    minutes = (time.time() - start) / 60
    print(f"Training done in {minutes:.1f} min; best weights -> {weights_path}")
    plot_history(csv_path, os.path.join(args.out_dir, "loss_curve.png"))
    return model


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--midi-dir", default="midi_files")
    p.add_argument("--cache", default=CACHE_PATH, help="token cache (JSON); '' disables it")
    p.add_argument("--rebuild-cache", action="store_true")
    p.add_argument("--out-dir", default="models")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seq-len", type=int, default=50)
    p.add_argument("--units", type=int, default=512, help="LSTM units per layer")
    p.add_argument("--layers", type=int, default=3, help="number of stacked LSTM layers")
    p.add_argument("--embedding-dim", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--val-split", type=float, default=0.1, help="fraction of pieces held out")
    p.add_argument("--patience", type=int, default=0, help="early-stopping patience in epochs (0 = off)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-files", type=int, default=None, help="use only the first N MIDI files")
    p.add_argument("--max-sequences", type=int, default=None, help="cap training/validation windows")
    p.add_argument("--steps-per-epoch", type=int, default=None)
    p.add_argument("--verbose", type=int, default=2, help="Keras fit verbosity (0, 1 or 2)")
    return p.parse_args(argv)


if __name__ == "__main__":
    train(parse_args())
