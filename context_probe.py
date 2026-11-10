"""How much context does a trained model use? Loss by position in the window.

The model predicts the token after every position of a window, so the loss at position
k (1-based) is its loss when it has seen k tokens of context. If the loss stops falling
after a few dozen tokens, the model is not using the rest of its window. Measured on
non-overlapping windows of the piano-midi.de held-out pieces listed in the model's
config.json. Prints a table and writes <model-dir>/context_probe.json and .png.

    python context_probe.py --model-dir results/stage2-rhythm-transpose/model
"""
import argparse
import json
import os

os.environ.setdefault("KERAS_BACKEND", "torch")

import keras  # noqa: E402
import numpy  # noqa: E402

from MidiGeneration import load_model  # noqa: E402
from MidiModelTraining import load_pieces  # noqa: E402

POSITIONS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]


def per_position_loss(model, vocab, config, pieces, names, seq_len):
    index = {t: i for i, t in enumerate(vocab)}
    windows = []
    for n in names:
        ids = [index[t] for t in pieces[n]]
        for s in range(0, len(ids) - seq_len - 1, seq_len):
            windows.append(ids[s:s + seq_len + 1])
    data = numpy.array(windows, dtype="int32")
    total = numpy.zeros(seq_len)
    for b in range(0, len(data), 32):
        x, y = data[b:b + 32, :-1], data[b:b + 32, 1:]
        probs = keras.ops.convert_to_numpy(model(x, training=False))
        p = numpy.take_along_axis(probs, y[..., None], axis=-1)[..., 0]
        total += -numpy.log(p + 1e-12).sum(axis=0)
    return total / len(data), len(data)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--midi-dir", default="midi_files")
    p.add_argument("--seq-len", type=int, default=None, help="window length (default: the training length)")
    args = p.parse_args(argv)

    model, vocab, config = load_model(args.model_dir)
    seq_len = args.seq_len or config["seq_len"]
    pieces = load_pieces(args.midi_dir)
    loss, n = per_position_loss(model, vocab, config, pieces, config["val_pieces"], seq_len)
    # Average over ranges [k, 2k) of positions: single positions are noisy because the
    # token type (time, pitch, length) at a given position varies between windows.
    table = {}
    for k in POSITIONS:
        if k <= seq_len:
            table[f"{k}-{min(2 * k, seq_len + 1) - 1}"] = round(float(loss[k - 1:min(2 * k - 1, seq_len)].mean()), 4)
    result = {"windows": n, "seq_len": seq_len, "mean_loss_by_context_range": table,
              "mean_loss": round(float(loss.mean()), 4)}
    print(json.dumps(result, indent=2))
    out = os.path.join(args.model_dir, f"context_probe_{seq_len}")
    with open(out + ".json", "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(numpy.arange(1, seq_len + 1), loss, alpha=0.3, label="single position")
    smooth = numpy.convolve(loss, numpy.ones(9) / 9, mode="same")
    ax.plot(numpy.arange(5, seq_len - 3), smooth[4:-4], label="9-position average")
    ax.legend()
    ax.set_xscale("log", base=2)
    ax.set_xlabel("tokens of context seen")
    ax.set_ylabel("held-out loss at that position")
    ax.set_title(os.path.basename(os.path.dirname(os.path.abspath(args.model_dir))))
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out + ".png", dpi=120)
    plt.close(fig)
    return result


if __name__ == "__main__":
    main()
