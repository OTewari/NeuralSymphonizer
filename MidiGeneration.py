"""Generate a piano MIDI file with a model trained by MidiModelTraining.py.

Generation starts from a random window of --prompt-len tokens taken from one of the
held-out pieces (the "prompt"), then repeatedly samples the next event token from the
model's softmax output at the given temperature, until --notes notes have been
generated. Only the generated tokens are written out, at a fixed tempo (--bpm), because
the tokens carry rhythm but not tempo.

Optionally writes a piano-roll PNG and prints simple structural statistics.
"""
import argparse
import json
import os
import random

os.environ.setdefault("KERAS_BACKEND", "torch")

import keras  # noqa: E402
import numpy  # noqa: E402

import music_tokens  # noqa: E402
from MidiModelTraining import build_from_config, load_pieces  # noqa: E402

DEFAULT_MODEL_DIR = os.path.join("results", "stage2-rhythm-transpose", "model")


def load_model(model_dir):
    model, config = build_from_config(model_dir)
    with open(os.path.join(model_dir, "vocab.json"), encoding="utf-8") as fh:
        vocab = json.load(fh)
    model.load_weights(os.path.join(model_dir, "model.weights.h5"))
    return model, vocab, config


def sample(probs, temperature, rng):
    """Draw an index from probs after temperature scaling (temperature 0 = argmax)."""
    probs = numpy.asarray(probs, dtype="float64")
    if temperature <= 0:
        return int(numpy.argmax(probs))
    logits = numpy.log(probs + 1e-12) / temperature
    logits -= logits.max()
    p = numpy.exp(logits)
    p /= p.sum()
    return int(rng.choice(len(p), p=p))


def generate_tokens(model, vocab, config, prompt, n_notes, temperature, seed, max_tokens=None):
    """Sample tokens after the prompt until n_notes P tokens have been produced."""
    rng = numpy.random.default_rng(seed)
    note_to_int = {t: i for i, t in enumerate(vocab)}
    window = [note_to_int[t] for t in prompt]
    out, notes = [], 0
    max_tokens = max_tokens or 8 * n_notes
    while notes < n_notes and len(out) < max_tokens:
        x = numpy.array([window[-config["seq_len"]:]], dtype="int32")
        probs = keras.ops.convert_to_numpy(model(x, training=False))[0, -1]
        idx = sample(probs, temperature, rng)
        out.append(vocab[idx])
        notes += vocab[idx][0] == "P"
        window.append(idx)
    return out


def choose_prompt(pieces, config, prompt_len, rng):
    """A random prompt_len-token window of a held-out piece, starting at a time shift."""
    names = [n for n in config.get("val_pieces") or sorted(pieces)
             if n in pieces and len(pieces[n]) > prompt_len + 1]
    if not names:
        raise SystemExit("No piece is long enough to provide a prompt")
    piece = rng.choice(names)
    toks = pieces[piece]
    starts = [i for i in range(len(toks) - prompt_len) if toks[i][0] == "S"]
    start = rng.choice(starts)
    return piece, start, toks[start:start + prompt_len]


def describe(tokens, pm):
    """Simple structural statistics; they say nothing about how the music sounds."""
    notes = sorted(pm.instruments[0].notes, key=lambda n: (n.start, n.pitch))
    onsets = {}
    for n in notes:
        onsets.setdefault(round(n.start, 4), []).append(n.pitch)
    seq = [tuple(v) for _, v in sorted(onsets.items())]
    longest, run = 1, 1
    for a, b in zip(seq, seq[1:]):
        run = run + 1 if a == b else 1
        longest = max(longest, run)
    ill = sum(1 for a, b in zip(tokens, tokens[1:] + ["S1"]) if a[0] == "P" and b[0] != "D")
    ill += sum(1 for a, b in zip(["S1"] + tokens, tokens) if b[0] == "D" and a[0] != "P")
    pitches = [n.pitch for n in notes]
    durations = [round((n.end - n.start) * 1000) for n in notes]
    return {
        "tokens": len(tokens),
        "notes": len(notes),
        "onsets": len(seq),
        "chord_onsets": sum(len(s) > 1 for s in seq),
        "distinct_pitches": len(set(pitches)),
        "pitch_range": [min(pitches), max(pitches)] if pitches else None,
        "distinct_note_lengths": len(set(durations)),
        "seconds": round(pm.get_end_time(), 1),
        "longest_run_of_identical_onsets": longest,
        "ill_formed_tokens": ill,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    p.add_argument("--midi-dir", default="midi_files", help="where the prompt pieces are read from")
    p.add_argument("--notes", type=int, default=300, help="number of notes to generate")
    p.add_argument("--prompt-len", type=int, default=128, help="prompt length in tokens")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--bpm", type=float, default=100.0, help="tempo of the written MIDI file")
    p.add_argument("--output", default=os.path.join("samples", "generated.mid"))
    p.add_argument("--piano-roll", action="store_true", help="also save <output>.png")
    args = p.parse_args(argv)

    keras.utils.set_random_seed(args.seed)
    model, vocab, config = load_model(args.model_dir)

    pieces = load_pieces(args.midi_dir)
    rng = random.Random(args.seed)
    piece, start, prompt = choose_prompt(pieces, config, args.prompt_len, rng)

    tokens = generate_tokens(model, vocab, config, prompt, args.notes, args.temperature, args.seed)
    pm = music_tokens.decode(tokens, bpm=args.bpm)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    pm.write(args.output)
    stats = describe(tokens, pm)
    stats.update({"prompt_piece": piece, "prompt_start": start, "seed": args.seed,
                  "temperature": args.temperature, "bpm": args.bpm, "output": args.output.replace("\\", "/")})
    print(json.dumps(stats, indent=2))

    if args.piano_roll:
        import render_audio
        png = os.path.splitext(args.output)[0] + ".png"
        notes = render_audio.read_notes(args.output)
        render_audio.piano_roll_frame(notes, max(n.end for n in notes), png,
                                      f"{os.path.basename(args.output)}  (seed {args.seed}, "
                                      f"temperature {args.temperature})")
        print(f"piano roll -> {png}")
    return stats


if __name__ == "__main__":
    main()
