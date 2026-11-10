"""Generate a piano MIDI file with a model trained by MidiModelTraining.py.

Generation starts from a random window of --seq-len tokens taken from one of the
training pieces (the "prompt"), then repeatedly samples the next token from the
model's softmax output at the given temperature. Only the generated tokens are
written out; every token becomes a note or chord of one fixed --step length
(default an eighth note), so the output has no rhythm of its own.

Optionally writes a piano-roll PNG and prints simple structural statistics.
"""
import argparse
import collections
import json
import os
import random

os.environ.setdefault("KERAS_BACKEND", "torch")

import keras  # noqa: E402
import music21  # noqa: E402
import numpy  # noqa: E402

from MidiModelTraining import CACHE_PATH, create_lstm, load_pieces  # noqa: E402


def load_model(model_dir):
    with open(os.path.join(model_dir, "config.json"), encoding="utf-8") as fh:
        config = json.load(fh)
    with open(os.path.join(model_dir, "vocab.json"), encoding="utf-8") as fh:
        vocab = json.load(fh)
    model = create_lstm(len(vocab), config["seq_len"], config["units"], config["layers"],
                        config["embedding_dim"], config["dropout"])
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


def generate_tokens(model, vocab, config, prompt, length, temperature, seed):
    rng = numpy.random.default_rng(seed)
    note_to_int = {t: i for i, t in enumerate(vocab)}
    window = [note_to_int[t] for t in prompt]
    out = []
    for _ in range(length):
        x = numpy.array([window[-config["seq_len"]:]], dtype="int32")
        probs = keras.ops.convert_to_numpy(model(x, training=False))[0]
        idx = sample(probs, temperature, rng)
        out.append(vocab[idx])
        window.append(idx)
    return out


def token_to_element(token, offset, step):
    """Turn a token back into a music21 Note or Chord placed at offset."""
    if "." in token or token.isdigit():
        # Chords were stored as pitch classes; place them in the octave above middle C.
        notes = [music21.note.Note(int(pc) + 60) for pc in token.split(".")]
        element = music21.chord.Chord(notes)
    else:
        element = music21.note.Note(token)
    element.offset = offset
    element.quarterLength = step
    element.storedInstrument = music21.instrument.Piano()
    return element


def tokens_to_midi(tokens, path, step=0.5):
    stream = music21.stream.Stream([music21.instrument.Piano()])
    for i, token in enumerate(tokens):
        stream.insert(i * step, token_to_element(token, i * step, step))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    stream.write("midi", fp=path)


def read_midi_events(path):
    """Read back (offset, duration, midi pitch) triples from a written MIDI file."""
    score = music21.converter.parse(path)
    events = []
    for el in score.flatten().notes:
        for p in el.pitches:
            events.append((float(el.offset), float(el.quarterLength), p.midi))
    return events


def describe(tokens, events):
    """Simple structural statistics; they say nothing about how the music sounds."""
    longest, run = 1, 1
    for a, b in zip(tokens, tokens[1:]):
        run = run + 1 if a == b else 1
        longest = max(longest, run)
    counts = collections.Counter(tokens)
    top = counts.most_common(1)[0][0]  # ties go to the token that appeared first
    pitches = [p for _, _, p in events]
    return {
        "tokens": len(tokens),
        "distinct_tokens": len(counts),
        "chord_tokens": sum("." in t or t.isdigit() for t in tokens),
        "most_common_token": top,
        "most_common_share": round(counts[top] / len(tokens), 3),
        "longest_repeat_run": longest,
        "midi_note_events": len(events),
        "distinct_pitches": len(set(pitches)),
        "pitch_range": [music21.pitch.Pitch(midi=min(pitches)).nameWithOctave,
                        music21.pitch.Pitch(midi=max(pitches)).nameWithOctave] if pitches else None,
    }


def plot_piano_roll(events, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 4))
    for offset, dur, pitch in events:
        ax.add_patch(plt.Rectangle((offset, pitch - 0.4), dur, 0.8, color="#3b6ea5"))
    if events:
        pitches = [p for _, _, p in events]
        ax.set_xlim(0, max(o + d for o, d, _ in events))
        ax.set_ylim(min(pitches) - 2, max(pitches) + 2)
    ax.set_xlabel("time (quarter notes)")
    ax.set_ylabel("MIDI pitch (60 = middle C)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", default="models")
    p.add_argument("--midi-dir", default="midi_files", help="used to build the prompt cache if missing")
    p.add_argument("--cache", default=CACHE_PATH)
    p.add_argument("--length", type=int, default=200, help="number of notes/chords to generate")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--step", type=float, default=0.5, help="duration of each token in quarter notes")
    p.add_argument("--output", default=os.path.join("samples", "generated.mid"))
    p.add_argument("--piano-roll", action="store_true", help="also save <output>.png")
    args = p.parse_args(argv)

    keras.utils.set_random_seed(args.seed)
    model, vocab, config = load_model(args.model_dir)

    # Prompt: a random window from a validation piece when there is one (the model never
    # trained on it), otherwise from any piece.
    pieces = load_pieces(args.midi_dir, args.cache)
    rng = random.Random(args.seed)
    candidates = [n for n in config.get("val_pieces") or sorted(pieces)
                  if n in pieces and len(pieces[n]) > config["seq_len"]]
    if not candidates:
        raise SystemExit("No piece is long enough to provide a prompt")
    piece = rng.choice(candidates)
    start = rng.randrange(len(pieces[piece]) - config["seq_len"])
    prompt = pieces[piece][start:start + config["seq_len"]]

    tokens = generate_tokens(model, vocab, config, prompt, args.length, args.temperature, args.seed)
    tokens_to_midi(tokens, args.output, args.step)
    events = read_midi_events(args.output)
    stats = describe(tokens, events)
    stats.update({"prompt_piece": piece, "prompt_start": start, "seed": args.seed,
                  "temperature": args.temperature, "output": args.output})
    print(json.dumps(stats, indent=2))

    if args.piano_roll:
        png = os.path.splitext(args.output)[0] + ".png"
        plot_piano_roll(events, png, f"{os.path.basename(args.output)}  (seed {args.seed}, "
                                     f"temperature {args.temperature})")
        print(f"piano roll -> {png}")
    return stats


if __name__ == "__main__":
    main()
