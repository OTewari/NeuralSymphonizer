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


def grammar_masks(vocab):
    """Allowed next tokens after each token type: P must be followed by its D; D and S
    by an S or a P. Returns {type of previous token: 0/1 mask over the vocabulary}."""
    kind = numpy.array([t[0] for t in vocab])
    return {"P": (kind == "D").astype("float64"),
            "D": (kind != "D").astype("float64"),
            "S": (kind != "D").astype("float64")}


def sample(probs, temperature, rng, top_k=0, top_p=1.0, allowed=None):
    """Draw an index from probs after temperature scaling (temperature 0 = argmax).

    allowed, if given, is a 0/1 mask; disallowed tokens get probability 0 before
    anything else (grammar-constrained decoding). top_k > 0 keeps only the k most
    likely tokens (Fan et al., 2018); top_p < 1 keeps the smallest set of most likely
    tokens whose probability adds up to at least top_p (nucleus sampling, Holtzman et
    al., 2019). They can be combined.
    """
    probs = numpy.asarray(probs, dtype="float64")
    if allowed is not None:
        probs = probs * allowed
        probs /= probs.sum()
    if temperature <= 0:
        return int(numpy.argmax(probs))
    logits = numpy.log(probs + 1e-12) / temperature
    logits -= logits.max()
    p = numpy.exp(logits)
    p /= p.sum()
    if top_k > 0 or top_p < 1.0:
        keep = numpy.argsort(-p, kind="stable")
        if top_k > 0:
            keep = keep[:top_k]
        if top_p < 1.0:
            keep = keep[:int(numpy.searchsorted(numpy.cumsum(p[keep]), top_p)) + 1]
        mask = numpy.zeros_like(p)
        mask[keep] = 1.0
        p = p * mask
        p /= p.sum()
    return int(rng.choice(len(p), p=p))


def generate_batch(model, vocab, config, prompts, n_notes, seeds, temperature=1.0, top_k=0, top_p=1.0,
                   max_tokens=None, grammar=False):
    """Sample after each prompt until it has n_notes P tokens; all prompts run as one batch.

    With a single prompt this is the plain one-sample generator. Prompts must have equal
    length. grammar=True masks tokens that would break the P-then-D token grammar.
    """
    masks = grammar_masks(vocab) if grammar else None
    rngs = [numpy.random.default_rng(s) for s in seeds]
    note_to_int = {t: i for i, t in enumerate(vocab)}
    windows = [[note_to_int[t] for t in p] for p in prompts]
    outs, notes = [[] for _ in prompts], [0] * len(prompts)
    max_tokens = max_tokens or 8 * n_notes
    active = [i for i in range(len(prompts))]
    while active:
        x = numpy.array([windows[i][-config["seq_len"]:] for i in active], dtype="int32")
        probs = keras.ops.convert_to_numpy(model(x, training=False))[:, -1]
        for row, i in enumerate(active):
            allowed = masks[vocab[windows[i][-1]][0]] if masks else None
            idx = sample(probs[row], temperature, rngs[i], top_k, top_p, allowed)
            outs[i].append(vocab[idx])
            notes[i] += vocab[idx][0] == "P"
            windows[i].append(idx)
        active = [i for i in active if notes[i] < n_notes and len(outs[i]) < max_tokens]
    return outs


def generate_tokens(model, vocab, config, prompt, n_notes, temperature, seed, max_tokens=None, top_k=0, top_p=1.0,
                    grammar=False):
    """Sample tokens after the prompt until n_notes P tokens have been produced."""
    return generate_batch(model, vocab, config, [prompt], n_notes, [seed], temperature, top_k, top_p, max_tokens,
                          grammar)[0]


def choose_prompt(pieces, config, prompt_len, rng, piece=None, start=None, names=None):
    """A prompt_len-token window starting at a time shift.

    By default a random window of a random held-out piece (or of one of `names`);
    piece and start pick one.
    """
    if piece is None:
        names = [n for n in (names or config.get("val_pieces") or sorted(pieces))
                 if n in pieces and len(pieces[n]) > prompt_len + 1]
        if not names:
            raise SystemExit("No piece is long enough to provide a prompt")
        piece = rng.choice(names)
    if piece not in pieces:
        raise SystemExit(f"Unknown prompt piece {piece!r}")
    toks = pieces[piece]
    if start is None:
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
    p.add_argument("--top-k", type=int, default=0, help="sample only from the k most likely tokens (0 = off)")
    p.add_argument("--top-p", type=float, default=1.0, help="nucleus sampling threshold (1.0 = off)")
    p.add_argument("--grammar", action="store_true",
                   help="never sample a token that breaks the P-then-D grammar (constrained decoding)")
    p.add_argument("--prompt-source", choices=["piano-midi", "maestro"], default="piano-midi",
                   help="take prompts from the piano-midi.de held-out pieces or the MAESTRO validation split")
    p.add_argument("--prompt-piece", default=None, help="file name in --midi-dir to take the prompt from")
    p.add_argument("--prompt-start", type=int, default=None, help="token index where the prompt starts")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--count", type=int, default=1,
                   help="generate this many samples (seeds seed..seed+count-1) in one batch; "
                        "outputs get a _seed<N> suffix")
    p.add_argument("--bpm", type=float, default=100.0, help="tempo of the written MIDI file")
    p.add_argument("--output", default=os.path.join("generated", "generated.mid"))
    p.add_argument("--piano-roll", action="store_true", help="also save <output>.png")
    args = p.parse_args(argv)

    keras.utils.set_random_seed(args.seed)
    model, vocab, config = load_model(args.model_dir)
    if args.prompt_source == "maestro":
        from MidiModelTraining import load_maestro
        val = load_maestro(splits=("validation",))["validation"]
        pieces = {n: [music_tokens.VOCAB[i] for i in ids] for n, ids in val.items()}
        names = sorted(pieces)
    else:
        pieces, names = load_pieces(args.midi_dir), None

    seeds = list(range(args.seed, args.seed + args.count))
    prompts, outputs = [], []
    for s in seeds:
        prompts.append(choose_prompt(pieces, config, args.prompt_len, random.Random(s),
                                     args.prompt_piece, args.prompt_start, names))
        base, ext = os.path.splitext(args.output)
        outputs.append(args.output if args.count == 1 else f"{base}_seed{s}{ext}")
    token_lists = generate_batch(model, vocab, config, [pr[2] for pr in prompts], args.notes, seeds,
                                 args.temperature, args.top_k, args.top_p, grammar=args.grammar)

    all_stats = []
    for s, (piece, start, _), tokens, out in zip(seeds, prompts, token_lists, outputs):
        pm = music_tokens.decode(tokens, bpm=args.bpm)
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        pm.write(out)
        stats = describe(tokens, pm)
        stats.update({"prompt_piece": piece, "prompt_start": start, "seed": s,
                      "temperature": args.temperature, "top_k": args.top_k, "top_p": args.top_p,
                      "grammar": args.grammar,
                      "bpm": args.bpm, "output": out.replace("\\", "/")})
        all_stats.append(stats)
        if args.piano_roll:
            import render_audio
            png = os.path.splitext(out)[0] + ".png"
            notes = render_audio.read_notes(out)
            sampling = f"temperature {args.temperature}"
            sampling += f", top-k {args.top_k}" if args.top_k else ""
            sampling += f", top-p {args.top_p}" if args.top_p < 1.0 else ""
            sampling += ", grammar" if args.grammar else ""
            render_audio.piano_roll_frame(notes, max(n.end for n in notes), png,
                                          f"{os.path.basename(out)}  (seed {s}, {sampling})")
    print(json.dumps(all_stats[0] if args.count == 1 else all_stats, indent=2))
    return all_stats[0] if args.count == 1 else all_stats


if __name__ == "__main__":
    main()
