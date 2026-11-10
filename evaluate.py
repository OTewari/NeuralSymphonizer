"""Compare generated MIDI files with real pieces using musical feature distributions.

Works on MIDI files only, so samples from any stage (or any model) can be scored the
same way. Times are measured in beats (quarter notes) from each file's own tempo map,
so the tempo a sample was written at does not matter.

For each set of files the notes are pooled into histograms of seven features:
    pitch             MIDI pitch 21-108 (register)
    pitch_class       12 pitch classes (key / harmony)
    interval          signed interval between the top notes of consecutive onsets, -12..+12
    polyphony         notes starting together (1..6+)
    ioi               time between consecutive onsets, snapped to the token time steps
    note_length       note durations, snapped to the same steps
    repetition        whether each 4-onset pattern (intervals + timing, so transposition-
                      invariant) already occurred earlier in the same file
Each generated histogram is compared with the reference histogram by overlap (the sum of
the bin-wise minimum of the two normalised histograms): 1.0 means identical
distributions, 0.0 means no overlap. This follows the spirit of the distribution-based
evaluation of Yang & Lerch (2020), simplified to pooled histograms.

    python evaluate.py --reference held-out --sets "stage 1=results/stage1-baseline/eval/*.mid" ...
"""
import argparse
import collections
import glob
import json
import os

import numpy
import pretty_midi

import music_tokens

STEPS = numpy.array(music_tokens.TIME_STEPS) / music_tokens.GRID  # in beats


def _snap(beats):
    return float(STEPS[numpy.argmin(numpy.abs(STEPS - beats))]) if beats > 0 else 0.0


def read_events(path):
    """Return a list of onsets: (beat, sorted pitches, durations in beats)."""
    pm = pretty_midi.PrettyMIDI(path)
    res = pm.resolution
    groups = collections.defaultdict(list)
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for n in inst.notes:
            on = pm.time_to_tick(n.start) / res
            off = pm.time_to_tick(n.end) / res
            groups[round(on * music_tokens.GRID)].append((n.pitch, off - on))
    return [(k / music_tokens.GRID, sorted(p for p, _ in v), [d for _, d in v]) for k, v in sorted(groups.items())]


def features(events):
    f = collections.defaultdict(collections.Counter)
    for beat, pitches, durs in events:
        f["pitch"].update(pitches)
        f["pitch_class"].update(p % 12 for p in pitches)
        f["polyphony"][min(len(pitches), 6)] += 1
        f["note_length"].update(_snap(d) for d in durs)
    for (b0, p0, _), (b1, p1, _) in zip(events, events[1:]):
        f["ioi"][_snap(b1 - b0)] += 1
        f["interval"][max(-12, min(12, p1[-1] - p0[-1]))] += 1
    seen = set()
    for i in range(3, len(events)):
        window = events[i - 3:i + 1]
        key = tuple((round(b1 - b0, 3), p1[-1] - p0[-1]) for (b0, p0, _), (b1, p1, _) in zip(window, window[1:]))
        f["repetition"]["repeat" if key in seen else "new"] += 1
        seen.add(key)
    return f


def pooled(paths):
    total = collections.defaultdict(collections.Counter)
    notes = onsets = 0
    density = []
    for p in paths:
        ev = read_events(p)
        onsets += len(ev)
        n = sum(len(x[1]) for x in ev)
        notes += n
        if len(ev) > 1:
            density.append(n / max(ev[-1][0] - ev[0][0], 1e-6))  # notes per beat
        for k, c in features(ev).items():
            total[k].update(c)
    total["_density"] = density
    return total, notes, onsets


def _spread(values):
    """Coefficient of variation (std / mean) of per-file note density."""
    return float(numpy.std(values) / numpy.mean(values)) if len(values) > 1 else 0.0


def _outlier_share(pitches, lo, hi):
    total = sum(pitches.values()) or 1
    return sum(c for p, c in pitches.items() if p < lo or p > hi) / total


def overlap(a, b):
    keys = set(a) | set(b)
    sa, sb = sum(a.values()) or 1, sum(b.values()) or 1
    return sum(min(a[k] / sa, b[k] / sb) for k in keys)


FEATURES = ["pitch", "pitch_class", "interval", "polyphony", "ioi", "note_length", "repetition"]


def evaluate(reference_paths, sets):
    ref, ref_notes, ref_onsets = pooled(reference_paths)
    # pitch range holding 98% of the reference notes (1st to 99th percentile)
    ref_p = numpy.repeat(list(ref["pitch"].keys()), list(ref["pitch"].values()))
    lo, hi = numpy.percentile(ref_p, 1), numpy.percentile(ref_p, 99)
    rows = {}
    for name, paths in sets.items():
        gen, n, o = pooled(paths)
        row = {k: round(overlap(gen[k], ref[k]), 3) for k in FEATURES}
        row["mean"] = round(float(numpy.mean([row[k] for k in FEATURES])), 3)
        rep = gen["repetition"]
        row["repeat_share"] = round(rep["repeat"] / max(1, sum(rep.values())), 3)
        row["outlier_share"] = round(_outlier_share(gen["pitch"], lo, hi), 4)
        row["density_spread"] = round(_spread(gen["_density"]), 3)
        row["files"], row["notes"], row["notes_per_onset"] = len(paths), n, round(n / max(1, o), 2)
        rows[name] = row
    rr = ref["repetition"]
    reference = {"files": len(reference_paths), "notes": ref_notes, "notes_per_onset": round(ref_notes / ref_onsets, 2),
                 "repeat_share": round(rr["repeat"] / max(1, sum(rr.values())), 3),
                 "outlier_share": round(_outlier_share(ref["pitch"], lo, hi), 4),
                 "density_spread": round(_spread(ref["_density"]), 3), "pitch_range_98pct": [float(lo), float(hi)]}
    return reference, rows


def markdown(reference, rows):
    head = ["set", *FEATURES, "mean", "repeat share", "outside 98% range", "density spread", "notes/onset", "files"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for name, r in rows.items():
        lines.append("| " + " | ".join([name] + [f"{r[k]:.3f}" for k in FEATURES + ["mean"]]
                                       + [f"{r['repeat_share']:.3f}", f"{r['outlier_share']:.3f}",
                                          f"{r['density_spread']:.2f}", f"{r['notes_per_onset']}", str(r["files"])]) + " |")
    lines.append(f"| reference ({reference['files']} real pieces) | " + " | ".join(["1.000"] * (len(FEATURES) + 1))
                 + f" | {reference['repeat_share']:.3f} | {reference['outlier_share']:.3f}"
                 f" | {reference['density_spread']:.2f} | {reference['notes_per_onset']} | {reference['files']} |")
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reference", nargs="+", required=True,
                   help="reference MIDI files or globs; 'held-out' = the validation pieces of --config; "
                        "'maestro-validation:N' = N seeded MAESTRO validation performances")
    p.add_argument("--config", default=os.path.join("results", "stage2-rhythm-transpose", "model", "config.json"))
    p.add_argument("--midi-dir", default="midi_files")
    p.add_argument("--sets", nargs="+", required=True, help='"name=glob" pairs of generated files')
    p.add_argument("--real-sample", type=int, default=0,
                   help="also score N real training pieces (seeded choice) as a ceiling")
    p.add_argument("--real-maestro", type=int, default=0,
                   help="also score N real MAESTRO training performances (seeded choice)")
    p.add_argument("--json", help="also write the results to this JSON file")
    args = p.parse_args(argv)

    with open(args.config, encoding="utf-8") as fh:
        config = json.load(fh)
    import random
    maestro_rows = None
    if args.reference[0].startswith("maestro-validation") or args.real_maestro:
        import csv
        root = os.path.join("data", "maestro", "maestro-v3.0.0")
        with open(os.path.join(root, "maestro-v3.0.0.csv"), encoding="utf-8") as fh:
            maestro_rows = [(r["split"], os.path.join(root, r["midi_filename"])) for r in csv.DictReader(fh)]
    if args.reference == ["held-out"]:
        ref = [os.path.join(args.midi_dir, n) for n in config["val_pieces"]]
    elif args.reference[0].startswith("maestro-validation"):
        # "maestro-validation:N" = N seeded MAESTRO validation performances
        n = int(args.reference[0].split(":")[1])
        ref = random.Random(0).sample(sorted(p for s, p in maestro_rows if s == "validation"), n)
    else:
        ref = sorted(f for g in args.reference for f in glob.glob(g))
    sets = {}
    if args.real_sample:
        chosen = random.Random(0).sample(sorted(config["train_pieces"]), args.real_sample)
        sets[f"real training pieces ({args.real_sample})"] = [os.path.join(args.midi_dir, n) for n in chosen]
    if args.real_maestro:
        chosen = random.Random(0).sample(sorted(p for s, p in maestro_rows if s == "train"), args.real_maestro)
        sets[f"real MAESTRO training performances ({args.real_maestro})"] = chosen
    for s in args.sets:
        name, pattern = s.split("=", 1)
        sets[name] = sorted(glob.glob(pattern))
        if not sets[name]:
            raise SystemExit(f"no files match {pattern!r}")
    reference, rows = evaluate(ref, sets)
    print(markdown(reference, rows))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({"reference": reference, "sets": rows}, fh, indent=2)
    return reference, rows


if __name__ == "__main__":
    main()
