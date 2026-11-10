"""Bootstrap confidence intervals for evaluate.py's mean overlap.

Resamples the generated files of each set with replacement (default 300 times) and
reports the mean overlap's 2.5th-97.5th percentile range, so differences between
sampling settings can be judged against the noise of a 20-file set.

    python evaluate_bootstrap.py --sets "name=glob" ...
"""
import argparse
import collections
import glob
import json
import os

import numpy

import evaluate


def file_features(path):
    ev = evaluate.read_events(path)
    return evaluate.features(ev)


def pool(feature_list):
    total = collections.defaultdict(collections.Counter)
    for f in feature_list:
        for k, c in f.items():
            total[k].update(c)
    return total


def mean_overlap(gen, ref):
    return float(numpy.mean([evaluate.overlap(gen[k], ref[k]) for k in evaluate.FEATURES]))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=os.path.join("results", "stage2-rhythm-transpose", "model", "config.json"))
    p.add_argument("--midi-dir", default="midi_files")
    p.add_argument("--sets", nargs="+", required=True)
    p.add_argument("--resamples", type=int, default=300)
    p.add_argument("--json")
    args = p.parse_args(argv)

    with open(args.config, encoding="utf-8") as fh:
        config = json.load(fh)
    ref = pool([file_features(os.path.join(args.midi_dir, n)) for n in config["val_pieces"]])
    rng = numpy.random.default_rng(0)
    out = {}
    for s in args.sets:
        name, pattern = s.split("=", 1)
        feats = [file_features(f) for f in sorted(glob.glob(pattern))]
        point = mean_overlap(pool(feats), ref)
        boots = [mean_overlap(pool([feats[i] for i in rng.integers(0, len(feats), len(feats))]), ref)
                 for _ in range(args.resamples)]
        lo, hi = numpy.percentile(boots, [2.5, 97.5])
        out[name] = {"mean_overlap": round(point, 3), "ci95": [round(float(lo), 3), round(float(hi), 3)]}
        print(f"| {name} | {point:.3f} | {lo:.3f} to {hi:.3f} |")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2)


if __name__ == "__main__":
    main()
