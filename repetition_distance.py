"""How far back does real music repeat itself? (Used in PAPER.md, Section 5.)

For every 4-onset pattern (top-note intervals and onset gaps, the same key evaluate.py
uses) in the piano-midi.de held-out pieces, find the nearest earlier occurrence in the
same piece and measure the distance in onsets and in stage-2 event tokens.

    python repetition_distance.py > results/analysis/repetition_distance.json
"""
import json
import os

import numpy

import evaluate
import music_tokens

CONFIG = os.path.join("results", "stage2-rhythm-transpose", "model", "config.json")


def main():
    with open(CONFIG, encoding="utf-8") as fh:
        names = json.load(fh)["val_pieces"]
    d_onsets, d_tokens, total = [], [], 0
    for name in names:
        path = os.path.join("midi_files", name)
        events = evaluate.read_events(path)
        tokens = music_tokens.encode_midi(path)
        # token index where each onset starts: its first P token after a time shift (or at 0)
        starts = [j for j, t in enumerate(tokens) if t[0] == "P" and (j == 0 or tokens[j - 1][0] == "S")]
        last = {}
        for i in range(3, len(events)):
            w = events[i - 3:i + 1]
            key = tuple((round(b1 - b0, 3), p1[-1] - p0[-1]) for (b0, p0, _), (b1, p1, _) in zip(w, w[1:]))
            total += 1
            if key in last:
                j = last[key]
                d_onsets.append(i - j)
                d_tokens.append(starts[i] - starts[j])
            last[key] = i
    d_tokens = numpy.array(d_tokens)
    out = {"pieces": len(names), "patterns": total, "repeat_share": round(len(d_onsets) / total, 3),
           "median_distance_onsets": int(numpy.median(d_onsets)),
           "median_distance_tokens": int(numpy.median(d_tokens))}
    for limit in (64, 256, 1024):
        out[f"share_of_patterns_repeating_from_more_than_{limit}_tokens_back"] = round(
            float((d_tokens > limit).sum()) / total, 3)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
