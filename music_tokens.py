"""Event tokens for piano MIDI: timing, exact pitches (with octave) and note lengths.

A piece becomes a stream of three kinds of token, on a grid of 1/24 of a quarter note:

    S<n>   advance time by n grid steps (shift); long gaps use several S tokens
    P<p>   start a note of MIDI pitch p (21-108, the piano range) at the current time
    D<n>   the length of the note just started, in grid steps

Notes that start together (a chord) share one time position and are written lowest
pitch first, each followed by its D token, for example
    S12 P48 D24 P55 D24 P64 D12
Shifts and lengths are snapped to the values in TIME_STEPS, which cover 32nd notes to
a whole note, including triplets. Tempo, velocity and pedalling are not represented.
"""
import pretty_midi

GRID = 24  # grid steps per quarter note
TIME_STEPS = [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 32, 36, 48, 72, 96]
PITCH_MIN, PITCH_MAX = 21, 108

VOCAB = ([f"S{n}" for n in TIME_STEPS] + [f"P{p}" for p in range(PITCH_MIN, PITCH_MAX + 1)]
         + [f"D{n}" for n in TIME_STEPS])


def _nearest_step(n):
    return min(TIME_STEPS, key=lambda s: (abs(s - n), s))


def _shift_tokens(n):
    """Greedy decomposition of a gap of n grid steps into S tokens."""
    out = []
    while n > 0:
        step = max(s for s in TIME_STEPS if s <= n)
        out.append(f"S{step}")
        n -= step
    return out


def encode_midi(path):
    """Read every non-drum note of a MIDI file (all tracks/channels) into event tokens."""
    pm = pretty_midi.PrettyMIDI(path)
    res = pm.resolution
    notes = []
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for n in inst.notes:
            if PITCH_MIN <= n.pitch <= PITCH_MAX:
                start = round(pm.time_to_tick(n.start) * GRID / res)
                end = round(pm.time_to_tick(n.end) * GRID / res)
                notes.append((start, n.pitch, _nearest_step(max(1, end - start))))
    notes.sort()
    tokens, now = [], 0
    for start, pitch, dur in notes:
        if start > now:
            tokens += _shift_tokens(start - now)
            now = start
        tokens += [f"P{pitch}", f"D{dur}"]
    return tokens


def decode(tokens, bpm=100.0, velocity=80, default_steps=12):
    """Turn event tokens back into a one-track piano PrettyMIDI object.

    Tolerant of ill-formed sequences: a P without a following D gets default_steps,
    and a D that does not follow a P is ignored.
    """
    sec_per_step = 60.0 / bpm / GRID
    pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    piano = pretty_midi.Instrument(program=0, name="Piano")
    now, pending = 0, None  # pending = (start step, pitch) waiting for its D token

    def flush(steps):
        start, pitch = pending
        piano.notes.append(pretty_midi.Note(velocity=velocity, pitch=pitch, start=start * sec_per_step,
                                            end=(start + steps) * sec_per_step))

    for tok in tokens:
        kind, value = tok[0], int(tok[1:])
        if kind == "D":
            if pending:
                flush(value)
                pending = None
            continue
        if pending:
            flush(default_steps)
            pending = None
        if kind == "S":
            now += value
        elif kind == "P":
            pending = (now, value)
    if pending:
        flush(default_steps)
    pm.instruments.append(piano)
    return pm


def transpose_table(vocab, semitones):
    """Index map that shifts every P token by `semitones`; -1 where it leaves the piano range."""
    table = []
    for i, tok in enumerate(vocab):
        if tok[0] == "P":
            p = int(tok[1:]) + semitones
            table.append(vocab.index(f"P{p}") if PITCH_MIN <= p <= PITCH_MAX else -1)
        else:
            table.append(i)
    return table


def pitch_of(tok):
    return int(tok[1:]) if tok[0] == "P" else None
