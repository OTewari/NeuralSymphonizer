# NeuralSymphonizer

An LSTM that learns to continue classical piano music from MIDI files, improved one
documented stage at a time. It began as a reproduction of Sigurður Skúli's 2017 Keras
tutorial (see [Credits](#credits)). **[`PAPER.md`](PAPER.md)** is the full write-up:
what limited each stage, what was changed and why, and the measured results.

## Stages and recordings

Every stage keeps its trained model, its samples (MIDI, piano roll, statistics) and MP3
recordings under `results/`. GitHub does not play audio inside a README, so the
recordings are linked; open one and use GitHub's download or raw button to listen.

| Stage | What changed | Listen |
|---|---|---|
| 1. Baseline | Tutorial tokens (one per note or chord, no timing), 3 × 512 LSTM | [1](results/stage1-baseline/samples/sample_seed1.mp3) · [2](results/stage1-baseline/samples/sample_seed2.mp3) · [3](results/stage1-baseline/samples/sample_seed3.mp3) |
| 2. Timing, octaves, transposition | Event tokens with note lengths and exact pitches, all notes read, random transposition into 12 keys; same model | [1](results/stage2-rhythm-transpose/samples/sample_seed1.mp3) · [2](results/stage2-rhythm-transpose/samples/sample_seed2.mp3) · [3](results/stage2-rhythm-transpose/samples/sample_seed3.mp3) |

Each stage's section in [`PAPER.md`](PAPER.md) gives the commit its code is at.

## How it works now (stage 2)

- `music_tokens.py` turns MIDI into event tokens. `P<pitch>` starts a note at its
  exact MIDI pitch, `D<n>` gives its length, and `S<n>` advances time, on a grid of
  1/24 of a beat. The vocabulary is 120 tokens, and a file decodes back to exactly the
  notes it started with.
- `MidiModelTraining.py` reads every note of the 95 files in `midi_files/` (about
  5 s), holds out 10 whole pieces, and trains an Embedding → 3 × LSTM 512 → Dense 256
  → softmax model. It predicts the next token at every position of 256-token windows,
  and each training window is transposed by a random −6 to +5 semitones.
- `MidiGeneration.py` takes a 128-token prompt from a held-out piece and samples until
  it has a given number of notes. It writes MIDI at a fixed tempo, with an optional
  piano roll.
- `render_audio.py` plays MIDI through a piano SoundFont and writes an MP3, plus an
  optional MP4 with a moving playhead.

## Setup

Tested on Windows 11 with Python 3.12.6 and an NVIDIA RTX 5070.

```bash
git clone https://github.com/OTewari/NeuralSymphonizer.git
cd NeuralSymphonizer
python -m venv .venv
```

Activate it (`.venv\Scripts\activate` on Windows, `source .venv/bin/activate`
elsewhere), then install PyTorch and the rest. With an NVIDIA GPU:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

CPU only:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

The scripts default `KERAS_BACKEND` to `torch`, so TensorFlow is not needed. On
Linux, install the PortAudio headers first (`sudo apt-get install portaudio19-dev`),
because `tinysoundfont` depends on `pyaudio`, which has no Linux wheel.

## Generate and listen

The stage-2 weights (`results/stage2-rhythm-transpose/model/model.weights.h5`,
22 MB) are in the repository, so this works right after cloning:

```bash
python MidiGeneration.py --seed 1 --notes 300 --temperature 1.0 --bpm 100 --output results/stage2-rhythm-transpose/samples/sample_seed1.mid --piano-roll
python render_audio.py results/stage2-rhythm-transpose/samples/sample_seed1.mid --video
```

`MidiGeneration.py` options: `--model-dir`, `--notes`, `--prompt-len`,
`--temperature`, `--seed`, `--bpm`, `--output` and `--piano-roll`. It prints
structural statistics for the sample. On the same machine the same seed gives the same
file; on other hardware it may not, because tiny numerical differences change which
token is sampled. `render_audio.py` uses `TimGM6mb.sf2` from `pretty_midi` by default,
which sounds basic; pass `--soundfont path/to/piano.sf2` for a better piano.
`.mp4` and `.wav` files are git-ignored.

## Train

The stage-2 run (5.8 minutes on the RTX 5070):

```bash
python MidiModelTraining.py --epochs 500 --patience 40 --units 512 --layers 3 --seq-len 256 --batch-size 32 --seed 42 --out-dir results/stage2-rhythm-transpose/model
```

Other options: `--embedding-dim`, `--dropout`, `--val-split`, `--transpose-min` and
`--transpose-max` (set both to 0 to turn transposition off), `--max-files`.
Training writes `best.weights.h5` (the checkpoint, including optimizer state;
git-ignored), `model.weights.h5` (weights only, used for generation), `vocab.json`,
`config.json`, `history.csv` and `loss_curve.png`.

## Test

```bash
python -m unittest discover -s tests -v
```

Three smoke tests run in about 5 seconds on CPU:
- the token round trip and the transposition table;
- a 16-unit model trained for one epoch on the two smallest files, then generating
  12 notes;
- rendering a few notes to MP3 and MP4.

The GitHub Actions workflow in `.github/workflows/smoke-test.yml` runs them on CPU.

## Limitations

See [`PAPER.md`](PAPER.md) for the measured limits of each stage. In short, for
stage 2:
- The model still overfits the 95-piece corpus.
- Tempo, dynamics (velocity) and pedalling are not modelled.
- Samples written at temperature 1.0 vary a lot in density and contain stray extreme
  notes.
- The statistics and audio checks show structure and correct rendering, not musical
  quality.

## Credits

- **Approach and original code structure:** Sigurður Skúli, *How to Generate Music
  using a LSTM Neural Network in Keras* (Towards Data Science, December 2017), and
  its repository [Skuldur/Classical-Piano-Composer](https://github.com/Skuldur/Classical-Piano-Composer)
  (MIT licence). Stage 1 follows it closely: the note/chord encoding, the stacked
  512-unit LSTM design and the checkpoint naming. Later stages replace the encoding
  and the training setup; see [`PAPER.md`](PAPER.md).
- **Training data:** the MIDI files in `midi_files/` are by Bernd Krueger,
  [piano-midi.de](http://www.piano-midi.de). Every file carries a copyright meta
  event naming him (for example "Copyright © 2002 by Bernd Krueger"). His site
  licenses the files under the Creative Commons Attribution-ShareAlike licence
  (Germany): use and adaptation are allowed with attribution to Bernd Krueger and
  piano-midi.de, and redistribution must be under the same licence. The files in
  this repository are redistributed on those terms. The compositions are by Chopin,
  Beethoven and Mozart.
- Whether the generated samples count as adaptations of the training files is not
  settled here; they are shared on the same CC BY-SA terms to be safe.
- **Audio rendering:** [TinySoundFont](https://pypi.org/project/tinysoundfont/)
  (Python bindings by Nathan Whitehead, MIT), with `TimGM6mb.sf2` (by Tim Brechbill,
  with additions by David Bolton; GPL-2 according to Debian's packaging) as bundled in
  [pretty_midi](https://github.com/craffel/pretty-midi), and the ffmpeg build from
  [imageio-ffmpeg](https://pypi.org/project/imageio-ffmpeg/). The SoundFont itself is
  not included in this repository.

## Licence

- **Code:** MIT, see [`LICENSE`](LICENSE). Parts of the code are adapted from
  Classical-Piano-Composer, which is also MIT-licensed; its copyright notice is
  reproduced in [`NOTICE.md`](NOTICE.md).
- **MIDI files and what is derived from them:** the MIT licence does not cover
  `midi_files/`. Those files keep the Creative Commons Attribution-ShareAlike terms
  described under [Credits](#credits). The trained weights and the generated files
  in `results/` are derived from them and are shared on the same terms to be safe.
