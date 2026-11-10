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
| 3. Evaluation and sampling | Objective comparison with real music; grammar-constrained top-k 10 sampling; stage-2 model | [1](results/stage3-sampling-eval/samples/sample_seed1.mp3) · [2](results/stage3-sampling-eval/samples/sample_seed2.mp3) · [3](results/stage3-sampling-eval/samples/sample_seed3.mp3) |
| 4. More data | The same model trained on MAESTRO's 962 recorded performances (27.6 times the tokens); stage-3 sampling | [1](results/stage4-maestro/samples/sample_seed1.mp3) · [2](results/stage4-maestro/samples/sample_seed2.mp3) · [3](results/stage4-maestro/samples/sample_seed3.mp3); MAESTRO-prompted: [1](results/stage4-maestro/samples-maestro/sample_seed1.mp3) · [2](results/stage4-maestro/samples-maestro/sample_seed2.mp3) · [3](results/stage4-maestro/samples-maestro/sample_seed3.mp3) |
| 5. How the LSTM is trained | LayerNorm after each LSTM, Adam with gradient clipping; piano-midi.de data again; ablations of each change | [1](results/stage5-lstm-training/samples/sample_seed1.mp3) · [2](results/stage5-lstm-training/samples/sample_seed2.mp3) · [3](results/stage5-lstm-training/samples/sample_seed3.mp3) |

Each stage's section in [`PAPER.md`](PAPER.md) gives the commit its code is at. In
brief, comparing each model with the 10 held-out real pieces:

| | held-out loss per event token | mean similarity (0–1) | patterns repeated |
|---|---|---|---|
| stage 1 | (different tokens) | 0.540 | 1.9% |
| stage 2 | 1.779 | 0.771 | 0.3% |
| stage 3 (stage-2 model, new sampling) | 1.779 | 0.756 | 0.9% |
| stage 4 (MAESTRO model) | 1.838 | 0.662 | 0.6% |
| stage 5 | **1.490** | 0.758 | 5.7% |
| real training pieces | — | 0.781 | 60.1% |
| the held-out pieces themselves | — | 1 | 66.0% |

## How it works now (stage 5)

- `music_tokens.py` turns MIDI into event tokens. `P<pitch>` starts a note at its
  exact MIDI pitch, `D<n>` gives its length, and `S<n>` advances time, on a grid of
  1/24 of a beat. The vocabulary is 120 tokens, and a file decodes back to exactly the
  notes it started with.
- `MidiModelTraining.py` reads every note of the 95 files in `midi_files/` (about
  5 s), holds out 10 whole pieces, and trains an Embedding → 3 × (LSTM 512 →
  LayerNorm) → Dense 256 → softmax model with Adam and gradient clipping. It predicts
  the next token at every position of 256-token windows, and each training window is
  transposed by a random −6 to +5 semitones. `--dataset maestro` trains on MAESTRO
  instead (`get_maestro.py` downloads it).
- `MidiGeneration.py` takes a 128-token prompt from a held-out piece and samples until
  it has a given number of notes. Sampling can use top-k, top-p and grammar-constrained
  decoding. It writes MIDI at a fixed tempo, with an optional piano roll.
- `render_audio.py` plays MIDI through a piano SoundFont and writes an MP3, plus an
  optional MP4 with a moving playhead.
- `evaluate.py` compares sets of MIDI files with the real held-out pieces through
  feature distributions, and `evaluate_bootstrap.py` gives 95% intervals.
  `context_probe.py` shows how much of its context window a model uses.

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

Every stage's weights are in the repository (22 MB each). The default model is the
stage-5 one (`results/stage5-lstm-training/b-layernorm-adam-clip/`), so this works
right after cloning:

```bash
python MidiGeneration.py --seed 1 --notes 300 --temperature 1.0 --top-k 10 --grammar --bpm 100 --output results/stage5-lstm-training/samples/sample_seed1.mid --piano-roll
python render_audio.py results/stage5-lstm-training/samples/sample_seed1.mid --video
```

`--top-k 10 --grammar` is the sampling chosen in stage 3. For an earlier stage's
model, pass `--model-dir`: `results/stage2-rhythm-transpose/model`, or
`results/stage4-maestro/model` with `--prompt-source maestro --bpm 120`. Stage 1 used
different code; check out commit `fc3f7a8` to run it.

`MidiGeneration.py` options: `--model-dir`, `--notes`, `--prompt-len`,
`--temperature`, `--top-k`, `--top-p`, `--grammar`, `--prompt-piece`,
`--prompt-start`, `--prompt-source`, `--count` (a batch of samples with consecutive
seeds), `--seed`,
`--bpm`, `--output` and `--piano-roll`. It prints
structural statistics for the sample. On the same machine the same seed gives the same
file; on other hardware it may not, because tiny numerical differences change which
token is sampled. `render_audio.py` uses `TimGM6mb.sf2` from `pretty_midi` by default,
which sounds basic; pass `--soundfont path/to/piano.sf2` for a better piano.
`.mp4` and `.wav` files are git-ignored.

## Train

The stage-5 model (8.4 minutes on the RTX 5070):

```bash
python MidiModelTraining.py --epochs 500 --patience 40 --units 512 --layers 3 --seq-len 256 --batch-size 32 --seed 42 --stage 5 --layer-norm --optimizer adam --learning-rate 0.001 --clipnorm 1.0 --out-dir results/stage5-lstm-training/b-layernorm-adam-clip
```

The stage-2 model is the same command without `--stage 5 --layer-norm --optimizer adam
--learning-rate 0.001 --clipnorm 1.0`. The stage-4 model (about 80 minutes; run
`python get_maestro.py` first):

```bash
python MidiModelTraining.py --dataset maestro --stage 4 --epochs 80 --steps-per-epoch 1000 --patience 10 --units 512 --layers 3 --seq-len 256 --batch-size 32 --seed 42 --out-dir results/stage4-maestro/model
```

Other options: `--embedding-dim`, `--dropout`, `--variational-dropout`,
`--val-split`, `--transpose-min` and `--transpose-max` (set both to 0 to turn
transposition off), `--max-files`.
Training writes `best.weights.h5` (the checkpoint, including optimizer state;
git-ignored), `model.weights.h5` (weights only, used for generation), `vocab.json`,
`config.json`, `history.csv` and `loss_curve.png`.

## Test

```bash
python -m unittest discover -s tests -v
```

Five smoke tests run in about 5 seconds on CPU:
- the token round trip and the transposition table;
- the top-k, top-p and grammar filters;
- a 16-unit model trained for one epoch on the two smallest files, then generating
  12 notes;
- rendering a few notes to MP3 and MP4.

The GitHub Actions workflow in `.github/workflows/smoke-test.yml` runs them on CPU.

## Limitations

See [`PAPER.md`](PAPER.md) for the measured limits of each stage. In short, as of
stage 5:
- The model rarely repeats earlier material. At most 5.7% of 4-onset patterns repeat
  in generated samples, against 66% in real pieces.
- Every model in the study uses only about the last 32 to 63 tokens of context.
- It still overfits the 95-piece corpus after about 25 epochs.
- Tempo, dynamics (velocity) and pedalling are not modelled.
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
- **Stage-4 training data:** MAESTRO v3.0.0 (Hawthorne et al., "Enabling Factorized
  Piano Music Modeling and Generation with the MAESTRO Dataset", ICLR 2019),
  [magenta.tensorflow.org/datasets/maestro](https://magenta.tensorflow.org/datasets/maestro),
  made available by Google LLC under CC BY-NC-SA 4.0. It is not included here;
  `get_maestro.py` downloads it.
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
- **Exception, stage 4:** `results/stage4-maestro/` (its model and samples) is
  derived from MAESTRO, not from `midi_files/`. It is shared under MAESTRO's
  CC BY-NC-SA 4.0 terms, so non-commercial use only.
