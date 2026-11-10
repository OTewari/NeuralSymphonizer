# Notices

The MIT licence in [`LICENSE`](LICENSE) covers the source code written for this
repository. Two things in the repository come from other people and keep their
own terms.

## Code adapted from Classical-Piano-Composer

The note and chord encoding and the stacked LSTM design in the first version of
this repository follow
[Skuldur/Classical-Piano-Composer](https://github.com/Skuldur/Classical-Piano-Composer).
That project is distributed under the MIT License, reproduced here as it
requires:

```
MIT License

Copyright (c) 2019 Sigurður Skúli Sigurgeirsson

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## MIDI files

The files in `midi_files/` are by Bernd Krueger, <http://www.piano-midi.de>,
and are distributed under a Creative Commons Attribution-ShareAlike licence
(Germany). They are not covered by the MIT licence. The trained weights and the
generated files in `results/` are derived from them and are shared on the same
terms to be safe, except `results/stage4-maestro/` (see below). See the Credits
section of [`README.md`](README.md) for the details.

## MAESTRO

`results/stage4-maestro/` holds a model trained on, and samples prompted from,
MAESTRO v3.0.0 (Hawthorne et al., ICLR 2019), made available by Google LLC under
the Creative Commons Attribution Non-Commercial Share-Alike 4.0 licence
(<https://magenta.tensorflow.org/datasets/maestro>). That folder is shared on the
same CC BY-NC-SA 4.0 terms. The MAESTRO files themselves are not in this
repository; `get_maestro.py` downloads them.
