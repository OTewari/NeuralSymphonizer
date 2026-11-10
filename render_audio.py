"""Render MIDI files to piano audio: MP3, optionally WAV and an MP4 with a moving playhead.

The MIDI notes are played through a SoundFont with TinySoundFont, offline and
sample-accurately, then encoded with the ffmpeg build that the imageio-ffmpeg
package ships. Nothing needs installing outside pip.

By default the General MIDI SoundFont bundled with pretty_midi (TimGM6mb.sf2,
about 6 MB, GPL-2) is used with preset 0, Acoustic Grand Piano. It is small and
sounds basic; pass --soundfont to use a better piano SoundFont.

Example:
    python render_audio.py samples/sample_seed1.mid --video
"""
import argparse
import os
import subprocess
import tempfile

import imageio_ffmpeg
import numpy
import pretty_midi
import tinysoundfont


def default_soundfont():
    return os.path.join(os.path.dirname(pretty_midi.__file__), "TimGM6mb.sf2")


def read_notes(midi_path):
    """Return the non-drum notes of a MIDI file, sorted by start time (seconds)."""
    pm = pretty_midi.PrettyMIDI(midi_path)
    notes = [n for inst in pm.instruments if not inst.is_drum for n in inst.notes]
    return sorted(notes, key=lambda n: (n.start, n.pitch))


def render(notes, soundfont, preset=0, bank=0, sample_rate=44100, tail=2.0):
    """Synthesize notes to a float32 array of shape (samples, 2), peak-normalised to -1 dBFS."""
    synth = tinysoundfont.Synth(samplerate=sample_rate)
    sfid = synth.sfload(soundfont)
    synth.program_select(0, sfid, bank, preset)

    # Note-offs sort before note-ons at the same instant, so a repeated pitch is re-struck.
    events = [(n.start, 1, n.pitch, n.velocity) for n in notes] + [(n.end, 0, n.pitch, 0) for n in notes]
    events.sort(key=lambda e: (e[0], e[1]))

    chunks, position = [], 0
    for time, is_on, pitch, velocity in events:
        target = int(round(time * sample_rate))
        if target > position:
            chunks.append(numpy.frombuffer(synth.generate(target - position), dtype=numpy.float32).copy())
            position = target
        if is_on:
            synth.noteon(0, pitch, velocity)
        else:
            synth.noteoff(0, pitch)
    chunks.append(numpy.frombuffer(synth.generate(int(tail * sample_rate)), dtype=numpy.float32).copy())

    audio = numpy.concatenate(chunks).reshape(-1, 2)
    peak = float(numpy.abs(audio).max())
    if peak == 0:
        raise ValueError("Rendered audio is silent; is the SoundFont/preset valid?")
    return audio * (10 ** (-1 / 20) / peak)


def _ffmpeg(args, stdin=None):
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y"] + args
    subprocess.run(cmd, input=stdin, check=True)


def encode(audio, path, sample_rate=44100, bitrate="192k"):
    """Encode float32 stereo audio to MP3 or WAV, chosen by the file extension."""
    codec = ["-c:a", "libmp3lame", "-b:a", bitrate] if path.lower().endswith(".mp3") else ["-c:a", "pcm_s16le"]
    _ffmpeg(["-f", "f32le", "-ar", str(sample_rate), "-ac", "2", "-i", "-"] + codec + [path],
            stdin=audio.astype("<f4").tobytes())


def piano_roll_frame(notes, duration, png_path, title):
    """Draw a 1280x480 piano roll whose x axis is seconds; return the axes box in pixels."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12.8, 4.8), dpi=100)
    for n in notes:
        ax.add_patch(plt.Rectangle((n.start, n.pitch - 0.4), n.end - n.start, 0.8, color="#3b6ea5"))
    pitches = [n.pitch for n in notes]
    ax.set_xlim(0, duration)
    ax.set_ylim(min(pitches) - 2, max(pitches) + 2)
    ax.set_xlabel("time (seconds)")
    ax.set_ylabel("MIDI pitch (60 = middle C)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(png_path, dpi=100)
    box = ax.get_window_extent()  # display pixels, origin bottom-left
    height = fig.get_size_inches()[1] * fig.dpi
    plt.close(fig)
    return int(box.x0), int(box.width), int(height - box.y1), int(box.height)


def make_video(notes, audio_path, video_path, title, seconds, fps=25):
    """MP4 of the piano roll with a red playhead moving in time with the audio (seconds long)."""
    duration = max(n.end for n in notes)
    with tempfile.TemporaryDirectory() as tmp:
        frame = os.path.join(tmp, "frame.png")
        x0, width, y0, height = piano_roll_frame(notes, duration, frame, title)
        playhead = (f"color=c=red@0.85:s=3x{height}:r={fps}[bar];"
                    f"[0:v][bar]overlay=x='{x0}+min(t/{duration:.3f},1)*{width}':y={y0}:shortest=1,"
                    f"format=yuv420p[v]")
        _ffmpeg(["-loop", "1", "-framerate", str(fps), "-i", frame, "-i", audio_path,
                 "-filter_complex", playhead, "-map", "[v]", "-map", "1:a",
                 "-c:v", "libx264", "-tune", "stillimage", "-crf", "28",
                 "-c:a", "aac", "-b:a", "160k", "-t", f"{seconds:.3f}", "-movflags", "+faststart",
                 video_path])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("midi", nargs="+", help="MIDI file(s) to render")
    p.add_argument("--soundfont", default=None, help="path to an .sf2/.sf3 file (default: pretty_midi's TimGM6mb)")
    p.add_argument("--preset", type=int, default=0, help="SoundFont preset (0 = Acoustic Grand Piano in GM)")
    p.add_argument("--bank", type=int, default=0)
    p.add_argument("--sample-rate", type=int, default=44100)
    p.add_argument("--bitrate", default="192k")
    p.add_argument("--tail", type=float, default=2.0, help="seconds of release after the last note")
    p.add_argument("--wav", action="store_true", help="also write a WAV file")
    p.add_argument("--video", action="store_true", help="also write an MP4 with a moving playhead")
    args = p.parse_args(argv)

    soundfont = args.soundfont or default_soundfont()
    results = []
    for midi_path in args.midi:
        notes = read_notes(midi_path)
        if not notes:
            raise SystemExit(f"{midi_path}: no notes")
        audio = render(notes, soundfont, args.preset, args.bank, args.sample_rate, args.tail)
        base = os.path.splitext(midi_path)[0]
        outputs = [base + ".mp3"] + ([base + ".wav"] if args.wav else [])
        for out in outputs:
            encode(audio, out, args.sample_rate, args.bitrate)
        seconds = len(audio) / args.sample_rate
        if args.video:
            outputs.append(base + ".mp4")
            make_video(notes, base + ".mp3", base + ".mp4", os.path.basename(base), seconds)
        print(f"{midi_path}: {len(notes)} notes, {seconds:.1f} s -> " + ", ".join(outputs))
        results.append({"midi": midi_path, "notes": len(notes), "seconds": seconds, "outputs": outputs})
    return results


if __name__ == "__main__":
    main()
