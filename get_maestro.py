"""Download the MAESTRO v3.0.0 MIDI files (58.4 MB) into data/maestro/ and verify them.

MAESTRO: Hawthorne et al., "Enabling Factorized Piano Music Modeling and Generation
with the MAESTRO Dataset", ICLR 2019. 1,276 performances from the International
Piano-e-Competition, about 199 hours. Released by Google LLC under the Creative
Commons Attribution Non-Commercial Share-Alike 4.0 licence (CC BY-NC-SA 4.0):
https://magenta.tensorflow.org/datasets/maestro

The files are not part of this repository; data/maestro/ is git-ignored.

    python get_maestro.py            # download, check SHA-256, unpack
"""
import hashlib
import os
import sys
import urllib.request
import zipfile

URL = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip"
SHA256 = "70470ee253295c8d2c71e6d9d4a815189e35c89624b76d22fce5a019d5dde12c"
DEST = os.path.join("data", "maestro")


def main():
    os.makedirs(DEST, exist_ok=True)
    archive = os.path.join(DEST, os.path.basename(URL))
    if not os.path.exists(archive):
        print(f"downloading {URL}")
        urllib.request.urlretrieve(URL, archive)
    digest = hashlib.sha256(open(archive, "rb").read()).hexdigest()
    if digest != SHA256:
        sys.exit(f"SHA-256 mismatch for {archive}: {digest}")
    print("SHA-256 OK")
    with zipfile.ZipFile(archive) as zf:
        for member in zf.namelist():
            target = os.path.realpath(os.path.join(DEST, member))
            if not target.startswith(os.path.realpath(DEST) + os.sep):
                sys.exit(f"refusing to extract {member!r} outside {DEST}")
        zf.extractall(DEST)
    root = os.path.join(DEST, "maestro-v3.0.0")
    n = sum(f.lower().endswith((".mid", ".midi")) for _, _, fs in os.walk(root) for f in fs)
    print(f"{n} MIDI files under {root}")


if __name__ == "__main__":
    main()
