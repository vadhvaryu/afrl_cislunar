"""Verify every original file has a relocated file and an exact archived copy."""
import hashlib
import json
from pathlib import Path
import zipfile


def main():
    here = Path(__file__).resolve().parent
    root = here.parents[1]
    manifest = json.loads((here / "original-sha256.json").read_text())
    moves = json.loads((here / "moves.json").read_text(encoding="utf-8-sig"))
    changed = []
    with zipfile.ZipFile(here / "original-files.zip") as archive:
        assert set(archive.namelist()) == set(manifest), "Archive inventory mismatch"
        for original, digest in manifest.items():
            assert hashlib.sha256(archive.read(original)).hexdigest() == digest, original
            current = original
            for source, target in moves:
                if current == source or current.startswith(source + "/"):
                    current = target + current[len(source):]
            destination = root / current
            assert destination.is_file(), f"Missing relocated file: {original} -> {current}"
            if hashlib.sha256(destination.read_bytes()).hexdigest() != digest:
                changed.append(current)
    print(json.dumps({"original_files": len(manifest), "archived_exactly": len(manifest),
                      "relocated_files_present": len(manifest), "edited_since_snapshot": changed}, indent=2))


if __name__ == "__main__":
    main()
