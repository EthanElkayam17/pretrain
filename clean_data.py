#!/usr/bin/env python3
import argparse
from pathlib import Path
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
import sys

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}

def is_image_path(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMAGE_EXTS

def looks_too_small(p: Path, min_bytes: int) -> bool:
    try:
        return p.stat().st_size < min_bytes
    except OSError:
        return True

def check_image(path: Path) -> bool:
    """
    Returns True if the file is a valid image, False otherwise.
    Uses Image.verify() (fast) + a reopen + load() (catches truncated files).
    """
    try:
        with Image.open(path) as im:
            im.verify()
        # Re-open and load to catch truncated images Pillow sometimes misses with verify()
        with Image.open(path) as im:
            im.load()
        return True
    except (UnidentifiedImageError, OSError, SyntaxError):
        return False

def gather_candidates(root: Path, recursive: bool, include_nonstd_ext: bool):
    it = root.rglob("*") if recursive else root.iterdir()
    if include_nonstd_ext:
        return [p for p in it if p.is_file()]
    else:
        return [p for p in it if is_image_path(p)]

def main():
    ap = argparse.ArgumentParser(
        description="Delete unidentifiable/bad image files in a directory (with progress bar)."
    )
    ap.add_argument("directory", type=Path, help="Directory to scan")
    ap.add_argument("-r", "--recursive", action="store_true", help="Recurse into subdirectories")
    ap.add_argument("--min-bytes", type=int, default=0,
                    help="Also delete images smaller than this many bytes (default: 0 = ignore)")
    ap.add_argument("--include-nonstd-ext", action="store_true",
                    help="Try ALL files (not only common image extensions).")
    ap.add_argument("--apply", action="store_true",
                    help="Actually delete files. Without this flag, runs in dry-run mode.")
    ap.add_argument("--no-progress", action="store_true",
                    help="Disable the progress bar.")
    args = ap.parse_args()

    if not args.directory.exists() or not args.directory.is_dir():
        print(f"Error: {args.directory} is not a directory", file=sys.stderr)
        sys.exit(1)

    # Build candidate list for accurate progress total
    candidates = gather_candidates(args.directory, args.recursive, args.include_nonstd_ext)

    total = len(candidates)
    bad = 0
    small = 0
    deleted = 0
    checked = 0

    iterator = candidates if args.no_progress else tqdm(candidates, desc="Scanning", unit="file")

    for p in iterator:
        checked += 1

        small_flag = args.min_bytes > 0 and looks_too_small(p, args.min_bytes)
        ok_flag = False if small_flag else check_image(p)

        if small_flag or not ok_flag:
            reason = "too small" if small_flag else "unidentifiable/bad"
            if small_flag:
                small += 1
            else:
                bad += 1

            if args.apply:
                try:
                    p.unlink()
                    deleted += 1
                    # Use tqdm.write so we don't break the progress bar formatting
                    tqdm.write(f"DELETED: {p}  ({reason})")
                except OSError as e:
                    tqdm.write(f"FAILED TO DELETE: {p}  ({reason}) -> {e}", file=sys.stderr)
            else:
                tqdm.write(f"[dry-run] Would delete: {p}  ({reason})")

    if not args.no_progress:
        # Clear the progress line before printing the summary
        sys.stderr.flush()

    print("\nSummary")
    print("-------")
    print(f"Scanned files:  {total}")
    print(f"Bad images:     {bad}")
    print(f"Too small:      {small} (threshold: {args.min_bytes} bytes)")
    if args.apply:
        print(f"Deleted:        {deleted}")
    else:
        print("Deleted:        0 (dry-run; use --apply to actually delete)")

if __name__ == "__main__":
    main()
