#!/usr/bin/env python3
import argparse
import csv
import shutil
from pathlib import Path

def ensure_unique_path(target: Path) -> Path:
    """Return a non-existing path by adding _1, _2, ... before suffix (files) or at end (dirs)."""
    if not target.exists():
        return target
    parent = target.parent
    if target.is_file() or target.suffix:
        stem, suffix = target.stem, target.suffix
        i = 1
        while True:
            cand = parent / f"{stem}_{i}{suffix}"
            if not cand.exists():
                return cand
            i += 1
    name = target.name
    i = 1
    while True:
        cand = parent / f"{name}_{i}"
        if not cand.exists():
            return cand
        i += 1

def dir_is_empty(p: Path) -> bool:
    try:
        next(p.iterdir())
        return False
    except StopIteration:
        return True

def copy_item(src: Path, dst_dir: Path, dry: bool):
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = ensure_unique_path(dst_dir / src.name)
    if dry:
        print(f"[DRY] COPY {src} -> {dst}")
        return
    if src.is_dir():
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)

def move_item(src: Path, dst_dir: Path, dry: bool):
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = ensure_unique_path(dst_dir / src.name)
    if dry:
        print(f"[DRY] MOVE {src} -> {dst}")
        return
    shutil.move(str(src), str(dst))

def merge_dirs(src_dir: Path, dst_dir: Path, mode: str, dry: bool):
    """
    Merge contents of src_dir into dst_dir (creating it if needed).
    Files are moved/copied uniquely; subdirectories are merged recursively by name.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    for child in src_dir.iterdir():
        if child.is_dir():
            target_dir = dst_dir / child.name
            if not target_dir.exists():
                if mode == "copy":
                    copy_item(child, dst_dir, dry)
                else:
                    move_item(child, dst_dir, dry)
            else:
                merge_dirs(child, target_dir, mode, dry)
                if mode == "move" and not dry:
                    try:
                        if dir_is_empty(child):
                            child.rmdir()
                    except Exception:
                        pass
        else:
            if mode == "copy":
                copy_item(child, dst_dir, dry)
            else:
                move_item(child, dst_dir, dry)

def load_mapping_csv(path: Path, casefold: bool) -> dict:
    """
    Load a CSV with no header: sub_product,product per row.
    - Ignores blank lines and lines starting with '#'
    - Trims whitespace
    - Last mapping for a given sub_product wins (warn if duplicates)
    """
    mapping = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader, start=1):
            if not row or (len(row) == 1 and not row[0].strip()):
                continue
            if row[0].lstrip().startswith("#"):
                continue
            if len(row) < 2:
                raise SystemExit(f"CSV mapping error at line {i}: expected 2 columns, got {len(row)}")
            subp = row[0].strip()
            prod = row[1].strip()
            if not subp or not prod:
                raise SystemExit(f"CSV mapping error at line {i}: empty sub_product or product")
            key = subp.casefold() if casefold else subp
            if key in mapping and mapping[key] != prod:
                print(f"Warning: duplicate mapping for '{subp}' at line {i}: "
                      f"overwriting '{mapping[key]}' -> '{prod}'")
            mapping[key] = prod
    return mapping

def map_name_or_skip(name: str, mapping: dict | None, strict: bool, casefold: bool) -> str | None:
    """
    Returns product name, or None to indicate 'skip this sub_product'.
    Rules:
      - If mapping is None: return original name (no mapping in use).
      - If mapping provided:
          * strict==True and unmapped -> error
          * strict==False and unmapped -> return None (skip)
    """
    if mapping is None:
        return name
    key = name.casefold() if casefold else name
    if key in mapping:
        return mapping[key]
    if strict:
        raise SystemExit(f"Unmapped sub_product encountered (strict): {name}")
    print(f"Info: sub_product '{name}' not in mapping; skipping (not strict).")
    return None

def main():
    ap = argparse.ArgumentParser(
        description="Pivot store->sub_product tree into product->(merged contents), using a CSV sub_product→product mapping."
    )
    ap.add_argument("input_root", type=Path, help="Root containing store directories.")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output root (default: INPUT_ROOT.parent / (INPUT_ROOT.name + '_rearranged'))")
    ap.add_argument("--mode", choices=["move", "copy"], default="move",
                    help="Move (default) or copy data into the output tree.")
    ap.add_argument("--dry-run", action="store_true", help="Print actions without changing anything.")
    ap.add_argument("--prune-empty-stores", action="store_true",
                    help="If moving: delete store dirs that become empty.")
    # CSV mapping options
    ap.add_argument("--map-csv", type=Path, default=None,
                    help="Path to CSV mapping with no header: sub_product,product per row.")
    ap.add_argument("--strict-unmapped", action="store_true",
                    help="Fail if a sub_product is not found in the mapping CSV.")
    ap.add_argument("--case-insensitive", action="store_true",
                    help="Treat sub_product names case-insensitively when applying the mapping.")

    args = ap.parse_args()

    input_root = args.input_root.resolve()
    if not input_root.is_dir():
        raise SystemExit(f"Input root does not exist or is not a directory: {input_root}")

    out_root = (args.out.resolve()
                if args.out is not None
                else (input_root.parent / f"{input_root.name}_rearranged").resolve())

    if args.dry_run:
        print(f"[DRY] Output root would be: {out_root}")
    else:
        out_root.mkdir(parents=True, exist_ok=True)

    mapping = None
    if args.map_csv is not None:
        mapping = load_mapping_csv(args.map_csv.resolve(), casefold=args.case_insensitive)

    # Every top-level directory is a store; every subdirectory inside a store is a sub_product
    stores = sorted([d for d in input_root.iterdir() if d.is_dir()])
    if not stores:
        print("No store directories found. Nothing to do.")
        return

    for store in stores:
        sub_products = sorted([d for d in store.iterdir() if d.is_dir()])
        for sub_dir in sub_products:
            product_name = map_name_or_skip(sub_dir.name, mapping, args.strict_unmapped, args.case_insensitive)

            # Skip unmapped sub_products when mapping is provided and not strict
            if product_name is None:
                if args.dry_run:
                    print(f"[DRY] SKIP (unmapped) {sub_dir}")
                # If moving, optionally clean up empty dirs later; we didn't touch contents.
                continue

            dest_product_root = out_root / product_name
            if args.dry_run:
                print(f"[DRY] Ensure product dir: {dest_product_root}")
            else:
                dest_product_root.mkdir(parents=True, exist_ok=True)

            # Merge this store's sub_product into its product root (no per-store subfolders)
            merge_dirs(sub_dir, dest_product_root, args.mode, args.dry_run)

            # After moving, remove the now-empty sub_dir
            if args.mode == "move" and not args.dry_run:
                try:
                    if dir_is_empty(sub_dir):
                        sub_dir.rmdir()
                except Exception:
                    pass

        # Optionally prune emptied store dirs
        if args.prune_empty_stores and args.mode == "move" and not args.dry_run:
            try:
                for sub in list(store.iterdir()):
                    if sub.is_dir() and dir_is_empty(sub):
                        sub.rmdir()
                if dir_is_empty(store):
                    store.rmdir()
            except Exception as e:
                print(f"Warning: could not prune {store}: {e}")

    print("Done.")

if __name__ == "__main__":
    main()
