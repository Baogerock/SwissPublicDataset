#!/usr/bin/env python3
import argparse
import os
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Match DSM/DTM tiles by coordinate token and remove unmatched files."
        )
    )
    parser.add_argument(
        "--dsm-dir",
        required=True,
        help="Directory containing DSM tiles (e.g., dataset/train/dsm)",
    )
    parser.add_argument(
        "--dtm-dir",
        required=True,
        help="Directory containing DTM tiles (e.g., dataset/train/dtm)",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan for files inside DSM/DTM directories",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show deletions without removing files",
    )
    return parser.parse_args()


def iter_files(root, recursive):
    if recursive:
        for dirpath, _, filenames in os.walk(root):
            for filename in filenames:
                yield os.path.join(dirpath, filename)
    else:
        for filename in os.listdir(root):
            path = os.path.join(root, filename)
            if os.path.isfile(path):
                yield path


def tile_key_from_name(path):
    base = os.path.basename(path)
    stem, _ = os.path.splitext(base)
    parts = stem.split("_")
    if len(parts) >= 4:
        return "_".join(parts[-4:])
    return stem


def build_tile_map(root, recursive):
    tile_map = defaultdict(list)
    for path in iter_files(root, recursive):
        tile_map[tile_key_from_name(path)].append(path)
    return tile_map


def delete_paths(paths, dry_run):
    for path in paths:
        if dry_run:
            print(f"[dry-run] remove {path}")
        else:
            os.remove(path)
            print(f"removed {path}")


def count_files(root, recursive):
    return sum(1 for _ in iter_files(root, recursive))


def main():
    args = parse_args()

    if not os.path.isdir(args.dsm_dir):
        raise SystemExit(f"DSM directory not found: {args.dsm_dir}")
    if not os.path.isdir(args.dtm_dir):
        raise SystemExit(f"DTM directory not found: {args.dtm_dir}")

    dsm_tiles = build_tile_map(args.dsm_dir, args.recursive)
    dtm_tiles = build_tile_map(args.dtm_dir, args.recursive)

    dsm_only = set(dsm_tiles) - set(dtm_tiles)
    dtm_only = set(dtm_tiles) - set(dsm_tiles)

    for tile in sorted(dsm_only):
        delete_paths(dsm_tiles[tile], args.dry_run)
    for tile in sorted(dtm_only):
        delete_paths(dtm_tiles[tile], args.dry_run)

    dsm_count = count_files(args.dsm_dir, args.recursive)
    dtm_count = count_files(args.dtm_dir, args.recursive)
    print(f"DSM files after sync: {dsm_count}")
    print(f"DTM files after sync: {dtm_count}")

    if dsm_count != dtm_count:
        print("Warning: DSM/DTM counts are still different.")


if __name__ == "__main__":
    main()
