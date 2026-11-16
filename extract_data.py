import argparse
import os
import shutil
import tarfile
import tempfile
import zipfile

from download_data import (
    DEFAULT_ARCHIVE_DIR,
    DEFAULT_MANIFEST,
    parse_manifest,
    select_entries,
)

DEFAULT_DATA_DIR = "./data"


def _is_within_directory(base_dir, target_path):
    abs_base = os.path.abspath(base_dir)
    abs_target = os.path.abspath(target_path)
    return abs_target == abs_base or abs_target.startswith(abs_base + os.sep)


def _safe_extract_zip(archive_path, staging_dir):
    with zipfile.ZipFile(archive_path, "r") as zf:
        for member in zf.infolist():
            dest = os.path.join(staging_dir, member.filename)
            if not _is_within_directory(staging_dir, dest):
                raise RuntimeError(f"Unsafe zip member path detected: {member.filename}")
        zf.extractall(staging_dir)


def _safe_extract_tar(archive_path, staging_dir):
    with tarfile.open(archive_path, "r:*") as tf:
        for member in tf.getmembers():
            dest = os.path.join(staging_dir, member.name)
            if not _is_within_directory(staging_dir, dest):
                raise RuntimeError(f"Unsafe tar member path detected: {member.name}")
        tf.extractall(staging_dir)


def _extract_archive(archive_path, staging_dir):
    if zipfile.is_zipfile(archive_path):
        _safe_extract_zip(archive_path, staging_dir)
    elif tarfile.is_tarfile(archive_path):
        _safe_extract_tar(archive_path, staging_dir)
    else:
        raise ValueError(
            f"Unsupported or corrupted archive format: {archive_path} (expected .zip, .tar, or .tar.gz)"
        )


def _find_train_test_root(staging_dir):
    candidates = []
    for root, dirs, _ in os.walk(staging_dir):
        dirs[:] = [d for d in sorted(dirs) if d != "__MACOSX" and not d.startswith(".")]
        if "train" in dirs and "test" in dirs:
            rel_depth = 0 if root == staging_dir else len(os.path.relpath(root, staging_dir).split(os.sep))
            candidates.append((rel_depth, root))
    if not candidates:
        top_items = sorted(os.listdir(staging_dir))
        raise RuntimeError(
            f"Could not find both 'train/' and 'test/' directories inside extracted archive. Top-level entries: {top_items}"
        )
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def _is_extracted_valid(target_dir):
    train_dir = os.path.join(target_dir, "train")
    test_dir = os.path.join(target_dir, "test")
    return os.path.isdir(train_dir) and os.path.isdir(test_dir)


def extract_entry(entry, archive_dir=DEFAULT_ARCHIVE_DIR, data_dir=DEFAULT_DATA_DIR, cleanup=False, force=False):
    archive_path = os.path.join(archive_dir, entry["filename"])
    target_dir = os.path.join(data_dir, entry["target_relpath"])

    if _is_extracted_valid(target_dir) and not force:
        print(f"[SKIP] {entry['dataset_key']} already extracted at {target_dir}")
        if cleanup and os.path.isfile(archive_path):
            os.remove(archive_path)
            print(f"[CLEANUP] Removed archive {archive_path}")
        return target_dir, False

    if not os.path.isfile(archive_path):
        raise FileNotFoundError(
            f"Archive not found: {archive_path}. Run download_data.py first."
        )

    os.makedirs(data_dir, exist_ok=True)
    staging_dir = tempfile.mkdtemp(prefix=f".extract_{entry['dataset_key']}_", dir=data_dir)
    try:
        print(f"[EXTRACT] {archive_path} -> {target_dir}")
        _extract_archive(archive_path, staging_dir)
        source_root = _find_train_test_root(staging_dir)

        os.makedirs(target_dir, exist_ok=True)
        for split in ("train", "test"):
            dest_split = os.path.join(target_dir, split)
            if os.path.exists(dest_split):
                if force:
                    shutil.rmtree(dest_split)
                else:
                    raise FileExistsError(f"Destination already exists: {dest_split}")
            shutil.move(os.path.join(source_root, split), dest_split)
    finally:
        if os.path.exists(staging_dir):
            shutil.rmtree(staging_dir, ignore_errors=True)

    if not _is_extracted_valid(target_dir):
        raise RuntimeError(f"Extraction verification failed: missing train/ or test/ in {target_dir}")

    if cleanup and os.path.isfile(archive_path):
        os.remove(archive_path)
        print(f"[CLEANUP] Removed archive {archive_path}")

    return target_dir, True


def setup_parser():
    parser = argparse.ArgumentParser(
        description="Extract and normalize continual learning dataset archives into ./data/."
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=DEFAULT_MANIFEST,
        help="Path to dataset manifest file (default: dataset.txt).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        nargs="+",
        default=None,
        help="Dataset name(s) to extract (e.g., cub imagenetr vtab). Defaults to all entries in manifest.",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default=None,
        help="Benchmark preset or cluster name from exps/benchmarks.json (e.g., fast, all_6).",
    )
    parser.add_argument(
        "--archive_dir",
        type=str,
        default=DEFAULT_ARCHIVE_DIR,
        help="Directory containing downloaded archive files (default: ./data/archives).",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=DEFAULT_DATA_DIR,
        help="Root data directory for normalized datasets (default: ./data).",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        default=False,
        help="Delete archive files after successful extraction and verification.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Re-extract and overwrite existing train/ and test/ directories.",
    )
    return parser


def main():
    args = setup_parser().parse_args()
    entries = parse_manifest(args.manifest)
    selected, notices = select_entries(entries, datasets=args.dataset, benchmark=args.benchmark)

    for notice in notices:
        print(f"[INFO] {notice}")

    if not selected:
        print("[INFO] No manifest archives to extract.")
        return

    for entry in selected:
        extract_entry(
            entry,
            archive_dir=args.archive_dir,
            data_dir=args.data_dir,
            cleanup=args.cleanup,
            force=args.force,
        )

    print("[DONE] Extraction step complete.")


if __name__ == "__main__":
    main()
