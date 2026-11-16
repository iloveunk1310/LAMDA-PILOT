import argparse
import json
import os
import sys
import urllib.parse
import urllib.request

DEFAULT_MANIFEST = "dataset.txt"
DEFAULT_ARCHIVE_DIR = "./data/archives"
BENCHMARKS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exps", "benchmarks.json")

# Canonical dataset definitions matching utils/data.py
DATASET_SPECS = {
    "cub": {
        "dataset_key": "cub",
        "target_relpath": "cub",
        "aliases": {"cub", "cub.zip"},
    },
    "imagenet-r": {
        "dataset_key": "imagenetr",
        "target_relpath": "imagenet-r",
        "aliases": {"imagenetr", "imagenet-r", "imagenet_r", "inr", "imagenet-r.zip"},
    },
    "imagenet-a": {
        "dataset_key": "imageneta",
        "target_relpath": "imagenet-a",
        "aliases": {"imageneta", "imagenet-a", "imagenet_a", "ina", "imagenet-a.zip"},
    },
    "omnibenchmark": {
        "dataset_key": "omnibenchmark",
        "target_relpath": "omnibenchmark",
        "aliases": {"omnibenchmark", "omnibench", "omnibenchmark.zip"},
    },
    "vtab": {
        "dataset_key": "vtab",
        "target_relpath": os.path.join("vtab-cil", "vtab"),
        "aliases": {"vtab", "vtab-cil", "vtab.zip"},
    },
}

CIFAR_ALIASES = {"cifar224", "cifar100", "cifar", "cifar_b0inc5"}


def _canonical_spec_for_filename(filename):
    stem = os.path.basename(filename).lower()
    for base in (stem, stem.rsplit(".", 1)[0], stem.replace(".tar.gz", "")):
        for spec_name, spec in DATASET_SPECS.items():
            if base == spec_name or base in spec["aliases"]:
                return spec_name, spec
    dataset_stem = stem.split(".")[0]
    return dataset_stem, {
        "dataset_key": dataset_stem,
        "target_relpath": dataset_stem,
        "aliases": {dataset_stem, stem},
    }


def parse_manifest(manifest_path=DEFAULT_MANIFEST):
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    entries = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line_num, raw_line in enumerate(f, 1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                raise ValueError(
                    f"Invalid manifest entry at {manifest_path}:{line_num}: expected '<url> <filename>', got {raw_line!r}"
                )
            url, filename = parts[0], parts[1]
            spec_name, spec = _canonical_spec_for_filename(filename)
            entries.append({
                "url": url,
                "filename": filename,
                "name": spec_name,
                "dataset_key": spec["dataset_key"],
                "target_relpath": spec["target_relpath"],
                "aliases": set(spec["aliases"]) | {spec_name, filename.lower()},
            })
    return entries


def _resolve_benchmark_datasets(benchmark_name, benchmarks_path=BENCHMARKS_PATH):
    with open(benchmarks_path, "r", encoding="utf-8") as f:
        benchmarks = json.load(f)

    presets = benchmarks.get("presets", {})
    clusters = benchmarks.get("clusters", {})

    if benchmark_name in clusters:
        preset_names = clusters[benchmark_name]
    elif benchmark_name in presets:
        preset_names = [benchmark_name]
    else:
        valid = sorted(list(presets.keys()) + list(clusters.keys()))
        raise ValueError(
            f"Unknown benchmark '{benchmark_name}'. Available presets/clusters: {', '.join(valid)}"
        )

    datasets = []
    for preset_name in preset_names:
        ds = presets[preset_name]["dataset"]
        if ds not in datasets:
            datasets.append(ds)
    return datasets


def select_entries(entries, datasets=None, benchmark=None, benchmarks_path=BENCHMARKS_PATH):
    requested = []
    if datasets:
        requested.extend(datasets)
    if benchmark:
        requested.extend(_resolve_benchmark_datasets(benchmark, benchmarks_path=benchmarks_path))

    if not requested:
        return list(entries), []

    selected = []
    selected_filenames = set()
    skipped_cifar = False
    unknown = []

    for item in requested:
        norm = item.strip().lower()
        if norm in CIFAR_ALIASES:
            skipped_cifar = True
            continue
        matched = [e for e in entries if norm in e["aliases"]]
        if not matched:
            unknown.append(item)
            continue
        for entry in matched:
            if entry["filename"] not in selected_filenames:
                selected.append(entry)
                selected_filenames.add(entry["filename"])

    if unknown:
        available = ", ".join(e["dataset_key"] for e in entries)
        raise ValueError(
            f"Unknown dataset filter(s): {', '.join(unknown)}. Available manifest datasets: {available} (and cifar224)."
        )

    notices = []
    if skipped_cifar:
        notices.append(
            "Skipping 'cifar224': CIFAR-100 is automatically downloaded by torchvision during training and does not require a dataset.txt entry."
        )
    return selected, notices


def _prepare_sharepoint_url(url):
    parsed = urllib.parse.urlsplit(url)
    query = urllib.parse.parse_qs(parsed.query, keep_blank_values=True)
    if "download" not in query:
        sep = "&" if parsed.query else "?"
        return f"{url}{sep}download=1"
    return url


def _check_not_html_file(file_path, url):
    with open(file_path, "rb") as f:
        head = f.read(512).lstrip().lower()
    if head.startswith(b"<!doctype html") or head.startswith(b"<html"):
        raise RuntimeError(
            f"Downloaded HTML error page instead of archive from '{url}'. "
            "Check link permissions/quota or update the URL in dataset.txt."
        )


def _download_google_drive(url, output_path):
    try:
        import gdown
    except ImportError as exc:
        raise RuntimeError(
            "The 'gdown' package is required for Google Drive links. Install it with: pip install gdown"
        ) from exc

    tmp_path = output_path + ".part"
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    result = gdown.download(url, tmp_path, fuzzy=True, quiet=False)
    if not result or not os.path.isfile(tmp_path) or os.path.getsize(tmp_path) == 0:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise RuntimeError(f"gdown failed to download archive from: {url}")

    try:
        _check_not_html_file(tmp_path, url)
    except Exception:
        os.remove(tmp_path)
        raise
    os.replace(tmp_path, output_path)


def _download_direct(url, output_path):
    download_url = _prepare_sharepoint_url(url) if (
        "sharepoint.com" in url.lower() or "onedrive" in url.lower() or "1drv.ms" in url.lower()
    ) else url

    tmp_path = output_path + ".part"
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        import requests

        with requests.get(download_url, headers=headers, stream=True, allow_redirects=True, timeout=60) as resp:
            resp.raise_for_status()
            content_type = resp.headers.get("Content-Type", "")
            if "text/html" in content_type.lower():
                raise RuntimeError(
                    f"Server returned HTML ('{content_type}') instead of archive from '{download_url}'. "
                    "Check link sharing permissions or update the URL in dataset.txt."
                )
            with open(tmp_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
    except ImportError:
        opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor())
        req = urllib.request.Request(download_url, headers=headers)
        with opener.open(req, timeout=60) as resp:
            content_type = resp.headers.get("Content-Type", "")
            if "text/html" in content_type.lower():
                raise RuntimeError(
                    f"Server returned HTML ('{content_type}') instead of archive from '{download_url}'. "
                    "Check link sharing permissions or update the URL in dataset.txt."
                )
            with open(tmp_path, "wb") as f:
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)

    if not os.path.isfile(tmp_path) or os.path.getsize(tmp_path) == 0:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise RuntimeError(f"Downloaded file is empty from: {download_url}")

    try:
        _check_not_html_file(tmp_path, download_url)
    except Exception:
        os.remove(tmp_path)
        raise
    os.replace(tmp_path, output_path)


def download_entry(entry, archive_dir, force=False):
    os.makedirs(archive_dir, exist_ok=True)
    output_path = os.path.join(archive_dir, entry["filename"])

    if os.path.isfile(output_path) and os.path.getsize(output_path) > 0 and not force:
        print(f"[SKIP] {entry['filename']} already exists at {output_path}")
        return output_path, False

    url = entry["url"]
    print(f"[DOWNLOAD] {entry['dataset_key']} -> {output_path}")
    if "drive.google.com" in url.lower():
        _download_google_drive(url, output_path)
    else:
        _download_direct(url, output_path)
    return output_path, True


def setup_parser():
    parser = argparse.ArgumentParser(
        description="Download continual learning dataset archives listed in dataset.txt."
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
        help="Dataset name(s) to download (e.g., cub imagenetr vtab). Defaults to all entries in manifest.",
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
        help="Directory to store downloaded archive files (default: ./data/archives).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Re-download archives even if they already exist.",
    )
    return parser


def main():
    args = setup_parser().parse_args()
    entries = parse_manifest(args.manifest)
    selected, notices = select_entries(entries, datasets=args.dataset, benchmark=args.benchmark)

    for notice in notices:
        print(f"[INFO] {notice}")

    if not selected:
        print("[INFO] No manifest archives to download.")
        return

    for entry in selected:
        download_entry(entry, archive_dir=args.archive_dir, force=args.force)

    print("[DONE] Download step complete.")


if __name__ == "__main__":
    main()
