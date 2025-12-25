#!/usr/bin/env python3
import argparse
import concurrent.futures
import os
import sys
import threading
import urllib.error
import urllib.request

CHUNK_SIZE = 1024 * 1024


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Download DSM/DTM tiles referenced by the CSV/TXT manifests. "
            "Outputs into dataset/<split>/<dsm|dtm> with region-prefixed filenames."
        )
    )
    parser.add_argument(
        "--source-root",
        default="dataset",
        help="Root directory containing DSMs/DTMs manifests (default: dataset)",
    )
    parser.add_argument(
        "--output-root",
        default="dataset",
        help="Output root directory (default: dataset)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=8,
        help="Number of download threads (default: 8)",
    )
    parser.add_argument(
        "--log-file",
        default="download_failures.log",
        help="Log file for failed downloads (default: download_failures.log)",
    )
    return parser.parse_args()


def read_url_map(csv_path):
    url_map = {}
    with open(csv_path, "r", encoding="utf-8") as handle:
        for line in handle:
            url = line.strip()
            if not url:
                continue
            filename = url.rsplit("/", 1)[-1]
            url_map[filename] = url
    return url_map


def split_from_manifest(filename):
    lowered = filename.lower()
    if "train" in lowered:
        return "train"
    if "val" in lowered:
        return "val"
    if "test" in lowered:
        return "test"
    return None


def collect_tasks(source_root, output_root):
    tasks = []
    missing_urls = []
    for folder_name, dtype in ("DSMs", "dsm"), ("DTMs", "dtm"):
        base_dir = os.path.join(source_root, folder_name)
        if not os.path.isdir(base_dir):
            continue
        for region in sorted(os.listdir(base_dir)):
            region_dir = os.path.join(base_dir, region)
            if not os.path.isdir(region_dir):
                continue
            csv_files = [
                entry
                for entry in os.listdir(region_dir)
                if entry.lower().endswith(".csv")
            ]
            if not csv_files:
                continue
            csv_path = os.path.join(region_dir, csv_files[0])
            url_map = read_url_map(csv_path)
            manifest_files = [
                entry
                for entry in os.listdir(region_dir)
                if entry.lower().endswith(".txt")
            ]
            for manifest in manifest_files:
                split = split_from_manifest(manifest)
                if split is None:
                    continue
                manifest_path = os.path.join(region_dir, manifest)
                with open(manifest_path, "r", encoding="utf-8") as handle:
                    for line in handle:
                        line = line.strip()
                        if not line:
                            continue
                        tile_name = line.split(",", 1)[0]
                        url = url_map.get(tile_name)
                        region_prefix = region.replace(".", "")
                        output_name = f"{region_prefix}_{tile_name}"
                        output_dir = os.path.join(output_root, split, dtype)
                        output_path = os.path.join(output_dir, output_name)
                        if url is None:
                            missing_urls.append((region, tile_name))
                            continue
                        tasks.append(
                            {
                                "region": region,
                                "url": url,
                                "output_path": output_path,
                            }
                        )
    return tasks, missing_urls


def get_remote_size(url):
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request, timeout=30) as response:
        length = response.headers.get("Content-Length")
    if length is None:
        return None
    try:
        return int(length)
    except ValueError:
        return None


def download_task(task, failures, lock):
    url = task["url"]
    output_path = task["output_path"]
    region = task["region"]
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    try:
        remote_size = get_remote_size(url)
    except urllib.error.URLError:
        remote_size = None

    if os.path.exists(output_path):
        local_size = os.path.getsize(output_path)
        if remote_size is None and local_size > 0:
            return
        if remote_size is not None and local_size == remote_size:
            return

    temp_path = output_path + ".part"
    if os.path.exists(temp_path):
        os.remove(temp_path)

    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            with open(temp_path, "wb") as handle:
                while True:
                    chunk = response.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    handle.write(chunk)
        if remote_size is not None:
            downloaded_size = os.path.getsize(temp_path)
            if downloaded_size != remote_size:
                raise IOError(
                    f"size mismatch: expected {remote_size} got {downloaded_size}"
                )
        os.replace(temp_path, output_path)
    except (urllib.error.URLError, IOError, TimeoutError) as exc:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        with lock:
            failures.append((region, url, str(exc)))


def main():
    args = parse_args()
    tasks, missing_urls = collect_tasks(args.source_root, args.output_root)
    failures = []
    lock = threading.Lock()

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = [executor.submit(download_task, task, failures, lock) for task in tasks]
        for future in concurrent.futures.as_completed(futures):
            future.result()

    if missing_urls:
        with lock:
            for region, tile_name in missing_urls:
                failures.append((region, f"MISSING_URL:{tile_name}", "not found"))

    if failures:
        with open(args.log_file, "w", encoding="utf-8") as handle:
            for region, url, reason in failures:
                handle.write(f"{region}\t{url}\t{reason}\n")
        print(f"Finished with {len(failures)} failures. See {args.log_file}.")
    else:
        print("Download completed without failures.")


if __name__ == "__main__":
    sys.exit(main())
