"""
Baguio City Ordinance scraper for https://data.baguio.gov.ph/

Downloads all ordinance PDFs from the "City Ordinances of Baguio City Series of YYYY"
datasets on the Baguio Open Data Portal, using its public JSON API.

Usage:
    python scrape_ordinances.py                       # 2020-2026 into ./baguio_ordinances
    python scrape_ordinances.py --start 2015 --end 2019
    python scrape_ordinances.py --out D:/ordinances

Re-running is safe: files already downloaded with the correct size are skipped.
Requires only the Python standard library (3.8+).
"""
import argparse
import csv
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

API = "https://data.baguio.gov.ph/api"
UA = "Mozilla/5.0 (compatible; ordinance-archiver/1.0)"
DELAY = 0.6  # seconds between requests; server rate limit is 120 req/min


def request(url, retries=5):
    for attempt in range(retries):
        req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json, */*"})
        try:
            return urllib.request.urlopen(req, timeout=120)
        except urllib.error.HTTPError as e:
            if e.code == 429 or e.code >= 500:
                wait = int(e.headers.get("Retry-After") or 0) or (5 * (attempt + 1))
                print(f"  HTTP {e.code}, waiting {wait}s...")
                time.sleep(wait)
                continue
            raise
        except (urllib.error.URLError, TimeoutError) as e:
            print(f"  network error ({e}), retrying...")
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"Failed after {retries} attempts: {url}")


def get_json(url):
    with request(url) as r:
        return json.load(r)


def list_ordinance_datasets():
    """Return {year: dataset} for every 'City Ordinances ... Series of YYYY' dataset."""
    found, page = {}, 1
    while True:
        q = urllib.parse.urlencode({"search": "ordinance", "limit": 100, "page": page})
        data = get_json(f"{API}/v1/dataset/s?{q}")
        for ds in data.get("data", []):
            m = re.search(r"City Ordinances of Baguio City Series of (\d{4})", ds.get("name", ""))
            if m:
                found[int(m.group(1))] = ds
        meta = data.get("meta", {})
        if meta.get("current_page", page) >= meta.get("last_page", page):
            break
        page += 1
        time.sleep(DELAY)
    return found


def safe_name(name):
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip() or "file"


def download(url, dest, expected_size=None):
    tmp = dest + ".part"
    with request(url) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 16):
            f.write(chunk)
    size = os.path.getsize(tmp)
    if expected_size and size != expected_size:
        os.remove(tmp)
        raise RuntimeError(f"size mismatch ({size} != {expected_size})")
    os.replace(tmp, dest)
    return size


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", type=int, default=2020)
    ap.add_argument("--end", type=int, default=2026)
    ap.add_argument("--out", default="baguio_ordinances")
    args = ap.parse_args()

    print("Fetching dataset list...")
    datasets = list_ordinance_datasets()
    years = [y for y in sorted(datasets) if args.start <= y <= args.end]
    missing = [y for y in range(args.start, args.end + 1) if y not in datasets]
    if missing:
        print(f"No dataset found for: {missing}")

    os.makedirs(args.out, exist_ok=True)
    manifest_rows, failures = [], []
    for year in years:
        ds = datasets[year]
        resources = ds.get("resources") or []
        year_dir = os.path.join(args.out, str(year))
        os.makedirs(year_dir, exist_ok=True)
        print(f"\n[{year}] {ds['name']} - {len(resources)} files")
        seen = set()
        for i, res in enumerate(resources, 1):
            url = (res.get("links") or {}).get("download")
            name = safe_name(res.get("name") or f"{res['id']}.{res.get('ext', 'bin')}")
            if name.lower() in seen:  # avoid clobbering duplicate filenames
                stem, ext = os.path.splitext(name)
                name = f"{stem}_{res['id'][:6]}{ext}"
            seen.add(name.lower())
            dest = os.path.join(year_dir, name)
            size = res.get("size")
            row = {"year": year, "file": name, "size": size, "dataset_id": ds["id"],
                   "resource_id": res.get("id"), "url": url, "status": ""}
            if not url:
                row["status"] = "no-download-link"
                failures.append(row)
            elif os.path.exists(dest) and (not size or os.path.getsize(dest) == size):
                row["status"] = "skipped-existing"
            else:
                try:
                    download(url, dest, size)
                    row["status"] = "downloaded"
                    print(f"  ({i}/{len(resources)}) {name}")
                except Exception as e:
                    row["status"] = f"error: {e}"
                    failures.append(row)
                    print(f"  ({i}/{len(resources)}) FAILED {name}: {e}")
                time.sleep(DELAY)
            manifest_rows.append(row)

    manifest = os.path.join(args.out, "manifest.csv")
    with open(manifest, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()) if manifest_rows else ["year"])
        w.writeheader()
        w.writerows(manifest_rows)

    ok = sum(r["status"] in ("downloaded", "skipped-existing") for r in manifest_rows)
    print(f"\nDone: {ok}/{len(manifest_rows)} files present in {os.path.abspath(args.out)}")
    print(f"Manifest: {os.path.abspath(manifest)}")
    if failures:
        print(f"{len(failures)} failed - re-run the script to retry them.")
        sys.exit(1)


if __name__ == "__main__":
    main()
