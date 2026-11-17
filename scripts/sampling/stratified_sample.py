"""
Stratified random sampling and 70/30 dev/test split of the Baguio City ordinance corpus.

Method (report this in Chapter 3):
  - Sampling frame: every PDF marked KEEP in screening_report.csv (726 documents, 2020-2026).
  - Strata: enactment year (7 strata). Equal allocation: N_PER_YEAR documents per year.
  - One random.Random(SEED) generator, years processed in ascending order.
    For each year: sort filenames, draw N_PER_YEAR without replacement (rng.sample),
    then shuffle the drawn files and assign the first DEV_PER_YEAR to dev, the rest to test.
  - Output: sample.csv, corpus/dev/<year>/*.pdf, corpus/test/<year>/*.pdf (read-only), SHA-256 hashes.

Usage:
  python stratified_sample.py <ordinances_dir> <output_dir>
"""
import csv
import hashlib
import os
import random
import shutil
import stat
import sys
from collections import Counter, defaultdict

SEED = 2026
N_PER_YEAR = 57          # 57 x 7 years = 399 documents
DEV_PER_YEAR = 40        # 40 dev / 17 test per year -> 280 dev / 119 test
YEARS = range(2020, 2027)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    src, out = sys.argv[1], sys.argv[2]
    report = os.path.join(src, "screening_report.csv")

    # Sampling frame from the screening report (year folder may carry a label, e.g. "2026 (as of ...)")
    frame, meta = defaultdict(list), {}
    with open(report, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["verdict"] != "KEEP":
                continue
            year = int(r["year"][:4])
            frame[year].append(r["file"])
            meta[(year, r["file"])] = r
    folder = {int(d[:4]): d for d in os.listdir(src) if os.path.isdir(os.path.join(src, d)) and d[:4].isdigit()}

    rng = random.Random(SEED)
    rows = []
    for year in YEARS:
        files = sorted(frame[year])
        if len(files) < N_PER_YEAR:
            sys.exit(f"Year {year} has only {len(files)} eligible documents; need {N_PER_YEAR}.")
        drawn = rng.sample(files, N_PER_YEAR)
        rng.shuffle(drawn)
        for i, name in enumerate(drawn):
            split = "dev" if i < DEV_PER_YEAR else "test"
            m = meta[(year, name)]
            rows.append({"year": year, "file": name, "split": split, "pages": int(m["pages"]),
                         "phys_dpi": m["phys_dpi_min"], "frame_size": len(files),
                         "source_path": os.path.join(src, folder[year], name)})

    if os.path.exists(os.path.join(out, "corpus")):
        sys.exit(f"{out}/corpus already exists; remove it first so the sample is not mixed with an old one.")

    for r in rows:
        dest_dir = os.path.join(out, "corpus", r["split"], str(r["year"]))
        os.makedirs(dest_dir, exist_ok=True)
        dest = os.path.join(dest_dir, r["file"])
        shutil.copy2(r["source_path"], dest)
        r["sha256"] = sha256(dest)
        if r["split"] == "test":
            os.chmod(dest, stat.S_IREAD)  # read-only: the sealed exam pile
        r["corpus_path"] = os.path.relpath(dest, out)

    rows.sort(key=lambda r: (r["split"], r["year"], r["file"]))
    fields = ["year", "file", "split", "pages", "phys_dpi", "frame_size", "sha256", "corpus_path"]
    with open(os.path.join(out, "sample.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    # Sanity checks
    by = Counter((r["year"], r["split"]) for r in rows)
    print(f"Seed {SEED}: {len(rows)} documents sampled\n")
    print("Year   Frame  Dev  Test  Total")
    for y in YEARS:
        print(f"{y}   {len(frame[y]):5d}  {by[(y, 'dev')]:3d}  {by[(y, 'test')]:4d}  {by[(y, 'dev')] + by[(y, 'test')]:5d}")
    for s in ("dev", "test"):
        sub = [r for r in rows if r["split"] == s]
        pages = [r["pages"] for r in sub]
        hi = [f"{r['year']}/{r['file']}" for r in sub if float(r["phys_dpi"]) > 150]
        print(f"\n{s}: {len(sub)} docs, {sum(pages)} pages, pages/doc mean {sum(pages) / len(pages):.2f}, "
              f"min {min(pages)}, max {max(pages)}; >150 DPI files: {hi or 'none'}")
    assert len(rows) == N_PER_YEAR * len(YEARS)
    assert len({(r['year'], r['file']) for r in rows}) == len(rows), "duplicate document in sample"


if __name__ == "__main__":
    main()
