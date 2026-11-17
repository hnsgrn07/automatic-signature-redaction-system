"""
Draw the annotation pilot (and the practice round) from the DEV set only.

Method (report this in Chapter 3):
  - Frame: the 280 dev documents in sample.csv. Test documents are never drawn.
  - Pilot: PILOT_PER_YEAR documents per enactment year (6 x 7 = 42, 10.5% of 399), drawn with
    random.Random("pilot-2026-r<round>"); years in ascending order, filenames sorted before sampling.
    A re-pilot (--round 2, 3, ...) uses a new seed and excludes every document drawn in earlier rounds.
  - Practice round: PRACTICE_DOCS documents from the Iteration 1 sample (iter1_sample.csv) that are
    not in any pilot round, seed "practice-2026". Practice boxes are discarded.
  - Output (under <annotation_dir>/pilot_r<round>/ and /practice/): a document list, copies of the page
    PNGs, Label Studio task files with paths relative to that folder, and a zip for the shared drive.

Usage:
  python -P -E draw_pilot.py <corpus_sample_dir> <annotation_dir> [--round N]
"""
import argparse
import csv
import json
import os
import random
import shutil
import sys

PILOT_PER_YEAR = 6
PRACTICE_DOCS = 5
YEARS = range(2020, 2027)


def write_package(name, docs, pages, ann_dir):
    """Copy the page PNGs of `docs` into <ann_dir>/<name>/ with a doc list, task file and zip."""
    root = os.path.join(ann_dir, name)
    if os.path.exists(root):
        sys.exit(f"{root} already exists; remove it first (or use a new --round).")
    keys = {(d["year"], d["file"]) for d in docs}
    sel = [p for p in pages if (p["year"], p["file"]) in keys]
    for p in sel:
        src = os.path.join(ann_dir, p["image"])
        rel = os.path.join("images", p["year"], os.path.basename(p["image"]))
        os.makedirs(os.path.join(root, os.path.dirname(rel)), exist_ok=True)
        shutil.copyfile(src, os.path.join(root, rel))  # copyfile: copies are writable, originals untouched
        p["pkg_image"] = rel.replace("\\", "/")

    with open(os.path.join(root, f"{name}_documents.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["year", "file", "pages"])
        w.writeheader()
        w.writerows(sorted(docs, key=lambda d: (d["year"], d["file"])))

    tasks = [{"data": {"image": f"/data/local-files/?d={p['pkg_image']}", "year": int(p["year"]),
                       "file": p["file"], "page": int(p["page"])}} for p in sel]
    with open(os.path.join(root, f"labelstudio_tasks_{name}.json"), "w", encoding="utf-8") as f:
        json.dump(tasks, f, indent=1)

    zip_path = shutil.make_archive(os.path.join(ann_dir, name), "zip", root_dir=ann_dir, base_dir=name)
    return root, zip_path, len(sel)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("corpus_dir", help="folder with sample.csv and iter1_sample.csv")
    ap.add_argument("annotation_dir", help="folder made by render_annotation_images.py (has pages.csv)")
    ap.add_argument("--round", type=int, default=1, help="pilot round (2+ = re-pilot on new documents)")
    args = ap.parse_args()

    with open(os.path.join(args.corpus_dir, "sample.csv"), encoding="utf-8") as f:
        dev = [r for r in csv.DictReader(f) if r["split"] == "dev"]
    with open(os.path.join(args.annotation_dir, "pages.csv"), encoding="utf-8") as f:
        pages = [p for p in csv.DictReader(f) if p["split"] == "dev"]
    pages_per_doc = {}
    for p in pages:
        pages_per_doc[(p["year"], p["file"])] = pages_per_doc.get((p["year"], p["file"]), 0) + 1

    # Documents used in earlier pilot rounds are excluded from this one.
    used = set()
    for r in range(1, args.round):
        prev = os.path.join(args.annotation_dir, f"pilot_r{r}", f"pilot_r{r}_documents.csv")
        if not os.path.exists(prev):
            sys.exit(f"Round {r} list not found ({prev}); run earlier rounds first.")
        with open(prev, encoding="utf-8") as f:
            used |= {(d["year"], d["file"]) for d in csv.DictReader(f)}

    rng = random.Random(f"pilot-2026-r{args.round}")
    pilot = []
    for y in YEARS:
        files = sorted(r["file"] for r in dev if int(r["year"]) == y and (str(y), r["file"]) not in used)
        if len(files) < PILOT_PER_YEAR:
            sys.exit(f"Year {y}: only {len(files)} unused dev documents left.")
        for name in sorted(rng.sample(files, PILOT_PER_YEAR)):
            pilot.append({"year": str(y), "file": name, "pages": pages_per_doc[(str(y), name)]})

    name = f"pilot_r{args.round}"
    root, zp, n_pages = write_package(name, pilot, pages, args.annotation_dir)
    print(f"{name}: {len(pilot)} documents, {n_pages} pages -> {root}\n  zip: {zp}")
    for y in YEARS:
        ds = [d for d in pilot if d["year"] == str(y)]
        print(f"  {y}: " + ", ".join(f"{d['file']} ({d['pages']}p)" for d in ds))

    # Practice round: only once, alongside round 1.
    if args.round == 1:
        pilot_keys = {(d["year"], d["file"]) for d in pilot}
        with open(os.path.join(args.corpus_dir, "iter1_sample.csv"), encoding="utf-8") as f:
            iter1 = [(r["year"], r["file"]) for r in csv.DictReader(f)]
        pool = sorted(k for k in iter1 if k not in pilot_keys)
        prac = sorted(random.Random("practice-2026").sample(pool, PRACTICE_DOCS))
        practice = [{"year": y, "file": f, "pages": pages_per_doc[(y, f)]} for y, f in prac]
        root, zp, n_pages = write_package("practice", practice, pages, args.annotation_dir)
        print(f"\npractice: {len(practice)} documents, {n_pages} pages -> {root}\n  zip: {zp}")
        for d in practice:
            print(f"  {d['year']}/{d['file']} ({d['pages']}p)")

    assert not {(d["year"], d["file"]) for d in pilot} & used, "pilot overlaps an earlier round"


if __name__ == "__main__":
    main()
