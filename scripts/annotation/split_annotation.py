"""
Split the full annotation (every corpus document not in the pilot) between the annotators.

Method (report this in Chapter 3):
  - Frame: the 399 corpus documents minus the pilot documents (pilot_r*_documents.csv), whose
    adjudicated boxes are already ground truth. 280 dev - 42 pilot = 238 dev + 119 test = 357 documents.
  - Two annotators: Ben and Geron. All test documents go to Geron, because test pages are annotated
    in one separate Label Studio project on one computer (protocol v1.0, Section 8).
  - Dev documents are shared so that both annotators get about the same total number of PAGES
    (test + dev), and each annotator has dev documents from every year. Geron's dev page target is
    half the total pages minus the test pages; the target is spread over the years in proportion to
    each year's dev pages.
  - Within each year, documents are shuffled with random.Random("split-2026") (filenames sorted first),
    then placed largest first, each going to the annotator furthest below their target for that year.
  - Cross-review (protocol Section 9): 10% of each annotator's documents (rounded up, at least one per
    split they hold), drawn with random.Random("xreview-2026"), are reviewed by the other annotator.
  - Output under <annotation_dir>/full/: assignment.csv, crossreview.csv, split_report.txt and one
    Label Studio package per annotator and split (full_<name>_<split>/ with images/, a document list
    and a task file). Only dev packages are zipped; test images never go on the shared drive.

Usage:
  python -P -E split_annotation.py <annotation_dir> [--dry-run]
"""
import argparse
import csv
import glob
import json
import math
import os
import random
import shutil
import sys

ANNOTATORS = ["Ben", "Geron"]
TEST_ANNOTATOR = "Geron"
YEARS = [str(y) for y in range(2020, 2027)]
SEED = "split-2026"
XREVIEW_SEED = "xreview-2026"
XREVIEW_SHARE = 0.10


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("annotation_dir", help="folder with pages.csv, images/ and pilot_r*/")
    ap.add_argument("--dry-run", action="store_true", help="print the split, write nothing")
    args = ap.parse_args()
    ann = args.annotation_dir
    out = os.path.join(ann, "full")
    if os.path.exists(out) and not args.dry_run:
        sys.exit(f"{out} already exists; the split is drawn once. Remove it only if nobody has started.")

    with open(os.path.join(ann, "pages.csv"), encoding="utf-8") as f:
        pages = list(csv.DictReader(f))
    docs = {}
    for p in pages:
        d = docs.setdefault((p["year"], p["file"]), {"split": p["split"], "year": p["year"], "file": p["file"], "pages": 0})
        d["pages"] += 1

    pilot = set()
    for path in glob.glob(os.path.join(ann, "pilot_r*", "pilot_r*_documents.csv")):
        with open(path, encoding="utf-8") as f:
            pilot |= {(r["year"], r["file"]) for r in csv.DictReader(f)}
    if not pilot:
        sys.exit("No pilot document lists found.")
    rest = [d for k, d in docs.items() if k not in pilot]

    test = [d for d in rest if d["split"] == "test"]
    dev = [d for d in rest if d["split"] == "dev"]
    for d in test:
        d["annotator"] = TEST_ANNOTATOR
    total = sum(d["pages"] for d in rest)
    test_pages = sum(d["pages"] for d in test)
    dev_pages = sum(d["pages"] for d in dev)
    # Page target per annotator over the whole split, then their dev share.
    dev_target = {a: total / len(ANNOTATORS) - (test_pages if a == TEST_ANNOTATOR else 0) for a in ANNOTATORS}
    if min(dev_target.values()) <= 0:
        sys.exit("Test pages alone exceed half the work; this split rule does not apply.")

    rng = random.Random(SEED)
    for y in YEARS:
        yd = sorted((d for d in dev if d["year"] == y), key=lambda d: d["file"])
        rng.shuffle(yd)
        yd.sort(key=lambda d: -d["pages"])  # stable: seeded order breaks ties
        ypages = sum(d["pages"] for d in yd)
        target = {a: dev_target[a] * ypages / dev_pages for a in ANNOTATORS}
        got = {a: 0 for a in ANNOTATORS}
        for d in yd:
            a = min(ANNOTATORS, key=lambda a: ((got[a] + d["pages"]) / target[a], a))
            d["annotator"] = a; got[a] += d["pages"]
        for a in ANNOTATORS:  # every annotator covers every year
            if not any(d["annotator"] == a for d in yd):
                d = min((d for d in yd), key=lambda d: d["pages"])
                got[d["annotator"]] -= d["pages"]; d["annotator"] = a; got[a] += d["pages"]

    # Cross-review sample
    xrng = random.Random(XREVIEW_SEED)
    xreview = []
    for a in ANNOTATORS:
        other = [b for b in ANNOTATORS if b != a][0]
        for split in ("dev", "test"):
            mine = sorted((d for d in rest if d["annotator"] == a and d["split"] == split),
                          key=lambda d: (d["year"], d["file"]))
            if not mine:
                continue
            for d in sorted(xrng.sample(mine, max(1, math.ceil(XREVIEW_SHARE * len(mine)))),
                            key=lambda d: (d["year"], d["file"])):
                xreview.append({**{k: d[k] for k in ("split", "year", "file", "pages")},
                                "annotator": a, "reviewer": other})

    # Report
    lines = [f"Full annotation split (seed '{SEED}'): {len(rest)} documents, {total} pages "
             f"(dev {len(dev)} docs / {dev_pages} p, test {len(test)} docs / {test_pages} p); "
             f"pilot excluded: {len(pilot)} docs", ""]
    lines.append(f"{'Annotator':<10}{'Docs':>6}{'Pages':>7}{'Dev docs':>10}{'Dev p':>7}{'Test docs':>11}{'Test p':>8}")
    for a in ANNOTATORS:
        m = [d for d in rest if d["annotator"] == a]
        md, mt = [d for d in m if d["split"] == "dev"], [d for d in m if d["split"] == "test"]
        lines.append(f"{a:<10}{len(m):>6}{sum(d['pages'] for d in m):>7}{len(md):>10}"
                     f"{sum(d['pages'] for d in md):>7}{len(mt):>11}{sum(d['pages'] for d in mt):>8}")
    lines.append("\nDev pages per year (" + " / ".join(ANNOTATORS) + "):")
    for y in YEARS:
        lines.append(f"  {y}: " + " / ".join(
            f"{sum(d['pages'] for d in dev if d['year'] == y and d['annotator'] == a)}p "
            f"({sum(1 for d in dev if d['year'] == y and d['annotator'] == a)} docs)" for a in ANNOTATORS))
    lines.append(f"\nCross-review (seed '{XREVIEW_SEED}', {XREVIEW_SHARE:.0%} per annotator and split):")
    for a in ANNOTATORS:
        for split in ("dev", "test"):
            x = [r for r in xreview if r["annotator"] == a and r["split"] == split]
            if x:
                lines.append(f"  {x[0]['reviewer']} reviews {len(x)} of {a}'s {split} docs "
                             f"({sum(r['pages'] for r in x)} pages)")
    report = "\n".join(lines)
    print(report)
    if args.dry_run:
        return

    # Write lists and packages
    os.makedirs(out)
    fields = ["split", "year", "file", "pages", "annotator"]
    with open(os.path.join(out, "assignment.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore"); w.writeheader()
        w.writerows(sorted(rest, key=lambda d: (d["annotator"], d["split"], d["year"], d["file"])))
    with open(os.path.join(out, "crossreview.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields + ["reviewer"]); w.writeheader(); w.writerows(xreview)

    owner = {(d["year"], d["file"]): d["annotator"] for d in rest}
    for a in ANNOTATORS:
        for split in ("dev", "test"):
            sel = [p for p in pages if p["split"] == split and owner.get((p["year"], p["file"])) == a]
            if not sel:
                continue
            name = f"full_{a.lower()}_{split}"
            root = os.path.join(out, name)
            tasks = []
            for p in sel:
                rel = f"images/{p['year']}/{os.path.basename(p['image'])}"
                os.makedirs(os.path.join(root, "images", p["year"]), exist_ok=True)
                shutil.copyfile(os.path.join(ann, p["image"]), os.path.join(root, rel))
                tasks.append({"data": {"image": f"/data/local-files/?d={rel}", "split": split,
                                       "year": int(p["year"]), "file": p["file"], "page": int(p["page"])}})
            with open(os.path.join(root, f"labelstudio_tasks_{name}.json"), "w", encoding="utf-8") as f:
                json.dump(tasks, f, indent=1)
            mine = sorted((d for d in rest if d["annotator"] == a and d["split"] == split),
                          key=lambda d: (d["year"], d["file"]))
            with open(os.path.join(root, f"{name}_documents.csv"), "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["year", "file", "pages"], extrasaction="ignore")
                w.writeheader(); w.writerows(mine)
            msg = f"{name}: {len(mine)} docs, {len(tasks)} pages -> {root}"
            if split == "dev" and a != TEST_ANNOTATOR:  # only packages that travel get a zip
                msg += f"\n  zip: {shutil.make_archive(root, 'zip', root_dir=out, base_dir=name)}"
            report += "\n" + msg
            print(msg)
    with open(os.path.join(out, "split_report.txt"), "w", encoding="utf-8") as f:
        f.write(report + "\n")


if __name__ == "__main__":
    main()
