"""
Set up the v1.1 annotation round (protocol v1.1 addendum: staff and margin signatures): a mini-pilot
for all three annotators, then a split of every corpus page between Ben, Geron and Ferrer.
Labels: staff_signature (mark over a small printed staff name) and margin_signature (any other
handwritten signature mark, mostly at the bottom of body pages). Initials/paraphs are no longer a
separate category (team decision 2026-11-18).

Why: the adviser ruled (2026-11-18) that handwritten marks over the small printed staff names on the
signature page (no anchor label, no title) must also be redacted. The existing full-signature ground
truth (dev 869 / test 372 boxes, test sealed) is NOT changed; staff marks become a separate label
`staff_mark` with their own ground-truth files.

Method (report this in Chapter 3):
  - Frame: all 399 corpus documents, ALL pages (1,333). Every page is shown, so the page selection
    never depends on the sealed test ground truth; pages without a staff mark are submitted empty.
  - Mini-pilot: 2 dev documents per year with <= 3 pages, drawn with random.Random("staff-pilot-2026")
    (per year: filenames sorted, shuffled, first 2 that qualify) = 14 documents, annotated by all three.
    Agreement is checked with compare_annotations.py before the full annotation starts; pilot boxes
    become ground truth after median-box adjudication (merge_adjudicate.py).
  - Full: the other 385 documents. All 119 test documents go to Geron, who annotated the test set
    before and keeps it on one computer. The dev documents are split Ferrer 60% / Ben 40% (team
    decision 2026-11-18: Ben develops the pipeline), measured by an effort estimate per document:
    60 s for the signature page + 4 s per other page. Within each year dev documents are shuffled
    with random.Random("staff-split-2026"), placed largest first, each to the annotator furthest
    below their target for that year.
  - Output under <annotation_dir>/staff/: assignment.csv, split_report.txt, labelstudio_config_staff.xml
    and one package per group (staff_pilot, staff_<name>_dev, staff_ferrer_test) with images/, a
    document list, a task file and the config. Dev packages are zipped for transfer; the test
    package is never zipped and never goes on the shared drive.

Usage:
  python -P -E split_staff_marks.py <annotation_dir> [--dry-run]
"""
import argparse
import csv
import json
import os
import random
import shutil
import sys

ANNOTATORS = ["Ben", "Geron", "Ferrer"]
TEST_ANNOTATOR = "Geron"
DEV_SHARE = {"Ben": 0.4, "Geron": 0.0, "Ferrer": 0.6}
YEARS = [str(y) for y in range(2020, 2027)]
PILOT_SEED = "staff-pilot-2026"
PILOT_PER_YEAR = 2
PILOT_MAX_PAGES = 3
SEED = "staff-split-2026"
SIG_PAGE_S, OTHER_PAGE_S = 60, 4

CONFIG = """<View>
  <Image name="image" value="$image" zoom="true" zoomControl="true"/>
  <RectangleLabels name="label" toName="image">
    <Label value="staff_signature" background="#1d4ed8"/>
    <Label value="margin_signature" background="#d97706"/>
  </RectangleLabels>
  <Header value="Notes (only if unsure): unsure if signature, unsure which label, marks touching, cut at edge, faint"/>
  <TextArea name="note" toName="image" rows="2" editable="true" maxSubmissions="1"/>
</View>
"""


def effort(d):
    return SIG_PAGE_S + OTHER_PAGE_S * (d["pages"] - 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("annotation_dir", help="folder with pages.csv and images/")
    ap.add_argument("--dry-run", action="store_true", help="print the split, write nothing")
    args = ap.parse_args()
    ann = args.annotation_dir
    out = os.path.join(ann, "staff")
    if os.path.exists(out) and not args.dry_run:
        sys.exit(f"{out} already exists; the split is drawn once. Remove it only if nobody has started.")

    with open(os.path.join(ann, "pages.csv"), encoding="utf-8") as f:
        pages = list(csv.DictReader(f))
    docs = {}
    for p in pages:
        d = docs.setdefault((p["year"], p["file"]),
                            {"split": p["split"], "year": p["year"], "file": p["file"], "pages": 0})
        d["pages"] += 1
    alldocs = list(docs.values())

    # Mini-pilot
    prng = random.Random(PILOT_SEED)
    pilot = []
    for y in YEARS:
        yd = sorted((d for d in alldocs if d["split"] == "dev" and d["year"] == y), key=lambda d: d["file"])
        prng.shuffle(yd)
        pilot += [d for d in yd if d["pages"] <= PILOT_MAX_PAGES][:PILOT_PER_YEAR]
    for d in pilot:
        d["annotator"] = "all"
    pkeys = {(d["year"], d["file"]) for d in pilot}
    rest = [d for d in alldocs if (d["year"], d["file"]) not in pkeys]

    test = [d for d in rest if d["split"] == "test"]
    dev = [d for d in rest if d["split"] == "dev"]
    for d in test:
        d["annotator"] = TEST_ANNOTATOR
    total_e = sum(effort(d) for d in rest)
    dev_e = sum(effort(d) for d in dev)
    test_e = sum(effort(d) for d in test)
    dev_target = {a: DEV_SHARE[a] * dev_e for a in ANNOTATORS}
    dev_annotators = [a for a in ANNOTATORS if DEV_SHARE[a] > 0]

    rng = random.Random(SEED)
    for y in YEARS:
        yd = sorted((d for d in dev if d["year"] == y), key=lambda d: d["file"])
        rng.shuffle(yd)
        yd.sort(key=lambda d: -d["pages"])  # stable: seeded order breaks ties
        ye = sum(effort(d) for d in yd)
        target = {a: dev_target[a] * ye / dev_e for a in dev_annotators}
        got = {a: 0 for a in dev_annotators}
        for d in yd:
            a = min(dev_annotators, key=lambda a: ((got[a] + effort(d)) / target[a], a))
            d["annotator"] = a
            got[a] += effort(d)

    # Report
    lines = [f"Staff-mark annotation split: {len(alldocs)} documents, {len(pages)} pages (all pages shown).",
             f"Mini-pilot (seed '{PILOT_SEED}', {PILOT_PER_YEAR} dev docs/year with <= {PILOT_MAX_PAGES} pages), "
             f"all three annotators: {len(pilot)} docs, {sum(d['pages'] for d in pilot)} pages",
             f"Full split (seed '{SEED}', dev shares {DEV_SHARE}), effort = {SIG_PAGE_S} s + {OTHER_PAGE_S} s per extra page:", "",
             f"{'Annotator':<10}{'Docs':>6}{'Pages':>7}{'Dev docs':>10}{'Dev p':>7}{'Test docs':>11}"
             f"{'Test p':>8}{'Est. h':>8}"]
    for a in ANNOTATORS:
        m = [d for d in rest if d["annotator"] == a]
        md, mt = [d for d in m if d["split"] == "dev"], [d for d in m if d["split"] == "test"]
        est = (sum(effort(d) for d in m) + effort_pilot(pilot)) / 3600
        lines.append(f"{a:<10}{len(m):>6}{sum(d['pages'] for d in m):>7}{len(md):>10}"
                     f"{sum(d['pages'] for d in md):>7}{len(mt):>11}{sum(d['pages'] for d in mt):>8}{est:>8.1f}")
    lines.append("(Est. h includes the mini-pilot.)")
    lines.append("\nDev docs per year (" + " / ".join(ANNOTATORS) + "):")
    for y in YEARS:
        lines.append(f"  {y}: " + " / ".join(
            str(sum(1 for d in dev if d["year"] == y and d["annotator"] == a)) for a in ANNOTATORS))
    lines.append("\nPilot documents: " + ", ".join(f"{d['year']}/{d['file']}" for d in pilot))
    report = "\n".join(lines)
    print(report)
    if args.dry_run:
        return

    os.makedirs(out)
    with open(os.path.join(out, "labelstudio_config_staff.xml"), "w", encoding="utf-8") as f:
        f.write(CONFIG)
    fields = ["split", "year", "file", "pages", "annotator"]
    with open(os.path.join(out, "assignment.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore"); w.writeheader()
        w.writerows(sorted(pilot + rest, key=lambda d: (d["annotator"], d["split"], d["year"], d["file"])))

    groups = [("staff_pilot", pilot, True)]
    for a in ANNOTATORS:
        for split in ("dev", "test"):
            sel = [d for d in rest if d["annotator"] == a and d["split"] == split]
            if sel:
                groups.append((f"staff_{a.lower()}_{split}", sel, split == "dev"))
    for name, sel, zip_it in groups:
        root = os.path.join(out, name)
        keys = {(d["year"], d["file"]) for d in sel}
        tasks = []
        for p in pages:
            if (p["year"], p["file"]) not in keys:
                continue
            rel = f"images/{p['year']}/{os.path.basename(p['image'])}"
            os.makedirs(os.path.join(root, "images", p["year"]), exist_ok=True)
            shutil.copyfile(os.path.join(ann, p["image"]), os.path.join(root, rel))
            tasks.append({"data": {"image": f"/data/local-files/?d={rel}", "split": p["split"],
                                   "year": int(p["year"]), "file": p["file"], "page": int(p["page"])}})
        with open(os.path.join(root, f"labelstudio_tasks_{name}.json"), "w", encoding="utf-8") as f:
            json.dump(tasks, f, indent=1)
        with open(os.path.join(root, f"{name}_documents.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["split", "year", "file", "pages"], extrasaction="ignore")
            w.writeheader(); w.writerows(sorted(sel, key=lambda d: (d["year"], d["file"])))
        shutil.copyfile(os.path.join(out, "labelstudio_config_staff.xml"),
                        os.path.join(root, "labelstudio_config_staff.xml"))
        msg = f"{name}: {len(sel)} docs, {len(tasks)} pages -> {root}"
        if zip_it:
            msg += f"\n  zip: {shutil.make_archive(root, 'zip', root_dir=out, base_dir=name)}"
        report += "\n" + msg
        print(msg)
    with open(os.path.join(out, "split_report.txt"), "w", encoding="utf-8") as f:
        f.write(report + "\n")


def effort_pilot(pilot):
    return sum(effort(d) for d in pilot)


if __name__ == "__main__":
    main()
