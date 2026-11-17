"""
Merge several annotators' Label Studio exports of the SAME pages into one ground truth (pilot adjudication).

Method (report this in Chapter 3):
  - Exports are loaded and boxes matched exactly as in compare_annotations.py (one-to-one, highest IoU
    first, match at IoU >= 0.50).
  - Per page, boxes are grouped into clusters: each annotator's boxes are matched to the clusters built so
    far (compared with the cluster's current median box); an unmatched box starts a new cluster.
  - A cluster drawn by at least 2 annotators is accepted automatically. Its ground-truth box is the
    coordinate-wise median of the members' corners (x1, y1, x2, y2); with 2 members this is their mean.
  - A cluster drawn by only 1 annotator is NOT accepted automatically. It is listed in
    adjudication_decisions.csv and the team decides keep / drop in the adjudication meeting, with a reason.
  - Ground truth is written only when every listed box has a decision.

Usage:
  python -P -E merge_adjudicate.py <exports_folder> <pages_csv> <out_folder> [--round pilot_r1]
  1st run: writes <out_folder>/adjudication_decisions.csv (fill 'decision' = keep/drop and 'reason').
  Re-run after filling it: writes <round>_ground_truth.csv, <round>_ground_truth_coco.json, merge_report.txt.
"""
import argparse
import csv
import datetime
import glob
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_annotations import load, match, iou  # noqa: E402  (same loading/matching as the agreement check)

DEC_FIELDS = ["year", "file", "page", "annotator", "x", "y", "w", "h", "decision", "reason"]


def corners(b):
    return [b[0], b[1], b[0] + b[2], b[1] + b[3]]


def median_box(boxes):
    c = [corners(b) for b in boxes]
    x1, y1, x2, y2 = (statistics.median(v[i] for v in c) for i in range(4))
    return [x1, y1, x2 - x1, y2 - y1]


def cluster_page(per_person):
    """per_person: {name: [boxes]} -> list of clusters {name: box}."""
    clusters = []
    for name, boxes in per_person.items():
        refs = [median_box(list(c.values())) for c in clusters]
        m, _, unmatched = match(refs, boxes)
        for ci, bj, _ in m:
            clusters[ci][name] = boxes[bj]
        for bj in unmatched:
            clusters.append({name: boxes[bj]})
    return clusters


def dec_key(year, file, page, who, b):
    return (int(year), file, int(page), who, *(round(float(v)) for v in b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exports"); ap.add_argument("pages_csv"); ap.add_argument("out")
    ap.add_argument("--round", default="pilot_r1")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    people, data = [], {}
    for f in sorted(glob.glob(os.path.join(args.exports, "*.json"))):
        name = os.path.splitext(os.path.basename(f))[0].rsplit("_", 1)[-1].capitalize()
        pages, _ = load(f)
        people.append(name); data[name] = pages
    if len(people) < 2:
        sys.exit("Need at least 2 exports.")
    keys = sorted(set().union(*[set(d) for d in data.values()]))
    for k in keys:
        for p in people:
            if data[p].get(k, {}).get("status") != "submitted":
                sys.exit(f"{k[0]}/{k[1]} p{k[2]}: not submitted by {p} - cannot merge.")

    # page images and sizes
    meta = {}
    with open(args.pages_csv, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            meta[(int(r["year"]), r["file"], int(r["page"]))] = r
    missing = [k for k in keys if k not in meta]
    if missing:
        sys.exit(f"{len(missing)} pages not found in {args.pages_csv}, e.g. {missing[0]}")

    # existing decisions
    dec_path = os.path.join(args.out, "adjudication_decisions.csv")
    decisions = {}
    if os.path.exists(dec_path):
        with open(dec_path, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                decisions[dec_key(r["year"], r["file"], r["page"], r["annotator"],
                                  [r["x"], r["y"], r["w"], r["h"]])] = r

    gt, dec_rows, pending, sizes = {}, [], 0, {}
    for k in keys:
        clusters = cluster_page({p: data[p][k]["boxes"] for p in people})
        out = []
        for c in clusters:
            sizes[len(c)] = sizes.get(len(c), 0) + 1
            if len(c) >= 2:
                mb = median_box(list(c.values()))
                out.append({"box": mb, "source": f"consensus {len(c)}/{len(people)}",
                            "members": "+".join(sorted(c)),
                            "min_iou_to_gt": min(iou(mb, b) for b in c.values())})
                continue
            (who, b), = c.items()
            prev = decisions.get(dec_key(*k, who, b), {})
            d = (prev.get("decision") or "").strip().lower()
            dec_rows.append({"year": k[0], "file": k[1], "page": k[2], "annotator": who,
                             "x": round(b[0]), "y": round(b[1]), "w": round(b[2]), "h": round(b[3]),
                             "decision": d, "reason": prev.get("reason", "")})
            if d == "keep":
                out.append({"box": b, "source": "adjudicated (single annotator, kept)",
                            "members": who, "min_iou_to_gt": 1.0})
            elif d != "drop":
                pending += 1
        gt[k] = out

    with open(dec_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=DEC_FIELDS); w.writeheader(); w.writerows(dec_rows)

    lines = [f"Merge of {args.round}: annotators {', '.join(people)}; {len(keys)} pages",
             "Clusters by number of annotators: " + ", ".join(f"{n} annotator(s): {c}" for n, c in sorted(sizes.items())),
             f"Single-annotator boxes needing a decision: {len(dec_rows)} (pending: {pending})"]
    notes = [(k, p, data[p][k]["notes"]) for k in keys for p in people if data[p][k]["notes"]]
    lines.append(f"Pages with annotator notes: {len(notes)}")
    lines += [f"  {k[0]}/{k[1]} p{k[2]} {p}: {n}" for k, p, n in notes]
    if pending:
        lines.append(f"\nNOT FINISHED - fill 'decision' (keep/drop) and 'reason' in {dec_path}, then re-run.")
        print("\n".join(lines)); return

    # write ground truth
    stamp = datetime.datetime.now().isoformat(timespec="seconds")
    rows, images, anns = [], [], []
    for img_id, k in enumerate(keys, 1):
        m = meta[k]
        images.append({"id": img_id, "file_name": m["image"], "width": int(m["width_px"]),
                       "height": int(m["height_px"]), "year": k[0], "pdf": k[1], "page": k[2],
                       "split": m["split"], "sha256": m["sha256"]})
        for g in gt[k]:
            x, y, w, h = (round(v, 2) for v in g["box"])
            anns.append({"id": len(anns) + 1, "image_id": img_id, "category_id": 1, "bbox": [x, y, w, h],
                         "area": round(w * h, 2), "iscrowd": 0, "source": g["source"], "annotators": g["members"]})
            rows.append({"split": m["split"], "year": k[0], "file": k[1], "page": k[2], "image": m["image"],
                         "x": x, "y": y, "w": w, "h": h, "source": g["source"], "annotators": g["members"],
                         "min_member_iou": round(g["min_iou_to_gt"], 3)})
    base = os.path.join(args.out, f"{args.round}_ground_truth")
    with open(base + ".csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    coco = {"info": {"description": f"Signature ground truth, {args.round} (adjudicated)",
                     "date_created": stamp, "annotators": people,
                     "method": "median of >=2 matched annotator boxes (IoU>=0.50); single boxes by team decision"},
            "images": images, "annotations": anns,
            "categories": [{"id": 1, "name": "signature", "supercategory": "handwriting"}]}
    with open(base + "_coco.json", "w", encoding="utf-8") as f:
        json.dump(coco, f, indent=1)
    kept = sum(1 for r in dec_rows if r["decision"] == "keep")
    lines += [f"Single boxes kept: {kept}, dropped: {len(dec_rows) - kept}",
              f"Ground truth: {len(anns)} signature boxes on {len(keys)} pages "
              f"({sum(1 for k in keys if not gt[k])} pages with no signature)",
              f"Lowest IoU of any member box to its consensus box: "
              f"{min(r['min_member_iou'] for r in rows):.3f}",
              f"Written: {base}.csv, {base}_coco.json  ({stamp})"]
    report = "\n".join(lines)
    with open(os.path.join(args.out, "merge_report.txt"), "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
