"""
Build the ground truth of the v1.1 mini-pilot (staff + margin signatures) from the three exports.

Method (report this in Chapter 3):
  - The raw exports (staff_pilot_<name>.json) were made with the old one-label config, so every box is
    labelled full_signature. They are not changed; cleaned copies are written to <exports>/clean/.
  - Cleaning: (1) boxes narrower or lower than 8 px are dropped (accidental clicks, protocol v1.0.1
    ruling); (2) boxes that match a v1.0.1 full signature of the same page (IoU >= 0.50 against the dev
    ground truth) are dropped, because full signatures are already ground truth and not part of v1.1.
  - The cleaned exports are merged with merge_adjudicate.py (median of >= 2 matched boxes; single-annotator
    boxes need a keep/drop decision), round name "staff_pilot".
  - Type: a box on a page that has at least one v1.0.1 full signature is a staff_signature, otherwise a
    margin_signature (checked by eye on all 30 pilot pages). This rule needs the ground truth, so it is
    used for the dev pilot only; the full annotation uses the two-label config.
  - Output: <out>/staff_pilot_ground_truth.csv / _coco.json with a 'type' column / category, clean_report.txt.

Usage:
  python -P -E build_v11_pilot_gt.py <exports_folder> <ground_truth_dev.csv> <pages_csv> <out_folder>
"""
import collections
import copy
import csv
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_annotations import iou  # noqa: E402

MIN_SIDE = 8
FULL_IOU = 0.50


def main():
    exports, gt_csv, pages_csv, out = sys.argv[1:5]
    gt = collections.defaultdict(list)
    with open(gt_csv, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["x"]:
                gt[(int(r["year"]), r["file"], int(r["page"]))].append([float(r[c]) for c in "xywh"])

    clean_dir = os.path.join(exports, "clean")
    os.makedirs(clean_dir, exist_ok=True)
    os.makedirs(out, exist_ok=True)
    lines = ["v1.1 pilot cleaning (raw exports unchanged)"]
    for name in sorted(os.listdir(exports)):
        if not (name.startswith("staff_pilot_") and name.endswith(".json")):
            continue
        tasks = json.load(open(os.path.join(exports, name), encoding="utf-8"))
        dropped = collections.Counter()
        for t in tasks:
            d = t["data"]
            k = (int(d["year"]), d["file"], int(d["page"]))
            for a in t.get("annotations", []):
                keep = []
                for r in a.get("result", []):
                    if r.get("type") == "rectanglelabels":
                        v, W, H = r["value"], r["original_width"], r["original_height"]
                        b = [v["x"] / 100 * W, v["y"] / 100 * H, v["width"] / 100 * W, v["height"] / 100 * H]
                        if b[2] < MIN_SIDE or b[3] < MIN_SIDE:
                            dropped["stray < 8 px"] += 1
                            continue
                        if any(iou(b, g) >= FULL_IOU for g in gt.get(k, [])):
                            dropped["full signature (already ground truth)"] += 1
                            continue
                    keep.append(r)
                a["result"] = keep
        with open(os.path.join(clean_dir, name), "w", encoding="utf-8") as f:
            json.dump(tasks, f, indent=1)
        lines.append(f"  {name}: dropped {dict(dropped) or 'nothing'}")
    with open(os.path.join(out, "clean_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))

    r = subprocess.run([sys.executable, "-P", "-E",
                        os.path.join(os.path.dirname(os.path.abspath(__file__)), "merge_adjudicate.py"),
                        clean_dir, pages_csv, out, "--round", "staff_pilot"])
    base = os.path.join(out, "staff_pilot_ground_truth")
    if r.returncode or not os.path.exists(base + ".csv"):
        sys.exit("Merge not finished (see adjudication_decisions.csv); re-run after the decisions.")

    # add the type
    def typ(year, file, page):
        return "staff_signature" if gt.get((int(year), file, int(page))) else "margin_signature"

    with open(base + ".csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["type"] = typ(row["year"], row["file"], row["page"])
    with open(base + ".csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    coco = json.load(open(base + "_coco.json", encoding="utf-8"))
    cats = {"staff_signature": 2, "margin_signature": 3}
    imgs = {i["id"]: i for i in coco["images"]}
    for a in coco["annotations"]:
        im = imgs[a["image_id"]]
        a["type"] = typ(im["year"], im["pdf"], im["page"])
        a["category_id"] = cats[a["type"]]
    coco["categories"] = [{"id": v, "name": k, "supercategory": "signature"} for k, v in cats.items()]
    coco["info"]["type_rule"] = "staff_signature if the page has >= 1 v1.0.1 full signature, else margin_signature"
    with open(base + "_coco.json", "w", encoding="utf-8") as f:
        json.dump(coco, f, indent=1)
    c = collections.Counter(row["type"] for row in rows)
    msg = f"Types: {dict(c)}"
    with open(os.path.join(out, "merge_report.txt"), "a", encoding="utf-8") as f:
        f.write(msg + "\n")
    print(msg)


if __name__ == "__main__":
    main()
