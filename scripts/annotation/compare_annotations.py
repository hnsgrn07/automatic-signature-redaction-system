"""
Compare signature boxes from several annotators (Label Studio JSON exports) on the same pages.

Method (report this in Chapter 3):
  - Each export file is named <round>_<annotator>.json (e.g. pilot_r1_ben.json).
  - Per page, the latest submitted (not skipped) annotation of each annotator is used. Boxes are
    converted from Label Studio percentages to image pixels.
  - For each pair of annotators, boxes on the same page are matched one-to-one, highest IoU first;
    a pair counts as matched when IoU >= MATCH_IOU (0.50).
  - Agreement per pair = mean IoU of matched boxes; the pilot passes when every pair has a mean
    IoU >= PASS_IOU (0.80). Unmatched boxes (drawn by one annotator only) are reported separately.
  - Near-identical boxes across annotators (IoU >= 0.995) are counted as a check on independence.

Usage:
  python -P -E compare_annotations.py <exports_folder>
Writes agreement_report.txt and agreement_pages.csv into that folder.
"""
import csv
import glob
import itertools
import json
import os
import statistics
import sys

MATCH_IOU = 0.50
PASS_IOU = 0.80
LOW_IOU = 0.80       # matched boxes below this are listed for discussion
SAME_IOU = 0.995     # practically identical boxes


def iou(a, b):
    ax1, ay1, ax2, ay2 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx1, by1, bx2, by2 = b[0], b[1], b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def load(path):
    """Return {page_key: {"boxes": [[x,y,w,h] px], "notes": str, "status": str}} and warnings."""
    tasks = json.load(open(path, encoding="utf-8"))
    pages, warns = {}, []
    for t in tasks:
        d = t["data"]
        key = (int(d["year"]), d["file"], int(d["page"]))
        anns = [a for a in t.get("annotations", []) if not a.get("was_cancelled")]
        skipped = any(a.get("was_cancelled") for a in t.get("annotations", []))
        if not anns:
            pages[key] = {"boxes": [], "notes": "", "status": "skipped" if skipped else "not submitted"}
            continue
        if len(anns) > 1:
            warns.append(f"{key[0]}/{key[1]} p{key[2]}: {len(anns)} annotations, using the latest")
        a = max(anns, key=lambda x: x.get("updated_at") or "")
        boxes, notes, seen = [], [], set()
        for r in a.get("result", []):
            # Label Studio sometimes repeats a region in the export; keep each region id once.
            rid = (r.get("id"), r.get("type"))
            if r.get("id") and rid in seen:
                warns.append(f"{key[0]}/{key[1]} p{key[2]}: duplicate region {r['id']} in export, counted once")
                continue
            seen.add(rid)
            if r.get("type") == "rectanglelabels":
                v, W, H = r["value"], r["original_width"], r["original_height"]
                if abs(v.get("rotation", 0)) > 0.01:
                    warns.append(f"{key[0]}/{key[1]} p{key[2]}: rotated box ({v['rotation']:.1f} deg) treated as upright")
                boxes.append([v["x"] / 100 * W, v["y"] / 100 * H, v["width"] / 100 * W, v["height"] / 100 * H])
            elif r.get("type") == "textarea":
                notes += [s for s in r["value"].get("text", []) if s.strip()]
        pages[key] = {"boxes": boxes, "notes": "; ".join(notes), "status": "submitted"}
    return pages, warns


def match(a, b):
    """Greedy one-to-one matching by highest IoU. Returns [(i, j, iou)], unmatched_a, unmatched_b."""
    pairs = sorted(((iou(x, y), i, j) for i, x in enumerate(a) for j, y in enumerate(b)), reverse=True)
    used_a, used_b, out = set(), set(), []
    for v, i, j in pairs:
        if v < MATCH_IOU:
            break
        if i not in used_a and j not in used_b:
            used_a.add(i); used_b.add(j); out.append((i, j, v))
    return out, [i for i in range(len(a)) if i not in used_a], [j for j in range(len(b)) if j not in used_b]


def fmt_box(b):
    return f"[x={b[0]:.0f}, y={b[1]:.0f}, w={b[2]:.0f}, h={b[3]:.0f}]"


def main():
    folder = sys.argv[1]
    files = sorted(glob.glob(os.path.join(folder, "*.json")))
    if len(files) < 2:
        sys.exit(f"Need at least 2 exports in {folder}, found {len(files)}.")
    people, data, lines = [], {}, []
    out = lines.append
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0].rsplit("_", 1)[-1].capitalize()
        pages, warns = load(f)
        people.append(name); data[name] = pages
        out(f"{name}: {os.path.basename(f)}, {len(pages)} pages, "
            f"{sum(len(p['boxes']) for p in pages.values())} boxes")
        for w in warns:
            out(f"  warning: {w}")

    all_keys = sorted(set().union(*[set(d) for d in data.values()]))
    out(f"\nPages: {len(all_keys)}")
    problems = []
    for k in all_keys:
        for p in people:
            st = data[p].get(k, {"status": "missing from export"})["status"]
            if st != "submitted":
                problems.append(f"  {k[0]}/{k[1]} p{k[2]}: {p} - {st}")
    out("Pages not submitted by everyone: " + ("none" if not problems else str(len(problems))))
    lines.extend(problems)

    # Pairwise agreement
    out(f"\nPAIRWISE AGREEMENT (match at IoU >= {MATCH_IOU:.2f}, pass at mean IoU >= {PASS_IOU:.2f})")
    out(f"{'Pair':<18}{'Matched':>8}{'Mean IoU':>10}{'Min IoU':>9}{'Only 1st':>10}{'Only 2nd':>10}{'Identical':>11}  Result")
    page_rows, verdicts = {}, []
    for a, b in itertools.combinations(people, 2):
        ious, only_a, only_b, same = [], 0, 0, 0
        for k in all_keys:
            pa, pb = data[a].get(k), data[b].get(k)
            if not pa or not pb or pa["status"] != "submitted" or pb["status"] != "submitted":
                continue
            m, ua, ub = match(pa["boxes"], pb["boxes"])
            ious += [v for _, _, v in m]
            only_a += len(ua); only_b += len(ub)
            same += sum(1 for _, _, v in m if v >= SAME_IOU)
            row = page_rows.setdefault(k, [])
            for i in ua:
                row.append(f"{a} only: box {fmt_box(pa['boxes'][i])} (no match from {b})")
            for j in ub:
                row.append(f"{b} only: box {fmt_box(pb['boxes'][j])} (no match from {a})")
            for i, j, v in m:
                if v < LOW_IOU:
                    row.append(f"{a} vs {b}: IoU {v:.2f}  {fmt_box(pa['boxes'][i])} vs {fmt_box(pb['boxes'][j])}")
        mean = statistics.mean(ious) if ious else 0.0
        ok = bool(ious) and mean >= PASS_IOU
        verdicts.append(ok)
        out(f"{a + '-' + b:<18}{len(ious):>8}{mean:>10.3f}{(min(ious) if ious else 0):>9.3f}"
            f"{only_a:>10}{only_b:>10}{same:>11}  {'PASS' if ok else 'BELOW 0.80'}")
    out("\nOverall: " + ("PASS - every pair has mean IoU >= 0.80" if all(verdicts)
                         else "NOT PASSED - at least one pair is below 0.80"))

    # Page-by-page list for the discussion meeting
    out("\nPAGES TO DISCUSS (box counts per annotator, then each disagreement)")
    csv_rows = []
    for k in all_keys:
        counts = {p: len(data[p].get(k, {"boxes": []})["boxes"]) for p in people}
        notes = {p: data[p][k]["notes"] for p in people if k in data[p] and data[p][k]["notes"]}
        issues = page_rows.get(k, [])
        flag = bool(issues) or len(set(counts.values())) > 1 or bool(notes)
        csv_rows.append({"year": k[0], "file": k[1], "page": k[2],
                         **{f"boxes_{p.lower()}": counts[p] for p in people},
                         "issues": len(issues), "notes": " | ".join(f"{p}: {n}" for p, n in notes.items())})
        if not flag:
            continue
        out(f"\n{k[0]}/{k[1]} page {k[2]}  boxes: " + ", ".join(f"{p} {c}" for p, c in counts.items()))
        for s in issues:
            out(f"  - {s}")
        for p, n in notes.items():
            out(f"  - note from {p}: {n}")

    with open(os.path.join(folder, "agreement_pages.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        w.writeheader(); w.writerows(csv_rows)
    report = "\n".join(lines)
    with open(os.path.join(folder, "agreement_report.txt"), "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
