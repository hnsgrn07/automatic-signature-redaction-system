"""Measure anchor detection on the DEV set against the dev ground truth.

A ground-truth signature is "reached" by an anchor when the gap between the two rectangles is at
most --reach inches (default 1.0 in). Reported per anchor family and method, plus:
  * unreached signatures (listed),
  * anchors on pages that have no signature (where stage 3 would search for nothing),
  * anchors that reach no signature (unsigned slots or false matches).

  D:\\doctr-env\\Scripts\\python.exe eval_anchors.py [--reach 1.0] [--out DIR]
"""
import argparse, csv, glob, json, os, sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from anchors import find_anchors, PAGE_WIDTH_IN

GT_DEV = r'D:\Thesis\ground_truth\final\ground_truth_dev.json'
CACHE = r'D:\Thesis\pipeline_cache\doctr_dev'


def gap(a, b):
    """a, b = [x0, y0, x1, y1]; Euclidean gap between rectangles, 0 if they touch or overlap."""
    dx = max(0, max(a[0], b[0]) - min(a[2], b[2]))
    dy = max(0, max(a[1], b[1]) - min(a[3], b[3]))
    return (dx * dx + dy * dy) ** 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reach', type=float, default=1.0)
    ap.add_argument('--out', default=r'D:\Thesis\pipeline_checks\anchor_eval')
    args = ap.parse_args()
    gt = json.load(open(GT_DEV, encoding='utf-8'))
    if any(im['split'] != 'dev' for im in gt['images']):
        sys.exit('refusing: non-dev image in GT file')
    sigs = defaultdict(list)
    for a in gt['annotations']:
        x, y, w, h = a['bbox']
        sigs[a['image_id']].append([x, y, x + w, y + h])

    rows, anchor_rows = [], []
    reached = Counter()
    fam_pages = Counter()
    missing = 0
    for im in gt['images']:
        path = os.path.join(CACHE, f"{im['year']}_{im['pdf'][:-4]}_p{im['page']:02d}.json")
        if not os.path.exists(path):
            missing += 1
            continue
        rec = json.load(open(path, encoding='utf-8'))
        ppi = rec['width'] / PAGE_WIDTH_IN
        anchors = find_anchors(rec)
        boxes = sigs[im['id']]
        for s in boxes:
            near = [a for a in anchors if gap(a['box'], s) / ppi <= args.reach]
            keys = {'label exact': any(a['kind'] == 'label' and a['method'] == 'exact' for a in near),
                    'label any': any(a['kind'] == 'label' for a in near),
                    'block': any(a['kind'] == 'block' for a in near)}
            keys['label or block'] = keys['label any'] or keys['block']
            for k, v in keys.items():
                reached[k] += v
            rows.append(dict(year=im['year'], file=im['pdf'], page=im['page'],
                             box=','.join(str(round(v)) for v in s), **{k: int(v) for k, v in keys.items()},
                             anchors_near=' | '.join(f"{a['kind']}:{a['role']}:{a['method']}" for a in near)))
        for a in anchors:
            hit = any(gap(a['box'], s) / ppi <= args.reach for s in boxes)
            fam_pages[(a['kind'], 'page has signatures' if boxes else 'page has NO signature')] += 1
            anchor_rows.append(dict(year=im['year'], file=im['pdf'], page=im['page'], kind=a['kind'], role=a['role'],
                                    method=a['method'], conf=round(a['conf'], 2), reaches_signature=int(hit),
                                    page_signatures=len(boxes), text=a['text'][:80],
                                    box=','.join(str(round(v)) for v in a['box'])))
    if missing:
        sys.exit(f'{missing} dev pages are not in the OCR cache yet')
    os.makedirs(args.out, exist_ok=True)
    for name, data in (('signatures.csv', rows), ('anchors.csv', anchor_rows)):
        with open(os.path.join(args.out, name), 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)

    n = len(rows)
    print(f'Dev signatures: {n}; reach = {args.reach} in')
    for k in ('label exact', 'label any', 'block', 'label or block'):
        print(f'  reached by {k:15s}: {reached[k]:4d}  ({reached[k] / n:.1%})')
    print('Anchors found:')
    for (kind, where), c in sorted(fam_pages.items()):
        print(f'  {kind:5s} on {where}: {c}')
    lone = Counter((r['kind'], r['role'], r['method']) for r in anchor_rows if not r['reaches_signature'])
    print('Anchors that reach no signature (by kind/role/method):')
    for k, c in lone.most_common():
        print(f'  {k}: {c}')
    print('Signatures reached by no anchor:')
    for r in rows:
        if not r['label or block']:
            print(f"  {r['year']} {r['file']} p{r['page']} box {r['box']}")


if __name__ == '__main__':
    main()
