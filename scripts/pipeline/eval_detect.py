"""Score stage 3 detections on the DEV set: precision, recall, F1 at IoU >= 0.50.

Matching is one-to-one, greedy by highest IoU (as in the annotation agreement check).
Writes per-page results and an error list; optional overlays for the worst pages.

  D:\\doctr-env\\Scripts\\python.exe eval_detect.py [--overlays N] [--out DIR]
"""
import argparse, csv, json, os, sys, time
from collections import Counter, defaultdict

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detect import detect
from eval_anchors import GT_DEV, CACHE

ANN = r'D:\Thesis\baguio_annotation'
IOU_MATCH = 0.50


def iou(a, b):
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    i = iw * ih
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)


def match(gts, dets):
    pairs = sorted(((iou(g, d), gi, di) for gi, g in enumerate(gts) for di, d in enumerate(dets)), reverse=True)
    ug, ud, m = set(), set(), []
    for v, gi, di in pairs:
        if v < IOU_MATCH:
            break
        if gi in ug or di in ud:
            continue
        ug.add(gi); ud.add(di); m.append((gi, di, v))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=r'D:\Thesis\pipeline_checks\detect_eval')
    ap.add_argument('--overlays', type=int, default=0, help='draw this many pages with the most errors')
    args = ap.parse_args()
    gt = json.load(open(GT_DEV, encoding='utf-8'))
    if any(im['split'] != 'dev' for im in gt['images']):
        sys.exit('refusing: non-dev image in GT file')
    sig = defaultdict(list)
    for a in gt['annotations']:
        x, y, w, h = a['bbox']
        sig[a['image_id']].append([x, y, x + w, y + h])

    tp = fp = fn = 0
    ious, rows, errs = [], [], []
    t0 = time.perf_counter()
    for im in gt['images']:
        rec = json.load(open(os.path.join(CACHE, f"{im['year']}_{im['pdf'][:-4]}_p{im['page']:02d}.json"), encoding='utf-8'))
        gray = cv2.imread(os.path.join(ANN, im['file_name']), cv2.IMREAD_GRAYSCALE)
        dets = detect(gray, rec)
        g = sig[im['id']]
        d = [x['box'] for x in dets]
        m = match(g, d)
        tp += len(m); fp += len(d) - len(m); fn += len(g) - len(m)
        ious += [v for _, _, v in m]
        mg, md = {a for a, _, _ in m}, {b for _, b, _ in m}
        for gi, b in enumerate(g):
            if gi not in mg:
                best = max((iou(b, x) for x in d), default=0)
                errs.append(dict(year=im['year'], file=im['pdf'], page=im['page'], type='FN', box=[round(v) for v in b],
                                 best_iou=round(best, 2)))
        for di, b in enumerate(d):
            if di not in md:
                best = max((iou(b, x) for x in g), default=0)
                errs.append(dict(year=im['year'], file=im['pdf'], page=im['page'], type='FP', box=[round(v) for v in b],
                                 best_iou=round(best, 2), anchor=f"{dets[di]['anchor']['kind']}:{dets[di]['anchor']['role']}"))
        rows.append(dict(year=im['year'], file=im['pdf'], page=im['page'], gt=len(g), det=len(d), tp=len(m),
                         errors=len(g) + len(d) - 2 * len(m), image=im['file_name'],
                         gt_boxes=json.dumps([[round(v) for v in b] for b in g]),
                         det_boxes=json.dumps([[round(v) for v in b] for b in d])))
    dt = time.perf_counter() - t0
    P = tp / max(1, tp + fp); R = tp / max(1, tp + fn); F = 2 * P * R / max(1e-9, P + R)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, 'pages.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with open(os.path.join(args.out, 'errors.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['year', 'file', 'page', 'type', 'box', 'best_iou', 'anchor'])
        w.writeheader(); w.writerows(errs)
    print(f'DEV: {len(gt["images"])} pages, {sum(len(v) for v in sig.values())} signatures, stage-3 time {dt:.0f}s')
    print(f'  TP {tp}  FP {fp}  FN {fn}')
    print(f'  precision {P:.3f}  recall {R:.3f}  F1 {F:.3f}   (targets: P >= 0.85, R >= 0.90, F1 >= 0.87)')
    if ious:
        ious.sort()
        print(f'  IoU of matches: mean {sum(ious) / len(ious):.3f}, median {ious[len(ious) // 2]:.3f}, min {ious[0]:.3f}')
    near = Counter((e['type'], 'near miss (0.3-0.5)' if 0.3 <= e['best_iou'] < 0.5 else
                    ('overlaps (<0.3)' if e['best_iou'] > 0 else 'no overlap')) for e in errs)
    print('  errors by type:', dict(near))
    if args.overlays:
        import pymupdf
        worst = sorted(rows, key=lambda r: -r['errors'])[:args.overlays]
        for r in worst:
            if not r['errors']:
                break
            path = os.path.join(ANN, r['image'])
            pix = pymupdf.Pixmap(path)
            doc = pymupdf.open(); pg = doc.new_page(width=pix.width, height=pix.height); pg.insert_image(pg.rect, filename=path)
            for b in json.loads(r['gt_boxes']):
                pg.draw_rect(pymupdf.Rect(*b), color=(1, 0, 0), width=3)
            for b in json.loads(r['det_boxes']):
                pg.draw_rect(pymupdf.Rect(*b), color=(0, 0.6, 0), width=3)
            pg.get_pixmap(matrix=pymupdf.Matrix(0.5, 0.5)).save(
                os.path.join(args.out, f"err_{r['year']}_{r['file'][:-4]}_p{r['page']:02d}.png"))


if __name__ == '__main__':
    main()
