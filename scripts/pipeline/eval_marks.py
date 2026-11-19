"""Score stage 3 + 3b on the DEV set against both ground truths (v1.0.1 full + v1.1 staff/margin).

Per type (full / staff / margin) and for ALL signatures together: precision, recall, F1 at IoU >= 0.50,
one-to-one greedy matching (eval_detect.match). For the ALL score every detection is matched against the
union of the ground-truth boxes, whatever its predicted type. Per-type precision uses the predicted type.

Refuses any ground-truth file that holds a non-dev image. Cached OCR (dev only) and the annotation PNGs
are used, as in eval_detect.py.

  D:\\doctr-env\\Scripts\\python.exe eval_marks.py [--out DIR] [--workers N]
"""
import argparse, csv, json, os, sys, time
from collections import Counter, defaultdict
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cv2
from paths import thesis_path
from eval_detect import match, iou
from eval_anchors import GT_DEV, CACHE
from detect import detect
from marks import detect_marks
from anchors import find_anchors

GT_V11_DEV = thesis_path('ground_truth', 'v11', 'ground_truth_v11_dev.json')
ANN = thesis_path('baguio_annotation')
TYPES = ('full', 'staff', 'margin')


def load_gt():
    gt = defaultdict(list)
    images = None
    for path, fixed in ((GT_DEV, 'full'), (GT_V11_DEV, None)):
        g = json.load(open(path, encoding='utf-8'))
        if any(im['split'] != 'dev' for im in g['images']):
            sys.exit(f'refusing: non-dev image in {path}')
        ids = {im['id']: (im['year'], im['pdf'], im['page']) for im in g['images']}
        cat = {c['id']: c['name'].replace('_signature', '') for c in g['categories']}
        for a in g['annotations']:
            x, y, w, h = a['bbox']
            gt[ids[a['image_id']]].append(([x, y, x + w, y + h], fixed or cat[a['category_id']]))
        if images is None:
            images = g['images']
    return images, gt


def run(im):
    rec = json.load(open(os.path.join(CACHE, f"{im['year']}_{im['pdf'][:-4]}_p{im['page']:02d}.json"), encoding='utf-8'))
    gray = cv2.imread(os.path.join(ANN, im['file_name']), cv2.IMREAD_GRAYSCALE)
    anc = find_anchors(rec)
    full = detect(gray, rec, anchors=anc)
    marks, _ = detect_marks(gray, rec, full, anchors=anc)
    return [(d['box'], 'full') for d in full] + [(d['box'], d['type'].replace('_signature', '')) for d in marks]


def prf(tp, nd, ng):
    p = tp / max(1, nd); r = tp / max(1, ng)
    return p, r, 2 * p * r / max(1e-9, p + r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=thesis_path('pipeline_checks', 'marks_eval'))
    ap.add_argument('--workers', type=int, default=max(1, min(6, (os.cpu_count() or 2) - 1)))
    args = ap.parse_args()
    images, gt = load_gt()
    t0 = time.perf_counter()
    with Pool(args.workers) as pool:
        dets = pool.map(run, images, chunksize=8)
    dt = time.perf_counter() - t0

    tp_t, nd_t, ng_t = Counter(), Counter(), Counter()
    tp_all = nd_all = ng_all = 0
    rows, errs = [], []
    for im, d in zip(images, dets):
        key = (im['year'], im['pdf'], im['page'])
        g = gt[key]
        for t in TYPES:                                   # per type, predicted type vs that GT type
            gb = [b for b, k in g if k == t]; db = [b for b, k in d if k == t]
            tp_t[t] += len(match(gb, db)); nd_t[t] += len(db); ng_t[t] += len(gb)
        m = match([b for b, _ in g], [b for b, _ in d])   # all signatures, type-blind
        tp_all += len(m); nd_all += len(d); ng_all += len(g)
        mg, md = {a for a, _, _ in m}, {b for _, b, _ in m}
        for gi, (b, k) in enumerate(g):
            if gi not in mg:
                errs.append(dict(year=key[0], file=key[1], page=key[2], err='FN', type=k, box=[round(v) for v in b],
                                 best_iou=round(max((iou(b, x) for x, _ in d), default=0), 2)))
        for di, (b, k) in enumerate(d):
            if di not in md:
                errs.append(dict(year=key[0], file=key[1], page=key[2], err='FP', type=k, box=[round(v) for v in b],
                                 best_iou=round(max((iou(b, x) for x, _ in g), default=0), 2)))
        rows.append(dict(year=key[0], file=key[1], page=key[2], gt=len(g), det=len(d), tp=len(m),
                         gt_by_type=json.dumps(Counter(k for _, k in g)), det_by_type=json.dumps(Counter(k for _, k in d))))
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, 'pages.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with open(os.path.join(args.out, 'errors.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['year', 'file', 'page', 'err', 'type', 'box', 'best_iou'])
        w.writeheader(); w.writerows(errs)
    lines = [f'DEV: {len(images)} pages, stages 3 + 3b in {dt:.0f}s ({args.workers} workers); IoU >= 0.50',
             f'{"type":8} {"GT":>5} {"det":>5} {"TP":>5}   P      R      F1']
    for t in TYPES:
        p, r, f1 = prf(tp_t[t], nd_t[t], ng_t[t])
        lines.append(f'{t:8} {ng_t[t]:5} {nd_t[t]:5} {tp_t[t]:5}   {p:.3f}  {r:.3f}  {f1:.3f}')
    p, r, f1 = prf(tp_all, nd_all, ng_all)
    lines.append(f'{"ALL":8} {ng_all:5} {nd_all:5} {tp_all:5}   {p:.3f}  {r:.3f}  {f1:.3f}   (type-blind matching)')
    lines.append('targets: P >= 0.85, R >= 0.90, F1 >= 0.87')
    with open(os.path.join(args.out, 'eval_marks_report.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
