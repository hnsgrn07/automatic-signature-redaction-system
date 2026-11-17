"""Score a redact_cli.py run against ground truth: precision, recall, F1 at IoU >= 0.50.

  python eval_cli.py RUN_DIR [--gt GT.json] [--compare-cache] [--out DIR]

RUN_DIR is a redact_cli.py output folder, run on a corpus folder whose sub-folders are the years
(e.g. corpus/dev), so every log is identified by (year, file). Pages are matched to the ground
truth by (year, file, page) - file names repeat across years.

Boxes are the DETECTED boxes from the logs (before redaction padding). Matching is the same
one-to-one greedy matching as eval_detect.py (functions imported from there).

Writes to --out (default RUN_DIR/eval): pages.csv and errors.csv in the same format as
eval_detect.py, plus eval_report.txt.

--compare-cache (dev only): also runs stage 3 on the cached OCR (pipeline_cache/doctr_dev) and the
annotation PNGs, i.e. exactly what eval_detect.py scores, and lists every page where the live-OCR
boxes differ from the cached-OCR boxes. Live OCR should give the same result as the cache; on a
different GPU small OCR differences are expected (see redact_cli.py commit notes).
"""
import argparse, csv, glob, json, os, sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_detect import iou, match, IOU_MATCH, ANN
from eval_anchors import GT_DEV, CACHE


def load_run(run_dir):
    """(year, file) -> log dict"""
    logs = {}
    for p in glob.glob(os.path.join(run_dir, 'logs', '**', '*.json'), recursive=True):
        lg = json.load(open(p, encoding='utf-8'))
        sub = lg.get('sub', os.path.basename(os.path.dirname(p)))
        year = os.path.basename(sub) if sub else ''
        if not year.isdigit():   # run on a single year folder: take the folder the input PDF was in
            year = os.path.basename(os.path.dirname(lg.get('input', '')))
        if not year.isdigit():
            sys.exit(f'{p}: cannot tell the year (run redact_cli.py on a folder whose sub-folders are the years)')
        logs[(int(year), lg.get('file') or os.path.basename(lg['input']))] = lg
    return logs


def scores(tp, fp, fn):
    P = tp / max(1, tp + fp); R = tp / max(1, tp + fn)
    return P, R, 2 * P * R / max(1e-9, P + R)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_dir')
    ap.add_argument('--gt', default=GT_DEV, help='COCO ground truth (default: dev)')
    ap.add_argument('--compare-cache', action='store_true', help='dev only: compare with cached-OCR detection')
    ap.add_argument('--run-docs-only', action='store_true',
                    help='score only the documents present in the run (for partial runs)')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    out = args.out or os.path.join(args.run_dir, 'eval')
    gt = json.load(open(args.gt, encoding='utf-8'))
    logs = load_run(args.run_dir)
    if args.run_docs_only:
        gt['images'] = [im for im in gt['images'] if (int(im['year']), im['pdf']) in logs]
        keep = {im['id'] for im in gt['images']}
        gt['annotations'] = [a for a in gt['annotations'] if a['image_id'] in keep]
    splits = sorted({im['split'] for im in gt['images']})
    if args.compare_cache and splits != ['dev']:
        sys.exit('--compare-cache is for the dev set only')
    sig = defaultdict(list)
    for a in gt['annotations']:
        x, y, w, h = a['bbox']
        sig[a['image_id']].append([x, y, x + w, y + h])

    if args.compare_cache:
        import cv2
        from detect import detect

    gt_docs = {(int(im['year']), im['pdf']) for im in gt['images']}
    missing_docs = sorted(gt_docs - set(logs))
    extra_docs = sorted(set(logs) - gt_docs)

    tp = fp = fn = 0
    ctp = cfp = cfn = 0
    ious, rows, errs, diffs = [], [], [], []
    for im in gt['images']:
        key = (int(im['year']), im['pdf'])
        lg = logs.get(key)
        g = sig[im['id']]
        pg = None
        if lg and 'pages' in lg:
            pg = next((p for p in lg['pages'] if p['page'] == im['page']), None)
        dets = pg['boxes'] if pg else []
        d = [b['box'] for b in dets]
        m = match(g, d)
        tp += len(m); fp += len(d) - len(m); fn += len(g) - len(m)
        ious += [v for _, _, v in m]
        mg, md = {a for a, _, _ in m}, {b for _, b, _ in m}
        for gi, b in enumerate(g):
            if gi not in mg:
                errs.append(dict(year=im['year'], file=im['pdf'], page=im['page'], type='FN', box=[round(v) for v in b],
                                 best_iou=round(max((iou(b, x) for x in d), default=0), 2),
                                 anchor='' if pg else 'page missing from run'))
        for di, b in enumerate(d):
            if di not in md:
                errs.append(dict(year=im['year'], file=im['pdf'], page=im['page'], type='FP', box=[round(v) for v in b],
                                 best_iou=round(max((iou(b, x) for x in g), default=0), 2),
                                 anchor=f"{dets[di]['anchor_kind']}:{dets[di]['anchor_role']}"))
        rows.append(dict(year=im['year'], file=im['pdf'], page=im['page'], gt=len(g), det=len(d), tp=len(m),
                         errors=len(g) + len(d) - 2 * len(m), image=im['file_name'],
                         gt_boxes=json.dumps([[round(v) for v in b] for b in g]),
                         det_boxes=json.dumps([[round(v) for v in b] for b in d])))
        if args.compare_cache:
            rec = json.load(open(os.path.join(CACHE, f"{im['year']}_{im['pdf'][:-4]}_p{im['page']:02d}.json"),
                                 encoding='utf-8'))
            gray = cv2.imread(os.path.join(ANN, im['file_name']), cv2.IMREAD_GRAYSCALE)
            c = [[round(v) for v in x['box']] for x in detect(gray, rec)]
            cm = match(g, c)
            ctp += len(cm); cfp += len(c) - len(cm); cfn += len(g) - len(cm)
            live = sorted([round(v) for v in b] for b in d)
            if live != sorted(c):
                pairs = match(c, d)
                diffs.append(dict(year=im['year'], file=im['pdf'], page=im['page'], cache_boxes=len(c), live_boxes=len(d),
                                  matched=len(pairs), min_iou=round(min((v for _, _, v in pairs), default=0), 3),
                                  tp_cache=len(cm), tp_live=len(m), gt=len(g)))

    if not rows:
        sys.exit('no ground-truth pages to score (wrong --gt, or --run-docs-only with no matching documents)')
    P, R, F = scores(tp, fp, fn)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, 'pages.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with open(os.path.join(out, 'errors.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['year', 'file', 'page', 'type', 'box', 'best_iou', 'anchor'])
        w.writeheader(); w.writerows(errs)
    status = Counter(lg.get('status') for k, lg in logs.items() if k in gt_docs)
    rep = [f'run: {os.path.abspath(args.run_dir)}',
           f'ground truth: {os.path.abspath(args.gt)} (split {",".join(splits)})',
           f'{len(gt["images"])} pages, {sum(len(v) for v in sig.values())} signatures, {len(gt_docs)} documents',
           f'documents in run: {len(logs)}; missing from run: {len(missing_docs)}; not in ground truth: {len(extra_docs)}',
           f'document status: {dict(status)}',
           f'LIVE OCR (CLI):   TP {tp}  FP {fp}  FN {fn}   precision {P:.3f}  recall {R:.3f}  F1 {F:.3f}'
           f'   (targets: P >= 0.85, R >= 0.90, F1 >= 0.87; IoU >= {IOU_MATCH})']
    if ious:
        ious.sort()
        rep.append(f'  IoU of matches: mean {sum(ious) / len(ious):.3f}, median {ious[len(ious) // 2]:.3f}, min {ious[0]:.3f}')
    near = Counter((e['type'], 'near miss (0.3-0.5)' if 0.3 <= e['best_iou'] < 0.5 else
                    ('overlaps (<0.3)' if e['best_iou'] > 0 else 'no overlap')) for e in errs)
    rep.append(f'  errors by type: {dict(near)}')
    for e in errs:
        rep.append(f"    {e['type']} {e['year']} {e['file']} p{e['page']} box {e['box']} best IoU {e['best_iou']} {e['anchor']}")
    if missing_docs:
        rep.append('missing documents: ' + ', '.join(f'{y}/{f}' for y, f in missing_docs))
    if args.compare_cache:
        cP, cR, cF = scores(ctp, cfp, cfn)
        rep.append(f'CACHED OCR:       TP {ctp}  FP {cfp}  FN {cfn}   precision {cP:.3f}  recall {cR:.3f}  F1 {cF:.3f}')
        rep.append(f'pages where live boxes != cached boxes: {len(diffs)} of {len(gt["images"])}')
        for x in diffs:
            rep.append(f"    {x['year']} {x['file']} p{x['page']}: cache {x['cache_boxes']} / live {x['live_boxes']} boxes, "
                       f"{x['matched']} paired (min IoU {x['min_iou']}), TP cache {x['tp_cache']} / live {x['tp_live']} of {x['gt']}")
        if diffs:
            with open(os.path.join(out, 'live_vs_cache.csv'), 'w', newline='', encoding='utf-8') as f:
                w = csv.DictWriter(f, fieldnames=list(diffs[0])); w.writeheader(); w.writerows(diffs)
    text = '\n'.join(rep)
    with open(os.path.join(out, 'eval_report.txt'), 'w', encoding='utf-8') as f:
        f.write(text + '\n')
    print(text)


if __name__ == '__main__':
    main()
