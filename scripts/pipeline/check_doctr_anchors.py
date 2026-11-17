"""Iteration 2 pre-check: can pretrained DocTR read the signature anchor labels on dev pages?

No tuning: default pretrained models, default settings. Dev pages only.

Picks one signature page per year (seed "doctr-check-2026") from the dev ground truth,
plus a few hand-picked hard dev pages, runs DocTR OCR, and reports for each page:
  * which anchor labels were found (CERTIFIED CORRECT / Attested / APPROVED), with confidence
  * OCR time per page
and writes an overlay PNG per page (anchors in blue, ground-truth signature boxes in red).

Usage (D:\\doctr-env):
  set DOCTR_CACHE_DIR=D:\\doctr-cache
  D:\\doctr-env\\Scripts\\python.exe check_doctr_anchors.py [--out DIR] [--cpu]
"""
import argparse, csv, json, os, random, re, sys, time
from collections import defaultdict

ANN = r'D:\Thesis\baguio_annotation'
GT_DEV = r'D:\Thesis\ground_truth\final\ground_truth_dev.json'
HARD = [  # (year, file, page, why)
    (2020, '069-2020.pdf', 2, 'Attested left unsigned'),
    (2021, '111-2021.pdf', 5, 'finance certification page, no anchor labels, 5 signatures'),
    (2026, '016-2026.pdf', 3, '~234 DPI scan (1992 x 3047 px)'),
]
ANCHORS = {  # label -> word sequence (upper-cased, punctuation stripped)
    'CERTIFIED CORRECT': ['CERTIFIED', 'CORRECT'],
    'Attested': ['ATTESTED'],
    'APPROVED': ['APPROVED'],
}


def norm(w):
    return re.sub(r'[^A-Z]', '', w.upper())


def find_anchors(words):
    """words: list of (text, conf, (x0,y0,x1,y1)) in reading order. Exact match after normalising."""
    found = []
    for label, seq in ANCHORS.items():
        for i in range(len(words) - len(seq) + 1):
            if all(norm(words[i + k][0]) == seq[k] for k in range(len(seq))):
                ws = words[i:i + len(seq)]
                box = (min(w[2][0] for w in ws), min(w[2][1] for w in ws),
                       max(w[2][2] for w in ws), max(w[2][3] for w in ws))
                found.append((label, min(w[1] for w in ws), box, ' '.join(w[0] for w in ws)))
    return found


def near_misses(words):
    """Words that look like an anchor but did not match exactly (to see OCR errors)."""
    out = []
    for t, c, b in words:
        n = norm(t)
        for target in ('CERTIFIED', 'CORRECT', 'ATTESTED', 'APPROVED'):
            if n != target and len(n) >= 5 and sum(a == b2 for a, b2 in zip(n, target)) >= len(target) - 2:
                out.append(f'{t!r}~{target} ({c:.2f})')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=r'D:\Thesis\pipeline_checks\doctr_anchor_check')
    ap.add_argument('--cpu', action='store_true')
    args = ap.parse_args()

    import numpy as np, torch, pymupdf
    from doctr.models import ocr_predictor

    gt = json.load(open(GT_DEV, encoding='utf-8'))
    if any(im['split'] != 'dev' for im in gt['images']):
        sys.exit('refusing: non-dev image in GT file')
    img_by_key = {(im['year'], im['pdf'], im['page']): im for im in gt['images']}
    boxes = defaultdict(list)
    for a in gt['annotations']:
        boxes[a['image_id']].append(a['bbox'])

    # one page with >= 3 signatures per year, seeded
    rng = random.Random('doctr-check-2026')
    pick = []
    for year in range(2020, 2027):
        cands = sorted(k for k, im in img_by_key.items() if k[0] == year and len(boxes[im['id']]) >= 3)
        pick.append(rng.choice(cands) + ('random, seed doctr-check-2026',))
    pick += HARD

    dev = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    print(f'torch {torch.__version__}, device {dev}' + (f' ({torch.cuda.get_device_name(0)})' if dev == 'cuda' else ''))
    t0 = time.perf_counter()
    model = ocr_predictor(pretrained=True).to(dev)   # DocTR 1.1 defaults: fast_base detection + crnn_vgg16_bn recognition
    t_load = time.perf_counter() - t0
    print(f'model load {t_load:.1f}s')
    os.makedirs(args.out, exist_ok=True)

    rows, report = [], []
    for year, pdf, page, why in pick:
        im = img_by_key[(year, pdf, page)]
        path = os.path.join(ANN, im['file_name'])
        pix = pymupdf.Pixmap(path)
        if pix.n == 1 or pix.alpha:
            pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
        arr = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3].copy()
        W, H = pix.width, pix.height
        if dev == 'cuda': torch.cuda.synchronize()
        t = time.perf_counter()
        res = model([arr])
        if dev == 'cuda': torch.cuda.synchronize()
        dt = time.perf_counter() - t
        words = []
        for blk in res.pages[0].blocks:
            for line in blk.lines:
                for w in line.words:
                    (x0, y0), (x1, y1) = w.geometry
                    words.append((w.value, float(w.confidence), (x0 * W, y0 * H, x1 * W, y1 * H)))
        found = find_anchors(words)
        labels = sorted({f[0] for f in found})
        nsig = len(boxes[im['id']])
        rows.append(dict(year=year, file=pdf, page=page, why=why, size=f'{W}x{H}', ocr_s=round(dt, 2),
                         words=len(words), gt_signatures=nsig,
                         certified_correct='CERTIFIED CORRECT' in labels, attested='Attested' in labels,
                         approved='APPROVED' in labels,
                         anchors=' | '.join(f'{f[0]}: "{f[3]}" conf {f[1]:.2f} at y={f[2][1]:.0f}' for f in found),
                         near_misses='; '.join(near_misses(words))))
        # overlay
        doc = pymupdf.open(); pg = doc.new_page(width=W, height=H); pg.insert_image(pg.rect, filename=path)
        for b in boxes[im['id']]:
            pg.draw_rect(pymupdf.Rect(b[0], b[1], b[0] + b[2], b[1] + b[3]), color=(1, 0, 0), width=3)
        for label, conf, b, text in found:
            pg.draw_rect(pymupdf.Rect(*b), color=(0, 0.3, 1), width=3)
            pg.insert_text((b[0], b[1] - 5), f'{label} {conf:.2f}', fontsize=22, color=(0, 0.3, 1))
        pg.get_pixmap(matrix=pymupdf.Matrix(0.6, 0.6)).save(
            os.path.join(args.out, f'{year}_{pdf[:-4]}_p{page:02d}_anchors.png'))
        print(f'{year} {pdf} p{page}: {dt:.2f}s, {len(words)} words, anchors {labels or "none"}, GT sigs {nsig}')

    with open(os.path.join(args.out, 'anchor_check.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f'\nmodel load {t_load:.1f}s; OCR per page: ' +
          ', '.join(f"{r['ocr_s']}" for r in rows) + f' s; written to {args.out}')


if __name__ == '__main__':
    main()
