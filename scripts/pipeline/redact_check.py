"""Stage 4 check on DEV documents: ingest -> detect (cached OCR) -> redact -> verify.

  * stage 1 pixels must equal the annotation PNG pixels (same grid as the ground truth)
  * every redacted PDF must pass redact.verify (black boxes, rest unchanged, no leftover images,
    single revision)
  * before/after images of each page with a redaction are written for a visual check

  D:\\doctr-env\\Scripts\\python.exe redact_check.py [--docs N] [--out DIR]
"""
import argparse, csv, json, os, random, sys, time
from paths import thesis_path

import cv2
import numpy as np
import pymupdf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ingest import load_pages, to_gray
from detect import detect
from redact import redact_pdf, verify

CORPUS = thesis_path('baguio_corpus_sample')
ANN = thesis_path('baguio_annotation')
CACHE = thesis_path('pipeline_cache', 'doctr_dev')
ALWAYS = [('2021', '111-2021.pdf'), ('2026', '016-2026.pdf'), ('2024', '082-2025.pdf'), ('2020', '069-2020.pdf')]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--docs', type=int, default=12, help='random dev docs (seed "redact-check-2026") plus 4 fixed ones')
    ap.add_argument('--out', default=thesis_path('pipeline_checks', 'redact_check'))
    args = ap.parse_args()
    rows = [r for r in csv.DictReader(open(os.path.join(CORPUS, 'sample.csv'), encoding='utf-8')) if r['split'] == 'dev']
    pick = random.Random('redact-check-2026').sample(rows, args.docs)
    pick += [r for r in rows if (r['year'], r['file']) in ALWAYS and r not in pick]
    os.makedirs(args.out, exist_ok=True)
    report, ok_all = [], True
    for r in pick:
        src = os.path.join(CORPUS, r['corpus_path'])
        if r['split'] != 'dev':
            sys.exit('refusing a non-dev document')
        stem = f"{r['year']}_{r['file'][:-4]}"
        scans = load_pages(src)
        boxes, same = {}, True
        for i, s in enumerate(scans):
            gray = to_gray(s['pix'])
            png = cv2.imread(os.path.join(ANN, 'images', 'dev', r['year'], f'{stem}_p{i + 1:02d}.png'), cv2.IMREAD_GRAYSCALE)
            same &= png is not None and png.shape == gray.shape and int(np.abs(png.astype(int) - gray.astype(int)).max()) <= 1
            rec = json.load(open(os.path.join(CACHE, f'{stem}_p{i + 1:02d}.json'), encoding='utf-8'))
            dets = detect(gray, rec)
            if dets:
                boxes[i] = [d['box'] for d in dets]
        dst = os.path.join(args.out, f'{stem}_redacted.pdf')
        t = time.perf_counter()
        done = redact_pdf(src, dst, boxes)
        t_red = time.perf_counter() - t
        ok, problems, st = verify(src, dst, done)
        ok_all &= ok and same
        n = sum(len(v) for v in done.values())
        line = (f"{r['year']} {r['file']}: {len(scans)} p, {n} boxes, redact {t_red:.2f}s, "
                f"stage-1 pixels = annotation PNG: {same}, verify: {'PASS' if ok else 'FAIL'} "
                f"(max outside diff {st.get('outside_diff_max', 0):.2f}), "
                f"size {os.path.getsize(src) // 1024} -> {os.path.getsize(dst) // 1024} KB")
        print(line); report.append(line)
        for p in problems:
            print('    ', p); report.append('    ' + p)
        with pymupdf.open(src) as a, pymupdf.open(dst) as b:
            for i in done:
                for tag, d in (('before', a), ('after', b)):
                    d[i].get_pixmap(matrix=pymupdf.Matrix(0.5, 0.5)).save(
                        os.path.join(args.out, f'{stem}_p{i + 1:02d}_{tag}.png'))
    with open(os.path.join(args.out, 'redact_check.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(report) + f'\nALL PASS: {ok_all}\n')
    print('ALL PASS:', ok_all)


if __name__ == '__main__':
    main()
