"""Run pretrained DocTR once over every DEV page image and cache the words as JSON.

One JSON per page in --out (default D:\\Thesis\\pipeline_cache\\doctr_dev):
  {"year", "file", "page", "width", "height", "ocr_s",
   "lines": [{"block": b, "line": l, "words": [{"text", "conf", "box": [x0, y0, x1, y1]}]}]}
Boxes are in image pixels. Already-cached pages are skipped, so the run can be resumed.
Refuses test pages: anchor rules are developed on dev only.

  set DOCTR_CACHE_DIR=D:\\doctr-cache
  D:\\doctr-env\\Scripts\\python.exe ocr_cache.py
"""
import argparse, csv, json, os, sys, time
from paths import thesis_path

ANN = thesis_path('baguio_annotation')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=thesis_path('pipeline_cache', 'doctr_dev'))
    args = ap.parse_args()
    import numpy as np, torch, pymupdf
    from doctr.models import ocr_predictor

    pages = [r for r in csv.DictReader(open(os.path.join(ANN, 'pages.csv'), encoding='utf-8'))
             if r['split'] == 'dev']
    if len(pages) != 976:
        sys.exit(f'expected 976 dev pages, got {len(pages)}')
    os.makedirs(args.out, exist_ok=True)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    t0 = time.perf_counter()
    model = ocr_predictor(pretrained=True).to(dev)
    print(f'device {dev}, model load {time.perf_counter() - t0:.1f}s', flush=True)

    done = 0
    for i, r in enumerate(pages, 1):
        name = f"{r['year']}_{r['file'][:-4]}_p{int(r['page']):02d}.json"
        dst = os.path.join(args.out, name)
        if os.path.exists(dst):
            continue
        pix = pymupdf.Pixmap(os.path.join(ANN, r['image']))
        if pix.n == 1 or pix.alpha:
            pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
        W, H = pix.width, pix.height
        arr = np.frombuffer(pix.samples, np.uint8).reshape(H, W, pix.n)[:, :, :3].copy()
        if dev == 'cuda': torch.cuda.synchronize()
        t = time.perf_counter()
        res = model([arr])
        if dev == 'cuda': torch.cuda.synchronize()
        dt = time.perf_counter() - t
        lines = []
        for b, blk in enumerate(res.pages[0].blocks):
            for l, line in enumerate(blk.lines):
                lines.append(dict(block=b, line=l, words=[
                    dict(text=w.value, conf=round(float(w.confidence), 4),
                         box=[round(w.geometry[0][0] * W, 1), round(w.geometry[0][1] * H, 1),
                              round(w.geometry[1][0] * W, 1), round(w.geometry[1][1] * H, 1)])
                    for w in line.words]))
        rec = dict(year=int(r['year']), file=r['file'], page=int(r['page']), width=W, height=H,
                   ocr_s=round(dt, 3), lines=lines)
        with open(dst + '.tmp', 'w', encoding='utf-8') as f:
            json.dump(rec, f)
        os.replace(dst + '.tmp', dst)
        done += 1
        if done % 50 == 0:
            print(f'{i}/{len(pages)} pages', flush=True)
    print(f'finished: {len(pages)} dev pages cached ({done} new) in {args.out}', flush=True)


if __name__ == '__main__':
    main()
