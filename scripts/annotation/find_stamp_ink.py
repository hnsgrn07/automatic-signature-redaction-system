import csv, sys, numpy as np, pymupdf as fitz
rows = [r for r in csv.DictReader(open('D:/Thesis/baguio_annotation/pages.csv')) if r['split'] == sys.argv[1]]
out = []
for r in rows:
    pix = fitz.Pixmap('D:/Thesis/baguio_annotation/' + r['image'])
    if pix.n < 3: continue
    a = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3].astype(int)
    R, G, B = a[..., 0], a[..., 1], a[..., 2]
    m = (B - G > 40) & (R - G > 15) & (B > 90)   # purple/violet stamp ink
    n = int(m.sum())
    if n > 1500: out.append((n, r['file'], r['page']))
for n, f, p in sorted(out, reverse=True): print(f, p, n)
print('colour pages scanned; stamp-ink pages:', len(out))
