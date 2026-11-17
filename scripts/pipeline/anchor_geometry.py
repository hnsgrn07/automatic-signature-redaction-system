"""Dev-only calibration: where do signatures sit relative to their anchors? (inches)

For every dev GT signature:
  label - the nearest label anchor whose top is above the signature's vertical centre and that
          overlaps it horizontally (or sits left of it within 1 in); offsets of the GT box from the
          label's top-left corner.
  block - the block whose printed-name line the GT box overlaps (or is nearest within 0.5 in);
          offsets of the GT box from the name line's box.
Prints percentiles; the search regions in detect.py are set from these.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from anchors import find_anchors, PAGE_WIDTH_IN
from eval_anchors import gap, GT_DEV, CACHE

gt = json.load(open(GT_DEV, encoding='utf-8'))
sig = {}
for a in gt['annotations']:
    x, y, w, h = a['bbox']; sig.setdefault(a['image_id'], []).append([x, y, x + w, y + h])
L, B = [], []
for im in gt['images']:
    if im['id'] not in sig:
        continue
    rec = json.load(open(os.path.join(CACHE, f"{im['year']}_{im['pdf'][:-4]}_p{im['page']:02d}.json"), encoding='utf-8'))
    ppi = rec['width'] / PAGE_WIDTH_IN
    an = find_anchors(rec)
    for s in sig[im['id']]:
        cy = (s[1] + s[3]) / 2
        labs = [a for a in an if a['kind'] == 'label' and a['box'][1] < cy
                and (min(a['box'][2], s[2]) > max(a['box'][0], s[0]) or 0 <= (s[0] - a['box'][2]) / ppi <= 1.0)]
        if labs:
            a = max(labs, key=lambda a: a['box'][1])
            ax, ay = a['box'][0], a['box'][1]
            L.append([(s[0] - ax) / ppi, (s[2] - ax) / ppi, (s[1] - ay) / ppi, (s[3] - ay) / ppi, (a['box'][2] - ax) / ppi])
        blks = [a for a in an if a['kind'] == 'block' and gap(a['name_box'], s) / ppi <= 0.5]
        if blks:
            a = min(blks, key=lambda a: gap(a['name_box'], s))
            n = a['name_box']
            B.append([(s[0] - n[0]) / ppi, (s[2] - n[2]) / ppi, (s[1] - n[1]) / ppi, (s[3] - n[3]) / ppi])
L, B = np.array(L), np.array(B)
q = [0, 1, 5, 50, 95, 99, 100]
print(f'label-anchored signatures: {len(L)}  (offsets from label top-left, inches)')
for i, n in enumerate(['sig x0 - label x0', 'sig x1 - label x0', 'sig y0 - label y0', 'sig y1 - label y0', 'label width']):
    print(f'  {n:20s}', ' '.join(f'p{p}={v:+.2f}' for p, v in zip(q, np.percentile(L[:, i], q))))
print(f'block-anchored signatures: {len(B)}  (offsets from printed-name line box, inches)')
for i, n in enumerate(['sig x0 - name x0', 'sig x1 - name x1', 'sig y0 - name y0', 'sig y1 - name y1']):
    print(f'  {n:20s}', ' '.join(f'p{p}={v:+.2f}' for p, v in zip(q, np.percentile(B[:, i], q))))
