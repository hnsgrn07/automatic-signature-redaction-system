"""Stage 3: find signature boxes near the anchors with OpenCV.

  1. binarise the page (dark ink only; the faint watermark falls out)
  2. connected components of ink; a component is PRINT when nearly all of its ink lies inside
     DocTR word boxes read with good confidence, otherwise HANDWRITING
  3. a search region per anchor (geometry measured on dev, see anchor_geometry.py)
  4. handwriting components touching the region are grouped into clusters; the cluster with the
     most ink is the signature, boxed at its full extent (tails outside the region are kept)
  5. too little ink -> no signature (unsigned slot); overlapping boxes from several anchors are merged

All distances in inches (page width 8.5 in). Settings are tuned on dev only.
"""
import cv2
import numpy as np

from anchors import find_anchors, PAGE_WIDTH_IN

INK_THRESHOLD = 150            # grey level; darker = ink
PRINT_WORD_CONF = 0.5          # OCR words at least this confident count as print
PRINT_INSIDE_SHARE = 0.85      # a component with this share of its ink inside print words is print
MIN_COMPONENT_PX_IN2 = 0.0004  # ignore specks smaller than this (in square inches)
CLUSTER_GAP_IN = 0.20          # handwriting components closer than this belong to one cluster
MIN_SIGNATURE_INK_IN2 = 0.010  # a cluster with less ink than this is not a signature
MIN_BOX_AREA_IN2 = 0.25        # smallest dev signature box is 0.30 in2; stray marks are <= 0.21 in2

# search regions and expected signature boxes, inches, from dev percentiles (anchor_geometry.py)
LABEL_REGION = dict(x0=-0.6, x1=3.3, y0=-0.4, y1=2.0)    # relative to the label's top-left corner
LABEL_EXPECT = dict(x0=0.50, x1=2.21, y0=0.22, y1=1.14)  # median signature box, same origin
BLOCK_REGION = dict(x0=-1.1, x1=1.3, y0=-1.1, y1=1.0)    # relative to the printed-name line box
BLOCK_EXPECT = dict(x0=0.10, x1=-0.07, y0=-0.40, y1=0.34)
DIST_SCALE_IN = 1.0            # cluster score = ink * exp(-distance to expected centre / this)
MERGE_OVERLAP = 0.5            # boxes overlapping this much (share of the smaller) are one signature


def binarise(gray):
    return (gray < INK_THRESHOLD).astype(np.uint8)


def print_mask(rec, shape):
    m = np.zeros(shape, np.uint8)
    for ln in rec['lines']:
        for w in ln['words']:
            if w['conf'] >= PRINT_WORD_CONF:
                x0, y0, x1, y1 = (int(round(v)) for v in w['box'])
                m[max(0, y0):y1 + 1, max(0, x0):x1 + 1] = 1
    return m


def components(ink, pmask, ppi):
    n, lab, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    inside = np.bincount(lab[pmask == 1].ravel(), minlength=n)
    min_px = MIN_COMPONENT_PX_IN2 * ppi * ppi
    comps = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < min_px:
            continue
        hand = inside[i] / area < PRINT_INSIDE_SHARE
        comps.append(dict(box=[x, y, x + w, y + h], area=int(area), hand=hand))
    return comps


def _rel(origin, r, ppi):
    """origin = (x0, y0, x1, y1) the offsets are taken from; r = offsets dict (x0/x1 from origin x0/x1 ...)."""
    return [origin[0] + r['x0'] * ppi, origin[1] + r['y0'] * ppi, origin[2] + r['x1'] * ppi, origin[3] + r['y1'] * ppi]


def geometry(anchor, ppi):
    """(search region, expected box) for one anchor, in pixels."""
    if anchor['kind'] == 'label':
        ax, ay = anchor['box'][0], anchor['box'][1]
        o = (ax, ay, ax, ay)
        return _rel(o, LABEL_REGION, ppi), _rel(o, LABEL_EXPECT, ppi)
    n = anchor['name_box']
    return _rel(n, BLOCK_REGION, ppi), _rel(n, BLOCK_EXPECT, ppi)


def centre(b):
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def inside(pt, b):
    return b[0] <= pt[0] <= b[2] and b[1] <= pt[1] <= b[3]


def touches(a, b, pad=0.0):
    return a[0] - pad <= b[2] and b[0] - pad <= a[2] and a[1] - pad <= b[3] and b[1] - pad <= a[3]


def anchor_groups(anchors, ppi):
    """Pair each block with the label whose expected box contains the block's name line:
    both describe the same signatory. The block's (tighter) geometry is used for the pair."""
    geo = [geometry(a, ppi) for a in anchors]
    labels = [i for i, a in enumerate(anchors) if a['kind'] == 'label']
    blocks = [i for i, a in enumerate(anchors) if a['kind'] == 'block']
    used, groups = set(), []
    for b in blocks:
        nc = centre(anchors[b]['name_box'])
        cand = [l for l in labels if l not in used and inside(nc, geo[l][0])]
        if cand:
            l = min(cand, key=lambda l: abs(centre(geo[l][1])[1] - nc[1]) + abs(centre(geo[l][1])[0] - nc[0]))
            used.add(l)
            reg = [min(geo[l][0][0], geo[b][0][0]), min(geo[l][0][1], geo[b][0][1]),
                   max(geo[l][0][2], geo[b][0][2]), max(geo[l][0][3], geo[b][0][3])]
            groups.append(dict(anchors=[anchors[l], anchors[b]], region=reg, expect=geo[b][1]))
        else:
            groups.append(dict(anchors=[anchors[b]], region=geo[b][0], expect=geo[b][1]))
    for l in labels:
        if l not in used:
            groups.append(dict(anchors=[anchors[l]], region=geo[l][0], expect=geo[l][1]))
    return groups


def clusters(comps, gap_px):
    """Group boxes whose rectangles come within gap_px of each other (union-find)."""
    parent = list(range(len(comps)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i in range(len(comps)):
        for j in range(i + 1, len(comps)):
            if touches(comps[i]['box'], comps[j]['box'], gap_px):
                parent[find(i)] = find(j)
    groups = {}
    for i, c in enumerate(comps):
        groups.setdefault(find(i), []).append(c)
    return list(groups.values())


def union(boxes):
    return [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]


def overlap_share(a, b):
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return iw * ih / max(1, small)


def detect(gray, rec, anchors=None):
    """gray: page image (H x W uint8). rec: DocTR page record. Returns a list of detections."""
    H, W = gray.shape
    ppi = W / PAGE_WIDTH_IN
    if anchors is None:
        anchors = find_anchors(rec)
    if not anchors:
        return []
    ink = binarise(gray)
    comps = [c for c in components(ink, print_mask(rec, gray.shape), ppi) if c['hand']]
    groups = anchor_groups(anchors, ppi)
    # each handwriting component goes to the anchor group whose expected box centre is nearest,
    # among the groups whose search region contains the component's centre
    owned = [[] for _ in groups]
    for c in comps:
        cc = centre(c['box'])
        cand = [i for i, g in enumerate(groups) if inside(cc, g['region'])]
        if cand:
            i = min(cand, key=lambda i: np.hypot(*(np.subtract(cc, centre(groups[i]['expect'])))))
            owned[i].append(c)
    found = []
    for g, mine in zip(groups, owned):
        if not mine:
            continue
        ec = centre(g['expect'])

        def score(cl):
            d = np.hypot(*(np.subtract(centre(union([c['box'] for c in cl])), ec))) / ppi
            return sum(c['area'] for c in cl) * np.exp(-d / DIST_SCALE_IN)
        best = max(clusters(mine, CLUSTER_GAP_IN * ppi), key=score)
        ink_px = sum(c['area'] for c in best)
        box = union([c['box'] for c in best])
        if ink_px < MIN_SIGNATURE_INK_IN2 * ppi * ppi or (box[2] - box[0]) * (box[3] - box[1]) < MIN_BOX_AREA_IN2 * ppi * ppi:
            continue
        found.append(dict(box=box, ink=ink_px, anchor=g['anchors'][-1]))
    found.sort(key=lambda d: -d['ink'])
    kept = []
    for d in found:
        if all(overlap_share(d['box'], k['box']) < MERGE_OVERLAP for k in kept):
            kept.append(d)
    return kept
