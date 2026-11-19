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

INK_THRESHOLD = 150            # grey level; darker = ink (seeds)
INK_WEAK = 190                 # lighter ink kept when connected to a seed (hysteresis); dev sweep 175-205 equal
TEXT_HEIGHT_IN = 0.25          # a component this short that lies inside ANY OCR word is print; dev sweep 0.25-0.30 equal
PRINT_WORD_CONF = 0.5          # OCR words at least this confident count as print
PRINT_INSIDE_SHARE = 0.85      # a component with this share of its ink inside print words is print
MIN_COMPONENT_PX_IN2 = 0.0004  # ignore specks smaller than this (in square inches)
CLUSTER_GAP_IN = 0.20          # handwriting components closer than this belong to one cluster
ROW_EXTEND_IN = 1.0            # also add owned handwriting on the same row within this gap (tails past the printed name); dev sweep 1.0-1.5 equal
ROW_OVERLAP = 0.5              # 'same row' = vertical overlap of at least this share of the smaller height
MIN_SIGNATURE_INK_IN2 = 0.010  # a cluster with less ink than this is not a signature
MIN_BOX_AREA_IN2 = 0.25        # smallest dev signature box is 0.30 in2; stray marks are <= 0.21 in2

# search regions and expected signature boxes, inches, from dev percentiles (anchor_geometry.py)
LABEL_REGION = dict(x0=-0.6, x1=3.3, y0=-0.4, y1=2.0)    # relative to the label's top-left corner
LABEL_EXPECT = dict(x0=0.50, x1=2.21, y0=0.22, y1=1.14)  # median signature box, same origin
BLOCK_REGION = dict(x0=-1.1, x1=1.3, y0=-1.1, y1=1.0)    # relative to the printed-name line box
BLOCK_EXPECT = dict(x0=0.10, x1=-0.07, y0=-0.40, y1=0.34)
DIST_SCALE_IN = 1.0            # cluster score = ink * exp(-distance to expected centre / this)
MERGE_OVERLAP = 0.5            # boxes overlapping this much (share of the smaller) are one signature
SPLIT_SHARED = True            # split one ink component that reaches into 2+ signatories' expected boxes ...
SPLIT_MIN_SHARE = 0.20         # ... each holding at least this share of its ink (dev sweep: 0.1 adds FPs, 0.3 no split)
SPLIT_MIN_PIECE_H_IN = 0.40    # every piece must be at least this tall (signature-sized)
SPLIT_SAME_ROW_GAP_IN = 0.40   # blocks side by side on one row closer than this = one signatory
WRITTEN_MAX_WORDS = 4         # a signature written out legibly is read as a short OCR line ...
WRITTEN_INSIDE = 0.5           # ... lying at least this much inside a block's expected signature box


def binarise(gray):
    strong = gray < INK_THRESHOLD
    if INK_WEAK <= INK_THRESHOLD:
        return strong.astype(np.uint8)
    weak = (gray < INK_WEAK).astype(np.uint8)
    n, lab = cv2.connectedComponents(weak, connectivity=8)
    keep = np.zeros(n, bool)
    keep[np.unique(lab[strong])] = True
    keep[0] = False
    return keep[lab].astype(np.uint8)


def written_lines(rec, anchors, ppi):
    """OCR lines that are a signature written out legibly ('Benna RTabargn' over DONNA R. TABANGIN):
    a short line, not a printed-name or title line and not a label, lying mostly inside a block
    anchor's expected signature box and ending above the bottom of its name line. Their words are
    not print."""
    from anchors import page_lines, is_name_line, role_of
    exp = [geometry(a, ppi)[1] + [a['name_box']] for a in anchors if a['kind'] == 'block']
    if not exp:
        return set()
    labels = [a['box'] for a in anchors if a['kind'] == 'label']
    out = set()
    for ln in page_lines(rec):
        b = ln['box']
        if len(ln['words']) > WRITTEN_MAX_WORDS or is_name_line(ln) or role_of(ln['text'])[0] or b in labels:
            continue
        area = max(1, (b[2] - b[0]) * (b[3] - b[1]))
        for e in exp:
            iw = min(b[2], e[2]) - max(b[0], e[0]); ih = min(b[3], e[3]) - max(b[1], e[1])
            if iw > 0 and ih > 0 and iw * ih / area >= WRITTEN_INSIDE and b[3] <= e[4][3] and b != e[4]:
                out.update(id(w) for w in ln['words'])
                break
    return out


def print_mask(rec, shape, min_conf=None, skip=()):
    min_conf = PRINT_WORD_CONF if min_conf is None else min_conf
    m = np.zeros(shape, np.uint8)
    for ln in rec['lines']:
        for w in ln['words']:
            if w['conf'] >= min_conf and id(w) not in skip:
                x0, y0, x1, y1 = (int(round(v)) for v in w['box'])
                m[max(0, y0):y1 + 1, max(0, x0):x1 + 1] = 1
    return m


def split_shared(ink, comps_hand, groups, ppi):
    """A handwriting component that reaches into the expected boxes of two or more signatories
    (one signature's stroke runs into a neighbour's, e.g. stacked finance-certification blocks)
    is split: each of its pixels goes to the nearest expected box (distance 0 inside a box)."""
    exp = [g['expect'] for g in groups]
    out = []
    for c in comps_hand:
        hit = [i for i, e in enumerate(exp) if touches(c['box'], e)]
        if len(hit) < 2:
            out.append(c)
            continue
        x0, y0, x1, y1 = c['box']
        sub = ink[y0:y1, x0:x1].astype(bool) & (c['lab'][y0:y1, x0:x1] == c['id'])
        ys, xs = np.nonzero(sub)
        xs = xs + x0; ys = ys + y0
        share = [((xs >= exp[i][0]) & (xs <= exp[i][2]) & (ys >= exp[i][1]) & (ys <= exp[i][3])).mean() for i in hit]
        hit = [i for i, s in zip(hit, share) if s >= SPLIT_MIN_SHARE]
        # one signatory read as two blocks side by side (name line cut by ink): not two signatories
        hit = [i for k, i in enumerate(hit) if not any(same_signatory(groups[i], groups[j], ppi) for j in hit[:k])]
        if len(hit) < 2:
            out.append(c)
            continue
        d = np.stack([np.hypot(np.maximum(0, np.maximum(exp[i][0] - xs, xs - exp[i][2])),
                               np.maximum(0, np.maximum(exp[i][1] - ys, ys - exp[i][3])))
                      + 1e-3 * np.hypot(xs - centre(exp[i])[0], ys - centre(exp[i])[1]) for i in hit])
        own = np.argmin(d, axis=0)
        pieces = []
        for k in range(len(hit)):
            m = own == k
            if m.any():
                pieces.append(dict(box=[int(xs[m].min()), int(ys[m].min()), int(xs[m].max()) + 1, int(ys[m].max()) + 1],
                                   area=int(m.sum()), hand=True, split=True))
        # every piece must be signature-sized, else a stroke merely reaches a neighbour (e.g. an unsigned slot)
        if len(pieces) < 2 or any(p['box'][3] - p['box'][1] < SPLIT_MIN_PIECE_H_IN * ppi for p in pieces):
            out.append(c)
        else:
            out.extend(pieces)
    return out


def same_signatory(g1, g2, ppi):
    n1 = next((a['name_box'] for a in g1['anchors'] if a['kind'] == 'block'), None)
    n2 = next((a['name_box'] for a in g2['anchors'] if a['kind'] == 'block'), None)
    if n1 is None or n2 is None:
        return False
    v = min(n1[3], n2[3]) - max(n1[1], n2[1])
    hgap = max(n1[0], n2[0]) - min(n1[2], n2[2])
    return v >= 0.5 * min(n1[3] - n1[1], n2[3] - n2[1]) and hgap <= SPLIT_SAME_ROW_GAP_IN * ppi


def components(ink, pmask, ppi, anymask=None, keep_labels=False):
    n, lab, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    inside = np.bincount(lab[pmask == 1].ravel(), minlength=n)
    inside_any = np.bincount(lab[anymask == 1].ravel(), minlength=n) if anymask is not None else None
    min_px = MIN_COMPONENT_PX_IN2 * ppi * ppi
    comps = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < min_px:
            continue
        hand = inside[i] / area < PRINT_INSIDE_SHARE
        if hand and inside_any is not None and h <= TEXT_HEIGHT_IN * ppi and inside_any[i] / area >= PRINT_INSIDE_SHARE:
            hand = False   # text-sized and inside a word OCR read with low confidence: still print
        c = dict(box=[int(x), int(y), int(x + w), int(y + h)], area=int(area), hand=bool(hand))
        if keep_labels:
            c.update(id=i, lab=lab)
        comps.append(c)
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
    skip = written_lines(rec, anchors, ppi) if WRITTEN_MAX_WORDS else set()
    anymask = print_mask(rec, gray.shape, min_conf=0.0, skip=skip) if TEXT_HEIGHT_IN > 0 else None
    comps = [c for c in components(ink, print_mask(rec, gray.shape, skip=skip), ppi, anymask,
                                   keep_labels=SPLIT_SHARED) if c['hand']]
    groups = anchor_groups(anchors, ppi)
    if SPLIT_SHARED:
        comps = split_shared(ink, comps, groups, ppi)
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
        groups_ = clusters(mine, CLUSTER_GAP_IN * ppi)
        best = max(groups_, key=score)
        if ROW_EXTEND_IN > 0:
            # a long tail separated from the signature by the printed name: same row, short gap
            grown = True
            while grown:
                grown = False
                bb = union([c['box'] for c in best])
                for other in groups_:
                    if other is best or any(c in best for c in other):
                        continue
                    ob = union([c['box'] for c in other])
                    v = min(bb[3], ob[3]) - max(bb[1], ob[1])
                    hgap = max(bb[0], ob[0]) - min(bb[2], ob[2])
                    if v >= ROW_OVERLAP * min(bb[3] - bb[1], ob[3] - ob[1]) and hgap <= ROW_EXTEND_IN * ppi:
                        best = best + other
                        grown = True
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
