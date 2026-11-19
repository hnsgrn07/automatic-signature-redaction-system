"""Stage 3b: staff and margin signatures (round v1.1), found after the full signatures (detect.py).

  staff_signature   initials / short signature on or over a small printed staff name (no label, no title),
                    lower-left of the signature page
  margin_signature  any other handwritten mark, nearly always in the bottom strip of a page

Steps
  1. ink as in detect.py (hysteresis binarisation); long straight lines (table rules, underlines)
     removed, because marks often touch them
  2. the full-signature boxes from detect.py (padded) are blanked, so a mark is never a full signature
  3. print vs handwriting: a component is print when it is letter-sized AND lies inside an OCR word that
     looks like real text (2+ characters read with good confidence, or 3+ characters read with any
     confidence - small italic staff names are read badly when a mark crosses them). Marks that DocTR
     reads as a 1-character "word" ('#', 'R') stay handwriting
  4. handwriting components are clustered (small gap)
  5. a cluster is a staff_signature when it lies in the zone of a staff-name anchor, a margin_signature
     when it lies in the bottom strip of the page and is mark-sized; everything else is dropped

Staff-name anchor (3rd anchor family): an all-caps OCR line of 8-60 letters, in the left 40 % of the page
and the lower 60 %, no taller than 0.26 in, not a label, not a signatory's printed name or title, not
ending in ':', and not above the topmost full-signature anchor (staff names only follow the signatories).

All distances in inches (page width 8.5 in). Settings were chosen on dev only (v1.1 dev ground truth);
see eval_marks.py. The sealed test set was not used.
"""
import re

import cv2
import numpy as np

import anchors as A
import detect as D

# candidates
LINE_LEN_IN = 0.6          # straight ink runs at least this long are table rules / underlines (removed)
STUB_LEN_IN = 0.15         # shorter straight runs joining removed rules AT BOTH ENDS (cell walls) also removed
WORD_CONF = 0.5            # an OCR word read at least this confidently ...
WORD_MIN_ALNUM = 2         # ... with at least this many letters/digits is text
WORD_ANYCONF_ALNUM = 3     # a word with this many letters/digits is text whatever its confidence ...
WORD_ANYCONF_CAPS = 0.8    # ... but inside a staff-name zone only if it is in capitals (names are printed in
                           # capitals; a mark read as 'sid' over JANICE D. EGUID is not text)
LETTER_H_IN = 0.22         # a print component is at most this tall (dev sweep 0.18-0.30 equal)
PRINT_INSIDE = 0.85        # share of a component's ink inside text words to call it print
GAP_IN = 0.06              # cluster gap (dev sweep 0.04-0.08: F1 0.913-0.901)
# size-aware clustering: two BIG pieces (each a mark on its own) are one mark only if their boxes overlap;
# small pieces (pen lifts, dots) join the nearest cluster within GAP_IN
BIG_BOX_IN2 = 0.06
BIG_H_IN = 0.20
BIG_OVERLAP = 0.25         # share of the smaller box
MIN_INK_IN2 = 0.0015       # a cluster needs at least this much ink
MIN_BOX_IN2 = 0.008        # ... and a box at least this large
FULL_PAD_IN = 0.05         # full-signature boxes are blanked with this padding
# staff-name anchors
STAFF_MIN_LETTERS, STAFF_MAX_LETTERS = 8, 60   # 60: names side by side on one row are read as one line
STAFF_MIN_CAPS = 0.8
STAFF_MAX_X0 = 0.40        # share of page width
STAFF_MIN_Y0 = 0.40        # share of page height
STAFF_MAX_WORDS = 10
STAFF_MAX_H_IN = 0.26
STAFF_FRAG_MIN_LETTERS = 4     # a shorter all-caps fragment counts when stacked under a staff name ...
STAFF_FRAG_GAP_IN = 0.9        # ... at most this far below it ...
STAFF_FRAG_X_IN = 0.3          # ... and starting within this of its left edge (OCR: 'GUID' for JANICE D. EGUID)
STAFF_ROW_GAP_IN = 1.5         # names side by side: an all-caps line on a staff name's row, this close to its right
# typing
STAFF_UP_IN, STAFF_DOWN_IN = 0.6, 0.3      # staff zone around a staff-name line (cluster centre inside)
STAFF_LEFT_IN, STAFF_RIGHT_IN = 0.5, 0.3
STAFF_MIN_AREA_IN2, STAFF_MIN_H_IN = 0.02, 0.10
MARGIN_Y = 0.90            # bottom strip: cluster centre below this share of the page height (sweep .88-.92)
MARGIN_MIN_AREA_IN2 = 0.02
MARGIN_MIN_H_IN, MARGIN_MIN_W_IN = 0.15, 0.08
MARGIN_LINE_W_IN, MARGIN_LINE_ASPECT = 0.15, 5   # thin and tall (a column of boxed list numbers) = a line, not a
                                                 # mark; no dev margin mark is <= 0.15 in wide and >= 5x taller
# margin marks above the bottom strip
REF_ZONE_X = 0.30          # left zone below the lowest signatory block (mark beside typed reference initials) ...
REF_ZONE_DOWN_IN = 2.0     # ... at most this far below the block
CORNER_X, CORNER_Y = 0.85, 0.85
# touching marks on stacked staff names (a large mark's loop reaching the mark on the next name)
STAFF_SPLIT = True


def remove_lines(ink, ppi):
    def straight(n):
        h = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (n, 1)))
        v = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, n)))
        return ((h | v) > 0).astype(np.uint8)
    rules = straight(max(3, int(LINE_LEN_IN * ppi)))
    if STUB_LEN_IN:
        stubs = straight(max(3, int(STUB_LEN_IN * ppi))) & (1 - rules)
        n, lab, st, _ = cv2.connectedComponentsWithStats(stubs, connectivity=8)
        near = cv2.dilate(rules, np.ones((5, 5), np.uint8)) > 0
        hit = np.zeros(n, bool)
        for i in range(1, n):
            x, y, w, h, _ = st[i]
            if h >= w:     # vertical stub: rule at its top and at its bottom
                ends = near[y, x:x + w].any() and near[y + h - 1, x:x + w].any()
            else:          # horizontal stub: rule at its left and at its right end
                ends = near[y:y + h, x].any() and near[y:y + h, x + w - 1].any()
            hit[i] = ends
        rules = rules | hit[lab].astype(np.uint8)
    return ink & (1 - rules)


def text_mask(rec, shape, zones=()):
    m = np.zeros(shape, np.uint8)
    for ln in rec['lines']:
        for w in ln['words']:
            an = re.sub(r'[^A-Za-z0-9]', '', w['text'])
            lt = re.sub(r'[^A-Za-z]', '', an)
            caps = sum(ch.isupper() for ch in lt) / len(lt) if lt else 1.0
            wc = ((w['box'][0] + w['box'][2]) / 2, (w['box'][1] + w['box'][3]) / 2)
            in_zone = any(D.inside(wc, z) for z in zones)
            if (w['conf'] >= WORD_CONF and len(an) >= WORD_MIN_ALNUM) or                     (len(an) >= WORD_ANYCONF_ALNUM and (caps >= WORD_ANYCONF_CAPS or not in_zone)):
                x0, y0, x1, y1 = (int(round(v)) for v in w['box'])
                m[max(0, y0):y1 + 1, max(0, x0):x1 + 1] = 1
    return m


def size_clusters(comps, ppi):
    if not BIG_BOX_IN2:
        return D.clusters(comps, GAP_IN * ppi)
    def big(c):
        b = c['box']
        return (b[2] - b[0]) * (b[3] - b[1]) >= BIG_BOX_IN2 * ppi * ppi and b[3] - b[1] >= BIG_H_IN * ppi
    parent = list(range(len(comps)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    isbig = [big(c) for c in comps]
    gap = GAP_IN * ppi
    order = sorted(range(len(comps)), key=lambda i: comps[i]['box'][0])
    for a in range(len(order)):
        i = order[a]
        for b in range(a + 1, len(order)):
            j = order[b]
            if comps[j]['box'][0] > comps[i]['box'][2] + gap:
                break
            if not D.touches(comps[i]['box'], comps[j]['box'], gap):
                continue
            if isbig[i] and isbig[j] and D.overlap_share(comps[i]['box'], comps[j]['box']) < BIG_OVERLAP:
                continue
            parent[find(i)] = find(j)
    # two big pieces must not end up in one cluster through small pieces in between
    groups = {}
    for i, c in enumerate(comps):
        groups.setdefault(find(i), []).append(i)
    out = []
    for g in groups.values():
        bigs = [i for i in g if isbig[i]]
        if len(bigs) <= 1:
            out.append([comps[i] for i in g]); continue
        # big pieces joined only through small pieces: regroup the big ones by touching boxes,
        # then attach each small piece to the nearest group
        merged = D.clusters([comps[i] for i in bigs], 0)
        boxes = [D.union([c['box'] for c in cl]) for cl in merged]
        for i in g:
            if isbig[i]:
                continue
            cc = D.centre(comps[i]['box'])
            k = min(range(len(boxes)), key=lambda k: max(0, boxes[k][0] - cc[0], cc[0] - boxes[k][2]) +
                    max(0, boxes[k][1] - cc[1], cc[1] - boxes[k][3]))
            merged[k].append(comps[i])
        out.extend(merged)
    return out


def staff_zone(a, ppi):
    b = a['box']
    return [b[0] - STAFF_LEFT_IN * ppi, b[1] - STAFF_UP_IN * ppi, b[2] + STAFF_RIGHT_IN * ppi, b[3] + STAFF_DOWN_IN * ppi]


def candidates(gray, rec, full_boxes, want_mask=False, zones=()):
    """Handwriting clusters outside the full signatures: list of dict(box, ink)
    (and, with want_mask, the handwriting-ink mask of the page)."""
    H, W = gray.shape
    ppi = W / A.PAGE_WIDTH_IN
    ink = remove_lines(D.binarise(gray), ppi)
    pad = FULL_PAD_IN * ppi
    for f in full_boxes:
        ink[max(0, int(f[1] - pad)):int(f[3] + pad) + 1, max(0, int(f[0] - pad)):int(f[2] + pad) + 1] = 0
    tm = text_mask(rec, gray.shape, zones)
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    inside = np.bincount(lab[tm == 1].ravel(), minlength=n)
    comps, keep = [], np.zeros(n, bool)
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if a < 0.2 * MIN_INK_IN2 * ppi * ppi:
            continue
        if inside[i] / a >= PRINT_INSIDE and h <= LETTER_H_IN * ppi:
            continue                                   # print
        comps.append(dict(box=[int(x), int(y), int(x + w), int(y + h)], area=int(a)))
        keep[i] = True
    out = []
    for cl in size_clusters(comps, ppi):
        b = D.union([c['box'] for c in cl])
        a = sum(c['area'] for c in cl)
        if a >= MIN_INK_IN2 * ppi * ppi and (b[2] - b[0]) * (b[3] - b[1]) >= MIN_BOX_IN2 * ppi * ppi:
            out.append(dict(box=b, ink=a))
    if want_mask:
        return out, keep[lab].astype(np.uint8)
    return out


def split_stacked(d, hand, sa):
    """Split a cluster that covers two or more stacked staff-name rows (a large mark touching the mark on
    the next name). Seeds: the cluster's ink on each name row. Every ink pixel goes to the seed that
    reaches it first along the ink (geodesic growth, all seeds at the same speed); pixels not connected
    to any seed go to the nearest name row. Returns the pieces, or [d] if there is nothing to split."""
    x0, y0, x1, y1 = d['box']
    rows = [a['box'] for a in sa if a['box'][0] < x1 and x0 < a['box'][2] and a['box'][1] < y1 and y0 < a['box'][3]]
    rows = sorted(rows, key=lambda b: b[1])
    rows = [r for k, r in enumerate(rows) if all(r[1] >= q[3] for q in rows[:k])]      # stacked, not overlapping
    if len(rows) < 2:
        return [d]
    m = hand[y0:y1, x0:x1].copy()
    lab = np.zeros(m.shape, np.int32)
    for k, r in enumerate(rows, 1):
        ra, rb = max(0, int(r[1]) - y0), min(m.shape[0], int(r[3]) - y0)
        band = np.zeros_like(m); band[ra:rb] = m[ra:rb]
        lab[(band > 0) & (lab == 0)] = k
    if len(np.unique(lab[lab > 0])) < 2:
        return [d]
    kern = np.ones((3, 3), np.uint8)
    while True:
        free = (m > 0) & (lab == 0)
        if not free.any():
            break
        grown = False
        new = lab.copy()
        for k in range(1, len(rows) + 1):
            reach = cv2.dilate((lab == k).astype(np.uint8), kern) > 0
            take = reach & free & (new == 0)
            if take.any():
                new[take] = k; grown = True
        lab = new
        if not grown:
            break
    ys, xs = np.nonzero((m > 0) & (lab == 0))
    if len(ys):
        cy = [((r[1] + r[3]) / 2) - y0 for r in rows]
        lab[ys, xs] = 1 + np.argmin(np.abs(ys[:, None] - np.array(cy)[None, :]), axis=1)
    out = []
    for k in range(1, len(rows) + 1):
        ys, xs = np.nonzero(lab == k)
        if len(ys):
            out.append(dict(box=[int(x0 + xs.min()), int(y0 + ys.min()), int(x0 + xs.max()) + 1, int(y0 + ys.max()) + 1],
                            ink=int(len(ys)), split=True))
    return out


def staff_anchors(rec, full_anchors):
    if not full_anchors:
        return []
    W, H = rec['width'], rec['height']
    ppi = W / A.PAGE_WIDTH_IN
    top = min(a['box'][1] for a in full_anchors)
    names = [a['name_box'] for a in full_anchors if a['kind'] == 'block']
    labels = [a['box'] for a in full_anchors if a['kind'] == 'label']
    out = []
    for ln in A.page_lines(rec):
        b, lt = ln['box'], A.letters(ln['text'])
        if not STAFF_MIN_LETTERS <= len(lt) <= STAFF_MAX_LETTERS or len(ln['words']) > STAFF_MAX_WORDS:
            continue
        if sum(ch.isupper() for ch in lt) / len(lt) < STAFF_MIN_CAPS:
            continue
        if b[0] > STAFF_MAX_X0 * W or b[1] < STAFF_MIN_Y0 * H or b[3] - b[1] > STAFF_MAX_H_IN * ppi:
            continue
        if b[1] < top or ln['text'].rstrip().endswith(':') or b in labels or A.role_of(ln['text'])[0]:
            continue
        if any(D.overlap_share(b, n) > 0.5 for n in names):
            continue
        out.append(dict(kind='staff', role='staff', box=b, text=ln['text']))
    if STAFF_FRAG_MIN_LETTERS and out:
        for ln in A.page_lines(rec):
            b, lt = ln['box'], A.letters(ln['text'])
            if not STAFF_FRAG_MIN_LETTERS <= len(lt) < STAFF_MIN_LETTERS or len(ln['words']) > 2:
                continue
            if sum(ch.isupper() for ch in lt) / len(lt) < STAFF_MIN_CAPS or b[3] - b[1] > STAFF_MAX_H_IN * ppi:
                continue
            if ln['text'].rstrip().endswith(':') or b in labels or A.role_of(ln['text'])[0]:
                continue
            below = any(o['box'][3] <= b[1] <= o['box'][3] + STAFF_FRAG_GAP_IN * ppi and
                        o['box'][0] - STAFF_FRAG_X_IN * ppi <= b[0] <= o['box'][2] for o in out)
            if below and not any(D.overlap_share(b, o['box']) > 0.3 for o in out):
                out.append(dict(kind='staff', role='staff', box=b, text=ln['text'], fragment=True))
    if STAFF_ROW_GAP_IN and out:
        grown = True
        while grown:                     # names side by side on one row, left to right
            grown = False
            for ln in A.page_lines(rec):
                b, lt = ln['box'], A.letters(ln['text'])
                if len(lt) < STAFF_FRAG_MIN_LETTERS or len(lt) > STAFF_MAX_LETTERS or len(ln['words']) > STAFF_MAX_WORDS:
                    continue
                if sum(ch.isupper() for ch in lt) / len(lt) < STAFF_MIN_CAPS or b[3] - b[1] > STAFF_MAX_H_IN * ppi:
                    continue
                if ln['text'].rstrip().endswith(':') or b in labels or A.role_of(ln['text'])[0]:
                    continue
                if any(D.overlap_share(b, o['box']) > 0.3 for o in out) or any(D.overlap_share(b, n) > 0.5 for n in names):
                    continue
                for o in out:
                    ob = o['box']
                    v = min(ob[3], b[3]) - max(ob[1], b[1])
                    if v >= 0.5 * min(ob[3] - ob[1], b[3] - b[1]) and 0 <= b[0] - ob[2] <= STAFF_ROW_GAP_IN * ppi:
                        out.append(dict(kind='staff', role='staff', box=b, text=ln['text'], same_row=True))
                        grown = True
                        break
    return out


def detect_marks(gray, rec, full_dets, anchors=None):
    """gray: page image; rec: DocTR page record; full_dets: detect.detect() output.
    Returns (marks, staff_anchors); each mark is dict(box, ink, type='staff_signature'|'margin_signature')."""
    H, W = gray.shape
    ppi = W / A.PAGE_WIDTH_IN
    if anchors is None:
        anchors = A.find_anchors(rec)
    sa = staff_anchors(rec, anchors)
    blocks = [a for a in anchors if a['kind'] == 'block']
    low = max(a['box'][3] for a in blocks) if blocks else None
    cands, hand = candidates(gray, rec, [f['box'] for f in full_dets], want_mask=True,
                             zones=[staff_zone(a, ppi) for a in sa])
    if STAFF_SPLIT and sa:
        cands = [p for d in cands for p in split_stacked(d, hand, sa)]
    out = []
    for d in cands:
        b = d['box']
        w, h = (b[2] - b[0]) / ppi, (b[3] - b[1]) / ppi
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        typ = None
        if any(a['box'][0] - STAFF_LEFT_IN * ppi <= cx <= a['box'][2] + STAFF_RIGHT_IN * ppi and
               a['box'][1] - STAFF_UP_IN * ppi <= cy <= a['box'][3] + STAFF_DOWN_IN * ppi for a in sa):
            if w * h >= STAFF_MIN_AREA_IN2 and h >= STAFF_MIN_H_IN:
                typ = 'staff_signature'
            else:
                continue
        elif w * h >= MARGIN_MIN_AREA_IN2 and h >= MARGIN_MIN_H_IN and w >= MARGIN_MIN_W_IN and                 not (w <= MARGIN_LINE_W_IN and h >= MARGIN_LINE_ASPECT * w) and (
                cy >= MARGIN_Y * H
                or (REF_ZONE_X and low is not None and cx <= REF_ZONE_X * W and low < cy <= low + REF_ZONE_DOWN_IN * ppi)
                or (CORNER_X and cx >= CORNER_X * W and cy >= CORNER_Y * H)):
            typ = 'margin_signature'
        if typ:
            out.append(dict(box=b, ink=d['ink'], type=typ))
    return out, sa
