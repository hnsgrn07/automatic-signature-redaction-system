"""Build the v1.1 DEV ground truth (staff + margin signatures).

Round v1.1 adds two labels on top of the v1.0.1 full-signature ground truth:
  staff_signature   mark on/over a small printed staff name (no label, no title)
  margin_signature  any other handwritten signature mark (paraphs, initials, short signatures,
                    marks beside typed reference initials, and handwritten signatures inside
                    RECEIVED stamps - team rule 2026-11-19)
The v1.0.1 files are read (for a safety check) but not changed. No test file is read.

Sources (280 dev docs, 976 pages, every page from exactly one source)
  - v1.1 pilot ground truth (14 docs / 30 pages, all three annotators, adjudicated):
      D:\\Thesis\\ground_truth\\staff_pilot\\staff_pilot_ground_truth.csv (hash checked)
  - Ben's export   staff_ben_dev.json    (107 docs, single annotator)
  - Ferrer's export staff_ferrer_dev.json (159 docs, single annotator)
  - Corrections:   D:\\Thesis\\exports\\staff_full\\corrections_v11_dev.csv
      every row must match exactly one box (within 2 px), otherwise the script stops.

Checks (script stops on failure): every dev page covered exactly once; one submitted annotation
per task; labels are the two v1.1 labels; no box under 10 px; after corrections no box matches a
v1.0.1 full signature (IoU >= 0.50). Unsubmitted drafts are ignored and listed in the report.

Outputs (in --out, default D:\\Thesis\\ground_truth\\v11)
  ground_truth_v11_dev.json (COCO, categories 1 staff_signature / 2 margin_signature)
  ground_truth_v11_dev.csv  (one row per box, column `type`)
  build_v11_dev_report.txt   SHA256SUMS_v11_dev.txt
Refuses to overwrite existing dev files unless --overwrite is given.

Usage
  python -P -E build_v11_dev_gt.py [--out DIR] [--overwrite]
"""
import argparse, csv, datetime, hashlib, json, os, sys
from collections import Counter, defaultdict

ANN = r'D:\Thesis\baguio_annotation'
EXP = r'D:\Thesis\exports\staff_full'
EXPORTS = {'Ben': os.path.join(EXP, 'staff_ben_dev.json'), 'Ferrer': os.path.join(EXP, 'staff_ferrer_dev.json')}
CORR = os.path.join(EXP, 'corrections_v11_dev.csv')
PILOT = r'D:\Thesis\ground_truth\staff_pilot\staff_pilot_ground_truth.csv'
PILOT_SHA = 'cdf98ddedcab9a2b915cf758b5ea87b5620301005d3de634d66ae92bae1adfe1'
V10_DEV = r'D:\Thesis\ground_truth\final\ground_truth_dev.csv'
LABELS = {'staff_signature': 1, 'margin_signature': 2}
MIN_SIDE_PX = 10
TOL_PX = 2
PROTOCOL = ('Signature Annotation Protocol v1.0.1 + addendum v1.1 additional signatures '
            '(docs/annotation_protocol_v1.1_additional_signatures.md)')


def die(msg):
    sys.exit('STOPPED: ' + msg)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def iou(a, b):
    ix = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    i = ix * iy
    return i / (a[2] * a[3] + b[2] * b[3] - i) if i else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=r'D:\Thesis\ground_truth\v11')
    ap.add_argument('--overwrite', action='store_true')
    a = ap.parse_args()
    outs = [os.path.join(a.out, 'ground_truth_v11_dev.' + e) for e in ('json', 'csv')]
    if any(os.path.exists(p) for p in outs) and not a.overwrite:
        die(f'{a.out} already holds a v1.1 dev ground truth; use --overwrite to rebuild.')

    pages = {}
    for r in csv.DictReader(open(os.path.join(ANN, 'pages.csv'), encoding='utf-8')):
        if r['split'] == 'dev':
            pages[(int(r['year']), r['file'], int(r['page']))] = r   # key by year: filenames repeat
    owner = {}
    for r in csv.DictReader(open(os.path.join(ANN, 'staff', 'assignment.csv'), encoding='utf-8')):
        if r['split'] == 'dev':
            owner[(int(r['year']), r['file'])] = 'pilot' if r['annotator'] == 'all' else r['annotator']
    if {k[:2] for k in pages} != set(owner):
        die('dev documents in pages.csv and staff/assignment.csv differ')

    boxes, src_of, notes = {}, {}, []

    # pilot ground truth
    if sha256(PILOT) != PILOT_SHA:
        die('pilot ground truth hash differs from SHA256SUMS.txt')
    for k in pages:
        if owner[k[:2]] == 'pilot':
            boxes[k], src_of[k] = [], 'pilot'
    for r in csv.DictReader(open(PILOT, encoding='utf-8')):
        k = (int(r['year']), r['file'], int(r['page']))
        if src_of.get(k) != 'pilot': die(f'pilot GT row {k} is not a pilot page')
        boxes[k].append(dict(b=[float(r[c]) for c in 'xywh'], type=r['type'], source=f'pilot v1.1 {r["source"]}',
                             annotators=r['annotators'], origin='staff_pilot_ground_truth.csv'))

    # single-annotator exports
    for who, path in EXPORTS.items():
        for t in json.load(open(path, encoding='utf-8')):
            k = (int(t['data']['year']), t['data']['file'], int(t['data']['page']))
            if k not in pages: die(f'{who}: {k} is not a dev page')
            if owner[k[:2]] != who: die(f'{who}: {k} is assigned to {owner[k[:2]]}')
            if k in src_of: die(f'{k} appears twice')
            src_of[k] = who
            if t.get('drafts'): notes.append(f'{who} {k}: unsubmitted draft ignored (submitted annotation used)')
            anns = [x for x in t.get('annotations', []) if not x.get('was_cancelled')]
            if len(anns) != 1: die(f'{who} {k}: {len(anns)} submitted annotations')
            W, H = int(pages[k]['width_px']), int(pages[k]['height_px'])
            boxes[k], ids = [], set()
            for r in anns[0]['result']:
                if r.get('type') != 'rectanglelabels': die(f'{who} {k}: result type {r.get("type")}')
                if r['id'] in ids: continue   # Label Studio repeats regions with the same id
                ids.add(r['id'])
                if (r['original_width'], r['original_height']) != (W, H): die(f'{who} {k}: image size')
                v = r['value']; lab = v.get('rectanglelabels') or []
                if len(lab) != 1 or lab[0] not in LABELS: die(f'{who} {k}: label {lab}')
                if v.get('rotation', 0): die(f'{who} {k}: rotated box')
                x0, y0 = max(0, v['x'] * W / 100), max(0, v['y'] * H / 100)
                x1, y1 = min(W, (v['x'] + v['width']) * W / 100), min(H, (v['y'] + v['height']) * H / 100)
                boxes[k].append(dict(b=[x0, y0, x1 - x0, y1 - y0], type=lab[0], source='single annotator',
                                     annotators=who, origin=os.path.basename(path)))
    missing = sorted(set(pages) - set(src_of))
    if missing: die(f'{len(missing)} dev pages not covered, e.g. {missing[:5]}')

    # corrections
    applied = []
    for c in csv.DictReader(open(CORR, encoding='utf-8')):
        k = (int(c['year']), c['file'], int(c['page']))
        cb = [float(c[x]) for x in 'xywh']
        hits = [b for b in boxes.get(k, []) if b['origin'] == c['source']
                and all(abs(b['b'][i] - cb[i]) <= TOL_PX for i in range(4))]
        if len(hits) != 1: die(f'correction {k} {cb} matches {len(hits)} boxes')
        b = hits[0]
        if c['action'] == 'drop':
            boxes[k].remove(b)
        elif c['action'].startswith('relabel '):
            new = c['action'].split()[1]
            if new not in LABELS or new == b['type']: die(f'correction {k}: bad relabel {c["action"]}')
            b['type'] = new
            b['source'] += ' (relabelled)'
        else:
            die(f'correction {k}: unknown action {c["action"]}')
        applied.append(f'{c["source"]} {k} {[round(v) for v in cb]}: {c["action"]} - {c["reason"]}')

    # safety checks
    full = defaultdict(list)
    for r in csv.DictReader(open(V10_DEV, encoding='utf-8')):
        full[(int(r['year']), r['file'], int(r['page']))].append([float(r[c]) for c in 'xywh'])
    for k, L in boxes.items():
        for b in L:
            if min(b['b'][2], b['b'][3]) < MIN_SIDE_PX: die(f'{k}: box under {MIN_SIDE_PX} px {b}')
            if any(iou(b['b'], g) >= 0.5 for g in full[k]): die(f'{k}: box matches a v1.0.1 full signature')

    # write
    os.makedirs(a.out, exist_ok=True)
    now = datetime.datetime.now().isoformat(timespec='seconds')
    images, annots, rows = [], [], []
    for i, k in enumerate(sorted(pages), 1):
        p = pages[k]
        images.append(dict(id=i, file_name=p['image'], width=int(p['width_px']), height=int(p['height_px']),
                           year=k[0], pdf=k[1], page=k[2], split='dev', sha256=p['sha256']))
        for b in sorted(boxes[k], key=lambda b: (b['b'][1], b['b'][0])):
            bb = [round(v, 2) for v in b['b']]
            annots.append(dict(id=len(annots) + 1, image_id=i, category_id=LABELS[b['type']], bbox=bb,
                               area=round(bb[2] * bb[3], 2), iscrowd=0, source=b['source'],
                               annotators=b['annotators']))
            rows.append(['dev', k[0], k[1], k[2], p['image'], *bb, b['source'], b['annotators'], b['type']])
    coco = dict(
        info=dict(description='Additional signature ground truth v1.1 (staff + margin), dev split',
                  date_created=now, protocol=PROTOCOL,
                  method='pilot pages: v1.1 pilot consensus (3 annotators, median box); other pages: single '
                         'annotator (Ben 107 docs, Ferrer 159 docs), two-label config; '
                         f'{len(applied)} logged corrections (corrections_v11_dev.csv)'),
        images=images, annotations=annots,
        categories=[dict(id=1, name='staff_signature', supercategory='handwriting'),
                    dict(id=2, name='margin_signature', supercategory='handwriting')])
    jp, cp = outs
    with open(jp, 'w', encoding='utf-8') as f:
        json.dump(coco, f, indent=1)
    with open(cp, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['split', 'year', 'file', 'page', 'image', 'x', 'y', 'w', 'h', 'source', 'annotators', 'type'])
        w.writerows(rows)

    by_type = Counter(r[-1] for r in rows)
    by_src = Counter((src_of[(r[1], r[2], r[3])], r[-1]) for r in rows)
    by_year = Counter((r[1], r[-1]) for r in rows)
    pg_src = Counter(src_of.values())
    empty = sum(1 for k in pages if not boxes[k])
    rep = [f'v1.1 additional-signature ground truth, DEV  ({now})', f'Protocol: {PROTOCOL}', '',
           f'DEV: {len({k[:2] for k in pages})} documents, {len(pages)} pages, {len(rows)} boxes, '
           f'{empty} pages with no box',
           '  by type: ' + ', '.join(f'{t} {by_type[t]}' for t in LABELS),
           '  by source (pages; staff/margin): ' + ', '.join(
               f'{s} {pg_src[s]} p; {by_src[(s, "staff_signature")]}/{by_src[(s, "margin_signature")]}'
               for s in ('pilot', 'Ben', 'Ferrer')),
           '  by year (staff/margin): ' + ', '.join(
               f'{y} {by_year[(y, "staff_signature")]}/{by_year[(y, "margin_signature")]}'
               for y in sorted({k[0] for k in pages})), '',
           f'Corrections applied ({len(applied)}):'] + ['  ' + s for s in applied]
    rep += ['', 'Notes:'] + ['  ' + s for s in notes or ['none']]
    ins = [PILOT, *EXPORTS.values(), CORR]
    sums = [f'{sha256(p)}  {os.path.basename(p)}' for p in outs] + [f'{sha256(p)}  input: {p}' for p in ins]
    rep += ['', 'SHA-256 (also in SHA256SUMS_v11_dev.txt):'] + ['  ' + s for s in sums]
    with open(os.path.join(a.out, 'SHA256SUMS_v11_dev.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(sums) + '\n')
    with open(os.path.join(a.out, 'build_v11_dev_report.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(rep) + '\n')
    print('\n'.join(rep))


if __name__ == '__main__':
    main()
