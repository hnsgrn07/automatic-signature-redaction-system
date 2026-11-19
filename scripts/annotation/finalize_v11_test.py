"""Build and seal the v1.1 TEST ground truth (staff + margin signatures).

Round v1.1 adds two labels on top of the sealed v1.0 full-signature ground truth:
  staff_signature   mark on/over a small printed staff name (no label, no title)
  margin_signature  any other handwritten signature mark (paraphs, initials, short signatures,
                    and handwritten signatures inside RECEIVED stamps - team rule 2026-11-19)
The v1.0 files are not read or changed.

Source: Geron's Label Studio export of package staff_geron_test (all 119 test docs, 357 pages,
one annotator). Fixes found in the export check were made in Label Studio before export, so no
corrections file is needed.

Outputs (in --out, default D:\\Thesis\\ground_truth\\v11)
  ground_truth_v11_test.json  (COCO, categories 1 staff_signature / 2 margin_signature)
  ground_truth_v11_test.csv   (one row per box, column `type`)
  finalize_v11_test_report.txt   SHA256SUMS_v11_test.txt
The two test files are made read-only. The script refuses to run if they already exist.

Usage
  python -I finalize_v11_test.py            # real run -> D:\\Thesis\\ground_truth\\v11
  python -I finalize_v11_test.py --out DIR  # trial run anywhere else
"""
import argparse, csv, datetime, hashlib, json, os, stat, sys
from collections import Counter

ANN = r'D:\Thesis\baguio_annotation'
EXPORT = r'D:\Thesis\exports\staff_full\staff_geron_test.json'
ANNOTATOR = 'Geron'
LABELS = {'staff_signature': 1, 'margin_signature': 2}
MIN_SIDE_PX = 10   # smaller boxes are accidental clicks (checked: none left in the final export)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=r'D:\Thesis\ground_truth\v11')
    out = ap.parse_args().out
    sealed = [os.path.join(out, 'ground_truth_v11_test.' + e) for e in ('json', 'csv')]
    if any(os.path.exists(p) for p in sealed):
        die(f'{out} already holds a v1.1 test ground truth. It is sealed; nothing was changed.')

    pages = {}
    for r in csv.DictReader(open(os.path.join(ANN, 'pages.csv'), encoding='utf-8')):
        if r['split'] == 'test':
            pages[(int(r['year']), r['file'], int(r['page']))] = r   # key by year: filenames repeat across years
    assign = {}
    for r in csv.DictReader(open(os.path.join(ANN, 'staff', 'assignment.csv'), encoding='utf-8')):
        assign[(int(r['year']), r['file'])] = (r['annotator'], r['split'])

    data = json.load(open(EXPORT, encoding='utf-8'))
    boxes, seen_pages = {}, set()
    for t in data:
        key = (int(t['data']['year']), t['data']['file'], int(t['data']['page']))
        if key not in pages: die(f'{key} is not a test page in pages.csv')
        if t['data'].get('split') != 'test': die(f'{key} split {t["data"].get("split")}')
        if assign.get(key[:2]) != (ANNOTATOR, 'test'): die(f'{key} not assigned to {ANNOTATOR}/test')
        if key in seen_pages: die(f'{key} appears twice')
        seen_pages.add(key)
        if t.get('drafts'): die(f'{key} has an unsubmitted draft')
        anns = [a for a in t.get('annotations', []) if not a.get('was_cancelled')]
        if len(anns) != 1: die(f'{key} has {len(anns)} submitted annotations')
        W, H = int(pages[key]['width_px']), int(pages[key]['height_px'])
        boxes[key] = []
        ids = set()
        for r in anns[0]['result']:
            if r.get('type') != 'rectanglelabels': die(f'{key} unexpected result type {r.get("type")}')
            if r['id'] in ids: continue
            ids.add(r['id'])
            if (r['original_width'], r['original_height']) != (W, H): die(f'{key} image size differs from pages.csv')
            v = r['value']
            lab = v.get('rectanglelabels') or []
            if len(lab) != 1 or lab[0] not in LABELS: die(f'{key} label {lab}')
            if v.get('rotation', 0): die(f'{key} rotated box')
            x0, y0 = v['x'] * W / 100, v['y'] * H / 100
            x1, y1 = x0 + v['width'] * W / 100, y0 + v['height'] * H / 100
            x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
            if min(x1 - x0, y1 - y0) < MIN_SIDE_PX: die(f'{key} box smaller than {MIN_SIDE_PX} px')
            boxes[key].append(dict(x=x0, y=y0, w=x1 - x0, h=y1 - y0, type=lab[0]))
    missing = sorted(set(pages) - seen_pages)
    if missing: die(f'{len(missing)} test pages missing from the export, e.g. {missing[:5]}')

    os.makedirs(out, exist_ok=True)
    now = datetime.datetime.now().isoformat(timespec='seconds')
    images, annots, rows = [], [], []
    for i, key in enumerate(sorted(pages), 1):
        p = pages[key]
        images.append(dict(id=i, file_name=p['image'], width=int(p['width_px']), height=int(p['height_px']),
                           year=key[0], pdf=key[1], page=key[2], split='test', sha256=p['sha256']))
        for b in sorted(boxes[key], key=lambda b: (b['y'], b['x'])):
            bb = [round(b['x'], 2), round(b['y'], 2), round(b['w'], 2), round(b['h'], 2)]
            annots.append(dict(id=len(annots) + 1, image_id=i, category_id=LABELS[b['type']], bbox=bb,
                               area=round(bb[2] * bb[3], 2), iscrowd=0,
                               source='single annotator', annotators=ANNOTATOR))
            rows.append(['test', key[0], key[1], key[2], p['image'], *bb, 'single annotator', ANNOTATOR, b['type']])
    coco = dict(
        info=dict(description='Additional signature ground truth v1.1 (staff + margin), test split (final)',
                  date_created=now, protocol=PROTOCOL,
                  method='single annotator (Geron, all test docs), Label Studio two-label config; '
                         'export checked and fixed in Label Studio before export'),
        images=images, annotations=annots,
        categories=[dict(id=1, name='staff_signature', supercategory='handwriting'),
                    dict(id=2, name='margin_signature', supercategory='handwriting')])
    jp, cp = sealed
    with open(jp, 'w', encoding='utf-8') as f:
        json.dump(coco, f, indent=1)
    with open(cp, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['split', 'year', 'file', 'page', 'image', 'x', 'y', 'w', 'h', 'source', 'annotators', 'type'])
        w.writerows(rows)

    by_type = Counter(r[-1] for r in rows)
    by_year = Counter((r[1], r[-1]) for r in rows)
    empty = sum(1 for k in pages if not boxes[k])
    report = [f'v1.1 additional-signature ground truth, TEST  ({now})', f'Protocol: {PROTOCOL}', '',
              f'TEST: {len({k[:2] for k in pages})} documents, {len(pages)} pages, {len(rows)} boxes, '
              f'{empty} pages with no box',
              '  by type: ' + ', '.join(f'{t} {by_type[t]}' for t in LABELS),
              '  by year (staff/margin): ' + ', '.join(
                  f'{y} {by_year[(y, "staff_signature")]}/{by_year[(y, "margin_signature")]}'
                  for y in sorted({k[0] for k in pages})), '']
    sums = [f'{sha256(p)}  {os.path.basename(p)}' for p in sealed]
    sums.append(f'{sha256(EXPORT)}  input: {EXPORT}')
    report += ['SHA-256 (also in SHA256SUMS_v11_test.txt):'] + ['  ' + s for s in sums]
    report += ['', 'Test files are read-only. Open them once, for the single final test run.']
    with open(os.path.join(out, 'SHA256SUMS_v11_test.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(sums) + '\n')
    with open(os.path.join(out, 'finalize_v11_test_report.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(report) + '\n')
    for p in sealed:
        os.chmod(p, stat.S_IREAD)
    print('\n'.join(report))


if __name__ == '__main__':
    main()
