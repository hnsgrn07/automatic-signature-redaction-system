"""Build the final signature ground truth (dev + test) and seal the test file.

Sources
  * Pilot r1 ground truth (42 dev docs, adjudicated median boxes, 3/3 consensus)
  * Full-annotation Label Studio exports (357 docs, one annotator each)
  * exports/full/corrections.csv  (logged drops, e.g. accidental-click boxes)

Outputs (in --out, default D:\\Thesis\\ground_truth\\final)
  ground_truth_dev.json   ground_truth_test.json   (COCO, same schema as the pilot file)
  ground_truth_dev.csv    ground_truth_test.csv    (one row per box)
  finalize_report.txt     SHA256SUMS.txt
  The test JSON/CSV are made read-only. The script refuses to run if they already exist.

Usage
  python -P -E finalize_ground_truth.py            # real run -> D:\\Thesis\\ground_truth\\final
  python -P -E finalize_ground_truth.py --out DIR  # trial run anywhere else
"""
import argparse, csv, datetime, hashlib, json, os, stat, sys
from collections import Counter, defaultdict

ANN = r'D:\Thesis\baguio_annotation'
EXPORTS = r'D:\Thesis\exports\full'
PILOT_CSV = r'D:\Thesis\ground_truth\pilot_r1\pilot_r1_ground_truth.csv'
PILOT_COCO = r'D:\Thesis\ground_truth\pilot_r1\pilot_r1_ground_truth_coco.json'
FULL_EXPORTS = {  # export file -> (annotator, split)
    'full_ben_dev.json': ('Ben', 'dev'),
    'full_geron_dev.json': ('Geron', 'dev'),
    'full_geron_test.json': ('Geron', 'test'),
}
LABEL = 'full_signature'
PROTOCOL = 'Signature Annotation Protocol v1.0.1 (https://claude.ai/artifact/WF5eQ3TruqqUekmMkWz1Md)'


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
    ap.add_argument('--out', default=r'D:\Thesis\ground_truth\final')
    args = ap.parse_args()
    out = args.out
    sealed = [os.path.join(out, 'ground_truth_test.' + e) for e in ('json', 'csv')]
    if any(os.path.exists(p) for p in sealed):
        die(f'{out} already holds a test ground truth. It is sealed; nothing was changed.')

    # ---- pages -------------------------------------------------------------
    pages = {}
    for r in csv.DictReader(open(os.path.join(ANN, 'pages.csv'), encoding='utf-8')):
        pages[(int(r['year']), r['file'], int(r['page']))] = r   # year needed: 2024 has files named 068-2025/082-2025

    # ---- pilot ground truth ------------------------------------------------
    coco = json.load(open(PILOT_COCO, encoding='utf-8'))
    pilot_pages = {(im['year'], im['pdf'], im['page']) for im in coco['images']}
    pilot_docs = {(y, f) for y, f, _ in pilot_pages}
    boxes = defaultdict(list)          # (year, file, page) -> [dict]
    covered = {}                       # (year, file, page) -> source
    for key in pilot_pages:
        if key not in pages: die(f'pilot page {key} not in pages.csv')
        if pages[key]['split'] != 'dev': die(f'pilot page {key} is not dev')
        covered[key] = 'pilot_r1'
    for r in csv.DictReader(open(PILOT_CSV, encoding='utf-8')):
        key = (int(r['year']), r['file'], int(r['page']))
        boxes[key].append(dict(x=float(r['x']), y=float(r['y']), w=float(r['w']), h=float(r['h']),
                               source='pilot r1 ' + r['source'], annotators=r['annotators']))

    # ---- assignment --------------------------------------------------------
    assign = {}
    for r in csv.DictReader(open(os.path.join(ANN, 'full', 'assignment.csv'), encoding='utf-8')):
        assign[(int(r['year']), r['file'])] = (r['annotator'], r['split'])
    if pilot_docs & set(assign): die(f'docs in both pilot and full: {sorted(pilot_docs & set(assign))}')

    # ---- full exports ------------------------------------------------------
    raw_count = Counter()
    for name, (who, split) in FULL_EXPORTS.items():
        data = json.load(open(os.path.join(EXPORTS, name), encoding='utf-8'))
        for t in data:
            key = (int(t['data']['year']), t['data']['file'], int(t['data']['page']))
            if key not in pages: die(f'{name}: {key} not in pages.csv')
            if assign.get(key[:2]) != (who, split): die(f'{name}: {key} not assigned to {who}/{split}')
            if key in covered: die(f'{name}: {key} already covered by {covered[key]}')
            anns = [a for a in t.get('annotations', []) if not a.get('was_cancelled')]
            if not anns: die(f'{name}: {key} was not submitted')
            covered[key] = name
            W, H = int(pages[key]['width_px']), int(pages[key]['height_px'])
            seen = set()
            for r in anns[-1]['result']:
                if r.get('type') != 'rectanglelabels' or r['id'] in seen:
                    continue
                seen.add(r['id'])
                v = r['value']
                if v.get('rectanglelabels') != [LABEL]: die(f'{name}: {key} label {v.get("rectanglelabels")}')
                if v.get('rotation', 0): die(f'{name}: {key} rotated box')
                x0, y0 = v['x'] * W / 100, v['y'] * H / 100
                x1, y1 = x0 + v['width'] * W / 100, y0 + v['height'] * H / 100
                x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)   # clip to image
                if x1 - x0 <= 0 or y1 - y0 <= 0: die(f'{name}: {key} empty box')
                boxes[key].append(dict(x=x0, y=y0, w=x1 - x0, h=y1 - y0, source='single annotator',
                                       annotators=who, export=name))
                raw_count[name] += 1

    missing = sorted(set(pages) - set(covered))
    if missing: die(f'{len(missing)} pages have no annotation, e.g. {missing[:5]}')

    # ---- corrections -------------------------------------------------------
    applied = []
    for c in csv.DictReader(open(os.path.join(EXPORTS, 'corrections.csv'), encoding='utf-8')):
        key = (int(c['year']), c['file'], int(c['page']))
        if c['action'] != 'drop': die(f'unknown correction action {c["action"]}')
        hit = [b for b in boxes[key] if b.get('export') == c['export']
               and all(abs(round(b[k]) - int(c[k])) <= 1 for k in 'xywh')]
        if len(hit) != 1: die(f'correction {key} {c["x"]},{c["y"]},{c["w"]},{c["h"]} matched {len(hit)} boxes')
        boxes[key].remove(hit[0])
        applied.append(f'{c["export"]} {c["file"]} p{c["page"]} ({c["x"]},{c["y"]},{c["w"]},{c["h"]}) dropped: {c["reason"]}')

    # ---- write -------------------------------------------------------------
    os.makedirs(out, exist_ok=True)
    now = datetime.datetime.now().isoformat(timespec='seconds')
    report = [f'Final signature ground truth  ({now})', f'Protocol: {PROTOCOL}', '']
    written = []
    for split in ('dev', 'test'):
        keys = sorted((k for k in pages if pages[k]['split'] == split),
                      key=lambda k: k)
        images, annots, rows = [], [], []
        for i, key in enumerate(keys, 1):
            p = pages[key]
            images.append(dict(id=i, file_name=p['image'], width=int(p['width_px']), height=int(p['height_px']),
                               year=key[0], pdf=key[1], page=key[2], split=split, sha256=p['sha256'],
                               source='pilot_r1' if covered[key] == 'pilot_r1' else 'full'))
            for b in sorted(boxes[key], key=lambda b: (b['y'], b['x'])):
                bb = [round(b['x'], 2), round(b['y'], 2), round(b['w'], 2), round(b['h'], 2)]
                annots.append(dict(id=len(annots) + 1, image_id=i, category_id=1, bbox=bb,
                                   area=round(bb[2] * bb[3], 2), iscrowd=0,
                                   source=b['source'], annotators=b['annotators']))
                rows.append([split, key[0], key[1], key[2], p['image'], *bb, b['source'], b['annotators']])
        coco_out = dict(
            info=dict(description=f'Signature ground truth, {split} split (final)', date_created=now,
                      protocol=PROTOCOL,
                      method='pilot r1 docs: median of matched boxes from 3 annotators; '
                             'other docs: single annotator (2-annotator design, no cross-review); '
                             'logged corrections applied'),
            images=images, annotations=annots,
            categories=[dict(id=1, name='signature', supercategory='handwriting')])
        jp = os.path.join(out, f'ground_truth_{split}.json')
        cp = os.path.join(out, f'ground_truth_{split}.csv')
        with open(jp, 'w', encoding='utf-8') as f:
            json.dump(coco_out, f, indent=1)
        with open(cp, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['split', 'year', 'file', 'page', 'image', 'x', 'y', 'w', 'h', 'source', 'annotators'])
            w.writerows(rows)
        written += [jp, cp]
        nd = len({k[:2] for k in keys})
        empty = sum(1 for k in keys if not boxes[k])
        by_src = Counter(a['source'] for a in annots)
        by_year = Counter(images[a['image_id'] - 1]['year'] for a in annots)
        report += [f'{split.upper()}: {nd} documents, {len(keys)} pages, {len(annots)} signature boxes, '
                   f'{empty} pages with no signature',
                   '  boxes by source: ' + ', '.join(f'{k} {v}' for k, v in sorted(by_src.items())),
                   '  boxes by year:   ' + ', '.join(f'{k} {v}' for k, v in sorted(by_year.items())), '']

    report += ['Inputs:', f'  pilot r1 ground truth: {len(pilot_docs)} docs, {len(pilot_pages)} pages']
    report += [f'  {n}: {raw_count[n]} boxes' for n in FULL_EXPORTS]
    report += ['Corrections applied: ' + str(len(applied))] + ['  ' + a for a in applied] + ['']

    # ---- seal --------------------------------------------------------------
    sums = []
    for p in written:
        sums.append(f'{sha256(p)}  {os.path.basename(p)}')
    for p in [PILOT_CSV, PILOT_COCO, os.path.join(EXPORTS, 'corrections.csv')] + \
             [os.path.join(EXPORTS, n) for n in FULL_EXPORTS]:
        sums.append(f'{sha256(p)}  input: {p}')
    report += ['SHA-256 (also in SHA256SUMS.txt):'] + ['  ' + s for s in sums]
    report += ['', 'Test files are read-only. Open them once, for the single final test run.']
    with open(os.path.join(out, 'SHA256SUMS.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(sums) + '\n')
    with open(os.path.join(out, 'finalize_report.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(report) + '\n')
    for p in sealed:
        os.chmod(p, stat.S_IREAD)
    print('\n'.join(report))


if __name__ == '__main__':
    main()
