import json, csv, sys, collections
exp, who, split = sys.argv[1], sys.argv[2], sys.argv[3]
base = 'D:/Thesis/baguio_annotation/'
assign = {r['file'] for r in csv.DictReader(open(base + 'full/assignment.csv'))
          if r['annotator'] == who and r['split'] == split}
pages = {(r['file'], int(r['page'])): r for r in csv.DictReader(open(base + 'pages.csv'))
         if r['split'] == split and r['file'] in assign}
data = json.load(open(exp, encoding='utf-8'))
print('export tasks', len(data), '| expected pages', len(pages), '| docs', len(assign))
seen = set(); nbox = 0; empty = 0; issues = []; notes = []
labels = collections.Counter(); perdoc = collections.Counter(); projects = set()
for t in data:
    d = t['data']; key = (d['file'], int(d['page'])); projects.add(t.get('project'))
    if key not in pages: issues.append(('not in assignment', key)); continue
    if key in seen: issues.append(('duplicate task', key))
    seen.add(key); p = pages[key]
    anns = [a for a in t.get('annotations', []) if not a.get('was_cancelled')]
    if not anns: issues.append(('skipped / not submitted', key)); continue
    if len(anns) > 1: issues.append(('multiple annotations', key, len(anns)))
    if t.get('drafts'): issues.append(('has draft', key))
    ids = set(); k = 0
    for r in anns[-1]['result']:
        v = r.get('value', {})
        if r.get('type') != 'rectanglelabels':
            notes.append((key, r.get('type'), v.get('text') or v.get('choices'))); continue
        if r['id'] in ids: continue
        ids.add(r['id']); k += 1
        for l in v.get('rectanglelabels', []): labels[l] += 1
        W, H = int(p['width_px']), int(p['height_px'])
        if r.get('original_width') and (r['original_width'], r['original_height']) != (W, H):
            issues.append(('size mismatch', key, r['original_width'], r['original_height'], W, H))
        x, y, w, h = v['x']*W/100, v['y']*H/100, v['width']*W/100, v['height']*H/100
        if x < -1 or y < -1 or x+w > W+1 or y+h > H+1:
            issues.append(('outside image', key, round(x), round(y), round(w), round(h)))
        if w < 20 or h < 10:
            issues.append(('tiny box', key, round(x), round(y), round(w), round(h)))
        if v.get('rotation', 0): issues.append(('rotated', key, v['rotation']))
    nbox += k; perdoc[d['file']] += k
    if k == 0: empty += 1
missing = sorted(set(pages) - seen)
print('projects', projects)
print('submitted pages', len(seen), '| boxes', nbox, '| empty pages', empty, '| labels', dict(labels))
print('missing pages', len(missing), missing[:20])
for i in issues: print('ISSUE', i)
for n in notes: print('NOTE', n)
for f in sorted(assign - set(perdoc)): perdoc[f] += 0
odd = {f: c for f, c in perdoc.items() if c != 3}
print('docs with != 3 boxes:', len(odd))
for f, c in sorted(odd.items()): print('   ', f, c)
