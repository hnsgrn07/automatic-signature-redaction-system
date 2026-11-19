"""Check a v1.1 (staff + margin) DEV Label Studio export before building the ground truth.

Reports: page/doc coverage against staff/assignment.csv, duplicates, drafts/skips, labels, empty pages,
tiny (< 10 px) and oversized boxes, overlapping own boxes, boxes on a v1.0.1 full signature, last pages
without a staff box, staff boxes in the right half, margin boxes in the staff zone of a last page.
Repeated Label Studio regions (same id) are counted once and flagged as 'dup-id'.

Usage: python -P -E check_v11_export.py <export.json> <Ben|Ferrer>
"""
import json,csv,collections,sys
exp,who=sys.argv[1],sys.argv[2]
ANN='D:/Thesis/baguio_annotation/'
pages={(int(r['year']),r['file'],int(r['page'])):r for r in csv.DictReader(open(ANN+'pages.csv',encoding='utf-8'))}
assign={(int(r['year']),r['file']):(r['annotator'],r['split'],int(r['pages'])) for r in csv.DictReader(open(ANN+'staff/assignment.csv',encoding='utf-8'))}
mine={k for k,v in assign.items() if v[0]==who}
exp_pages={(y,f,p) for (y,f,p) in pages if (y,f) in mine}
d=json.load(open(exp,encoding='utf-8'))
keys=collections.Counter((int(t['data']['year']),t['data']['file'],int(t['data']['page'])) for t in d)
print('tasks',len(d),'unique',len(keys),'expected',len(exp_pages),'docs',len({k[:2] for k in keys}),'expected docs',len(mine))
print('dups',[k for k,v in keys.items() if v>1]); print('missing',sorted(exp_pages-set(keys))[:10],'extra',sorted(set(keys)-exp_pages)[:10])
print('splits',collections.Counter(t['data']['split'] for t in d))
full=collections.defaultdict(list)
for r in csv.DictReader(open('D:/Thesis/ground_truth/final/ground_truth_dev.csv',encoding='utf-8')):
    full[(int(r['year']),r['file'],int(r['page']))].append([float(r[c]) for c in 'xywh'])
def ov(a,b):
    ix=max(0,min(a[0]+a[2],b[0]+b[2])-max(a[0],b[0]));iy=max(0,min(a[1]+a[3],b[1]+b[3])-max(a[1],b[1]));i=ix*iy
    return i/(a[2]*a[3]+b[2]*b[3]-i), i/(a[2]*a[3])
bad=[];labels=collections.Counter();tiny=[];big=[];empty=0;fullhit=[];dupov=[];by={}
for t in d:
    k=(int(t['data']['year']),t['data']['file'],int(t['data']['page']))
    anns=[a for a in t['annotations'] if not a.get('was_cancelled')]
    if len(anns)!=1 or len(t['annotations'])!=1 or t.get('drafts'): bad.append((k,len(t['annotations']),len(anns),bool(t.get('drafts'))))
    if not anns: continue
    res=anns[-1]['result']; _seen=set(); rs=[]
    for r in res:
        if r.get('type')=='rectanglelabels' and r['id'] not in _seen: _seen.add(r['id']); rs.append(r)
    res=[r for r in res if r.get('type')!='rectanglelabels' or True]; rs_all=[r for r in res if r.get('type')=='rectanglelabels']
    if len(rs_all)!=len(rs): bad.append((k,'dup-id',len(rs_all)-len(rs)))
    res=[r for r in res if r.get('type')!='rectanglelabels']+rs
    if len(rs)!=len(res): bad.append((k,'non-rect',[r.get('type') for r in res if r.get('type')!='rectanglelabels']))
    if not rs: empty+=1
    W,H=int(pages[k]['width_px']),int(pages[k]['height_px'])
    bx=[]
    for r in rs:
        v=r['value'];lab=tuple(v['rectanglelabels']);labels[lab]+=1
        if (r['original_width'],r['original_height'])!=(W,H): bad.append((k,'size'))
        if v.get('rotation',0): bad.append((k,'rot'))
        b=[v['x']*W/100,v['y']*H/100,v['width']*W/100,v['height']*H/100]
        if min(b[2],b[3])<10: tiny.append((k,lab,round(b[2]),round(b[3])))
        if b[2]>450 or b[3]>250: big.append((k,lab,round(b[2]),round(b[3])))
        for g in full[k]:
            iou,cov=ov(b,g)
            if iou>0.3 or cov>0.6: fullhit.append((k,lab[0],round(iou,2),round(cov,2)))
        bx.append((lab[0],b))
    for i,(l1,a) in enumerate(bx):
        for l2,b2 in bx[i+1:]:
            if ov(a,b2)[0]>0.3: dupov.append(k)
    by[k]=bx
print('problems',bad); print('empty pages',empty,'labels',dict(labels),'total',sum(labels.values()))
print('tiny',tiny); print('big',big); print('overlapping own boxes',dupov)
print('boxes on a v1.0 FULL signature',len(fullhit)); [print(' ',x) for x in fullhit]
last=collections.defaultdict(int)
for (y,f,p) in keys: last[(y,f)]=max(last[(y,f)],p)
nost=[df for df,lp in last.items() if not any(l=='staff_signature' for l,_ in by.get((df[0],df[1],lp),[]))]
print('docs: no staff box on last page',len(nost),sorted(nost))
odd=[(k,round(b[0]/int(pages[k]['width_px'])*100),round(b[1]/int(pages[k]['height_px'])*100)) for k,L in by.items() for l,b in L if l=='staff_signature' and (b[0]/int(pages[k]['width_px'])>0.5)]
print('staff boxes in right half',odd)
lm=[(k,sum(1 for l,b in L if l=='margin_signature')) for k,L in by.items() if k[2]==last[k[:2]] and any(l=='margin_signature' and b[0]/int(pages[k]['width_px'])<0.4 and 0.4<b[1]/int(pages[k]['height_px'])<0.92 for l,b in L)]
print('last pages with margin box in staff zone',lm)
