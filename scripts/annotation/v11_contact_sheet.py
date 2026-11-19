"""Draw a v1.1 export's boxes on dev page images, side by side (red = staff, blue = margin).

Usage (doctr-env Python, needs OpenCV):
  python v11_contact_sheet.py <export.json> <out.png> year:file:page[:draft] ...
  e.g. 2022:065-2022.pdf:2   (':draft' draws the unsubmitted draft instead of the submitted annotation)
"""
import json,sys,cv2,numpy as np
d=json.load(open(sys.argv[1],encoding='utf-8'))
T={(int(t['data']['year']),t['data']['file'],int(t['data']['page'])):t for t in d}
def img(y,f,p,src='ann',scale=0.45):
    im=cv2.imread(f'D:/Thesis/baguio_annotation/images/dev/{y}/{y}_{f[:-4]}_p{p:02d}.png')
    H,W=im.shape[:2]; t=T.get((y,f,p))
    if t:
        res = t['annotations'][0]['result'] if src=='ann' else t['drafts'][0]['result']
        for r in res:
            v=r['value']
            if 'x' not in v: continue
            c=(0,200,200) if 'rectanglelabels' not in v else (0,0,255) if v['rectanglelabels'][0]=='staff_signature' else (255,0,0)
            x,yy,w,h=[int(v[k]*s/100) for k,s in (('x',W),('y',H),('width',W),('height',H))]
            cv2.rectangle(im,(x,yy),(x+w,yy+h),c,4)
    im=cv2.resize(im,None,fx=scale,fy=scale)
    cv2.putText(im,f'{y} {f} p{p} {src}',(10,40),0,1.2,(0,160,0),3); return im
out=sys.argv[2]; items=[a.split(':') for a in sys.argv[3:]]
ims=[img(int(y),f,int(p),*(rest[:1] or ['ann'])) for y,f,p,*rest in items]
h=max(i.shape[0] for i in ims); ims=[cv2.copyMakeBorder(i,0,h-i.shape[0],0,10,0,value=(255,255,255)) for i in ims]
cv2.imwrite(out,np.hstack(ims))
