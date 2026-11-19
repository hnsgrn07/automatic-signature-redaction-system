"""Live demo: a local web page that runs the redaction pipeline on one PDF and shows every step.

  set DOCTR_CACHE_DIR=D:\\doctr-cache
  D:\\doctr-env\\Scripts\\python.exe scripts\\demo\\demo_server.py [--port 8765] [--no-browser]

DocTR is loaded and warmed up once at start-up, so the first run in front of an audience is fast.
The page lists the PDFs in <THESIS_ROOT>\\demo\\unseen (portal ordinances that are NOT in the
399-document corpus, so the system has never seen them) and also accepts a dropped-in PDF.

For each run it shows: live stage progress; the anchors and signature boxes found (full signatures,
staff signatures, margin signatures - each in its own colour); before/after; and the recoverability
comparison - PyMuPDF's ordinary redaction (looks black on screen, but the signature is still in the
image data) vs. this system (the image pixels themselves are black). The side-by-side crops are shown
for the full signatures; the recovery test itself is run on every redacted box.
--no-marks redacts full signatures only (the demo as it was before round v1.1).
Sealed TEST documents (by SHA-256 in sample.csv) are refused. Outputs: <THESIS_ROOT>\\demo\\runs\\.
Listens on 127.0.0.1 only.
"""
import argparse, csv, hashlib, json, os, re, sys, threading, time, traceback, uuid, webbrowser
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs, unquote

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'pipeline'))

import cv2
import numpy as np
import pymupdf

from paths import thesis_path
from ingest import load_pages, page_scan, to_gray, to_rgb, px_to_pt
from anchors import find_anchors
from detect import detect
from marks import detect_marks
from redact import redact_pdf, verify, padded, PAD_IN, PAGE_WIDTH_IN, BLACK_MAX

DEMO_DIR = thesis_path('demo', 'unseen')
UPLOAD_DIR = thesis_path('demo', 'uploads')
RUNS_DIR = thesis_path('demo', 'runs')
SAMPLE_CSV = thesis_path('baguio_corpus_sample', 'sample.csv')
VIEW_W = 900            # width of page images shown in the browser
CROP_MARGIN_IN = 0.25
MAX_UPLOAD = 60 * 1024 * 1024

OCR_ENGINE = None
GPU_LOCK = threading.Lock()
JOBS = {}
SPLIT_BY_HASH = {}


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def load_splits():
    if os.path.exists(SAMPLE_CSV):
        for r in csv.DictReader(open(SAMPLE_CSV, encoding='utf-8')):
            SPLIT_BY_HASH[r['sha256']] = r['split']


def provenance(path):
    split = SPLIT_BY_HASH.get(sha256_bytes(open(path, 'rb').read()))
    return {'test': 'sealed test document', 'dev': 'dev set (rules were tuned on it)'}.get(split, 'never seen by the system')


def bgr(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def save_view(img_bgr, path, width=VIEW_W):
    h, w = img_bgr.shape[:2]
    if w > width:
        img_bgr = cv2.resize(img_bgr, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(path, img_bgr)


def black_share(gray, box):
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    inner = gray[y0:y1, x0:x1]
    return float((inner <= BLACK_MAX).mean()) if inner.size else 1.0


def crop(img, box, ppi, H, W):
    m = CROP_MARGIN_IN * ppi
    x0, y0 = max(0, int(box[0] - m)), max(0, int(box[1] - m))
    x1, y1 = min(W, int(box[2] + m)), min(H, int(box[3] + m))
    return img[y0:y1, x0:x1], (x0, y0, x1, y1)


def decoded_images(doc):
    """Every image object stored in the file (drawn or not): [(xref, drawn?, grey array, BGR array)]."""
    drawn = {i['xref'] for p in doc for i in p.get_image_info(xrefs=True)}
    out = []
    for x in range(1, doc.xref_length()):
        if doc.xref_get_key(x, 'Subtype')[1] != '/Image':
            continue
        try:
            pix = pymupdf.Pixmap(doc, x)
            if pix.alpha:
                pix = pymupdf.Pixmap(pix, 0)
            if pix.colorspace and pix.colorspace.n not in (1, 3):
                pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
            rgb = to_rgb(pix)
            out.append((x, x in drawn, to_gray(pix), bgr(rgb)))
        except Exception:
            continue
    return out


def best_copy(images, orig_gray, box):
    """The stored image that looks most like the ORIGINAL inside box -> (mean grey diff, image tuple)."""
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    ref = orig_gray[y0:y1, x0:x1].astype(int)
    best = None
    for im in images:
        g = im[2]
        if g.shape != orig_gray.shape:
            continue
        diff = float(np.abs(g[y0:y1, x0:x1].astype(int) - ref).mean())
        if best is None or diff < best[0]:
            best = (diff, im)
    return best


USE_MARKS = True
TYPE_NAMES = {'full': 'full signature', 'staff': 'staff signature', 'margin': 'margin signature'}
TYPE_BGR = {'full': (40, 170, 40), 'staff': (170, 60, 150), 'margin': (200, 110, 30)}   # green, purple, blue
STAFF_ANCHOR_BGR = (150, 110, 190)
RECOVERABLE_DIFF = 5.0   # a stored image this close to the original inside the box still holds the signature


def naive_redact(src, dst, boxes, scans):
    """PyMuPDF's ordinary redaction, the way most tutorials do it."""
    with pymupdf.open(src) as doc:
        for i, bs in boxes.items():
            for b in bs:
                doc[i].add_redact_annot(px_to_pt(b, scans[i]['to_pt']), fill=(0, 0, 0))
            doc[i].apply_redactions(images=pymupdf.PDF_REDACT_IMAGE_PIXELS)
        doc.save(dst, garbage=4, deflate=True)


def run_job(job, pdf_path):
    out = job['dir']
    step = lambda msg: job['log'].append(dict(t=round(time.perf_counter() - job['t0'], 2), msg=msg))
    T = dict(ingest=0.0, ocr=0.0, detect=0.0, redact=0.0, verify=0.0)
    with GPU_LOCK:
        job['t0'] = time.perf_counter()
        job['state'] = 'running'
        try:
            t = time.perf_counter()
            scans = load_pages(pdf_path)
            T['ingest'] = time.perf_counter() - t
            n = len(scans)
            s0 = scans[0]['pix']
            step(f'Stage 1 - ingest: {n} page(s), scanned image {s0.width}x{s0.height} px '
                 f'(~{s0.width / PAGE_WIDTH_IN:.0f} DPI), no text layer')
            boxes, types, pages = {}, {}, []
            for i, s in enumerate(scans):
                rec, dt = OCR_ENGINE.page_record(s['pix'])
                T['ocr'] += dt
                t = time.perf_counter()
                gray = to_gray(s['pix'])
                anc = find_anchors(rec)
                dets = detect(gray, rec, anchors=anc)
                mk, sanc = detect_marks(gray, rec, dets, anchors=anc) if USE_MARKS else ([], [])
                T['detect'] += time.perf_counter() - t
                nwords = sum(len(l['words']) for l in rec['lines'])
                typed = ([('full', [int(v) for v in d['box']]) for d in dets] +
                         [(m['type'].replace('_signature', ''), [int(v) for v in m['box']]) for m in mk])
                cnt = {k: sum(1 for t_, _ in typed if t_ == k) for k in TYPE_NAMES}
                step(f'Page {i + 1}/{n} - OCR {dt:.2f}s ({nwords} words), '
                     f'{len(anc)} anchor(s): ' + (', '.join(sorted({a["role"] for a in anc})) or 'none')
                     + (f', {len(sanc)} staff name(s)' if sanc else '')
                     + f' -> {cnt["full"]} full signature(s)'
                     + (f', {cnt["staff"]} staff, {cnt["margin"]} margin' if USE_MARKS else ''))
                if typed:
                    boxes[i] = [b for _, b in typed]
                    types[i] = [t_ for t_, _ in typed]
                pages.append(dict(anchors=anc, staff=sanc, dets=dets))
            nsig = sum(len(v) for v in boxes.values())
            by_type = {k: sum(v.count(k) for v in types.values()) for k in TYPE_NAMES}
            step(f'Stage 3 - detection done: {by_type["full"]} full signature(s)'
                 + (f', {by_type["staff"]} staff and {by_type["margin"]} margin signature(s)' if USE_MARKS else '')
                 + f' on {len(boxes)} page(s)')

            red = os.path.join(out, 'redacted.pdf')
            t = time.perf_counter()
            done = redact_pdf(pdf_path, red, boxes)
            T['redact'] = time.perf_counter() - t
            step(f'Stage 4 - redaction: scan pixels overwritten under {nsig} padded box(es), image objects replaced')
            t = time.perf_counter()
            ok, problems, vstats = verify(pdf_path, red, done)
            T['verify'] = time.perf_counter() - t
            step(f'Verification: {"PASSED" if ok else "FAILED"}' + (f' - {"; ".join(problems)}' if problems else ''))
            pipeline_s = sum(T.values())

            # comparison with ordinary PyMuPDF redaction (not part of the pipeline timing)
            naive = os.path.join(out, 'naive_pymupdf.pdf')
            naive_redact(pdf_path, naive, boxes, scans)
            step('Comparison: same boxes redacted with PyMuPDF apply_redactions (the usual method)')

            res = dict(file=os.path.basename(pdf_path), provenance=job['provenance'], pages=n, signatures=nsig,
                       by_type=by_type, marks=USE_MARKS,
                       boxes={str(i + 1): [dict(box=b, type=t_) for b, t_ in zip(boxes[i], types[i])] for i in boxes},
                       ok=ok, problems=problems, timings={k: round(v, 2) for k, v in T.items()},
                       pipeline_s=round(pipeline_s, 2), verify=vstats, views=[], proof=[], recovery=[],
                       downloads=dict(redacted='redacted.pdf', naive='naive_pymupdf.pdf', original='original.pdf'))
            with pymupdf.open(pdf_path) as d_orig, pymupdf.open(red) as d_red, pymupdf.open(naive) as d_nav:
                show = sorted(boxes) or [n - 1]
                for i in show:
                    s = scans[i]
                    W, H = s['pix'].width, s['pix'].height
                    ppi = W / PAGE_WIDTH_IN
                    over = bgr(to_rgb(s['pix']))
                    for a in pages[i]['anchors']:
                        x0, y0, x1, y1 = (int(v) for v in a['box'])
                        cv2.rectangle(over, (x0, y0), (x1, y1), (20, 120, 230), 3)
                    for a in pages[i]['staff']:
                        x0, y0, x1, y1 = (int(v) for v in a['box'])
                        cv2.rectangle(over, (x0, y0), (x1, y1), STAFF_ANCHOR_BGR, 2)
                    for b, t_ in zip(boxes.get(i, []), types.get(i, [])):
                        cv2.rectangle(over, (b[0], b[1]), (b[2], b[3]), TYPE_BGR[t_], 5 if t_ == 'full' else 4)
                    save_view(over, os.path.join(out, f'p{i + 1}_found.png'))
                    after = bgr(to_rgb(page_scan(d_red, d_red[i])['pix']))
                    save_view(after, os.path.join(out, f'p{i + 1}_after.png'))
                    tl = types.get(i, [])
                    res['views'].append(dict(page=i + 1, found=f'p{i + 1}_found.png', after=f'p{i + 1}_after.png',
                                             anchors=[f"{a['kind']}: {a['text']}" for a in pages[i]['anchors']],
                                             counts={k: tl.count(k) for k in TYPE_NAMES}))
                    if i not in boxes:
                        continue
                    if not job.get('imgs'):
                        job['imgs'] = (decoded_images(d_nav), decoded_images(d_red))
                        n_nav, n_red = job['imgs']
                        step(f'Searching every image stored in each file: usual method {len(n_nav)} image object(s) '
                             f'({sum(not x[1] for x in n_nav)} not shown on any page), this system {len(n_red)} '
                             f'({sum(not x[1] for x in n_red)} not shown)')
                    imgs_nav, imgs_red = job['imgs']
                    g_orig = to_gray(s['pix'])
                    zoom = W / d_nav[i].rect.width
                    own_red = page_scan(d_red, d_red[i])['pix']          # this page's own stored image
                    k = -1
                    for b, t_ in zip(boxes[i], types[i]):
                        pb = padded(b, PAD_IN * ppi, W, H)
                        bn = best_copy(imgs_nav, g_orig, pb)
                        br = best_copy(imgs_red, g_orig, pb)
                        res['recovery'].append(dict(type=t_, naive=bool(bn[0] < RECOVERABLE_DIFF),
                                                    ours=bool(br[0] < RECOVERABLE_DIFF)))
                        if t_ != 'full':
                            continue          # side-by-side crops only for the full signatures
                        k += 1
                        c_orig, win = crop(bgr(to_rgb(s['pix'])), pb, ppi, H, W)
                        rect = px_to_pt(win, s['to_pt'])
                        c_screen = bgr(to_rgb(d_nav[i].get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), clip=rect)))
                        c_nav, _ = crop(bn[1][3], pb, ppi, H, W)
                        c_red, _ = crop(bgr(to_rgb(own_red)), pb, ppi, H, W)
                        tag = f'p{i + 1}_s{k + 1}'
                        for name, im in (('orig', c_orig), ('screen', c_screen), ('naive', c_nav), ('ours', c_red)):
                            save_view(im, os.path.join(out, f'{tag}_{name}.png'), width=420)
                        res['proof'].append(dict(
                            page=i + 1, sig=k + 1, orig=f'{tag}_orig.png', screen=f'{tag}_screen.png',
                            naive=f'{tag}_naive.png', ours=f'{tag}_ours.png',
                            naive_recoverable=bn[0] < RECOVERABLE_DIFF, naive_diff=round(bn[0], 1),
                            naive_hidden=not bn[1][1], naive_xref=bn[1][0],
                            ours_recoverable=br[0] < RECOVERABLE_DIFF, ours_diff=round(br[0], 1),
                            ours_black=round(black_share(to_gray(own_red), pb), 3)))
            if res['recovery']:
                rv = res['recovery']

                def tally(sel):
                    xs = [x for x in rv if sel(x['type'])]
                    return dict(naive=sum(x['naive'] for x in xs), ours=sum(x['ours'] for x in xs), total=len(xs))
                res['recoverable'] = tally(lambda t_: t_ == 'full')
                res['recoverable_marks'] = tally(lambda t_: t_ != 'full')
                rc, rm = res['recoverable'], res['recoverable_marks']
                step(f"Recovery test: usual method - {rc['naive']}/{rc['total']} full signature(s)"
                     + (f" and {rm['naive']}/{rm['total']} staff/margin signature(s)" if rm['total'] else '')
                     + f" recoverable from the file; this system - {rc['ours']}/{rc['total']}"
                     + (f" and {rm['ours']}/{rm['total']}" if rm['total'] else ''))
            job.pop('imgs', None)
            sizes = dict(original=os.path.getsize(pdf_path), redacted=os.path.getsize(red), naive=os.path.getsize(naive))
            res['sizes_kb'] = {k: v // 1024 for k, v in sizes.items()}
            with open(os.path.join(out, 'result.json'), 'w', encoding='utf-8') as f:
                json.dump(res, f, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
            job['result'] = json.loads(json.dumps(res, default=lambda o: o.item() if hasattr(o, 'item') else str(o)))
            step(f'Done - pipeline time {pipeline_s:.1f}s for {n} page(s)')
            job['state'] = 'done'
        except Exception as e:
            job['state'] = 'error'
            job['error'] = f'{type(e).__name__}: {e}'
            step('ERROR ' + job['error'])
            traceback.print_exc()


def start_job(pdf_path, display_name):
    prov = provenance(pdf_path)
    if prov == 'sealed test document':
        return None, 'This is a sealed TEST document. It is not used before the single final test run.'
    jid = time.strftime('%Y%m%d_%H%M%S_') + uuid.uuid4().hex[:6]
    d = os.path.join(RUNS_DIR, jid)
    os.makedirs(d)
    with open(pdf_path, 'rb') as fi, open(os.path.join(d, 'original.pdf'), 'wb') as fo:
        fo.write(fi.read())
    job = dict(id=jid, dir=d, name=display_name, provenance=prov, state='queued', log=[], result=None, error=None,
               t0=time.perf_counter())
    JOBS[jid] = job
    threading.Thread(target=run_job, args=(job, os.path.join(d, 'original.pdf')), daemon=True).start()
    return jid, None


def demo_docs():
    out = []
    if os.path.isdir(DEMO_DIR):
        for f in sorted(os.listdir(DEMO_DIR)):
            if f.lower().endswith('.pdf'):
                with pymupdf.open(os.path.join(DEMO_DIR, f)) as doc:
                    out.append(dict(name=f, pages=len(doc)))
    return out


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, code, body, ctype='application/json'):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == '/':
            return self.send(200, open(os.path.join(HERE, 'demo_page.html'), 'rb').read(), 'text/html; charset=utf-8')
        if u.path == '/api/docs':
            return self.send(200, dict(docs=demo_docs(), device=OCR_ENGINE.device, device_name=OCR_ENGINE.device_name))
        m = re.fullmatch(r'/api/job/([\w]+)', u.path)
        if m and m.group(1) in JOBS:
            j = JOBS[m.group(1)]
            return self.send(200, {k: j[k] for k in ('id', 'name', 'provenance', 'state', 'log', 'result', 'error')})
        m = re.fullmatch(r'/files/([\w]+)/([\w.\-]+)', u.path)
        if m and m.group(1) in JOBS:
            p = os.path.join(JOBS[m.group(1)]['dir'], m.group(2))
            if os.path.isfile(p):
                ctype = 'image/png' if p.endswith('.png') else 'application/pdf'
                return self.send(200, open(p, 'rb').read(), ctype)
        self.send(404, dict(error='not found'))

    def do_POST(self):
        u = urlparse(self.path)
        if u.path == '/api/run':
            name = os.path.basename(parse_qs(u.query).get('name', [''])[0])
            p = os.path.join(DEMO_DIR, name)
            if not name or not os.path.isfile(p):
                return self.send(404, dict(error='no such demo document'))
            jid, err = start_job(p, name)
        elif u.path == '/api/upload':
            n = int(self.headers.get('Content-Length', 0))
            if n <= 0 or n > MAX_UPLOAD:
                return self.send(400, dict(error='empty or too large (limit 60 MB)'))
            data = self.rfile.read(n)
            if not data.startswith(b'%PDF'):
                return self.send(400, dict(error='not a PDF'))
            name = re.sub(r'[^\w.\-]', '_', os.path.basename(unquote(self.headers.get('X-Filename', 'upload.pdf'))))
            os.makedirs(UPLOAD_DIR, exist_ok=True)
            p = os.path.join(UPLOAD_DIR, f'{sha256_bytes(data)[:12]}_{name}')
            with open(p, 'wb') as f:
                f.write(data)
            jid, err = start_job(p, name)
        else:
            return self.send(404, dict(error='not found'))
        if err:
            return self.send(403, dict(error=err))
        self.send(200, dict(job=jid))


def main():
    global OCR_ENGINE
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=8765)
    ap.add_argument('--no-browser', action='store_true')
    ap.add_argument('--cpu', action='store_true')
    ap.add_argument('--no-marks', action='store_true', help='full signatures only (skip staff/margin signatures)')
    args = ap.parse_args()
    global USE_MARKS
    USE_MARKS = not args.no_marks
    from redact_cli import OCR
    load_splits()
    os.makedirs(RUNS_DIR, exist_ok=True)
    print('Loading DocTR ...', flush=True)
    OCR_ENGINE = OCR(cpu=args.cpu)
    t = time.perf_counter()
    blank = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 1275, 1950), False)
    blank.set_rect(blank.irect, (255, 255, 255))
    OCR_ENGINE.page_record(blank)       # warm-up: first GPU call is slow
    print(f'DocTR on {OCR_ENGINE.device} ({OCR_ENGINE.device_name}); import {OCR_ENGINE.import_s:.1f}s, '
          f'load {OCR_ENGINE.load_s:.1f}s, warm-up {time.perf_counter() - t:.1f}s', flush=True)
    srv = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    url = f'http://127.0.0.1:{args.port}/'
    print(f'Demo ready: {url}   (Ctrl+C to stop)', flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
