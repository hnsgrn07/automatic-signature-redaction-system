"""Batch CLI: redact the full signatures in scanned ordinance PDFs.

  python redact_cli.py INPUT [--out DIR] [--cpu] [--save-ocr] [--limit N]

INPUT is one PDF or a folder (searched recursively for *.pdf). For every document:

  stage 1 ingest   - extract each page's scan pixel for pixel (ingest.load_pages)
  stage 2 OCR      - live DocTR OCR of every page (model loaded ONCE per run)
         anchors   - labels and printed name/title blocks (anchors.find_anchors)
  stage 3 detect   - signature boxes (detect.detect)
  stage 4 redact   - overwrite the scan pixels under the boxes (redact.redact_pdf)
         verify    - re-open the output and prove nothing is recoverable (redact.verify)

Output folder (default: Desktop/Redacted_Outputs/run_YYYYMMDD_HHMMSS):
  redacted/<sub>/<name>_redacted.pdf      verify PASSED, every page a single full-page scan
  needs_review/<sub>/<name>_NOT_SAFE.pdf  verify FAILED or a page could not be handled
                                          automatically -> redact by hand; do NOT publish as is
  logs/<sub>/<name>.json                  per page: size, method, anchors, boxes; verify result
  ocr/<sub>/<name>_pNN.json               (--save-ocr) live OCR, same format as ocr_cache.py
  timings.csv                             one row per document, seconds per stage + wall time
  run_info.json                           commit, versions, device, settings, model load time
  summary.txt                             totals and the list of documents that need review

<sub> mirrors the input's sub-folders (e.g. the year), because file names repeat across years.
Timing: library import and model load are logged once (run_info.json). Per document, wall_s runs from the start of
ingest to the end of verify; ocr_s is GPU-synchronised. The run total includes the model load.
"""
import argparse, csv, datetime, json, os, platform, subprocess, sys, time, traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pymupdf

import anchors as anchors_mod
import detect as detect_mod
import redact as redact_mod
from ingest import load_pages, to_gray, to_rgb
from anchors import find_anchors
from detect import detect
from redact import redact_pdf, verify

HERE = os.path.dirname(os.path.abspath(__file__))
STATUS_SAFE = 'SAFE'
STATUS_REVIEW = 'NEEDS_REVIEW'


def default_out():
    home = os.path.expanduser('~')
    desktop = os.path.join(home, 'Desktop')
    if not os.path.isdir(desktop):
        desktop = home
    return os.path.join(desktop, 'Redacted_Outputs', datetime.datetime.now().strftime('run_%Y%m%d_%H%M%S'))


def find_pdfs(inp):
    """[(absolute path, sub-folder relative to the input root)]"""
    if os.path.isfile(inp):
        return [(os.path.abspath(inp), '')]
    inp = os.path.abspath(inp)
    out = []
    for root, dirs, files in os.walk(inp):
        dirs.sort()
        for f in sorted(files):
            if f.lower().endswith('.pdf'):
                rel = os.path.relpath(root, inp)
                out.append((os.path.join(root, f), '' if rel == '.' else rel))
    return out


def git_info():
    def run(*a):
        try:
            return subprocess.run(['git', *a], cwd=HERE, capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception:
            return ''
    return dict(commit=run('rev-parse', 'HEAD'), branch=run('rev-parse', '--abbrev-ref', 'HEAD'),
                tag=run('describe', '--tags', '--exact-match'), dirty=bool(run('status', '--porcelain', '--', '.')))


def module_settings(mod):
    """Upper-case constants of a module (the tuned settings), for the run record."""
    out = {}
    for k, v in vars(mod).items():
        if k.isupper() and isinstance(v, (int, float, str, dict, list, tuple)):
            out[k] = v
    return out


class OCR:
    """DocTR loaded once. page_record() returns the same structure as ocr_cache.py writes."""

    def __init__(self, cpu=False):
        t = time.perf_counter()
        import torch
        from doctr.models import ocr_predictor
        self.import_s = time.perf_counter() - t     # loading the torch / DocTR libraries
        self.torch = torch
        self.device = 'cpu' if cpu or not torch.cuda.is_available() else 'cuda'
        self._sync()
        t = time.perf_counter()
        self.model = ocr_predictor(pretrained=True).to(self.device)
        self._sync()
        self.load_s = time.perf_counter() - t
        self.device_name = torch.cuda.get_device_name(0) if self.device == 'cuda' else platform.processor()

    def _sync(self):
        if self.device == 'cuda':
            self.torch.cuda.synchronize()

    def page_record(self, pix):
        arr = to_rgb(pix)
        H, W = arr.shape[:2]
        self._sync()
        t = time.perf_counter()
        res = self.model([arr])
        self._sync()
        dt = time.perf_counter() - t
        lines = []
        for b, blk in enumerate(res.pages[0].blocks):
            for l, line in enumerate(blk.lines):
                lines.append(dict(block=b, line=l, words=[
                    dict(text=w.value, conf=round(float(w.confidence), 4),
                         box=[round(w.geometry[0][0] * W, 1), round(w.geometry[0][1] * H, 1),
                              round(w.geometry[1][0] * W, 1), round(w.geometry[1][1] * H, 1)])
                    for w in line.words]))
        return dict(width=W, height=H, ocr_s=round(dt, 3), lines=lines), dt


def process(path, sub, ocr, out, save_ocr):
    name = os.path.splitext(os.path.basename(path))[0]
    t_doc = time.perf_counter()
    times = dict(ingest_s=0.0, ocr_s=0.0, detect_s=0.0, redact_s=0.0, verify_s=0.0)
    log = dict(input=path, sub=sub, file=os.path.basename(path), pages=[], status=None, problems=[])

    t = time.perf_counter()
    scans = load_pages(path)
    times['ingest_s'] = time.perf_counter() - t

    boxes, manual_pages = {}, []
    for i, s in enumerate(scans):
        pix = s['pix']
        rec, dt = ocr.page_record(pix)
        times['ocr_s'] += dt
        if save_ocr:
            d = os.path.join(out, 'ocr', sub)
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, f'{name}_p{i + 1:02d}.json'), 'w', encoding='utf-8') as f:
                json.dump(dict(file=os.path.basename(path), sub=sub, page=i + 1, **rec), f)
        t = time.perf_counter()
        anc = find_anchors(rec)
        dets = detect(to_gray(pix), rec, anchors=anc)
        times['detect_s'] += time.perf_counter() - t
        if s['method'] != 'extracted':
            manual_pages.append(i + 1)
        if dets:
            boxes[i] = [d['box'] for d in dets]
        log['pages'].append(dict(
            page=i + 1, width=pix.width, height=pix.height, method=s['method'],
            anchors=[dict(kind=a['kind'], role=a['role'], text=a.get('text'), box=[round(v, 1) for v in a['box']])
                     for a in anc],
            boxes=[dict(box=[int(v) for v in d['box']], ink_px=int(d['ink']),
                        anchor_kind=d['anchor']['kind'], anchor_role=d['anchor']['role']) for d in dets]))

    tmp = os.path.join(out, '_tmp', sub)
    os.makedirs(tmp, exist_ok=True)
    tmp_pdf = os.path.join(tmp, f'{name}.pdf')
    ok, problems, done = False, [], {}
    try:
        t = time.perf_counter()
        done = redact_pdf(path, tmp_pdf, boxes)
        times['redact_s'] = time.perf_counter() - t
        t = time.perf_counter()
        ok, problems, vstats = verify(path, tmp_pdf, done)
        times['verify_s'] = time.perf_counter() - t
        log['verify'] = dict(ok=ok, problems=problems, **{k: v for k, v in vstats.items()})
    except ValueError as e:          # redact.py refuses pages that are not a single full-page scan
        problems = [f'redaction refused: {e}']
        log['verify'] = dict(ok=False, problems=problems)
    if manual_pages:
        problems = problems + [f'page(s) {manual_pages} are not a single full-page scan: redact by hand']
    safe = ok and not manual_pages
    log['status'] = STATUS_SAFE if safe else STATUS_REVIEW
    log['problems'] = problems
    log['redacted_boxes'] = {str(k + 1): v for k, v in done.items()}   # padded pixel boxes, 1-based page keys

    if os.path.exists(tmp_pdf):
        if safe:
            dst = os.path.join(out, 'redacted', sub, f'{name}_redacted.pdf')
        else:
            dst = os.path.join(out, 'needs_review', sub, f'{name}_NOT_SAFE.pdf')
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        os.replace(tmp_pdf, dst)
        log['output'] = dst
    else:
        log['output'] = None
    wall = time.perf_counter() - t_doc

    row = dict(sub=sub, file=os.path.basename(path), status=log['status'], pages=len(scans),
               boxes=sum(len(v) for v in boxes.values()),
               **{k: round(v, 3) for k, v in times.items()}, wall_s=round(wall, 3),
               size_in_kb=os.path.getsize(path) // 1024,
               size_out_kb=os.path.getsize(log['output']) // 1024 if log['output'] else '',
               problems=' | '.join(problems))
    log['timings'] = {k: row[k] for k in ('ingest_s', 'ocr_s', 'detect_s', 'redact_s', 'verify_s', 'wall_s')}
    d = os.path.join(out, 'logs', sub)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, f'{name}.json'), 'w', encoding='utf-8') as f:
        json.dump(log, f, indent=1)
    return row


def main():
    ap = argparse.ArgumentParser(description='Redact full signatures in scanned ordinance PDFs.')
    ap.add_argument('input', help='a PDF file or a folder of PDFs (searched recursively)')
    ap.add_argument('--out', default=None, help='output folder (default: Desktop/Redacted_Outputs/run_<time>)')
    ap.add_argument('--cpu', action='store_true', help='run DocTR on the CPU even if a GPU is available')
    ap.add_argument('--save-ocr', action='store_true', help='also save the live OCR of every page (for checks)')
    ap.add_argument('--limit', type=int, default=0, help='process only the first N PDFs (for quick tests)')
    args = ap.parse_args()

    t_run = time.perf_counter()
    started = datetime.datetime.now().isoformat(timespec='seconds')
    pdfs = find_pdfs(args.input)
    if args.limit:
        pdfs = pdfs[:args.limit]
    if not pdfs:
        sys.exit(f'no PDF found in {args.input}')
    out = os.path.abspath(args.out or default_out())
    if os.path.exists(os.path.join(out, 'timings.csv')):
        sys.exit(f'{out} already holds a run; choose another --out')
    os.makedirs(out, exist_ok=True)

    ocr = OCR(cpu=args.cpu)
    print(f'{len(pdfs)} PDF(s) -> {out}', flush=True)
    print(f'DocTR on {ocr.device} ({ocr.device_name}), library import {ocr.import_s:.1f}s, '
          f'model load {ocr.load_s:.1f}s', flush=True)

    import torch, doctr, cv2
    info = dict(started=started, input=os.path.abspath(args.input), output=out, documents=len(pdfs),
                device=ocr.device, device_name=ocr.device_name, library_import_s=round(ocr.import_s, 3),
                model_load_s=round(ocr.load_s, 3),
                git=git_info(), python=platform.python_version(), platform=platform.platform(),
                versions=dict(torch=torch.__version__, doctr=doctr.__version__, pymupdf=pymupdf.VersionBind,
                              opencv=cv2.__version__, numpy=np.__version__),
                settings=dict(detect=module_settings(detect_mod), anchors=module_settings(anchors_mod),
                              redact=module_settings(redact_mod)))

    fields = ['sub', 'file', 'status', 'pages', 'boxes', 'ingest_s', 'ocr_s', 'detect_s', 'redact_s',
              'verify_s', 'wall_s', 'size_in_kb', 'size_out_kb', 'problems']
    rows = []
    with open(os.path.join(out, 'timings.csv'), 'w', newline='', encoding='utf-8') as fcsv:
        w = csv.DictWriter(fcsv, fieldnames=fields)
        w.writeheader()
        for k, (path, sub) in enumerate(pdfs, 1):
            try:
                row = process(path, sub, ocr, out, args.save_ocr)
            except Exception as e:
                row = dict(sub=sub, file=os.path.basename(path), status=STATUS_REVIEW,
                           problems=f'ERROR {type(e).__name__}: {e}')
                d = os.path.join(out, 'logs', sub)
                os.makedirs(d, exist_ok=True)
                with open(os.path.join(d, os.path.splitext(os.path.basename(path))[0] + '.json'), 'w',
                          encoding='utf-8') as f:
                    json.dump(dict(input=path, status=STATUS_REVIEW, error=traceback.format_exc()), f, indent=1)
            rows.append(row)
            w.writerow(row)
            fcsv.flush()
            flag = '' if row['status'] == STATUS_SAFE else '   <-- NEEDS REVIEW'
            print(f"[{k}/{len(pdfs)}] {os.path.join(sub, row['file'])}: {row.get('pages', '?')} p, "
                  f"{row.get('boxes', '?')} boxes, {row.get('wall_s', '?')}s, {row['status']}{flag}", flush=True)

    try:
        os.removedirs(os.path.join(out, '_tmp'))
    except OSError:
        pass
    total = time.perf_counter() - t_run
    review = [r for r in rows if r['status'] != STATUS_SAFE]
    walls = [r['wall_s'] for r in rows if isinstance(r.get('wall_s'), float)]
    info.update(finished=datetime.datetime.now().isoformat(timespec='seconds'), run_total_s=round(total, 3),
                safe=len(rows) - len(review), needs_review=len(review))
    with open(os.path.join(out, 'run_info.json'), 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=1)
    lines = [f'Run {started}  ->  {out}',
             f'commit {info["git"]["commit"][:7]} ({info["git"]["branch"]}{", uncommitted changes" if info["git"]["dirty"] else ""})',
             f'DocTR on {ocr.device} ({ocr.device_name}); library import {ocr.import_s:.1f}s, model load {ocr.load_s:.1f}s',
             f'documents {len(rows)}: SAFE {len(rows) - len(review)}, NEEDS_REVIEW {len(review)}',
             f'pages {sum(r.get("pages", 0) for r in rows)}, boxes {sum(r.get("boxes", 0) for r in rows)}',
             f'per-document wall time: mean {np.mean(walls):.2f}s, median {np.median(walls):.2f}s, max {max(walls):.2f}s'
             if walls else 'no timings',
             f'run total (incl. model load) {total:.1f}s']
    if review:
        lines.append('\nNEEDS REVIEW (do not publish these outputs as is):')
        lines += [f"  {os.path.join(r['sub'], r['file'])}: {r['problems']}" for r in review]
    text = '\n'.join(lines)
    with open(os.path.join(out, 'summary.txt'), 'w', encoding='utf-8') as f:
        f.write(text + '\n')
    print('\n' + text)


if __name__ == '__main__':
    main()
