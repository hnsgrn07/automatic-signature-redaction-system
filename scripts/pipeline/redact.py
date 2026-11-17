"""Stage 4: permanent redaction with PyMuPDF, and a check that it really is permanent.

redact_pdf  - one black redaction per detected box (padded slightly), applied so that the IMAGE
              PIXELS under the box are overwritten (not just covered by a drawn rectangle). The
              result is saved as a fresh file (no incremental update) with unused objects removed,
              so the original scan cannot survive inside the file.
verify      - re-opens the output and checks, page by page:
                1. every padded box is solid black in the actual image pixels;
                2. the pixels outside the boxes still match the input scan (nothing else changed);
                3. the file holds no image objects beyond the ones the pages draw (no orphaned
                   copy of an original scan) and has a single revision (one %%EOF);
                4. same page count and page sizes.
"""
import os

import numpy as np
import pymupdf

from ingest import load_pages, px_to_pt, to_gray

PAD_IN = 0.03            # extra margin around each box, inches (anti-aliased stroke edges)
BLACK_MAX = 8            # a pixel counts as black when its grey level is at most this (JPEG noise)
INSIDE_BLACK_MIN = 0.999 # share of padded-box pixels that must be black
OUTSIDE_DIFF_MAX = 6.0   # mean absolute grey difference allowed outside the boxes (re-encoding)
PAGE_WIDTH_IN = 8.5


def padded(box, pad_px, w, h):
    return [max(0, box[0] - pad_px), max(0, box[1] - pad_px), min(w, box[2] + pad_px), min(h, box[3] + pad_px)]


def _image_users(doc):
    """xref -> number of pages that draw that image (through any form XObject)."""
    users = {}
    for page in doc:
        for info in page.get_image_info(xrefs=True):
            if info['xref']:
                users[info['xref']] = users.get(info['xref'], 0) + 1
    return users


def redact_pdf(pdf_in, pdf_out, boxes_by_page, pad_in=PAD_IN):
    """boxes_by_page: {page_index (0-based): [[x0, y0, x1, y1] in image pixels, ...]}.
    The scan's pixels under each padded box are set to black and the image object in the PDF is
    REPLACED by the edited, losslessly stored image. (PyMuPDF's apply_redactions only drew black
    rectangles over these scans, because they sit inside FPDI template objects: the signature
    stayed in the image data.) Returns {page_index: [padded pixel boxes actually redacted]}."""
    scans = load_pages(pdf_in)
    done = {}
    with pymupdf.open(pdf_in) as doc:
        users = _image_users(doc)
        for i, page in enumerate(doc):
            boxes = boxes_by_page.get(i) or []
            if not boxes:
                continue
            s = scans[i]
            if s['method'] != 'extracted':
                raise ValueError(f'page {i + 1}: not a single full-page scan; cannot redact pixels safely')
            if users.get(s['xref'], 0) != 1:
                raise ValueError(f'page {i + 1}: image xref {s["xref"]} is drawn on {users.get(s["xref"])} pages')
            src = s['pix']
            w, h, n = src.width, src.height, src.n
            ncol = src.colorspace.n if src.colorspace else 1     # colour channels (1 grey, 3 RGB)
            a = np.frombuffer(src.samples, np.uint8).reshape(h, w, n)[:, :, :ncol].copy()
            pad_px = pad_in * w / PAGE_WIDTH_IN
            done[i] = []
            for b in boxes:
                x0, y0, x1, y1 = padded(b, pad_px, w, h)
                x0, y0, x1, y1 = int(np.floor(x0)), int(np.floor(y0)), int(np.ceil(x1)), int(np.ceil(y1))
                a[y0:y1, x0:x1, :] = 0
                done[i].append([x0, y0, x1, y1])
            cs = pymupdf.csGRAY if ncol == 1 else pymupdf.csRGB
            new = pymupdf.Pixmap(cs, w, h, np.ascontiguousarray(a).tobytes(), False)
            page.replace_image(s['xref'], pixmap=new)
        doc.save(pdf_out, garbage=4, deflate=True, clean=True, incremental=False)
    return done


def _image_objects(doc):
    n = 0
    for x in range(1, doc.xref_length()):
        if doc.xref_get_key(x, 'Subtype')[1] == '/Image':
            n += 1
    return n


def verify(pdf_in, pdf_out, redacted):
    """redacted: {page_index: [padded pixel boxes]}. Returns (ok, list of problem strings, stats)."""
    problems, stats = [], dict(boxes=0, pages=0)
    a_pages, b_pages = load_pages(pdf_in), load_pages(pdf_out)
    if len(a_pages) != len(b_pages):
        return False, [f'page count {len(a_pages)} -> {len(b_pages)}'], stats
    with pymupdf.open(pdf_in) as da, pymupdf.open(pdf_out) as db:
        for i, (pa, pb) in enumerate(zip(da, db)):
            if abs(pa.rect.width - pb.rect.width) > 0.01 or abs(pa.rect.height - pb.rect.height) > 0.01:
                problems.append(f'p{i + 1}: page size changed')
        drawn = sum(len(p.get_image_info()) for p in db)
        objs = _image_objects(db)
        if objs > drawn:
            problems.append(f'{objs} image objects in file but pages draw {drawn}: orphaned image data')
    with open(pdf_out, 'rb') as f:
        raw = f.read()
    if raw.count(b'%%EOF') != 1:
        problems.append('more than one revision (%%EOF) in the output file')
    # no fragment of a redacted page's ORIGINAL image stream may remain anywhere in the file.
    # Only fragments UNIQUE to that stream count: JPEGs from the same scanner share byte runs
    # (e.g. encoded blank paper), and unredacted pages keep their own JPEGs legitimately.
    others_by_page = {i: b''.join(p.get('raw') or b'' for j, p in enumerate(a_pages) if j != i) for i in redacted}
    for i in redacted:
        orig = a_pages[i].get('raw')
        if not orig:
            continue
        step = max(1, len(orig) // 200)
        probes = [orig[k:k + 64] for k in range(256, max(257, len(orig) - 64), step)]
        # skip low-information probes (e.g. runs of zero bytes, found in any compressed stream)
        probes = [pr for pr in probes if len(set(pr)) >= 16 and pr not in others_by_page[i]]
        hits = sum(pr in raw for pr in probes)
        if hits:
            problems.append(f'p{i + 1}: {hits}/{len(probes)} unique fragments of the original scan found in the output')
        stats['fragment_probes'] = stats.get('fragment_probes', 0) + len(probes)

    for i, (sa, sb) in enumerate(zip(a_pages, b_pages)):
        ga, gb = to_gray(sa['pix']), to_gray(sb['pix'])
        if ga.shape != gb.shape:
            problems.append(f'p{i + 1}: image size {ga.shape} -> {gb.shape}')
            continue
        mask = np.zeros(ga.shape, bool)
        for box in redacted.get(i, []):
            x0, y0, x1, y1 = (int(round(v)) for v in box)
            inner = gb[y0:y1, x0:x1]
            share = (inner <= BLACK_MAX).mean() if inner.size else 1.0
            if share < INSIDE_BLACK_MIN:
                problems.append(f'p{i + 1}: box {box} only {share:.1%} black')
            mask[y0:y1, x0:x1] = True
            stats['boxes'] += 1
        # outside the boxes (with a 3 px rim excluded) the scan must be unchanged
        grow = np.zeros_like(mask)
        ys, xs = np.nonzero(mask)
        if len(ys):
            for box in redacted.get(i, []):
                x0, y0, x1, y1 = (int(round(v)) for v in box)
                grow[max(0, y0 - 3):y1 + 3, max(0, x0 - 3):x1 + 3] = True
        out = ~grow
        diff = np.abs(ga[out].astype(int) - gb[out].astype(int)).mean() if out.any() else 0.0
        if diff > OUTSIDE_DIFF_MAX:
            problems.append(f'p{i + 1}: outside the boxes the scan changed (mean grey diff {diff:.2f})')
        stats['pages'] += 1
        stats.setdefault('outside_diff_max', 0.0)
        stats['outside_diff_max'] = max(stats['outside_diff_max'], float(diff))
    return not problems, problems, stats
