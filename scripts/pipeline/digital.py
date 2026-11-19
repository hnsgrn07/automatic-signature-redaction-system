"""Stage 4 for DIGITAL pages (typed / exported PDFs that are not a single scanned image).

Detection is unchanged: ingest renders such a page to an image (ingest.page_scan, method 'rendered'), and
stages 2-3b run on that image exactly as on a scan. Only the redaction differs. Two methods:

flatten (default)  The whole document is rebuilt from page images: every page becomes ONE losslessly
                   stored image (scanned pages: the extracted scan; digital pages: the rendered page) with
                   the padded boxes set to black. Nothing else is carried over - no text layer, vector
                   graphics, annotations, form / signature fields, links, attachments, JavaScript or
                   metadata - so no hidden copy of a signature can survive. Cost: text is no longer
                   selectable or searchable, and files get bigger.
keep_text          PyMuPDF's own redaction on the original file: under every padded box, image pixels are
                   blanked, vector paths touching the box and text inside it are removed, and a black box
                   is drawn; annotations and form / signature fields that overlap a box are deleted;
                   metadata and attachments are removed. Text elsewhere stays selectable.

verify_flat / verify_keep check the output; both return (ok, problems, stats) like redact.verify.
keep_text cannot be proven as strictly as flatten (a PDF can hide content in many places), so its
check also searches the saved file for any surviving copy of an original image or image stream that
touched a box - the failure found with PyMuPDF on the portal's FPDI scans.
"""
import hashlib

import numpy as np
import pymupdf

from ingest import load_pages, px_to_pt, to_gray, _plain
from redact import padded, PAD_IN, BLACK_MAX, INSIDE_BLACK_MIN, OUTSIDE_DIFF_MAX, PAGE_WIDTH_IN

KEEP_OUTSIDE_DIFF_MAX = 3.0    # keep_text: mean grey difference allowed outside the boxes (rendered pages)


def is_digital(scans):
    """True when any page is not a single full-page scan."""
    return any(s['method'] != 'extracted' for s in scans)


def _boxes_px(boxes, w, h, pad_in):
    pad_px = pad_in * w / PAGE_WIDTH_IN
    out = []
    for b in boxes:
        x0, y0, x1, y1 = padded(b, pad_px, w, h)
        out.append([int(np.floor(x0)), int(np.floor(y0)), int(np.ceil(x1)), int(np.ceil(y1))])
    return out


def _array(pix):
    pix = _plain(pix)
    ncol = pix.colorspace.n if pix.colorspace else 1
    a = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :ncol].copy()
    return a, ncol


# ---------------------------------------------------------------------------------------------- flatten
def flatten_redact(pdf_in, pdf_out, boxes_by_page, scans=None, pad_in=PAD_IN):
    """Rebuild pdf_in as image-only pages with the padded boxes black. Returns {page: [padded px boxes]}."""
    scans = scans or load_pages(pdf_in)
    done = {}
    with pymupdf.open(pdf_in) as src:
        sizes = [(p.rect.width, p.rect.height) for p in src]
    out = pymupdf.open()
    for i, s in enumerate(scans):
        a, ncol = _array(s['pix'])
        h, w = a.shape[:2]
        bx = _boxes_px(boxes_by_page.get(i) or [], w, h, pad_in)
        for x0, y0, x1, y1 in bx:
            a[y0:y1, x0:x1, :] = 0
        if bx:
            done[i] = bx
        cs = pymupdf.csGRAY if ncol == 1 else pymupdf.csRGB
        pix = pymupdf.Pixmap(cs, w, h, np.ascontiguousarray(a).tobytes(), False)
        pg = out.new_page(width=sizes[i][0], height=sizes[i][1])
        pg.insert_image(pg.rect, pixmap=pix)
    out.set_metadata({})
    out.del_xml_metadata()
    out.save(pdf_out, garbage=4, deflate=True, clean=True, incremental=False)
    out.close()
    return done


def _file_checks(db, raw, problems):
    if raw.count(b'%%EOF') != 1:
        problems.append('more than one revision (%%EOF) in the output file')
    if db.embfile_count():
        problems.append(f'{db.embfile_count()} attachment(s) left in the file')
    meta = {k: v for k, v in (db.metadata or {}).items() if v and k not in ('format', 'producer', 'creator',
                                                                          'creationDate', 'modDate', 'encryption')}
    if meta:
        problems.append(f'metadata left: {sorted(meta)}')
    if db.get_xml_metadata():
        problems.append('XMP metadata left')


def _chars_inside(page, rects):
    """Characters of the text layer whose centre lies inside any rect: [(char, bbox)]."""
    out = []
    for r in rects:
        raw = page.get_text('rawdict', clip=r)
        for bl in raw['blocks']:
            for ln in bl.get('lines', []):
                for sp in ln['spans']:
                    for c in sp['chars']:
                        bb = pymupdf.Rect(c['bbox'])
                        if c['c'].strip() and r.contains(pymupdf.Point((bb.x0 + bb.x1) / 2, (bb.y0 + bb.y1) / 2)):
                            out.append((c['c'], bb))
    return out


def _page_chars(page):
    """All non-blank characters in reading order: [(char, centre point)]."""
    out = []
    for bl in page.get_text('rawdict')['blocks']:
        for ln in bl.get('lines', []):
            for sp in ln['spans']:
                for c in sp['chars']:
                    if c['c'].strip():
                        bb = c['bbox']
                        out.append((c['c'], pymupdf.Point((bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2)))
    return out


def _text_check(pa, pb, rects):
    """Did any character that was INSIDE a box survive? PyMuPDF shifts the rest of a line left when it removes
    characters from its middle, so a later character can slide into a box: compare sequences, not positions.
    Returns (survivors, max shift in points of the remaining text)."""
    orig = _page_chars(pa)
    expected = [(c, pt) for c, pt in orig if not any(r.contains(pt) for r in rects)]
    after = _page_chars(pb)
    k, shift, extra = 0, 0.0, []
    for c, pt in after:                       # greedy: after must be a subsequence of expected
        while k < len(expected) and expected[k][0] != c:
            k += 1
        if k == len(expected):
            extra.append(c)
            continue
        shift = max(shift, abs(expected[k][1].x - pt.x), abs(expected[k][1].y - pt.y))
        k += 1
    return extra, shift


def _image_fingerprints(doc):
    """{sha256 of decoded pixels} for every image object in a document."""
    out = set()
    for x in range(1, doc.xref_length()):
        if doc.xref_get_key(x, 'Subtype')[1] != '/Image':
            continue
        try:
            pix = _plain(pymupdf.Pixmap(doc, x))
            out.add(hashlib.sha256(pix.samples).hexdigest())
        except Exception:
            continue
    return out


def _touching_images(da, scans, redacted):
    """Image objects of the ORIGINAL file drawn under a redacted box: {xref: (pixel hash, raw stream)}."""
    out = {}
    for i, boxes in redacted.items():
        page = da[i]
        rects = [px_to_pt(b, scans[i]['to_pt']) for b in boxes]
        for info in page.get_image_info(xrefs=True):
            if info['xref'] and any(pymupdf.Rect(info['bbox']).intersects(r) for r in rects):
                x = info['xref']
                if x not in out:
                    try:
                        pix = _plain(pymupdf.Pixmap(da, x))
                        out[x] = (hashlib.sha256(pix.samples).hexdigest(), da.xref_stream_raw(x))
                    except Exception:
                        pass
    return out


def _fragment_hits(orig_raw, others, raw):
    if not orig_raw or len(orig_raw) < 400:
        return 0, 0
    step = max(1, len(orig_raw) // 200)
    probes = [orig_raw[k:k + 64] for k in range(128, max(129, len(orig_raw) - 64), step)]
    probes = [pr for pr in probes if len(set(pr)) >= 16 and pr not in others]
    return sum(pr in raw for pr in probes), len(probes)


def verify_flat(pdf_in, pdf_out, redacted, scans=None):
    """Output must be image-only: per page exactly one full-page image and nothing else; boxes black;
    pixels outside the boxes identical to the page image the boxes were drawn on; no image that touched a
    box survives; no text, annotations, fields, links, attachments or metadata."""
    problems, stats = [], dict(boxes=0, pages=0, method='flatten')
    scans = scans or load_pages(pdf_in)
    with pymupdf.open(pdf_in) as da, pymupdf.open(pdf_out) as db:
        if len(da) != len(db):
            return False, [f'page count {len(da)} -> {len(db)}'], stats
        raw = open(pdf_out, 'rb').read()
        _file_checks(db, raw, problems)
        touching = _touching_images(da, scans, redacted)
        out_fp = _image_fingerprints(db)
        for x, (fp, sraw) in touching.items():
            if fp in out_fp:
                problems.append(f'original image {x} (under a box) is still in the file')
            hits, n = _fragment_hits(sraw, b'', raw)
            stats['fragment_probes'] = stats.get('fragment_probes', 0) + n
            if hits:
                problems.append(f'{hits}/{n} fragments of original image {x} found in the output')
        objs = sum(1 for x in range(1, db.xref_length()) if db.xref_get_key(x, 'Subtype')[1] == '/Image')
        if objs != len(db):
            problems.append(f'{objs} image objects for {len(db)} pages')
        for i, (pa, pb) in enumerate(zip(da, db)):
            if abs(pa.rect.width - pb.rect.width) > 0.01 or abs(pa.rect.height - pb.rect.height) > 0.01:
                problems.append(f'p{i + 1}: page size changed')
            if pb.get_text().strip():
                problems.append(f'p{i + 1}: text layer present')
            if list(pb.annots()) or list(pb.widgets()) or pb.get_links():
                problems.append(f'p{i + 1}: annotations, fields or links present')
            infos = pb.get_image_info(xrefs=True)
            if len(infos) != 1 or (pymupdf.Rect(infos[0]['bbox']) & pb.rect).get_area() < 0.999 * pb.rect.get_area():
                problems.append(f'p{i + 1}: not exactly one full-page image')
                continue
            ref, _ = _array(scans[i]['pix'])
            got, _ = _array(pymupdf.Pixmap(db, infos[0]['xref']))
            if ref.shape != got.shape:
                problems.append(f'p{i + 1}: image size {ref.shape} -> {got.shape}')
                continue
            gr, gg = ref.mean(axis=2), got.mean(axis=2)
            mask = np.zeros(gr.shape, bool)
            for x0, y0, x1, y1 in redacted.get(i, []):
                inner = gg[y0:y1, x0:x1]
                share = (inner <= BLACK_MAX).mean() if inner.size else 1.0
                if share < INSIDE_BLACK_MIN:
                    problems.append(f'p{i + 1}: box {[x0, y0, x1, y1]} only {share:.1%} black')
                mask[y0:y1, x0:x1] = True
                stats['boxes'] += 1
            diff = float(np.abs(gr[~mask] - gg[~mask]).mean()) if (~mask).any() else 0.0
            if diff > 0.5:     # lossless storage: outside the boxes the pixels must be identical
                problems.append(f'p{i + 1}: outside the boxes the page changed (mean diff {diff:.2f})')
            stats['pages'] += 1
            stats['outside_diff_max'] = max(stats.get('outside_diff_max', 0.0), diff)
    return not problems, problems, stats


# -------------------------------------------------------------------------------------------- keep text
def keep_text_redact(pdf_in, pdf_out, boxes_by_page, scans=None, pad_in=PAD_IN):
    """PyMuPDF redaction on the original file (see module doc). Returns {page: [padded px boxes]}."""
    scans = scans or load_pages(pdf_in)
    done = {}
    with pymupdf.open(pdf_in) as doc:
        for i, page in enumerate(doc):
            boxes = boxes_by_page.get(i) or []
            if not boxes:
                continue
            s = scans[i]
            w, h = s['pix'].width, s['pix'].height
            bx = _boxes_px(boxes, w, h, pad_in)
            rects = [px_to_pt(b, s['to_pt']) for b in bx]
            for wdg in list(page.widgets()):
                if any(wdg.rect.intersects(r) for r in rects):
                    page.delete_widget(wdg)
            for an in list(page.annots()):
                if any(an.rect.intersects(r) for r in rects):
                    page.delete_annot(an)
            for r in rects:
                page.add_redact_annot(r, fill=(0, 0, 0))
            for _, bb in _chars_inside(page, rects):     # PyMuPDF keeps some characters whose centre is in a box
                page.add_redact_annot(bb, fill=(0, 0, 0))
            page.apply_redactions(images=pymupdf.PDF_REDACT_IMAGE_PIXELS,
                                  graphics=pymupdf.PDF_REDACT_LINE_ART_REMOVE_IF_TOUCHED,
                                  text=pymupdf.PDF_REDACT_TEXT_REMOVE)
            done[i] = bx
        for name in list(doc.embfile_names()):
            doc.embfile_del(name)
        doc.set_metadata({})
        doc.del_xml_metadata()
        doc.save(pdf_out, garbage=4, deflate=True, clean=True, incremental=False)
    return done


def verify_keep(pdf_in, pdf_out, redacted, scans=None):
    """Rendered output: boxes black, everything else (nearly) unchanged; no text, annotation or field left
    in a box; no original image that touched a box survives anywhere in the file (pixels or stream)."""
    problems, stats = [], dict(boxes=0, pages=0, method='keep_text')
    scans = scans or load_pages(pdf_in)
    with pymupdf.open(pdf_in) as da, pymupdf.open(pdf_out) as db:
        if len(da) != len(db):
            return False, [f'page count {len(da)} -> {len(db)}'], stats
        raw = open(pdf_out, 'rb').read()
        _file_checks(db, raw, problems)
        touching = _touching_images(da, scans, redacted)
        out_fp = _image_fingerprints(db)
        for x, (fp, sraw) in touching.items():
            if fp in out_fp:
                problems.append(f'original image {x} (under a box) is still in the file - hidden copy')
            hits, n = _fragment_hits(sraw, b'', raw)
            stats['fragment_probes'] = stats.get('fragment_probes', 0) + n
            if hits:
                problems.append(f'{hits}/{n} fragments of original image {x} found in the output')
        for i, (pa, pb) in enumerate(zip(da, db)):
            if abs(pa.rect.width - pb.rect.width) > 0.01 or abs(pa.rect.height - pb.rect.height) > 0.01:
                problems.append(f'p{i + 1}: page size changed')
            s = scans[i]
            zoom = 1 / s['to_pt'][2]
            ga = to_gray(pa.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False))
            gb = to_gray(pb.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False))
            if ga.shape != gb.shape:
                problems.append(f'p{i + 1}: rendered size {ga.shape} -> {gb.shape}')
                continue
            grow = np.zeros(ga.shape, bool)
            for b in redacted.get(i, []):
                x0, y0, x1, y1 = b
                r = px_to_pt(b, s['to_pt'])
                inner = gb[y0 + 1:y1 - 1, x0 + 1:x1 - 1]      # 1 px rim: anti-aliased edge of the drawn box
                share = (inner <= BLACK_MAX).mean() if inner.size else 1.0
                if share < INSIDE_BLACK_MIN:
                    problems.append(f'p{i + 1}: box {b} only {share:.1%} black when rendered')
                if any(a.rect.intersects(r) for a in pb.annots()) or any(w.rect.intersects(r) for w in pb.widgets()):
                    problems.append(f'p{i + 1}: annotation or field left on box {b}')
                grow[max(0, y0 - 3):y1 + 3, max(0, x0 - 3):x1 + 3] = True
                stats['boxes'] += 1
            if redacted.get(i):
                extra, shift = _text_check(pa, pb, [px_to_pt(b, s['to_pt']) for b in redacted[i]])
                if extra:
                    problems.append(f'p{i + 1}: text that was inside a box survived: {"".join(extra)!r}')
                if shift > 0.5:
                    stats.setdefault('notices', []).append(
                        f'p{i + 1}: text after a removed part moved by up to {shift:.1f} pt (PyMuPDF re-flows the line)')
            out = ~grow
            diff = float(np.abs(ga[out].astype(int) - gb[out].astype(int)).mean()) if out.any() else 0.0
            if diff > KEEP_OUTSIDE_DIFF_MAX:
                problems.append(f'p{i + 1}: outside the boxes the page changed (mean grey diff {diff:.2f})')
            stats['pages'] += 1
            stats['outside_diff_max'] = max(stats.get('outside_diff_max', 0.0), diff)
    return not problems, problems, stats


def redact_digital(pdf_in, pdf_out, boxes_by_page, mode='flatten', scans=None):
    """-> (done, ok, problems, stats)."""
    scans = scans or load_pages(pdf_in)
    if mode == 'keep_text':
        done = keep_text_redact(pdf_in, pdf_out, boxes_by_page, scans)
        ok, problems, stats = verify_keep(pdf_in, pdf_out, done, scans)
    else:
        done = flatten_redact(pdf_in, pdf_out, boxes_by_page, scans)
        ok, problems, stats = verify_flat(pdf_in, pdf_out, done, scans)
    return done, ok, problems, stats
