"""Stage 1: read the scanned pages of an ordinance PDF.

Each page of the portal PDFs is one upright full-page scanned image (no text layer). The image is
EXTRACTED pixel for pixel, exactly as for the annotation images (render_annotation_images.py),
so detection boxes, ground truth and redaction share one pixel grid. Pages that are not a single
upright full-page image are RENDERED at the scan's own pixel density instead.

For every page the mapping from image pixels to PDF points is returned, so stage 4 can place
redactions: x_pt = x0 + x_px * sx, y_pt = y0 + y_px * sy.
"""
import numpy as np
import pymupdf

COVERAGE_MIN = 0.98   # the image must cover at least 98% of the page to be extracted as-is


def _single_upright_image(page):
    if page.rotation != 0:
        return None
    infos = page.get_image_info(xrefs=True)
    if len(infos) != 1 or not infos[0]['xref']:
        return None
    info = infos[0]
    a, b, c, d, _, _ = info['transform']
    if not (a > 0 and d > 0 and abs(b) < 1e-6 and abs(c) < 1e-6):
        return None
    bbox = pymupdf.Rect(info['bbox'])
    if (bbox & page.rect).get_area() < COVERAGE_MIN * page.rect.get_area():
        return None
    return info


def _plain(pix):
    if pix.alpha:
        pix = pymupdf.Pixmap(pix, 0)
    if pix.colorspace and pix.colorspace.n not in (1, 3):
        pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
    return pix


def to_gray(pix):
    """Pixmap -> H x W uint8 grey array."""
    a = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 1:
        return a[:, :, 0].copy()
    return (0.299 * a[:, :, 0] + 0.587 * a[:, :, 1] + 0.114 * a[:, :, 2]).round().astype(np.uint8)


def to_rgb(pix):
    """Pixmap -> H x W x 3 uint8 array (for DocTR)."""
    a = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 1:
        return np.repeat(a, 3, axis=2)
    return a[:, :, :3].copy()


def page_scan(doc, page):
    """dict(pix, method, xref, to_pt=(x0, y0, sx, sy)) for one page."""
    info = _single_upright_image(page)
    if info:
        pix = _plain(pymupdf.Pixmap(doc, info['xref']))
        bbox = pymupdf.Rect(info['bbox'])
        return dict(pix=pix, method='extracted', xref=info['xref'], raw=doc.xref_stream_raw(info['xref']),
                    to_pt=(bbox.x0, bbox.y0, bbox.width / pix.width, bbox.height / pix.height))
    infos = [x for x in page.get_image_info() if x['width'] and pymupdf.Rect(x['bbox']).width > 0]
    if infos:
        big = max(infos, key=lambda x: x['width'] * x['height'])
        zoom = big['width'] / pymupdf.Rect(big['bbox']).width
    else:
        zoom = 150 / 72.0
    pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
    return dict(pix=pix, method='rendered', xref=None, to_pt=(0.0, 0.0, 1 / zoom, 1 / zoom))


def load_pages(pdf_path):
    """List of page scans for a PDF (opened and closed here)."""
    with pymupdf.open(pdf_path) as doc:
        return [page_scan(doc, page) for page in doc]


def px_to_pt(box, to_pt):
    x0, y0, sx, sy = to_pt
    return pymupdf.Rect(x0 + box[0] * sx, y0 + box[1] * sy, x0 + box[2] * sx, y0 + box[3] * sy)
