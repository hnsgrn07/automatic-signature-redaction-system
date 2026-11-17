"""
Render every page of the 399-document benchmark corpus to PNG for ground-truth annotation.

Method (report this in Chapter 3):
  - Input: sample.csv from stratified_sample.py (split, year, file, corpus_path, phys_dpi).
  - Each ordinance page is one full-page scanned image. When the image is placed upright and
    covers the page, it is EXTRACTED pixel for pixel (no resampling), so annotation coordinates
    are the scan's own pixel grid (1275 x 1950 px for 150 DPI long bond).
  - Otherwise (rotated page, several images, partial coverage) the page is RENDERED with PyMuPDF
    at the scan's physical resolution, and the method is recorded per page.
  - Output: images/<split>/<year>/<year>_<stem>_pNN.png, pages.csv (one row per page with size,
    DPI, method, SHA-256), and Label Studio task files for local-file serving.
  - Test images are written read-only and kept in their own folder (sealed set).

Usage:
  python -P -E render_annotation_images.py <corpus_sample_dir> <output_dir> [--split dev|test|all] [--limit N]

  --limit N renders only the first N documents of each split (for a quick check).
"""
import argparse
import csv
import hashlib
import json
import os
import stat
import sys

import pymupdf

COVERAGE_MIN = 0.98   # image must cover >= 98% of the page area to be extracted as-is


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def extractable_image(doc, page):
    """Return the xref of the single upright full-page image, or None if the page must be rendered."""
    if page.rotation != 0:
        return None
    infos = page.get_image_info(xrefs=True)
    if len(infos) != 1 or not infos[0]["xref"]:
        return None
    info = infos[0]
    a, b, c, d, _, _ = info["transform"]
    if not (a > 0 and d > 0 and abs(b) < 1e-6 and abs(c) < 1e-6):
        return None  # flipped, mirrored or rotated placement
    bbox = pymupdf.Rect(info["bbox"]) & page.rect
    if bbox.get_area() < COVERAGE_MIN * page.rect.get_area():
        return None
    return info["xref"]


def to_plain_pixmap(pix):
    """Drop alpha and convert CMYK/other colour spaces to RGB so the PNG is plain grey or RGB."""
    if pix.alpha:
        pix = pymupdf.Pixmap(pix, 0)
    if pix.colorspace and pix.colorspace.n not in (1, 3):
        pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
    return pix


def page_image(doc, page, phys_dpi):
    xref = extractable_image(doc, page)
    if xref:
        return to_plain_pixmap(pymupdf.Pixmap(doc, xref)), "extracted"
    # Fallback: render at the scan's own pixel density, so 1 output pixel = 1 scan pixel.
    # (The portal's PDF page size implies 96 DPI, so phys_dpi / 72 would be wrong here.)
    infos = [x for x in page.get_image_info() if x["width"] and pymupdf.Rect(x["bbox"]).width > 0]
    if infos:
        big = max(infos, key=lambda x: x["width"] * x["height"])
        zoom = big["width"] / pymupdf.Rect(big["bbox"]).width
    else:
        zoom = phys_dpi / 72.0  # no image at all: assume true paper size in points
    return page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False), "rendered"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("corpus_dir", help="folder holding sample.csv and corpus/")
    ap.add_argument("out_dir", help="output folder for images/, pages.csv and task files")
    ap.add_argument("--split", choices=["dev", "test", "all"], default="all")
    ap.add_argument("--limit", type=int, default=0, help="only the first N documents per split (testing)")
    args = ap.parse_args()

    with open(os.path.join(args.corpus_dir, "sample.csv"), encoding="utf-8") as f:
        docs = list(csv.DictReader(f))
    splits = ["dev", "test"] if args.split == "all" else [args.split]

    rows, problems = [], []
    for split in splits:
        todo = [r for r in docs if r["split"] == split]
        if args.limit:
            todo = todo[: args.limit]
        img_root = os.path.join(args.out_dir, "images", split)
        if os.path.exists(img_root) and os.listdir(img_root):
            sys.exit(f"{img_root} already has files; remove it first so old and new images are not mixed.")

        for n, r in enumerate(todo, 1):
            pdf_path = os.path.join(args.corpus_dir, r["corpus_path"])
            if sha256(pdf_path) != r["sha256"]:
                sys.exit(f"SHA-256 mismatch for {pdf_path}: the corpus file changed since sampling.")
            stem = os.path.splitext(r["file"])[0]
            out_dir = os.path.join(img_root, r["year"])
            os.makedirs(out_dir, exist_ok=True)
            phys_dpi = float(r["phys_dpi"])

            with pymupdf.open(pdf_path) as doc:
                if doc.page_count != int(r["pages"]):
                    problems.append(f"{r['year']}/{r['file']}: {doc.page_count} pages, sample.csv says {r['pages']}")
                for i, page in enumerate(doc, 1):
                    pix, method = page_image(doc, page, phys_dpi)
                    name = f"{r['year']}_{stem}_p{i:02d}.png"
                    dest = os.path.join(out_dir, name)
                    pix.save(dest)
                    if split == "test":
                        os.chmod(dest, stat.S_IREAD)
                    rows.append({
                        "split": split, "year": r["year"], "file": r["file"], "page": i,
                        "image": os.path.relpath(dest, args.out_dir).replace("\\", "/"),
                        "width_px": pix.width, "height_px": pix.height,
                        "phys_dpi_doc": phys_dpi, "method": method,
                        "page_w_pt": round(page.rect.width, 3), "page_h_pt": round(page.rect.height, 3),
                        "sha256": sha256(dest),
                    })
            print(f"[{split}] {n}/{len(todo)} {r['year']}/{r['file']}", flush=True)

    # pages.csv (merged with any existing rows from a previous run of the other split)
    pages_csv = os.path.join(args.out_dir, "pages.csv")
    fields = ["split", "year", "file", "page", "image", "width_px", "height_px",
              "phys_dpi_doc", "method", "page_w_pt", "page_h_pt", "sha256"]
    old = []
    if os.path.exists(pages_csv):
        with open(pages_csv, encoding="utf-8") as f:
            old = [x for x in csv.DictReader(f) if x["split"] not in splits]
    all_rows = sorted(old + rows, key=lambda x: (x["split"], int(x["year"]), x["file"], int(x["page"])))
    with open(pages_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_rows)

    # Label Studio tasks (Local files storage; document root = out_dir)
    for split in splits:
        tasks = [{"data": {"image": f"/data/local-files/?d={x['image']}", "split": x["split"],
                           "year": int(x["year"]), "file": x["file"], "page": int(x["page"])}}
                 for x in rows if x["split"] == split]
        with open(os.path.join(args.out_dir, f"labelstudio_tasks_{split}.json"), "w", encoding="utf-8") as f:
            json.dump(tasks, f, indent=1)

    # Summary
    for split in splits:
        sub = [x for x in rows if x["split"] == split]
        sizes = {}
        for x in sub:
            sizes[(x["width_px"], x["height_px"])] = sizes.get((x["width_px"], x["height_px"]), 0) + 1
        rendered = [f"{x['year']}/{x['file']} p{x['page']}" for x in sub if x["method"] == "rendered"]
        print(f"\n{split}: {len({(x['year'], x['file']) for x in sub})} docs, {len(sub)} pages")
        print("  image sizes (px): " + ", ".join(f"{w}x{h} x{c}" for (w, h), c in sorted(sizes.items(), key=lambda t: -t[1])))
        print(f"  rendered instead of extracted: {rendered or 'none'}")
    for p in problems:
        print("WARNING:", p)


if __name__ == "__main__":
    main()
