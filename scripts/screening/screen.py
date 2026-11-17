"""Screen ordinance PDFs: digitally native vs scanned, and scan resolution (DPI)."""
import pymupdf, sys, glob, csv, os, statistics
from collections import Counter

root, out = sys.argv[1], sys.argv[2]
rows = []
for f in sorted(glob.glob(os.path.join(root, "*", "*.pdf"))):
    year = os.path.basename(os.path.dirname(f))
    r = {"year": year, "file": os.path.basename(f), "pages": 0, "producer": "", "page_kinds": "",
         "img_px_min": "", "pdf_dpi_min": "", "img_meta_dpi": "", "phys_dpi_min": "", "paper": "",
         "text_chars": 0, "verdict": "", "notes": ""}
    try:
        d = pymupdf.open(f)
    except Exception as e:
        r["verdict"] = "UNREADABLE"; r["notes"] = str(e); rows.append(r); continue
    r["pages"] = d.page_count; r["producer"] = d.metadata.get("producer") or ""
    kinds, pdf_dpis, phys_dpis, meta, sizes, papers = [], [], [], set(), [], set()
    for p in d:
        area = p.rect.width * p.rect.height
        txt = p.get_text().strip()
        r["text_chars"] += len(txt)
        imgs = p.get_image_info(xrefs=True)
        cover = 0; big = None
        for im in imgs:
            b = pymupdf.Rect(im["bbox"]) & p.rect
            a = b.width * b.height
            cover += a
            if not big or a > big[0]: big = (a, im, b)
        frac = min(cover / area, 1) if area else 0
        if frac >= 0.8:
            kind = "scan+text" if len(txt) > 20 else "scan"
        elif len(txt) > 20:
            kind = "native"
        else:
            kind = "other"
        kinds.append(kind)
        if big and big[2].width > 0:
            im, b = big[1], big[2]
            pdf_dpis.append(min(im["width"] / (b.width / 72), im["height"] / (b.height / 72)))
            sizes.append((im["width"], im["height"]))
            meta.add((im.get("xres"), im.get("yres")))
            # physical paper: match aspect ratio to common PH paper sizes
            ar = im["height"] / im["width"]
            paper = min({"long bond 8.5x13": (8.5, 13), "letter 8.5x11": (8.5, 11), "A4 8.27x11.69": (8.27, 11.69),
                         "legal 8.5x14": (8.5, 14)}.items(), key=lambda kv: abs(kv[1][1] / kv[1][0] - ar))
            papers.add(paper[0])
            phys_dpis.append(min(im["width"] / paper[1][0], im["height"] / paper[1][1]))
    c = Counter(kinds)
    r["page_kinds"] = ";".join(f"{k}:{v}" for k, v in sorted(c.items()))
    if sizes: r["img_px_min"] = "x".join(map(str, min(sizes)))
    if pdf_dpis: r["pdf_dpi_min"] = round(min(pdf_dpis), 1)
    if phys_dpis: r["phys_dpi_min"] = round(min(phys_dpis), 1)
    r["img_meta_dpi"] = ";".join(f"{a}x{b}" for a, b in sorted(meta, key=str))
    r["paper"] = ";".join(sorted(papers))
    reasons = []
    if "native" in c: reasons.append("digitally native page(s)")
    if "other" in c: reasons.append("page(s) with no scan image and no text")
    if phys_dpis and min(phys_dpis) < 150: reasons.append("scan < 150 DPI")
    r["verdict"] = "EXCLUDE: " + ", ".join(reasons) if reasons else "KEEP"
    if c.get("scan+text"): r["notes"] = "scan has a text layer (OCR?)"
    rows.append(r)

with open(out, "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

print("files:", len(rows))
print("verdicts:", Counter(r["verdict"] for r in rows))
print("page kinds:", Counter(k.split(":")[0] for r in rows for k in r["page_kinds"].split(";") if k))
print("producers:", Counter(r["producer"] for r in rows))
print("image sizes (min per doc):", Counter(r["img_px_min"] for r in rows).most_common(8))
print("PDF-implied DPI:", Counter(r["pdf_dpi_min"] for r in rows).most_common(6))
print("physical DPI:", Counter(r["phys_dpi_min"] for r in rows).most_common(8))
print("embedded image DPI metadata:", Counter(r["img_meta_dpi"] for r in rows).most_common(5))
print("paper:", Counter(r["paper"] for r in rows))
print("docs with any text:", sum(1 for r in rows if r["text_chars"] > 0))
for r in rows:
    if r["verdict"] != "KEEP" or r["notes"]: print("  ", r["year"], r["file"], r["verdict"], r["page_kinds"], r["img_px_min"], r["phys_dpi_min"], r["notes"])
