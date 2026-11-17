# Live demo

A local web page that runs the full pipeline on one PDF and shows each step: the live stage log, the anchors and signature boxes found, before and after, the automatic verification, and a recovery test against PyMuPDF's usual redaction.

## Run it

Use the pipeline environment (Python 3.12 with torch, python-doctr 1.1.0, PyMuPDF, OpenCV, rapidfuzz). If your data is not under `D:\Thesis`, set `THESIS_ROOT` first.

```
set DOCTR_CACHE_DIR=D:\doctr-cache
python scripts\demo\get_demo_docs.py      # once: puts the 14 demo PDFs in <THESIS_ROOT>\demo\unseen
python scripts\demo\demo_server.py         # loads DocTR, then opens http://127.0.0.1:8765
```

On Geron's laptop, `D:\Thesis\demo\start_demo.bat` does the second step. Stop the server with Ctrl+C.

- **Demo documents:** 14 Baguio City ordinances, 2 per year from 2020 to 2026, drawn with seed `demo-2026` from the portal PDFs that are **not** in the 399-document corpus. The system has never seen them. `get_demo_docs.py` copies them from `<THESIS_ROOT>\baguio_ordinances` or downloads them from the public portal, and checks each SHA-256.
- **Your own PDF:** drop it on the page. Dev documents are labelled "dev set (rules were tuned on it)". **Sealed test documents are refused.**
- **Outputs:** `<THESIS_ROOT>\demo\runs\<time_id>\` holds the original, redacted and usual-method PDFs, the images shown, and `result.json`.
- **Access:** the server listens on 127.0.0.1 only, so it is reachable from this computer and nowhere else.

## The recovery test

The same boxes are also redacted with PyMuPDF's usual method, `add_redact_annot` followed by `apply_redactions(images=PDF_REDACT_IMAGE_PIXELS)`. Every image object stored in each saved file is then decoded and compared with the original inside each box. A mean grey difference under 5 counts as recoverable.

On these portal PDFs, the usual method blanks the image the page shows. The original scan stays in the file, though, as a hidden image object that is still referenced by the FPDI template, so the signature can be recovered from it. In the rehearsal on 2026-11-17, all 43 signatures in the 14 documents were recoverable after the usual method and none were recoverable after this system.
