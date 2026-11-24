# Live demo

A local web page that runs the full pipeline on one PDF and shows each step: the live stage log, the anchors and signature boxes found, before and after, the automatic verification, and a recovery test against PyMuPDF's usual redaction.

## Using the page (demo v2)

1. **Choose a PDF:** drop it on the *Redact a PDF* area or click it to browse. The area then shows the chosen file (name and size) and takes no other file until you press **Remove**.
2. **Pick the option for digital (typed) PDFs:** *Flatten to images* or *Keep text selectable* (see below). Scanned PDFs are not affected by this choice.
3. Press **Redact**. The run starts and the button shows *Redacting…* until it ends.
4. **This session** lists every document run since the server was started, newest first, with its status (safe, check by eye, NOT SAFE, error) and signature counts. Click one to reopen its results without running it again. The list is kept in memory and is cleared when the server stops; the files of every run stay in `<THESIS_ROOT>\demo\runs\`.

The page no longer lists the 14 demo documents. To show one, drop it from `<THESIS_ROOT>\demo\unseen\` (filled by `get_demo_docs.py`). Its provenance badge still reads "never seen by the system".

Since round v1.1 the demo redacts full signatures **and** staff and margin signatures (stage 3b, `scripts/pipeline/marks.py`), each drawn in its own colour (green full, purple staff, blue margin; staff-name anchors in pink). The recovery test runs on every redacted box; the side-by-side crops are shown for the full signatures only. Start with `--no-marks` to redact full signatures only.

Digital (typed / exported) PDFs are redacted too (`scripts/pipeline/digital.py`): choose **Flatten to images** (default; output is image-only, nothing else kept, strictly verified) or **Keep text selectable** (PyMuPDF removes images, strokes and text under each box; verified less strictly, but the file is searched for surviving copies) in the sidebar. A document with no signature found shows CHECK BY EYE. Test files: `<THESIS_ROOT>\demo\digital_tests\` and `out_of_scope_tests\`.

The usual PyMuPDF method is run twice for the comparison: **(a) on the signature pages only** (the full-signature boxes; the way signatures are usually redacted by hand) and **(b) on exactly the same boxes as this system**. Its leak needs a page left unredacted: those pages still reference the original scans of the redacted pages. When every page of a document has a full signature (for example a one-page ordinance), (a) redacts every page too, so (a) and (b) give the same result and neither leaks; the page says so instead of claiming that (a) leaks. On the 14 demo documents (2026-11-19): full signatures recoverable (a) 43/43, (b) 0/43, this system 0/43 (marks: (b) 0/95, this system 0/95).

## Run it

Use the pipeline environment (Python 3.12 with torch, python-doctr 1.1.0, PyMuPDF, OpenCV, rapidfuzz). If your data is not under `D:\Thesis`, set `THESIS_ROOT` first.

```
set DOCTR_CACHE_DIR=D:\doctr-cache
python scripts\demo\get_demo_docs.py      # once: puts the 14 demo PDFs in <THESIS_ROOT>\demo\unseen
python scripts\demo\demo_server.py         # loads DocTR, then opens http://127.0.0.1:8765
```

On Geron's laptop, `D:\Thesis\demo\start_demo.bat` does the second step. Stop the server with Ctrl+C.

## Offline Windows app

`packaging/` builds the demo as a double-click Windows app that runs with no internet and no Python install (see `packaging/README.md`). The app loads this folder's code at start-up, so edits to `demo_server.py`, `demo_page.html` or `scripts/pipeline/*.py` apply the next time it is started, without a rebuild.

## Notes

- **Demo documents:** 14 Baguio City ordinances, 2 per year from 2020 to 2026, drawn with seed `demo-2026` from the portal PDFs that are **not** in the 399-document corpus. The system has never seen them. `get_demo_docs.py` copies them from `<THESIS_ROOT>\baguio_ordinances` or downloads them from the public portal, and checks each SHA-256.
- **Your own PDF:** drop it on the page. Dev documents are labelled "dev set (rules were tuned on it)". **Sealed test documents are refused.**
- **Outputs:** `<THESIS_ROOT>\demo\runs\<time_id>\` holds the original, redacted and usual-method PDFs, the images shown, and `result.json`.
- **Access:** the server listens on 127.0.0.1 only, so it is reachable from this computer and nowhere else.

## The recovery test

The same boxes are also redacted with PyMuPDF's usual method, `add_redact_annot` followed by `apply_redactions(images=PDF_REDACT_IMAGE_PIXELS)`. Every image object stored in each saved file is then decoded and compared with the original inside each box. A mean grey difference under 5 counts as recoverable.

On these portal PDFs, the usual method blanks the image the page shows. The original scan stays in the file, though, as a hidden image object that is still referenced by the FPDI template, so the signature can be recovered from it. In the rehearsal on 2026-11-17, all 43 signatures in the 14 documents were recoverable after the usual method and none were recoverable after this system. That rehearsal redacted the signature pages only; when every page of a document is redacted, the usual method leaves no hidden copy (see above).
