# automatic-signature-redaction-system

Proof of concept for Thesis Writing 1: **Automatic Redaction of Signatures on Public LGU Documents to Prevent Forgery Using OCR and Computer Vision** (University of the Cordilleras, BSCS).

The system finds handwritten signatures (full, staff and margin signatures) on scanned Baguio City ordinances and removes them permanently: DocTR anchor detection, OpenCV contour analysis, and PyMuPDF redaction with automatic verification. This repository holds the data preparation and ground truth, the pipeline, a batch tool, and a live demo that can be run as an offline Windows app.

## Run the demo

- **From Python:** `python scripts\demo\demo_server.py` opens a local page at `http://127.0.0.1:8765`. Choose a PDF, pick the option for digital PDFs, press **Redact**; earlier runs of the session stay listed under *This session*. Details in [`scripts/demo/README.md`](scripts/demo/README.md).
- **As an offline app:** `packaging/` builds `SignatureRedactor.exe`, which needs no internet or Python install and still runs the editable code in this repository. See [`packaging/README.md`](packaging/README.md).
- **Batch:** `python scripts\pipeline\redact_cli.py INPUT` redacts one PDF or a whole folder and writes a log per document.

## Data

- **Source:** City Ordinances of Baguio City, Series of 2020–2026, from the city's open data portal (https://data.baguio.gov.ph). There are 726 PDFs. Every page is a scanned image with no text layer, mostly at 150 DPI.
- **Benchmark corpus:** 399 documents, 57 per year, drawn with seed `2026`. Each year is split 70/30 into **dev** (280 documents, 976 pages) and **test** (119 documents, 357 pages).
- **Not in this repo:** the PDFs and page images, which are large and available from the portal. The scripts and lists here rebuild them.

## Ground truth

| Split | Documents | Pages | Signature boxes | Files |
|---|---|---|---|---|
| Dev | 280 | 976 | 869 | `data/ground_truth/final/ground_truth_dev.{json,csv}` (COCO) |
| Test | 119 | 357 | sealed | SHA-256 only, in `data/ground_truth/final/SHA256SUMS.txt` |

The **test ground truth is sealed and deliberately not in this repository.** It is opened once, for the single final evaluation run. Its hash was committed here before any pipeline tuning, so anyone can check that the file used in that run has not changed:

```
a9ecaed6c23c6c47fb1d9a700e7aa9a2488d7e2022fe023dbabe08ee4664cfb5  ground_truth_test.json
```

Annotation followed the [annotation protocol](docs/annotation_protocol_v1.0.1.md):
- **Pilot:** 42 dev documents, annotated independently by all three annotators. Pairwise mean IoU was at least 0.985, and the adjudicated median boxes are kept as ground truth.
- **Full annotation:** the remaining 357 documents, one annotator each.

Boxes are in image pixels of the embedded scan, origin top-left. Files under `data/` are stored byte-for-byte (`.gitattributes`), so their SHA-256 values match `SHA256SUMS.txt`.

**Note:** filenames are not unique across years. The 2024 folder contains `068-2025.pdf` and `082-2025.pdf` (misnamed 2024 ordinances), so always identify a page by **(year, file, page)**.

## Layout

| Path | Contents |
|---|---|
| `scripts/scrape_ordinances.py` | Download the ordinance PDFs from the portal API |
| `scripts/screening/` | DPI and text-layer screening |
| `scripts/sampling/` | Stratified sample, 70/30 split, Iteration 1 sample |
| `scripts/annotation/` | Page-image extraction, pilot draw, agreement, adjudication, split, export checks, final ground truth |
| `scripts/pipeline/` | The pipeline: ingest, OCR anchors, full-signature detection, staff/margin signatures, redaction (scanned and digital PDFs), verification, batch CLI and evaluation scripts |
| `scripts/demo/` | Live demo: local web server and page (choose a PDF, Redact, session history, recovery comparison) |
| `packaging/` | Builds the demo as an offline Windows app (`SignatureRedactor.exe`) with PyInstaller |
| `labelstudio/` | Label Studio labelling config and start script |
| `data/corpus/` | Download manifest, screening report, sample with split and SHA-256 |
| `data/annotation/` | Page list, pilot and split assignments, raw dev Label Studio exports, corrections |
| `data/ground_truth/` | Pilot r1 adjudicated ground truth, and the final dev ground truth with its report and hashes |
| `docs/` | Annotation protocol |

The scripts use absolute paths on the annotation laptop (`D:\Thesis\...`). Adjust them before running elsewhere.
