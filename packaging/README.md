# Offline Windows app

Builds the live demo (`scripts/demo/demo_server.py`) as `SignatureRedactor.exe`: a folder you can copy to any Windows laptop and start with a double-click. No internet and no Python install are needed. OCR runs on the CPU unless an NVIDIA GPU is present (about 3–5 s per page on a recent laptop CPU).

## How it works

The exe bundles Python 3.12, the libraries (PyTorch, DocTR, PyMuPDF, OpenCV, RapidFuzz) and the DocTR model weights. **The thesis code is not frozen into it.** At start-up `launcher.py` loads `scripts/demo` and `scripts/pipeline` from the `app` folder next to the exe, so the code stays editable:

```
SignatureRedactor\
  SignatureRedactor.exe     double-click to start; a console window shows the log, close it to stop
  _internal\                Python, libraries, DocTR weights (from the build)
  app\                      a clone of this repository (the code that runs)
```

On start the launcher:

- points DocTR at the bundled weights (`DOCTR_CACHE_DIR`), so nothing is downloaded;
- uses `Documents\SignatureRedactor` as `THESIS_ROOT` (outputs in `demo\runs\`), unless `THESIS_ROOT` is already set;
- on first run copies the bundled demo PDFs to `demo\unseen\` and `sample.csv` to `baguio_corpus_sample\` (so sealed test documents are still refused);
- starts the demo server and opens `http://127.0.0.1:8765` (pass `--port N` for another port, e.g. to run two versions side by side).

## Build

Needs Python **3.12** (PyTorch has no wheels for newer versions yet) and about 3 GB free.

```
py -3.12 -m venv venv
venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
venv\Scripts\pip install "python-doctr==1.1.0" pymupdf opencv-python-headless rapidfuzz numpy pyinstaller
```

Put the build inputs next to `launcher.py` and `SignatureRedactor.spec` (they are not in the repository):

```
stage\doctr-cache\        DocTR weights: run once with DOCTR_CACHE_DIR=stage\doctr-cache
                          python -c "from doctr.models import ocr_predictor; ocr_predictor(pretrained=True)"
stage\seed\demo\unseen\   the 14 demo PDFs: THESIS_ROOT=stage\seed python scripts\demo\get_demo_docs.py
stage\seed\baguio_corpus_sample\sample.csv   copy of data\corpus\sample.csv
```

Then:

```
venv\Scripts\pyinstaller --noconfirm SignatureRedactor.spec
```

The result is `dist\SignatureRedactor\`. Clone this repository into `dist\SignatureRedactor\app` and the app is ready (about 1 GB).

## When to rebuild

Only when the code starts importing a library that is not bundled yet. Install it in the venv, add it to `HIDDEN` in `SignatureRedactor.spec`, rebuild, and replace `SignatureRedactor.exe` and `_internal\` (keep `app\`). Changes to the pipeline, the server or the page need no rebuild.

## Known issue fixed in the spec

torchvision 0.29 names its compiled operators `_C_stable.pyd` / `image_stable.pyd`, which the PyInstaller hook misses (`RuntimeError: operator torchvision::nms does not exist`). The spec bundles every `.pyd` / `.dll` of torchvision explicitly.
