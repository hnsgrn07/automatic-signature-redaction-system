"""Offline launcher for the signature redaction demo (scripts/demo/demo_server.py).

Packaged with PyInstaller. The exe carries only Python, the libraries and the DocTR weights.
The thesis code is NOT frozen: it is loaded at every start from the editable folder
    <exe folder>/app/scripts/demo      (demo_server.py, demo_page.html)
    <exe folder>/app/scripts/pipeline  (ingest, anchors, detect, marks, redact, digital, ...)
so edits there take effect the next time the exe is started (no rebuild). `app` is a git clone
of the thesis repository. A rebuild is only needed when the code starts using a NEW library.

At start-up it also:
  - points DocTR at the bundled weights (no download, works offline);
  - keeps data in Documents/SignatureRedactor (THESIS_ROOT): demo/unseen, demo/uploads,
    demo/runs (outputs), baguio_corpus_sample/sample.csv (sealed-test refusal list);
  - on first run copies the bundled demo PDFs and sample.csv there.
"""
import os, shutil, sys

FROZEN = getattr(sys, 'frozen', False)
BUNDLE = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
EXE_DIR = os.path.dirname(sys.executable) if FROZEN else os.path.dirname(os.path.abspath(__file__))
APP = os.environ.get('REDACTOR_APP_DIR', os.path.join(EXE_DIR, 'app'))
DATA = os.path.join(os.path.expanduser('~'), 'Documents', 'SignatureRedactor')

os.environ['DOCTR_CACHE_DIR'] = os.path.join(BUNDLE, 'doctr-cache')
os.environ.setdefault('THESIS_ROOT', DATA)
os.environ.setdefault('DOCTR_MULTIPROCESSING_DISABLE', 'TRUE')
DATA = os.environ['THESIS_ROOT']

# the editable source goes FIRST on the path, ahead of anything frozen into the exe
for sub in ('pipeline', 'demo'):
    sys.path.insert(0, os.path.join(APP, 'scripts', sub))


def first_run_copy():
    seed = os.path.join(BUNDLE, 'seed')
    for rel in (('demo', 'unseen'), ('baguio_corpus_sample',)):
        src, dst = os.path.join(seed, *rel), os.path.join(DATA, *rel)
        if not os.path.isdir(src):
            continue
        os.makedirs(dst, exist_ok=True)
        for f in os.listdir(src):
            if not os.path.exists(os.path.join(dst, f)):
                shutil.copy2(os.path.join(src, f), os.path.join(dst, f))


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    print('Automatic Signature Redaction System (offline)')
    if not os.path.isfile(os.path.join(APP, 'scripts', 'demo', 'demo_server.py')):
        print(f'ERROR: the app code is missing. Expected it in: {APP}')
        input('Press Enter to close.')
        sys.exit(1)
    print(f'Code:             {APP}')
    print(f'Data and outputs: {DATA}')
    first_run_copy()
    try:
        import demo_server
        print(f'Loaded {demo_server.__file__}')
        print('Keep this window open while you use the app. Close it to stop.', flush=True)
        demo_server.main()
    except Exception:
        import traceback
        traceback.print_exc()
        input('\nThe app stopped with an error (above). Press Enter to close.')
