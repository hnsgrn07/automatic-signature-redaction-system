# PyInstaller recipe for SignatureRedactor.exe
# Build:  venv\Scripts\pyinstaller.exe --noconfirm SignatureRedactor.spec
#
# The thesis code (scripts/demo, scripts/pipeline) is deliberately NOT frozen: launcher.py loads it at
# run time from <exe folder>/app, so it stays editable. This recipe bundles only what that code imports:
# the libraries below, the DocTR weights (stage/doctr-cache) and the first-run data (stage/seed).
# If the code starts importing a NEW library, add it to HIDDEN, install it in venv, and rebuild.
from PyInstaller.utils.hooks import collect_submodules, collect_data_files

HIDDEN = (
    # third-party libraries used by scripts/demo and scripts/pipeline
    ['numpy', 'cv2', 'pymupdf', 'rapidfuzz', 'rapidfuzz.fuzz', 'rapidfuzz.distance', 'torch', 'torchvision']
    + collect_submodules('doctr') + collect_submodules('rapidfuzz')
    # standard library modules the thesis code imports (not seen by the analysis, since that code is external)
    + ['argparse', 'csv', 'datetime', 'glob', 'hashlib', 'http.server', 'json', 'multiprocessing', 'platform',
       'random', 're', 'shutil', 'subprocess', 'threading', 'traceback', 'urllib.parse', 'urllib.request',
       'uuid', 'webbrowser', 'collections', 'socketserver', 'email', 'html']
)

# torchvision 0.29 names its compiled ops _C_stable.pyd / image_stable.pyd, which the PyInstaller hook misses
# ("operator torchvision::nms does not exist"). Bundle every .pyd/.dll of torchvision explicitly.
import glob, os, torchvision
TV = os.path.dirname(torchvision.__file__)
TV_BINS = [(p, 'torchvision') for ext in ('*.pyd', '*.dll') for p in glob.glob(os.path.join(TV, ext))]

a = Analysis(
    ['launcher.py'],
    binaries=TV_BINS,
    hiddenimports=HIDDEN,
    datas=[('stage/doctr-cache', 'doctr-cache'), ('stage/seed', 'seed')] + collect_data_files('doctr'),
    excludes=['tkinter', 'matplotlib', 'IPython', 'jupyter', 'tensorflow', 'pymupdf4llm', 'pymupdf_layout'],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name='SignatureRedactor', console=True)
coll = COLLECT(exe, a.binaries, a.datas, name='SignatureRedactor')
