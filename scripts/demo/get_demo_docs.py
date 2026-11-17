"""Put the 14 demo ordinances into <THESIS_ROOT>\\demo\\unseen (THESIS_ROOT defaults to D:\\Thesis).

The list is demo_docs.csv: 2 per year (2020-2026), drawn with seed "demo-2026" from the portal
ordinances that are NOT in the 399-document corpus, so the system has never seen them.
Each file is copied from <THESIS_ROOT>\\baguio_ordinances if it is there, otherwise downloaded from
the public Baguio open data portal. Every file is checked against its SHA-256.

  python get_demo_docs.py
"""
import csv, glob, hashlib, os, shutil, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'pipeline'))
from paths import thesis_path

OUT = thesis_path('demo', 'unseen')
UA = 'Mozilla/5.0 (compatible; ordinance-archiver/1.0)'


def sha256(path):
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = list(csv.DictReader(open(os.path.join(HERE, 'demo_docs.csv'), encoding='utf-8')))
    bad = 0
    for r in rows:
        dst = os.path.join(OUT, r['demo_name'])
        if os.path.exists(dst) and sha256(dst) == r['sha256']:
            print(f"{r['demo_name']}: already there")
            continue
        local = [p for p in glob.glob(thesis_path('baguio_ordinances', r['year'] + '*', r['file']))
                 if sha256(p) == r['sha256']]
        if local:
            shutil.copy2(local[0], dst)
            how = 'copied'
        else:
            req = urllib.request.Request(r['url'], headers={'User-Agent': UA})
            with urllib.request.urlopen(req, timeout=120) as resp, open(dst + '.part', 'wb') as f:
                shutil.copyfileobj(resp, f)
            os.replace(dst + '.part', dst)
            how = 'downloaded'
            time.sleep(0.6)       # portal rate limit: 120 requests per minute
        ok = sha256(dst) == r['sha256']
        bad += not ok
        print(f"{r['demo_name']}: {how}, SHA-256 {'OK' if ok else 'MISMATCH'}")
        if not ok:
            os.remove(dst)
    print(f'{len(rows) - bad}/{len(rows)} demo documents ready in {OUT}')
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
