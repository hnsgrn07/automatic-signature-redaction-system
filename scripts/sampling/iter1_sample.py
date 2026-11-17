"""Iteration 1 characterization sample: 10 dev documents per year (70 total), seed 2026. Dev set only."""
import csv, random, sys, os
import pymupdf
base = sys.argv[1]
rows = [r for r in csv.DictReader(open(os.path.join(base, "sample.csv"), encoding="utf-8")) if r["split"] == "dev"]
rng = random.Random("iter1-2026")
out = []
for y in range(2020, 2027):
    files = sorted(r["file"] for r in rows if int(r["year"]) == y)
    for f in sorted(rng.sample(files, 10)):
        path = os.path.join(base, "corpus", "dev", str(y), f)
        n = pymupdf.open(path).page_count
        out.append({"year": y, "file": f, "pages": n})
with open(os.path.join(base, "iter1_sample.csv"), "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=["year", "file", "pages"]); w.writeheader(); w.writerows(out)
print(len(out), "docs,", sum(o["pages"] for o in out), "pages")
for o in out: print(o["year"], o["file"], o["pages"])
