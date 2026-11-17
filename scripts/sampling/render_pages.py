import pymupdf, sys, os
out = sys.argv[1]; os.makedirs(out, exist_ok=True)
for f in sys.argv[2:]:
    d = pymupdf.open(f)
    for p in d:
        p.get_pixmap(dpi=40).save(os.path.join(out, f"{os.path.basename(f)[:-4]}_p{p.number+1}.png"))
    print(os.path.basename(f), d.page_count)
