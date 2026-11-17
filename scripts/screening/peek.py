import fitz, sys, glob, random
files = sorted(glob.glob(sys.argv[1] + "/*/*.pdf"))
random.seed(1)
for f in random.sample(files, 8):
    d = fitz.open(f)
    print("==", f.split("baguio_ordinances")[1], "pages", d.page_count, d.metadata.get("producer"), "|", d.metadata.get("creator"))
    for p in list(d)[:3]:
        imgs = p.get_image_info()
        txt = p.get_text().strip()
        pr = p.rect
        info = [(i["width"], i["height"], round(i["bbox"][2]-i["bbox"][0]), round(i["bbox"][3]-i["bbox"][1])) for i in imgs]
        print("  page", p.number, "size", round(pr.width), round(pr.height), "imgs", info[:4], "textlen", len(txt))
