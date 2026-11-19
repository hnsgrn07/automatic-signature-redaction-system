# Annotation protocol v1.1 — addendum: staff and margin signatures

Adds two labels to protocol v1.0.1. Everything in v1.0.1 stays as it is: the `full_signature` boxes (dev 869, test 372) are not changed and are not redone, and the sealed test file stays sealed.

**Reason:** after the adviser meeting on 2026-11-18, the team dropped "initials and paraphs" as a separate, out-of-scope category. **Every handwritten signature mark on an ordinance is a signature and is redacted.** Round v1.0.1 boxed the full signatures. This round boxes the two kinds that were left out. In the thesis they are all "signatures"; the label only records which kind each box is, so results can be broken down by type.

## 1. What to box

| Label | Key | What it is | Where |
|---|---|---|---|
| `staff_signature` | 1 | Initials or a short signature on, over or right next to a **small printed staff name** that has no label and no title (e.g. ROGER P. LAOLAWI, JONATHAN M. FONTANOS, GRACE B. SALIO, FELISA D. RIOTOC) | Lower part of the signature page; names stacked or side by side in one row |
| `margin_signature` | 2 | Any other handwritten signature mark that is **not** on a printed name or under a label: shortened signatures of the signatories, small initials, paraphs; a mark beside typed reference initials (e.g. "/fvd") on a certification page; a **handwritten signature inside a RECEIVED stamp** (ruling 2026-11-19) | Mostly the bottom strip of body pages (usually 2–3 marks per page, often one small mark in the lower-right corner); sometimes the side margins |

Rules for both:

- **One box per mark.** Box all of that mark's strokes tightly: loops, tails, strokes crossing printed text or table lines. Leave out printed text that has no ink over it.
- **Marks that touch:** still one box per mark. Split where the strokes most likely belong, and add a note.
- **Every page is in the package.** A page with none of these marks is submitted empty, without a box. Never click Skip.

## 2. What NOT to box

- **Full signatures** over CERTIFIED CORRECT, Attested or APPROVED, and other signatures with a printed name **and** a title. They are already in the ground truth as `full_signature`.
- A staff name with no handwriting on it.
- Handwritten **page numbers** and **dates**, date stamps, dry seals, rubber stamps without handwriting, and the background watermark.
- Handwritten corrections or notes written as words inside the body text.

If you are not sure whether a mark counts, or which label it gets, box it and write `unsure: <reason>` in the note.

## 3. Procedure

1. **Mini-pilot first (all three):** package `staff_pilot` (14 dev documents, 30 pages). Export it as `staff_pilot_<name>.json` (`ben` / `geron` / `ferrer`) and send it to Geron. Agreement is checked with `compare_annotations.py`, per label. Small boxes make IoU sensitive, so the pass rule is a pairwise mean IoU of at least **0.70** for each label, with no systematic disagreement about *which* marks to box. If the pilot fails, the rules are fixed before anyone continues.
2. **Full annotation:** start only after the pilot passes. Ben → `staff_ben_dev` (40% of dev, because he is building the pipeline), Ferrer → `staff_ferrer_dev` (60% of dev), Geron → `staff_geron_test` (all test). Export each package as `staff_<name>_<split>.json`.
3. **Test:** the `staff_geron_test` package is never put on the shared drive and never goes to Ben's or Ferrer's laptop. It is annotated on Geron's laptop only. Its ground truth is finalised and **sealed with its own SHA-256 before any code is written to detect staff or margin signatures**. The pipeline is not used to pre-fill boxes on any package.
4. Ground truth: `ground_truth_v11_{dev,test}.{json,csv}` in `D:\Thesis\ground_truth\v11\` (the v1.0.1 files in `final\` do not change). Test: `finalize_v11_test.py` (sealed 2026-11-19, read-only). Dev: `build_v11_dev_gt.py` = pilot consensus + Ben's and Ferrer's exports + logged corrections (`corrections_v11_dev.csv`). Evaluation uses the union of both rounds, with the type (full / staff / margin) kept per box.

## 4. Version log

| Version | Date | Change |
|---|---|---|
| v1.1 | 2026-11-18 | Initials/paraphs are no longer a separate out-of-scope category. New labels `staff_signature` and `margin_signature`. All pages shown; mini-pilot pass rule mean IoU ≥ 0.70 per label. Split: Geron all test, dev Ferrer 60% / Ben 40%. |
| v1.1.1 | 2026-11-19 | Ruling: a handwritten signature inside a RECEIVED stamp = `margin_signature`; a mark beside typed reference initials on a certification page = `margin_signature`. Ground-truth folder corrected to `v11\`. Test v1.1 GT sealed (961 boxes); dev v1.1 GT built (2,634 boxes, 6 logged corrections). |
