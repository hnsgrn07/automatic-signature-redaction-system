# Signature Annotation Protocol

Nov 16, 2026 · @Hans

## 1. Purpose and status

Every annotator draws one box around every **full handwritten signature** on every page of the 399-document benchmark corpus, following the rules below and nothing else.

- **Status:** **v1.0, frozen on 2026-11-16** after pilot r1 passed (Section 8). From now on no rule changes except through the version log (Section 10), and pipeline thresholds are tuned only on dev ground truth made under this version.
- **Annotators:** Ben, Ferrer, Geron.
- **What the boxes are for:** they are the ground truth for the third specific objective. Each system detection is matched to a box at IoU ≥ 0.50 to count true positives, false positives and false negatives, and to compute precision, recall and F1-score. The dev boxes also give the signature size and aspect-ratio statistics for the content analysis.
- **Corpus:** 399 documents, 1,333 pages: dev 280 documents (976 pages) and test 119 documents (357 pages).

## 2. Definitions

The tool has **one label: `full_signature`**. Roles and location decide what counts, not how the ink looks.

| Term | Meaning in this protocol | Boxed? |
| --- | --- | --- |
| Full signature | The complete handwritten signature of an ordinance signatory: the Presiding Officer (Vice Mayor) over CERTIFIED CORRECT, the Secretary to the Sangguniang Panlungsod over Attested, the City Mayor over APPROVED, or any other person who signs in full (e.g. an acting officer, a second approving official) | Yes |
| Initials | Short handwritten marks (paraphs) by reviewing staff: the staff block at the lower left of the signature page above the small printed staff names, and marks at the bottom or margin of body pages. **Coded as initials whatever their size or style.** | No (delimitation l) |
| Printed name or title | Typed text such as a signatory's name or position | No |
| Seal, stamp, watermark | Dry seal, rubber stamp, date stamp (e.g. "JAN 22 2026"), background city seal watermark | No |
| Other handwriting | Handwritten page numbers, dates, corrections, notes | No |

## 3. What gets a box

Box a mark only if **all three** are true:

1. It is **handwritten** ink (not typed, not printed, not stamped).
2. It is a **full signature**, as defined in Section 2.
3. It is **not in the staff initials block** at the lower left of the signature page and not a marginal or bottom-of-page initial.

Never box: initials, printed names and titles, dry seals, rubber stamps, date stamps, the background watermark, handwritten page numbers or notes. The system boxing any of these counts as a **false positive**, so the ground truth must not contain them.

One signature = one box. A signatory who signs twice on the same page (e.g. signed again beside a correction) gets two boxes.

## 4. How to draw the box

Draw the **smallest axis-aligned rectangle that contains every ink stroke of the signature**. Zoom to at least 100% before you place the edges.

- **Loops and tails:** include them, even when they run far past the printed name or into the line above. A tail that is cut off leaves ink visible after redaction.
- **Printed text under the signature:** ignore it when placing the edges. The box follows the ink, so it may cover part of the printed name, or the printed name may stick out of the box. Neither is a problem.
- **Gaps inside the signature:** a signature written in two or more separate pieces (first name and surname apart) is still one box around all the pieces.
- **Occlusion:** if a stamp or seal covers part of the signature, draw the box around the full extent you can see or reasonably infer from the strokes on both sides. Do not shrink it to the visible part only.
- **Faint or broken strokes:** include them if you can see them at 200% zoom. Do not stretch the box for dust or scanner noise.
- **Touching signatures:** if two signatures touch or overlap, draw two boxes, one per signatory. The boxes may overlap.
- **Tolerance:** about 2–3 px of white margin around the ink is fine. Do not pad more than that.

## 5. Hard cases

When a case is not listed here, draw your best box, add a comment (Section 6), and bring it to the pilot discussion. New rulings are added to this table before v1.0.

| Case | Ruling |
| --- | --- |
| Staff mark in the lower-left block looks like a full signature | Initials. Do not box. Note it in the coding sheet ("staff mark resembles full signature"). |
| Signature with no anchor label (e.g. an extra signatory) | Box it. It is a full signature even without CERTIFIED CORRECT, Attested or APPROVED. |
| "For" or "By authority of" signature (someone signing on behalf of an official) | Box it. It is a full signature of the person who signed. |
| Date stamp overlapping the signature | Box the signature as usual. The box may include the stamp; do not cut the signature to avoid it. |
| Dry seal over the signature | Box the full signature extent (occlusion rule, Section 4). |
| Signature crosses into the next signature block | Each signature gets its own box; boxes may overlap. |
| Signature cut off at the page edge | Box up to the page edge. Add a comment "cut at edge". |
| Handwritten correction with initials in the body text | Initials. Do not box. |
| Scribble or crossed-out signature | Box it if it is a full signature attempt by a signatory; add a comment. |
| Page is too faint or damaged to tell | Box what you can see, add a comment "uncertain". It is settled at adjudication. |
| Tiny stray box with no handwriting inside (an accidental click or drag, a few pixels in size) | Delete it before you submit. If one reaches adjudication, it is dropped from the ground truth. (Pilot r1 ruling: 2 such boxes dropped.) |
| Handwritten signature in the "By:" field of a receiving stamp (e.g. a RECEIVED stamp on an attached letter) | Box it. It is a full signature of the person who received the document. (Ruling 2026-11-17.) |

## 6. Empty pages and uncertain cases

**Every page is submitted, including pages with no signature.** In Label Studio, a page with no boxes is submitted as it is, not skipped. A skipped page is missing from the ground truth, and any false positive the system makes on it would go uncounted.

If you are unsure about a mark:

- Draw the box you think is right, then add a comment on the task starting with one of these tags: `uncertain`, `cut at edge`, `occluded`, `possible initials`.
- Do not ask the other annotators during the pilot. Unsure cases are settled together at adjudication (Section 8).
- After v1.0, settle unsure cases with the hard-case table first. If it doesn't cover the case, all three decide together and the case is logged in the version log.

## 7. Images, names and coordinates

Annotators work on PNG page images made by `render_annotation_images.py`, never on the PDFs. Each PNG is the **scanned image stored in the PDF, copied pixel for pixel**: 1275 × 1950 px for 150 DPI pages, larger for the five \~234 DPI documents. The pipeline must read pages at the same size so its boxes and the ground truth share one coordinate system.

| Item | Rule |
| --- | --- |
| File name | `<year>_<file stem>_p<page, 2 digits>.png`, e.g. `2023_045-2023_p03.png` |
| Folders | `images/dev/<year>/` and `images/test/<year>/`, kept apart |
| Page list | `pages.csv`: split, year, file, page, image path, width and height in px, DPI, SHA-256 of the PNG |
| Box format | COCO JSON, `bbox = [x, y, width, height]` in pixels, origin at the top-left corner |
| Label | `full_signature` (COCO category id 1) |
| Back to the PDF | x in points = x in px × page width in points ÷ image width in px (same for y with heights) |

Test images never go on the shared drive. They stay on the one computer used for the test project.

## 8. Workflow

&#91;embedded content: annotation workflow · 6 steps, 1 decision\]

The full annotation starts only after the pilot passes and the protocol is frozen; a failed pilot loops back through a rule fix and a new pilot.

1. **Practice round.** Each annotator boxes the same 3–5 Iteration 1 documents (not in the pilot). Compare together, fix unclear wording, then delete these boxes.
2. **Pilot.** 42 dev documents (6 per year, 10.5% of 399), drawn by script with the seed string `pilot-2026`-r1. Each annotator boxes every page **alone**. No discussion until all three have submitted.
3. **Agreement.** For each pair of annotators, match boxes one-to-one by highest IoU; a pair counts as matched at IoU ≥ 0.50. Report the mean IoU of matched pairs and the number of unmatched boxes. The pilot **passes when every pair has a mean IoU ≥ 0.80**.
4. **If it fails.** Find the rule behind most disagreements, revise it (v0.9.x), and re-pilot on 42 **new** dev documents.
5. **Adjudicate and freeze.** Where two or three annotators matched, the final box is the median of their coordinates. Boxes drawn by one annotator only, and "uncertain" comments, are settled by discussion. Then date the protocol as v1.0.
6. **Annotate the rest.** 238 dev + 119 test documents, annotated by two annotators (Ferrer took part in the pilot only). Ben: 165 dev documents (601 pages). Geron: 73 dev + all 119 test documents (596 pages). Split by script (seed split-2026), balanced by page count, with each annotator covering every year; test pages stay on Geron's laptop.

**Pilot r1 result (2026-11-16): passed.** 42 dev documents, 136 pages, every page submitted by all three annotators. Mean IoU of matched boxes: Ben–Ferrer 0.989, Ben–Geron 0.985, Ferrer–Geron 0.987 (lowest single match 0.870). All three annotators drew the same 129 boxes. The only disagreements were 2 tiny boxes drawn by one annotator with no handwriting inside; the team dropped both (new hard-case ruling, Section 5). The adjudicated pilot ground truth has 129 boxes, and 91 of the 136 pages have no signature.

**Test-set rules.** Test pages are annotated with the same protocol, in a separate Label Studio project on one computer. What annotators notice on test pages is never used to change pipeline rules or thresholds.

## 9. Quality checks, export and sealing

The ground truth is accepted only when every check below passes.

- [ ] **Cross-review:** not done (team decision, 2026-11-17). The planned check was a review of a random 10% of the other annotator's documents, with more than 1 change per 20 boxes triggering a full re-check. Quality of the full annotation rests on the pilot agreement (every pair mean IoU ≥ 0.985) and the automated checks below; single annotation without cross-review is reported as a limitation.
- [ ] **Completeness:** every page in `pages.csv` appears in the export, including empty pages (1,333 images: 976 dev, 357 test).
- [ ] **Validity:** every box lies inside its image, has width and height > 0, and uses label `full_signature`.
- [ ] **Coding-sheet match:** for the 70 Iteration 1 documents, the box count per page equals "Full signatures (#)" in the coding sheet. Every mismatch is resolved and logged.
- [ ] **Export:** `ground_truth_dev.json` and `ground_truth_test.json` in COCO format.
- [ ] **Seal:** record the SHA-256 of `ground_truth_test.json`, commit the protocol, the dev file and the hash to Git, and store the test file read-only. It is opened once, for the single final test run.

## 10. Version log

After v1.0, a rule change creates a new version (v1.1, …), and every page already annotated is re-checked against the changed rule. Report any version after v1.0 in Chapter 3.

| Version | Date | Change |
| --- | --- | --- |
| v0.9 | 2026-11-16 | First draft for the practice round and pilot |
| v1.0 | 2026-11-16 | Frozen after pilot r1 adjudication (all pairs mean IoU ≥ 0.985). Added the hard-case ruling on tiny stray boxes; corrected the pilot seed to pilot-2026-r1 |
| v1.0.1 | 2026-11-17 | Hard-case ruling: a signature in a receiving stamp's "By:" field is boxed. Full annotation done by two annotators (Section 8, step 6). Cross-review not done; recorded as a deviation (Section 9). No change to what gets boxed for the existing annotations. |
