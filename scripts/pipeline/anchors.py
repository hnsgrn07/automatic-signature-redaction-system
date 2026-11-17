"""Stage 2: find signature anchors on a page from DocTR words.

Two anchor families
  label  - the printed labels CERTIFIED CORRECT / Attested / APPROVED, the label alone on its line.
           Matching tolerates OCR errors (edit distance on letters only), because signature
           strokes often run through the label.
  block  - a signatory block: a short title line ("City Mayor", "Secretary to the Sanggunian",
           "City Treasurer", ...) directly under an all-caps printed-name line. The signature is
           written over the name line. Covers pages with no labels (certifications, finance forms).

Input is the page record written by ocr_cache.py (or the same structure built live).
Every distance is in inches (page width = 8.5 in), so ~150 and ~234 DPI scans behave the same.
"""
import re
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

PAGE_WIDTH_IN = 8.5

# label -> (letters-only target, max edit distance)
LABELS = {
    'CERTIFIED CORRECT': ('CERTIFIEDCORRECT', 3),
    'Attested': ('ATTESTED', 3),   # often misread: Allested, Allcsted, Attcslcd
    'APPROVED': ('APPROVED', 1),
}
LABEL_EXTRA_LETTERS = 3          # a label line may carry at most this many other letters

# official-title vocabulary for the title line under a printed name (letters only, upper case).
# Matched fuzzily (partial ratio), because signature strokes often garble the title.
ROLES = [
    ('presiding_officer', ('VICEMAYOR', 'PRESIDINGOFFICER')),
    ('secretary', ('SECRETARY', 'SANGGUNIAN')),
    ('mayor', ('CITYMAYOR',)),
    ('other_official', ('TREASURER', 'ACCOUNTANT', 'ASSESSOR', 'BUDGETOFFICER', 'PLANNING', 'DEVELOPMENTOFFICER',
                        'CITYENGINEER', 'PUBLICSERVANT', 'OFFICERINCHARGE')),
]
TITLE_MIN_SCORE = 80             # rapidfuzz partial_ratio, 0-100
TITLE_MAX_WORDS = 7              # a title line is short
NAME_MIN_WORDS = 2               # a printed name has at least two words
# words that occur in headings and table headers but not in signatory names
NAME_STOPWORDS = {'OF', 'THE', 'AND', 'FOR', 'WITH', 'ORDINANCE', 'SECTION', 'CHAPTER',
                  'AMOUNT', 'OFFICE', 'APPROPRIATIONS', 'REMARKS', 'PRESENT', 'ABSENT', 'XXX'}
NAME_MIN_UPPER = 0.75            # share of upper-case letters in a printed-name line
NAME_MIN_LETTERS = 6
NAME_MAX_WORDS = 6
NAME_TITLE_GAP_IN = 0.35         # name line bottom to title line top, at most
NAME_TITLE_OVERLAP_IN = 0.15     # ink makes OCR boxes taller: name and title boxes may overlap this much
ROW_JOIN_GAP_IN = 1.5            # a name split into pieces on one row (ink in between) is re-joined
NAME_TITLE_X_OVERLAP = 0.3       # horizontal overlap of name and title (share of the shorter)


def letters(s):
    return re.sub(r'[^A-Za-z]', '', s)


def union(boxes):
    return [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]


def page_lines(rec):
    """Lines as dicts with text, box, words, sorted top to bottom."""
    out = []
    for ln in rec['lines']:
        ws = ln['words']
        if not ws:
            continue
        out.append(dict(text=' '.join(w['text'] for w in ws), box=union([w['box'] for w in ws]),
                        words=ws, conf=min(w['conf'] for w in ws)))
    out.sort(key=lambda l: (l['box'][1], l['box'][0]))
    return out


def find_labels(lines):
    found = []
    for ln in lines:
        up = letters(ln['text']).upper()
        for label, (target, kmax) in LABELS.items():
            if len(up) < len(target) - kmax or len(up) > len(target) + LABEL_EXTRA_LETTERS:
                continue
            best = min(Levenshtein.distance(up[:n], target)
                       for n in range(max(1, len(target) - kmax), min(len(up), len(target) + kmax) + 1))
            if best <= kmax:
                if re.search(r'[0-9]', ln['text']) or (label == 'Attested' and best >= 2 and ':' not in ln['text']):
                    continue
                first = letters(ln['text'])[:1]
                if first.upper() != target[0] or (label == 'APPROVED' and not first.isupper()):
                    continue
                found.append(dict(kind='label', role=label, box=ln['box'], text=ln['text'],
                                  conf=ln['conf'], method='exact' if best == 0 else f'fuzzy d={best}'))
    return found


def is_name_line(ln):
    t = ln['text'].strip()
    lt = letters(t)
    if len(lt) < NAME_MIN_LETTERS or len(ln['words']) > NAME_MAX_WORDS:
        return False
    # digits only as a lone character (a middle initial "O." is often read as 0 or 9)
    if any(re.search(r'[0-9]', w) and len(w.strip('.,')) > 1 for w in t.split()) or t.endswith(':'):
        return False
    words = [w for w in re.split(r'\s+', t) if letters(w)]
    if len(words) < NAME_MIN_WORDS or any(letters(w).upper() in NAME_STOPWORDS for w in words):
        return False
    return sum(c.isupper() for c in lt) / len(lt) >= NAME_MIN_UPPER


def row_join(lines, j, ppi):
    """Line j joined with the pieces on the same row next to it (a name split by signature ink)."""
    a = lines[j]
    h = a['box'][3] - a['box'][1]
    parts = [a]
    for k, b in enumerate(lines):
        if k == j:
            continue
        v = min(a['box'][3], b['box'][3]) - max(a['box'][1], b['box'][1])
        hgap = max(a['box'][0], b['box'][0]) - min(a['box'][2], b['box'][2])
        if v >= 0.6 * min(h, b['box'][3] - b['box'][1]) and hgap / ppi <= ROW_JOIN_GAP_IN:
            parts.append(b)
    parts.sort(key=lambda l: l['box'][0])
    return dict(text=' '.join(p['text'] for p in parts), box=union([p['box'] for p in parts]),
                words=[w for p in parts for w in p['words']], conf=min(p['conf'] for p in parts))


def x_overlap(a, b):
    ov = min(a[2], b[2]) - max(a[0], b[0])
    return ov / max(1e-6, min(a[2] - a[0], b[2] - b[0]))


def role_of(text):
    """(role, score) of the best-matching official title in a line, or (None, score)."""
    up = letters(text).upper()
    best = (None, 0)
    for role, keys in ROLES:
        for k in keys:
            sc = 100 if k in up else (fuzz.partial_ratio(k, up) if len(k) >= 6 and len(up) >= len(k) - 2 else 0)
            if sc > best[1]:
                best = (role, sc)
    return best if best[1] >= TITLE_MIN_SCORE else (None, best[1])


def find_blocks(lines, ppi):
    found, used = [], set()
    for i, title in enumerate(lines):
        if len(title['words']) > TITLE_MAX_WORDS or is_name_line(title):
            continue
        if re.match(r'\s*(Hon|HON)\b', title['text']):   # attendance lists: "Hon. ..., City Vice-Mayor"
            continue
        role, score = role_of(title['text'])
        if role is None:
            continue
        # nearest printed-name line just above the title, overlapping horizontally
        for j in range(i - 1, -1, -1):
            name = lines[j]
            gap = (title['box'][1] - name['box'][3]) / ppi
            if gap > NAME_TITLE_GAP_IN:
                break
            if (gap < -NAME_TITLE_OVERLAP_IN or (name['box'][1] + name['box'][3]) >= (title['box'][1] + title['box'][3])
                    or x_overlap(name['box'], title['box']) < NAME_TITLE_X_OVERLAP):
                continue
            if (is_name_line(name) or is_name_line(row_join(lines, j, ppi))):
                if j not in used:
                    used.add(j)
                    found.append(dict(kind='block', role=role, box=union([name['box'], title['box']]),
                                      name_box=name['box'], text=f"{name['text']} / {title['text']}",
                                      conf=min(name['conf'], title['conf']), method=f'name+title {score:.0f}'))
                break
            if role_of(name['text'])[0] is None:
                break      # only another title line (a two-line title) may sit between name and title
    return found


def find_anchors(rec):
    ppi = rec['width'] / PAGE_WIDTH_IN
    lines = page_lines(rec)
    return find_labels(lines) + find_blocks(lines, ppi)
