"""
Check every DOI in a reference list against Crossref (needs internet; run on your own computer).
Usage:  python verify_references.py refs.md
Prints one line per reference: OK, or MISMATCH with what Crossref says (first author, year, title).
A MISMATCH does not always mean an error (e.g. preprints), but every one should be looked at by hand.
"""
import re, sys, json, difflib, urllib.request, urllib.parse, time
refs = [r.strip() for r in open(sys.argv[1], encoding='utf-8').read().split('\n\n') if r.strip()]
for r in refs:
    m = re.search(r'doi:\s*(10\.\S+?)[\s.,;]*$', r) or re.search(r'(10\.\d{4,9}/[^\s]+?)[\s.,;]*$', r)
    first = r.split(',')[0].strip(); year = (re.search(r'\((\d{4})\)', r) or [None, '?'])[1]
    if not m: print(f'NO DOI   {r[:70]}'); continue
    doi = m.group(1)
    try:
        with urllib.request.urlopen('https://api.crossref.org/works/' + urllib.parse.quote(doi), timeout=20) as f:
            w = json.load(f)['message']
    except Exception as e:
        print(f'NOT FOUND {doi}  ({e})'); continue
    fa = (w.get('author') or [{}])[0].get('family', '?'); y = str((w.get('issued', {}).get('date-parts') or [[None]])[0][0])
    title = (w.get('title') or [''])[0]
    sim = difflib.SequenceMatcher(None, title.lower(), r.lower()).find_longest_match(0, len(title), 0, len(r)).size / max(len(title), 1)
    ok = fa.lower() == first.lower() and y == year and sim > 0.6
    print(('OK       ' if ok else 'MISMATCH ') + f'{doi} | yours: {first} {year} | Crossref: {fa} {y} | "{title[:70]}"')
    time.sleep(0.3)
