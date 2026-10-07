"""Look up every ref.bib entry on Crossref by title + first author; print the best matches."""
import json
import re
import time
import urllib.parse
import urllib.request

bib = open("ref.bib", encoding="utf-8").read()
entries = re.findall(r"@(\w+)\{([^,]+),(.*?)\n\}", bib, flags=re.S)
out = {}
for typ, key, body in entries:
    title = re.search(r"title\s*=\s*\{(.+?)\},?\n", body).group(1)
    author = re.search(r"author\s*=\s*\{(.+?)\},?\n", body).group(1).split(" and ")[0].split(",")[0]
    q = urllib.parse.quote(f"{title} {author}")
    url = f"https://api.crossref.org/works?query.bibliographic={q}&rows=3&mailto=research@example.org"
    items = []
    for attempt in range(6):
        try:
            items = json.load(urllib.request.urlopen(url, timeout=30))["message"]["items"]
            break
        except Exception as e:
            time.sleep(5 * (attempt + 1))
    else:
        print(key, "ERROR after retries")
    cands = []
    for it in items:
        cands.append(dict(doi=it.get("DOI"), title=(it.get("title") or [""])[0],
                          venue=(it.get("container-title") or [""])[0],
                          year=(it.get("issued", {}).get("date-parts") or [[None]])[0][0],
                          volume=it.get("volume"), issue=it.get("issue"), page=it.get("page"),
                          type=it.get("type"),
                          authors=[a.get("family") for a in it.get("author", [])][:8]))
    out[key] = dict(bib_title=title, candidates=cands)
    c = cands[0] if cands else {}
    print(f"{key:28s} | {c.get('doi')} | {c.get('year')} | {c.get('title','')[:70]} | {c.get('venue','')[:50]}")
    time.sleep(3)
json.dump(out, open("crossref_results.json", "w"), indent=1)
