import json
import re
import time
import urllib.parse
import urllib.request

KEYS = ["shokri2017membership", "luo2021feature", "bonawitz2017secureagg", "wu2020pivot",
        "manzoor2022fedclamp", "manzoor2023defending", "shabbir2024resilience", "shabbir2024robustness",
        "manzoor2025novel", "manzoor2024survey", "chen2026class", "manzoor2024centralised", "manzoor2024enhanced"]
bib = open("ref.bib", encoding="utf-8").read()
ent = {k: b for _, k, b in re.findall(r"@(\w+)\{([^,]+),(.*?)\n\}", bib, flags=re.S)}
out = {}
for key in KEYS:
    body = ent[key]
    title = re.search(r"title\s*=\s*\{(.+?)\},?\n", body).group(1).replace("\\", "")
    url = ("https://api.crossref.org/works?query.title=" + urllib.parse.quote(title)
           + "&rows=3&select=DOI,title,container-title,issued,volume,issue,page,type,author")
    req = urllib.request.Request(url, headers={"User-Agent": "bibcheck/1.0 (mailto:research@example.org)"})
    items = None
    for attempt in range(8):
        try:
            items = json.load(urllib.request.urlopen(req, timeout=40))["message"]["items"]
            break
        except Exception as e:
            time.sleep(15 * (attempt + 1))
    out[key] = dict(bib_title=title, items=items)
    if items:
        it = items[0]
        print(key, "|", it.get("DOI"), "|", (it.get("title") or [""])[0][:90], "|", (it.get("container-title") or [""])[0][:60],
              "|", (it.get("issued", {}).get("date-parts") or [[None]])[0][0], "| v", it.get("volume"), "n", it.get("issue"), "p", it.get("page"),
              "|", ",".join((a.get("family") or "") for a in it.get("author", [])), flush=True)
    else:
        print(key, "| NO RESULT", flush=True)
    json.dump(out, open("crossref_slow.json", "w"), indent=1)
    time.sleep(12)
print("finished", flush=True)
