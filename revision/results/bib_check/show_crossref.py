import json
d = json.load(open("crossref_results.json"))
for k, v in d.items():
    print("==", k, "|", v["bib_title"][:90])
    for c in v["candidates"][:2]:
        print("    ", c["doi"], c["year"], c["type"], "|", c["title"][:80], "|", c["venue"][:45],
              "| v", c["volume"], "n", c["issue"], "p", c["page"], "|", ",".join(a or "" for a in c["authors"]))
