import re
t = open("arxiv.xml", encoding="utf-8").read()
for e in re.findall(r"<entry>(.*?)</entry>", t, re.S):
    i = re.search(r"<id>https?://arxiv.org/abs/([^<]+)</id>", e).group(1)
    ti = " ".join(re.search(r"<title>(.*?)</title>", e, re.S).group(1).split())
    au = [" ".join(a.split()) for a in re.findall(r"<name>(.*?)</name>", e)]
    print(i, "|", ti, "|", ", ".join(au))
