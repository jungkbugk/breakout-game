"""questions.json + template.html -> index.html (단일 파일)"""
import os
h = os.path.dirname(os.path.abspath(__file__))
data = open(os.path.join(h, "questions.json"), encoding="utf-8").read()
t = open(os.path.join(h, "template.html"), encoding="utf-8").read()
open(os.path.join(h, "index.html"), "w", encoding="utf-8").write(t.replace("__DATA__", data))
print("ok", round(os.path.getsize(os.path.join(h, "index.html")) / 1e6, 2), "MB")
