"""Write data/label_menu.json (186 names + one-sentence definitions) for the Colab benchmark. Local only: contains book definitions."""
import json

import llm_baseline as lb

names, menu = lb.label_menu()
json.dump({"names": names, "menu": menu}, open("data/label_menu.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(len(names), "names;", len(menu.split()), "words of menu")
