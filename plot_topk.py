"""Plot top-k accuracy against k for every run (reads data/topk_*.json, writes topk_curves.png and topk_table.csv)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter

DATA = Path(__file__).parent / "data"
# (file key, legend label, colour slot, marker). Colours follow the validated categorical order; baseline is neutral gray.
RUNS = [("lora", "v1: 1,222 ex, r16, 3 ep", "#2a78d6", "o"), ("A", "A: r16, 3 ep", "#eb6834", "s"),
        ("B", "B: r32, 5 ep", "#1baf7a", "^"), ("C", "C: r16, early stop", "#eda100", "D"),
        ("D", "D: r32, early stop", "#e87ba4", "v"), ("E", "E: r16, 10 ep", "#008300", "P"),
        ("F", "F: r32, 10 ep", "#6250d6", "X"), ("baseline", "Untrained base model", "#8a8983", "None")]
KS = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 25, 30]
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3de", "#fcfcfb"

data = {k: json.loads((DATA / f"topk_{k}.json").read_text()) for k, *_ in RUNS}
fig, ax = plt.subplots(figsize=(10, 6), dpi=160, facecolor=SURFACE)
ax.set_facecolor(SURFACE)
for key, label, color, marker in sorted(RUNS, key=lambda r: -data[r[0]]["acc"]["1"]):
    y = [100 * data[key]["acc"][str(k)] for k in KS]
    ax.plot(KS, y, color=color, lw=2, marker=None if marker == "None" else marker, ms=6.5,
            markeredgecolor=SURFACE, markeredgewidth=1.2, label=f"{label}  (top-1 {y[0]:.1f}%)",
            ls="--" if key == "baseline" else "-", zorder=3 if key != "baseline" else 2)
ax.set_xscale("log")
ax.xaxis.set_major_locator(FixedLocator([1, 2, 3, 5, 10, 20, 30]))
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}"))
ax.xaxis.set_minor_locator(FixedLocator([]))
ax.set_ylim(0, 100)
ax.set_xlim(0.9, 33)
ax.set_xlabel("k (the gold fallacy is among the model's k most likely names)", color=MUTED)
ax.set_ylabel("Top-k accuracy (%), 135 eval examples", color=MUTED)
ax.set_title("Top-k accuracy by run", loc="left", color=INK, fontsize=15, fontweight="bold", pad=22)
ax.text(0, 1.02, "Most strict errors are near-misses: runs A-F reach 85-90% by k=3.",
        transform=ax.transAxes, color=MUTED, fontsize=10.5)
ax.grid(axis="y", color=GRID, lw=1)
ax.tick_params(colors=MUTED)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(GRID)
leg = ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.56), frameon=True, facecolor=SURFACE, edgecolor=GRID,
                fontsize=9, labelcolor=INK)
fig.tight_layout()
fig.savefig(DATA.parent / "topk_curves.png", facecolor=SURFACE)

shown = (1, 2, 3, 5, 10, 20, 30)
lines = ["run," + ",".join(f"top{k}" for k in shown)]
for key, label, *_ in RUNS:
    lines.append(label.split(":")[0].replace(",", "") + "," + ",".join(f"{100 * data[key]['acc'][str(k)]:.1f}" for k in shown))
(DATA.parent / "topk_table.csv").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
