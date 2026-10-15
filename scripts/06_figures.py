"""
Step 6 (optional): figures for the paper, 300 dpi PNG, in figures/.

Run after steps 1-4:  python scripts/06_figures.py
"""
from pathlib import Path
import json

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data" / "processed"
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

INK, INK2, RULE = "#1C2530", "#4A5562", "#C9C2B5"
SEV = {"Slight": "#E99A74", "Serious": "#D2532F", "Fatal": "#9E1C2A"}
DENS = ["#D99A72", "#CC774A", "#B55730", "#8F3D22", "#622513"]
RESP = ["#72B5B9", "#4F9EA8", "#348496", "#27677E", "#1E4A62"]
EHC = {"Persistent": "#A0202E", "Intensifying": "#A0202E", "Consecutive": "#D8692A", "New": "#8A6A00",
       "Sporadic": "#C9B9A6", "Diminishing": "#8FB3DA", "Historical": "#8FB3DA"}
LIC = {"High-High": "#A0202E", "High-Low": "#E7A0A8", "Low-High": "#8FB3DA", "Low-Low": "#2F6FB3"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2, "text.color": INK,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2})

edges = gpd.read_file(P / "athens.gpkg", layer="edges")
boundary = gpd.read_file(P / "athens.gpkg", layer="boundary")
col = gpd.read_file(P / "access.gpkg", layer="collisions")
fac = gpd.read_file(P / "access.gpkg", layer="facilities")
hosp = gpd.read_file(P / "access.gpkg", layer="hospitals")
lix_h = gpd.read_file(P / "hotspots.gpkg", layer="lixels")
lix_a = gpd.read_file(P / "access.gpkg", layer="lixels")
hexes = gpd.read_file(P / "hotspots.gpkg", layer="hexes")
cells = gpd.read_file(P / "model.gpkg", layer="cells")
top = gpd.read_file(P / "hotspots.gpkg", layer="top_all")
model = json.loads((P / "model.json").read_text())
BX = boundary.total_bounds


def base(ax, streets=True):
    if streets:
        edges.plot(ax=ax, color="#D9D3C7", linewidth=0.25, zorder=1)
    boundary.boundary.plot(ax=ax, color=INK2, linewidth=0.8, linestyle=(0, (3, 2)), zorder=5)
    ax.set_xlim(BX[0] - 300, BX[2] + 300)
    ax.set_ylim(BX[1] - 300, BX[3] + 300)
    ax.set_axis_off()
    ax.set_aspect("equal")
    # scale bar 1 km
    x0, y0 = BX[0] + 200, BX[1] + 100
    ax.plot([x0, x0 + 1000], [y0, y0], color=INK, linewidth=2, solid_capstyle="butt", zorder=6)
    ax.text(x0 + 500, y0 + 180, "1 km", ha="center", fontsize=7.5, color=INK, zorder=6)
    ax.annotate("N", xy=(BX[2] - 200, BX[3] - 300), xytext=(BX[2] - 200, BX[3] - 1100), ha="center",
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.2), fontsize=9, fontweight="bold", zorder=6)


def save(fig, name):
    fig.savefig(OUT / name, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", name)


# Figure 1: study area, collisions and facilities
fig, ax = plt.subplots(figsize=(7, 7.6))
base(ax)
for s in ["Slight", "Serious", "Fatal"]:
    d = col[col.severity == s]
    d.plot(ax=ax, color=SEV[s], markersize={"Slight": 4, "Serious": 14, "Fatal": 22}[s], edgecolor="white",
           linewidth=0.3, zorder=3)
mk = {"hospital": ("s", "#FFFFFF", 46), "police": ("o", "#2F6FB3", 26), "fire": ("D", "#D8692A", 30)}
for t, (m, c, sz) in mk.items():
    d = fac[(fac["type"] == t)]
    if t == "hospital":
        d = d[d["name"].isin(hosp.loc[hosp["kind"].isin(["general", "children"]), "name"])]
    ax.scatter(d.geometry.x, d.geometry.y, marker=m, s=sz, facecolor=c, edgecolor=INK, linewidth=0.9, zorder=4)
handles = [Line2D([], [], marker="o", ls="", color=SEV[s], markeredgecolor="white", markersize=ms, label=l)
           for s, ms, l in [("Fatal", 6.5, "Fatal (29)"), ("Serious", 5.5, "Serious (80)"), ("Slight", 3.5, "Slight (1,206)")]]
handles += [Line2D([], [], marker="s", ls="", markerfacecolor="white", markeredgecolor=INK, markersize=6, label="Public hospital"),
            Line2D([], [], marker="o", ls="", markerfacecolor="#2F6FB3", markeredgecolor=INK, markersize=5, label="Police station"),
            Line2D([], [], marker="D", ls="", markerfacecolor="#D8692A", markeredgecolor=INK, markersize=5, label="Fire station")]
ax.legend(handles=handles, loc="lower right", frameon=True, framealpha=0.95, edgecolor=RULE, fontsize=8)
ax.set_title("Pedestrian collisions in the Municipality of Athens, 2008–2017", loc="left", fontsize=11)
save(fig, "fig1_study_area.png")

# Figure 2: collisions per year by severity
yrs = list(range(2008, 2018))
tab = pd.crosstab(col.year, col.severity).reindex(index=yrs, columns=["Slight", "Serious", "Fatal"], fill_value=0)
fig, ax = plt.subplots(figsize=(7, 3.4))
bottom = np.zeros(len(yrs))
for s in ["Slight", "Serious", "Fatal"]:
    ax.bar(yrs, tab[s], bottom=bottom, color=SEV[s], width=0.68, label=s, edgecolor="white", linewidth=0.8)
    bottom += tab[s].values
for x, v in zip(yrs, bottom):
    ax.text(x, v + 3, f"{int(v)}", ha="center", fontsize=8, color=INK)
ax.set_xticks(yrs)
ax.set_ylabel("Collisions")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", color="#E5E0D6", linewidth=0.6)
ax.set_axisbelow(True)
ax.legend(frameon=False, ncol=3, loc="upper right", fontsize=8)
ax.set_title("Collisions fell by 55% between 2008 and 2017 (located collisions, n = 1,315)", loc="left", fontsize=10)
save(fig, "fig2_by_year.png")

# Figure 3: network kernel density, all and KSI
fig, axs = plt.subplots(1, 2, figsize=(11, 6.4))
for ax, colname, cuts, ttl in [(axs[0], "kde_all", [0.3, 0.6, 1.0, 1.6, 2.6], "(a) All collisions"),
                               (axs[1], "kde_ksi", [0.06, 0.12, 0.2, 0.35, 0.6], "(b) Killed or seriously injured")]:
    base(ax)
    d = lix_h[lix_h[colname] >= cuts[0]].copy()
    d["b"] = np.digitize(d[colname], cuts[1:])
    for b in range(5):
        dd = d[d.b == b]
        if len(dd):
            dd.plot(ax=ax, color=DENS[b], linewidth=0.9 + b * 0.7, zorder=2 + b * 0.01)
    labs = [f"{cuts[i]:g}–{cuts[i + 1]:g}" for i in range(4)] + [f"> {cuts[4]:g}"]
    ax.legend(handles=[Line2D([], [], color=DENS[i], lw=2.5, label=labs[i]) for i in range(5)],
              title="per km of street per year", loc="lower right", fontsize=7.5, title_fontsize=7.5,
              frameon=True, framealpha=0.95, edgecolor=RULE)
    ax.set_title(ttl, loc="left", fontsize=10)
for i, r in enumerate(top.itertuples(), start=1):
    c = r.geometry.centroid
    axs[0].text(c.x, c.y, str(i), fontsize=7, fontweight="bold", color="white", ha="center", va="center", zorder=8,
                bbox=dict(boxstyle="circle,pad=0.25", fc=INK, ec="white", lw=0.6))
save(fig, "fig3_network_kde.png")

# Figure 4: emerging hotspots and Local Moran
fig, axs = plt.subplots(1, 2, figsize=(11, 6.4))
base(axs[0])
d = hexes[hexes.emerging.isin(EHC)]
d.plot(ax=axs[0], color=d.emerging.map(EHC), edgecolor="white", linewidth=0.4, zorder=3)
cnt = hexes.emerging.value_counts()
eh_items = [(EHC[k], f"{k} ({cnt.get(k, 0)})") for k in ["Persistent", "Consecutive", "New", "Sporadic"] if cnt.get(k, 0)]
fading = cnt.get("Diminishing", 0) + cnt.get("Historical", 0)
if fading:
    eh_items.append((EHC["Historical"], f"Diminishing or historical ({fading})"))
axs[0].legend(handles=[Patch(color=c, label=l) for c, l in eh_items],
              loc="lower right", fontsize=7.5, frameon=True, framealpha=0.95, edgecolor=RULE)
axs[0].set_title("(a) Emerging hotspots, 200 m hexagons", loc="left", fontsize=10)
base(axs[1])
d = hexes[hexes.lisa.isin(LIC)]
d.plot(ax=axs[1], color=d.lisa.map(LIC), edgecolor="white", linewidth=0.4, zorder=3)
cnt = hexes.lisa.value_counts()
axs[1].legend(handles=[Patch(color=LIC[k], label=f"{k} ({cnt.get(k, 0)})") for k in LIC],
              loc="lower right", fontsize=7.5, frameon=True, framealpha=0.95, edgecolor=RULE)
axs[1].set_title("(b) Local Moran's I clusters and outliers (p ≤ 0.05)", loc="left", fontsize=10)
save(fig, "fig4_space_time.png")

# Figure 5: rate ratios
names = {"Alcohol": "Alcohol venues", "Education": "Education", "Traffic": "Traffic points", "Transport": "Transport stops"}
nb, single = model["nb"], model["single"]
fig, ax = plt.subplots(figsize=(7, 3.0))
for i, (r, s) in enumerate(zip(nb, single)):
    y = len(nb) - 1 - i
    ax.plot([s["lo"], s["hi"]], [y - 0.16] * 2, color="#9AA3AD", lw=1.6)
    ax.plot(s["irr_per_sd"], y - 0.16, "o", mfc="white", mec="#6E7884", ms=6, mew=1.4)
    sig = r["p"] < 0.05
    ax.plot([r["lo"], r["hi"]], [y + 0.16] * 2, color=INK, lw=2)
    ax.plot(r["irr_per_sd"], y + 0.16, "o", mfc="#9E1C2A" if sig else "white", mec=INK, ms=7, mew=1.4)
    ax.text(1.93, y + 0.16, f"{r['irr_per_sd']:.2f} ({r['lo']:.2f}–{r['hi']:.2f})", va="center", fontsize=8)
ax.axvline(1, color=INK, lw=1, ls=(0, (4, 3)))
ax.set_yticks(range(len(nb)))
ax.set_yticklabels([names[r["variable"]] for r in nb][::-1])
ax.set_xscale("log")
ax.set_xticks([0.8, 1, 1.25, 1.5, 1.75])
ax.set_xticklabels(["0.8", "1", "1.25", "1.5", "1.75"])
ax.set_xlim(0.78, 1.92)
ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set_ylim(-0.6, len(nb) - 0.4)
ax.set_xlabel("Rate ratio per standard deviation (log scale)")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(handles=[Line2D([], [], marker="o", color=INK, mfc="#9E1C2A", label="All four together (filled = p < 0.05)"),
                   Line2D([], [], marker="o", color="#9AA3AD", mfc="white", mec="#6E7884", label="Each on its own")],
          loc="upper center", bbox_to_anchor=(0.45, -0.28), ncol=2, fontsize=7.5, frameon=False)
ax.set_title("Negative binomial model, 990 hexagons, exposure = km of street", loc="left", fontsize=10)
save(fig, "fig5_rate_ratios.png")

# Figure 6: GWR local rate ratios
fig, axs = plt.subplots(1, 2, figsize=(11, 6.4))
cuts = [1.0, 1.2, 1.4, 1.6, 1.8]
for ax, g, ttl in [(axs[0], "Traffic", "(a) Traffic points"), (axs[1], "Transport", "(b) Transport stops")]:
    base(ax, streets=False)
    cells.plot(ax=ax, color="#EFEBE3", edgecolor="#DDD6C9", linewidth=0.3, zorder=2)
    d = cells[cells[f"gwr_{g}_sig"] & (cells[f"gwr_{g}"] > 1)].copy()
    d["b"] = np.clip(np.digitize(d[f"gwr_{g}"], cuts[1:]), 0, 4)
    d.plot(ax=ax, color=[DENS[b] for b in d.b], edgecolor="white", linewidth=0.3, zorder=3)
    labs = ["1.0–1.2", "1.2–1.4", "1.4–1.6", "1.6–1.8", "> 1.8"]
    ax.legend(handles=[Patch(color=DENS[i], label=labs[i]) for i in range(5)] + [Patch(color="#EFEBE3", label="not significant")],
              title="local rate ratio per SD", loc="lower right", fontsize=7.5, title_fontsize=7.5,
              frameon=True, framealpha=0.95, edgecolor=RULE)
    ax.set_title(ttl, loc="left", fontsize=10)
save(fig, "fig6_gwr.png")

# Figure 7: minutes to the nearest public general hospital
fig, axs = plt.subplots(1, 2, figsize=(11, 6.4))
lh = np.load(P / "lixel_hospital_m.npy")
gen = hosp.index[hosp["kind"] == "general"].values
ev = hosp.index[hosp["name"].str.startswith("Evangelismos")][0]
mpm = 30 * 1000 / 60
for ax, dist, ttl, which in [(axs[0], lh[:, gen].min(axis=1), "(a) All seven public general hospitals on duty", gen),
                             (axs[1], lh[:, ev], "(b) Only Evangelismos on duty", [ev])]:
    base(ax, streets=False)
    m = dist / mpm
    b = np.digitize(m, [2, 4, 6, 8])
    for k in range(5):
        lix_a[b == k].plot(ax=ax, color=RESP[k], linewidth=0.6, zorder=2)
    hh = hosp.loc[which]
    ax.scatter(hh.geometry.x, hh.geometry.y, marker="s", s=46, facecolor="white", edgecolor=INK, linewidth=1, zorder=5)
    labs = ["≤ 2 min", "2–4", "4–6", "6–8", "> 8"]
    ax.legend(handles=[Line2D([], [], color=RESP[i], lw=2.5, label=labs[i]) for i in range(5)],
              title="drive at 30 km/h, no traffic", loc="lower right", fontsize=7.5, title_fontsize=7.5,
              frameon=True, framealpha=0.95, edgecolor=RULE)
    ax.set_title(ttl, loc="left", fontsize=10)
save(fig, "fig7_hospital_access.png")
