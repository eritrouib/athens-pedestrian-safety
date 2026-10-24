"""Bivariate maps, as in the thesis (figures 4.51-4.56).

For each kind of place (alcohol venues, transport stops, traffic points, education), every 200 m hexagon
gets a class on two axes: collisions (none / some / many) and that kind of place (none / some / many).
"Some" and "many" split the non-zero hexagons at their median, because most hexagons have none.
Then a bivariate Local Moran's I (999 permutations) asks where many places of that kind sit next to
many collisions (high-high), or the opposite.
Output: data/processed/bivariate.json, read by 05_build_site.py.
"""
import json
from pathlib import Path
import numpy as np, geopandas as gpd
import libpysal
from esda.moran import Moran_Local_BV, Moran_BV

P = Path(__file__).resolve().parents[1] / "data" / "processed"
cells = gpd.read_file(P / "model.gpkg", layer="cells")
GROUPS = ["Alcohol", "Transport", "Traffic", "Education"]

def three(v):
    v = np.asarray(v, float); out = np.zeros(len(v), int)
    nz = v[v > 0]
    if len(nz):
        med = np.median(nz)
        out[(v > 0) & (v <= med)] = 1; out[v > med] = 2
        return out, float(med)
    return out, 0.0

ycls, ymed = three(cells["n"])
w = libpysal.weights.Queen.from_dataframe(cells, use_index=False)
w.transform = "r"
res = {"y": ycls.tolist(), "y_split": ymed, "groups": {}}
QUAD = {1: "High-High", 2: "Low-High", 3: "Low-Low", 4: "High-Low"}
for g in GROUPS:
    xcls, xmed = three(cells[g])
    lbv = Moran_Local_BV(cells[g].to_numpy(float), cells["n"].to_numpy(float), w, permutations=999, seed=42)
    sig = lbv.p_sim < 0.05
    lab = [QUAD[q] if s else "Not significant" for q, s in zip(lbv.q, sig)]
    glob = Moran_BV(cells[g].to_numpy(float), cells["n"].to_numpy(float), w, permutations=999)
    both = ((xcls == 2) & (ycls == 2))
    res["groups"][g] = {"x": xcls.tolist(), "x_split": xmed, "lisa": lab,
                        "moran_bv": round(float(glob.I), 3), "moran_bv_p": round(float(glob.p_sim), 3),
                        "hh": int(sum(l == "High-High" for l in lab)), "many_many": int(both.sum()),
                        "count": {k: int(sum(l == k for l in lab)) for k in QUAD.values()}}
    print(g, "split", xmed, "| many-many", int(both.sum()), "| BV Moran", round(glob.I, 3), "p", glob.p_sim, "|", res["groups"][g]["count"])
json.dump(res, open(P / "bivariate.json", "w"))
print("collisions split", ymed, "classes", np.bincount(ycls))
