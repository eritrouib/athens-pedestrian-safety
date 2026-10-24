"""
Step 5: build the website in docs/ (served by GitHub Pages).

  docs/index.html   the emergency services dashboard
  docs/story.html   the story of the analysis
  docs/data/*.js    the data both pages use (small, pre-computed)

Optional live traffic on the dashboard: get a free TomTom key (developer.tomtom.com), restrict it to your
website's address, and run once with  --traffic-key YOUR_KEY  (saved in config/traffic_key.txt).
Without a key, the dashboard shows a note instead of the traffic layer.

Run:  python scripts/05_build_site.py
"""
from pathlib import Path
import argparse
import datetime
import json
import shutil

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import shapely

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data" / "processed"
DOCS = ROOT / "docs"
OUTD = DOCS / "data"
CFG = ROOT / "config"
SPEED_KMH = 30
YEARS = list(range(2008, 2018))
SEV = {"Slight": 0, "Serious": 1, "Fatal": 2}

# The ten locations visited in the thesis (section 4.4), with its Street View links. Each is placed on the
# collisions recorded at that street (thesis addresses), not typed-in coordinates.
#   (name, thesis type, link, streets, how to pick the spot, note)
#   pick: "dense" = densest cluster on those streets; "east"/"west"/"south"/"north" = densest cluster in that
#   third of the street; ("meet", other streets) = where the two sets of collisions are closest
THESIS_PLACES = [
    ("Athinas Street", "Persistent", "https://goo.gl/maps/75Zw52DETkx98J4U9", ["ΑΘΗΝΑΣ"], "dense",
     "Busy all day; jaywalking between the crossings is common."),
    ("Peiraios Street near Omonia", "Persistent", "https://goo.gl/maps/JPvYPMa5Pdvuqh7p7",
     ["ΠΕΙΡΑΙΩΣ", "ΤΣΑΛΔΑΡΗ ΠΑΝΑΓΗ"], "east",
     "Two lanes one way, lined with shops; signals and crossings, but heavy jaywalking."),
    ("28is Oktovriou (Patision) near Omonia", "Persistent", "https://goo.gl/maps/X7AwccNSTgSKQg5Z7",
     ["ΠΑΤΗΣΙΩΝ", "28ΗΣ ΟΚΤΩΒΡΙΟΥ"], "south", "3+1 lanes with one signal at El. Venizelou; crossing away from it is frequent."),
    ("Peiraios Street near Kerameikos", "Persistent", "https://goo.gl/maps/Cp1BUB73s4yHXKe76",
     ["ΠΕΙΡΑΙΩΣ", "ΤΣΑΛΔΑΡΗ ΠΑΝΑΓΗ"], "west", "2+2 lanes with lights and crossings."),
    ("Vasilissis Amalias Avenue near Syntagma", "Persistent", "https://goo.gl/maps/L8jZcH7BEdGSgBUo7",
     ["ΛΕΩΦΟΡΟΣ ΒΑΣΙΛΙΣΣΗΣ ΑΜΑΛΙΑΣ"], "dense", "4+1 lanes split by the tram line."),
    ("Alexandras and Vasilissis Sofias junction", "Persistent", "https://goo.gl/maps/8nZ8pXM1KNQ2s7p76",
     ["ΛΕΩΦΟΡΟΣ ΑΛΕΞΑΝΔΡΑΣ"], ("meet", ["ΛΕΩΦΟΡΟΣ ΒΑΣΙΛΙΣΣΗΣ ΣΟΦΙΑΣ"]),
     "Crossings and lights mostly at the junction itself."),
    ("Alexandras Avenue near the Apostolos Nikolaidis stadium", "Persistent", "https://goo.gl/maps/vxzAtJgtAS6G69yP8",
     ["ΛΕΩΦΟΡΟΣ ΑΛΕΞΑΝΔΡΑΣ"], ("near", 37.9875, 23.7535), "3+3 lanes separated by an island."),
    ("28is Oktovriou (Patision) and Galatsiou Avenue", "Persistent", "https://goo.gl/maps/KokR3PFn8ijS53Em9",
     ["ΛΕΩΦΟΡΟΣ ΓΑΛΑΤΣΙΟΥ"], ("meet", ["ΠΑΤΗΣΙΩΝ"]), "Mostly residential; lights and crossings at the junction."),
    ("Kypselis Street near Kypseli Square", "Intensifying", "https://goo.gl/maps/vQJ99hPZi8dhtwYRA",
     ["ΚΥΨΕΛΗΣ"], "dense", "Dense population and mixed uses; jaywalking is constant."),
    ("Syggrou Avenue near Syggrou-Fix", "Intensifying", "https://goo.gl/maps/dC7FZSX69H7f6Sbi6",
     ["ΛΕΩΦΟΡΟΣ ΣΥΓΓΡΟΥ"], "north", "3+3 lanes with lights and crossings."),
]


def place_points(col):
    """Locate each thesis place from the collisions on its streets (EPSG:2100 in, list of dicts out)."""
    out, prev = [], None
    xy_all = np.c_[col.geometry.x, col.geometry.y]
    for name, kind, url, streets, pick, note in THESIS_PLACES:
        sel = col[col["street"].isin(streets)]
        xy = np.c_[sel.geometry.x, sel.geometry.y]
        if isinstance(pick, tuple) and pick[0] == "near":   # densest cluster within 400 m of a landmark
            lm = gpd.GeoSeries([shapely.Point(pick[2], pick[1])], crs=4326).to_crs(2100).iloc[0]
            cand = np.sqrt((xy[:, 0] - lm.x) ** 2 + (xy[:, 1] - lm.y) ** 2) <= 400
            dens = (np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1)) <= 150).sum(1)
            centre = xy[np.where(cand, dens, -1).argmax()]
        elif isinstance(pick, tuple):
            other = col[col["street"].isin(pick[1])]
            oxy = np.c_[other.geometry.x, other.geometry.y]
            d = np.sqrt(((xy[:, None, :] - oxy[None, :, :]) ** 2).sum(-1))
            i, k = np.unravel_index(d.argmin(), d.shape)
            centre = (xy[i] + oxy[k]) / 2
        else:
            cand = np.ones(len(xy), bool)
            if pick in ("east", "west", "north", "south"):
                v = xy[:, 0] if pick in ("east", "west") else xy[:, 1]
                lo, hi = np.quantile(v, [1 / 3, 2 / 3])
                cand = v >= hi if pick in ("east", "north") else v <= lo
            dens = (np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1)) <= 150).sum(1)
            dens = np.where(cand, dens, -1)
            centre = xy[dens.argmax()]
        near = np.sqrt(((xy_all - centre) ** 2).sum(1)) <= 150
        centre = xy_all[near].mean(axis=0)
        prev = centre
        out.append({"name": name, "type": kind, "url": url, "note": note, "x": float(centre[0]),
                    "y": float(centre[1]), "n": int(near.sum())})
    return out


def js(name, obj):
    OUTD.mkdir(parents=True, exist_ok=True)
    txt = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    (OUTD / f"{name}.js").write_text(f'window.ATH=window.ATH||{{}};ATH["{name}"]={txt};', encoding="utf-8")
    return len(txt)


def coords(geom, nd=5):
    return [[round(y, nd), round(x, nd)] for x, y in geom.coords]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traffic-key", help="TomTom API key for the live traffic layer (saved in config/)")
    args = ap.parse_args()
    CFG.mkdir(exist_ok=True)
    kf = CFG / "traffic_key.txt"
    if args.traffic_key:
        kf.write_text(args.traffic_key.strip() + "\n")
    traffic_key = kf.read_text().strip() if kf.exists() else ""

    edges = gpd.read_file(P / "athens.gpkg", layer="edges")
    boundary = gpd.read_file(P / "athens.gpkg", layer="boundary")
    col = gpd.read_file(P / "access.gpkg", layer="collisions")
    fac = gpd.read_file(P / "access.gpkg", layer="facilities")
    hosp = gpd.read_file(P / "access.gpkg", layer="hospitals")
    lix = gpd.read_file(P / "hotspots.gpkg", layer="lixels")
    hexes = gpd.read_file(P / "hotspots.gpkg", layer="hexes")
    cells = gpd.read_file(P / "model.gpkg", layer="cells")
    top_all = gpd.read_file(P / "hotspots.gpkg", layer="top_all")
    top_ksi = gpd.read_file(P / "hotspots.gpkg", layer="top_ksi")
    centres = gpd.read_file(P / "hotspots.gpkg", layer="mean_centres")
    model = json.loads((P / "model.json").read_text())
    trend = pd.read_csv(P / "trend.csv").iloc[0]
    zones = pd.read_csv(P / "walk_zones.csv", index_col=0)
    col_h = np.load(P / "collision_hospital_m.npy")

    # hospitals that can be "on duty": public general hospitals and the children's hospitals
    duty = hosp[hosp["kind"].isin(["general", "children"])].copy()
    duty_idx = duty["hid"].astype(int).tolist()

    # distances at street midpoints, per on-duty hospital, and to the nearest police / fire station
    G = nx.Graph()
    for u, v, L in zip(edges["u"], edges["v"], edges["length"]):
        if not G.has_edge(u, v) or G[u][v]["length"] > L:
            G.add_edge(u, v, length=L)
    n_nodes = int(max(edges["u"].max(), edges["v"].max())) + 1

    def from_sources(src):
        H = G.copy()
        for nd, d0 in src.items():
            H.add_edge("S", int(nd), length=float(d0))
        d = nx.single_source_dijkstra_path_length(H, "S", weight="length")
        out = np.full(n_nodes, np.inf)
        for k, v in d.items():
            if k != "S":
                out[k] = v
        u, v, L = edges["u"].values, edges["v"].values, edges["length"].values
        return np.minimum(out[u] + L / 2, out[v] + L / 2)

    start = lambda r: r["snap_m"] * 1.3 if r["snap_m"] > 60 else r["snap_m"]
    edge_h = [from_sources({r["node"]: start(r)}) for _, r in duty.iterrows()]
    edge_police = from_sources({r["node"]: start(r) for _, r in fac[fac["type"] == "police"].iterrows()})
    edge_fire = from_sources({r["node"]: start(r) for _, r in fac[fac["type"] == "fire"].iterrows()})

    e4326 = edges.to_crs(4326)
    e4326["geometry"] = e4326.geometry.simplify(0.000008)
    to10 = lambda a: [int(round(x / 10)) for x in a]
    size = js("streets", {
        "c": [coords(g) for g in e4326.geometry],
        "h": [to10(a) for a in edge_h],             # per on-duty hospital, tens of metres
        "p": to10(edge_police), "f": to10(edge_fire)})
    print(f"streets.js: {len(edges):,} streets, {size / 1e6:.1f} MB")

    # street hotspots (network kernel density), only pieces with some density
    keep = lix[(lix["kde_all"] >= 0.3) | (lix["kde_ksi"] >= 0.06)].to_crs(4326)
    size = js("kde", {"c": [coords(g) for g in keep.geometry],
                      "a": [round(float(v), 2) for v in keep["kde_all"]],
                      "k": [round(float(v), 2) for v in keep["kde_ksi"]]})
    print(f"kde.js: {len(keep):,} street pieces, {size / 1e6:.1f} MB")

    c4326 = col.to_crs(4326)
    js("collisions", {
        "lat": [round(p.y, 5) for p in c4326.geometry], "lon": [round(p.x, 5) for p in c4326.geometry],
        "year": col["year"].astype(int).tolist(), "sev": col["severity"].map(SEV).astype(int).tolist(),
        "persons": col["persons"].astype(int).tolist(),
        "street": [f"{s.title()} {n}".strip() for s, n in zip(col["street"], col["number"].replace("nan", ""))],
        "h": [to10(col_h[:, i]) for i in duty_idx],
        "p": to10(col["d_police"]), "f": to10(col["d_fire"])})

    f4326 = fac.to_crs(4326)
    hk = dict(zip(hosp["name"], hosp["kind"]))
    js("facilities", [{
        "type": r["type"], "kind": hk.get(r["name"], r["type"]) if r["type"] == "hospital" else r["type"],
        "name": r["name"], "local": r["name_local"], "lat": round(r.geometry.y, 5), "lon": round(r.geometry.x, 5),
        "source": r["source"], "duty": (int(duty_idx.index(int(hosp.loc[hosp["name"] == r["name"], "hid"].iloc[0])))
                                        if r["type"] == "hospital" and r["name"] in set(duty["name"]) else -1)}
        for _, r in f4326.iterrows()])

    h4326 = hexes.to_crs(4326)
    cl = cells.set_index(cells.geometry.centroid.round(0).astype(str))
    cells_c = cells.to_crs(4326)
    js("hexes", {
        "c": [coords(g.exterior, 5)[:-1] for g in h4326.geometry],
        "n": hexes["n"].astype(int).tolist(), "ksi": hexes["n_ksi"].astype(int).tolist(),
        "gi": hexes["gi_class"].tolist(), "lisa": hexes["lisa"].tolist(), "eh": hexes["emerging"].tolist(),
        "years": [[int(r[f"y{y}"]) for y in YEARS] for _, r in hexes.iterrows()]})
    js("cells", {
        "c": [coords(g.exterior, 5)[:-1] for g in cells_c.geometry],
        "gwr": {g: [round(float(v), 3) for v in cells[f"gwr_{g}"]] for g in ["Alcohol", "Education", "Traffic", "Transport"]
                if f"gwr_{g}" in cells},
        "sig": {g: cells[f"gwr_{g}_sig"].astype(int).tolist() for g in ["Alcohol", "Education", "Traffic", "Transport"]
                if f"gwr_{g}_sig" in cells},
        "expected": [round(float(v), 2) for v in cells["expected"]], "n": cells["n"].astype(int).tolist(),
        "pois": {g: cells[g].astype(int).tolist() for g in ["Alcohol", "Education", "Traffic", "Transport"]},
        # bivariate classes and bivariate Local Moran's I from 07_bivariate.py (optional)
        "bv": json.load(open(P / "bivariate.json")) if (P / "bivariate.json").exists() else None})

    def tops(t):
        t = t.to_crs(4326)
        return [{"lat": round(g.centroid.y, 5), "lon": round(g.centroid.x, 5), "label": l.title(), "n": int(n),
                 "kde": round(float(k), 2)} for g, l, n, k in
                zip(t.geometry, t["label"], t["collisions_150m"], t[t.columns[t.columns.str.startswith("kde_")][0]])]
    pl = place_points(col)
    pts = gpd.GeoSeries(gpd.points_from_xy([q["x"] for q in pl], [q["y"] for q in pl]), crs=2100).to_crs(4326)
    places = [{k: q[k] for k in ("name", "type", "url", "note", "n")} | {"lat": round(p.y, 5), "lon": round(p.x, 5)}
              for q, p in zip(pl, pts)]
    for q in places:
        print(f"  thesis place: {q['name']}: {q['n']} collisions within 150 m  ({q['lat']}, {q['lon']})")
    mc = centres.to_crs(4326)
    b = boundary.to_crs(4326).geometry.iloc[0]
    b = max(b.geoms, key=lambda g: g.area) if b.geom_type == "MultiPolygon" else b
    js("context", {
        "boundary": coords(b.exterior.simplify(0.0001)),
        "top_all": tops(top_all), "top_ksi": tops(top_ksi),
        "centres": [{"year": int(y), "lat": round(p.y, 5), "lon": round(p.x, 5)} for y, p in zip(mc["year"], mc.geometry)],
        "places": places,
    })

    ksi = col["severity"].isin(["Fatal", "Serious"])
    meta = {
        "built": datetime.date.today().strftime("%d %B %Y"), "years": YEARS, "speed_kmh": SPEED_KMH,
        "traffic_key": traffic_key,
        "total_in_data": 1318, "located": int(len(col)),
        "by_year": col.groupby("year").size().reindex(YEARS, fill_value=0).astype(int).tolist(),
        "by_year_sev": [[int(((col.year == y) & (col.severity == s)).sum()) for s in SEV] for y in YEARS],
        "trend_tau": round(float(trend["tau"]), 2), "trend_p": float(trend["p"]),
        "duty": [{"name": r["name"], "kind": r["kind"]} for _, r in duty.iterrows()],
        "access": {t: {"median": round(float(col[f"min_{t}"].median()), 1),
                       "p90": round(float(col[f"min_{t}"].quantile(.9)), 1),
                       "ksi_median": round(float(col.loc[ksi, f"min_{t}"].median()), 1)}
                   for t in ("hosp", "police", "fire")},
        "streets_within_5min": round(float((lix_min := (gpd.read_file(P / "access.gpkg", layer="lixels")["min_hosp"])).le(5).mean()), 3),
        "model": model,
        "zones": {"rows": zones.index.tolist(), "cols": zones.columns.tolist(), "values": zones.values.tolist()},
        "emerging": hexes["emerging"].value_counts().to_dict(),
        "lisa": hexes["lisa"].value_counts().to_dict(),
        "gi_hot": int(hexes["gi_class"].str.startswith("Hot").sum()),
        "gi_hot_fdr": int((hexes["gi_fdr"] & (hexes["gi_z"] > 0)).sum()),
    }
    js("meta", meta)

    for page in ("dashboard_template.html", "story_template.html"):
        src = ROOT / "scripts" / page
        if src.exists():
            dst = DOCS / ("index.html" if page.startswith("dashboard") else "story.html")
            shutil.copy(src, dst)
    (DOCS / ".nojekyll").write_text("")
    print(f"Built docs/ ({meta['built']}); live traffic {'on' if traffic_key else 'off (no key)'}")


if __name__ == "__main__":
    main()
