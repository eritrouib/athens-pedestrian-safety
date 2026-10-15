"""
Step 3: network analysis.

  A. Emergency access. Distance along the street network from every street piece and every collision to each
     hospital, and to the nearest police and fire station. Times use an assumed average emergency driving
     speed (SPEED_KMH), with no traffic: they are best-case, comparable figures, not predictions.
     Each hospital's distances are kept separately, so the dashboard can recompute "nearest on-duty hospital"
     for whichever hospitals are on the emergency rota (εφημερίες) that day.
  B. Walkability zones (thesis section 4.3): how many points of interest of each kind lie within 100, 250 and
     500 m walking distance of a collision, measured along the network instead of in straight lines.

The thesis network has no road types or one-way information, so routes may use streets in either direction.

Run:  python scripts/03_access.py
"""
from pathlib import Path
import warnings

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import shapely

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "athens.gpkg"
HOT = ROOT / "data" / "processed" / "hotspots.gpkg"
OUT = ROOT / "data" / "processed" / "access.gpkg"
SPEED_KMH = 30          # assumed average emergency driving speed in central Athens, without live traffic
APPROACH = 1.3          # sites off the study network: straight-line gap x 1.3 as an approach leg
ZONES = [100, 250, 500]


def log(msg=""):
    print(msg, flush=True)


def graph(edges):
    G = nx.Graph()
    for u, v, L in zip(edges["u"], edges["v"], edges["length"]):
        if G.has_edge(u, v):
            G[u][v]["length"] = min(G[u][v]["length"], L)
        else:
            G.add_edge(u, v, length=L)
    return G


def node_distances(G, sources, n_nodes):
    """sources: {node: starting distance}. Returns distance from the nearest source to every node."""
    d = nx.multi_source_dijkstra_path_length(G, set(sources), weight="length") if not any(sources.values()) else None
    if d is None:   # sources with different starting distances: add a virtual super-source
        H = G.copy()
        for nd, d0 in sources.items():
            H.add_edge("SRC", nd, length=float(d0))
        d = nx.single_source_dijkstra_path_length(H, "SRC", weight="length")
    out = np.full(n_nodes, np.inf)
    for k, v in d.items():
        if k != "SRC":
            out[k] = v
    return out


def at_points(dist_nodes, edges, edge_idx, frac):
    """Distance to points part-way along edges: through whichever end is closer."""
    u = edges["u"].values[edge_idx]
    v = edges["v"].values[edge_idx]
    L = edges["length"].values[edge_idx]
    return np.minimum(dist_nodes[u] + frac * L, dist_nodes[v] + (1 - frac) * L)


def main():
    edges = gpd.read_file(DATA, layer="edges")
    nodes = gpd.read_file(DATA, layer="nodes")
    col = gpd.read_file(DATA, layer="collisions")
    fac = gpd.read_file(DATA, layer="facilities")
    pois = gpd.read_file(DATA, layer="pois")
    lix = gpd.read_file(HOT, layer="lixels")
    OUT.unlink(missing_ok=True)
    G = graph(edges)
    n_nodes = int(max(edges["u"].max(), edges["v"].max())) + 1
    lix_mid = (lix["f0"].values + lix["f1"].values) / 2
    m_per_min = SPEED_KMH * 1000 / 60

    # ---------------- A. emergency access ----------------
    fac = fac.reset_index(drop=True)
    fac["start_m"] = np.where(fac["snap_m"] > 60, fac["snap_m"] * APPROACH, fac["snap_m"])
    hosp = fac[fac["type"] == "hospital"].reset_index(drop=True)
    hosp["hid"] = np.arange(len(hosp))
    lix_h = np.zeros((len(lix), len(hosp)), dtype=np.float32)
    col_h = np.zeros((len(col), len(hosp)), dtype=np.float32)
    for i, r in hosp.iterrows():
        dn = node_distances(G, {int(r["node"]): float(r["start_m"])}, n_nodes)
        lix_h[:, i] = at_points(dn, edges, lix["edge"].values, lix_mid)
        col_h[:, i] = at_points(dn, edges, col["edge"].values, col["frac"].values)
    general = hosp.index[hosp["kind"] == "general"].values
    lix["d_hosp"] = lix_h[:, general].min(axis=1)
    lix["hosp_id"] = general[lix_h[:, general].argmin(axis=1)]
    col["d_hosp"] = col_h[:, general].min(axis=1)
    col["hosp_id"] = general[col_h[:, general].argmin(axis=1)]
    for t in ("police", "fire"):
        f = fac[fac["type"] == t]
        dn = node_distances(G, {int(n): float(s) for n, s in zip(f["node"], f["start_m"])}, n_nodes)
        lix[f"d_{t}"] = at_points(dn, edges, lix["edge"].values, lix_mid)
        col[f"d_{t}"] = at_points(dn, edges, col["edge"].values, col["frac"].values)
    for t in ("hosp", "police", "fire"):
        col[f"min_{t}"] = col[f"d_{t}"] / m_per_min
        lix[f"min_{t}"] = lix[f"d_{t}"] / m_per_min
    ksi = col["severity"].isin(["Fatal", "Serious"])
    log(f"Assumed speed {SPEED_KMH} km/h, no traffic.")
    for t, lab in (("hosp", "nearest public general hospital"), ("police", "nearest police station"),
                   ("fire", "nearest fire station")):
        m = col[f"min_{t}"]
        log(f"  {lab}: median {m.median():.1f} min, 90% within {m.quantile(.9):.1f} min, "
            f"longest {m.max():.1f} min (KSI median {m[ksi].median():.1f})")
    share = (lix["min_hosp"] <= 5).mean()
    log(f"  {share:.0%} of streets are within 5 minutes of a public general hospital")
    busiest = col.groupby("hosp_id").size().sort_values(ascending=False)
    log("  nearest hospital for: " + ", ".join(f"{hosp.loc[i, 'name']} {n}" for i, n in busiest.items()))

    # ---------------- B. walkability zones around collisions ----------------
    # distance from each point of interest to the nearest collision, walking along the network
    src = {}
    for e, f in zip(col["edge"].values, col["frac"].values):
        u, v, L = edges.loc[e, ["u", "v", "length"]]
        src[int(u)] = min(src.get(int(u), np.inf), f * L)
        src[int(v)] = min(src.get(int(v), np.inf), (1 - f) * L)
    dn = node_distances(G, src, n_nodes)
    j = gpd.sjoin_nearest(pois, edges[["geometry"]], how="left", distance_col="off_m")
    j = j[~j.index.duplicated()]
    pe = j["index_right"].values
    pf = shapely.line_locate_point(edges.geometry.values[pe], pois.geometry.values, normalized=True)
    pois["walk_m"] = at_points(dn, edges, pe, pf) + j["off_m"].values
    bins = [0] + ZONES + [np.inf]
    labels = ["0-100 m", "100-250 m", "250-500 m", "over 500 m"]
    pois["zone"] = pd.cut(pois["walk_m"], bins, labels=labels, right=True, include_lowest=True)
    zt = pd.crosstab(pois["zone"], pois["group"]).reindex(labels, fill_value=0)
    zt["Sum"] = zt.sum(axis=1)
    zt["%"] = (100 * zt["Sum"] / zt["Sum"].sum()).round(1)
    log("\nPoints of interest by walking distance from the nearest collision:\n" + zt.to_string())
    zt.to_csv(ROOT / "data" / "processed" / "walk_zones.csv")

    keep = ["edge", "f0", "f1", "d_hosp", "hosp_id", "d_police", "d_fire", "min_hosp", "min_police", "min_fire",
            "geometry"]
    lix[keep].to_file(OUT, layer="lixels", driver="GPKG")
    col.to_file(OUT, layer="collisions", driver="GPKG")
    hosp.to_file(OUT, layer="hospitals", driver="GPKG")
    fac.to_file(OUT, layer="facilities", driver="GPKG")
    pois.to_file(OUT, layer="pois", driver="GPKG")
    np.save(ROOT / "data" / "processed" / "lixel_hospital_m.npy", lix_h)
    np.save(ROOT / "data" / "processed" / "collision_hospital_m.npy", col_h)
    log(f"\nSaved {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
