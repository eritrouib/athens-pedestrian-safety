"""
Step 1: prepare the Athens data.

Reads the thesis GeoPackage (data/raw/athens_thesis.gpkg) and writes data/processed/athens.gpkg with:
  collisions   1,315 located pedestrian collisions, 2008-2017 (3 of the 1,318 have no location)
  edges        the thesis street network, one row per street segment, with its two end nodes
  boundary     the Municipality of Athens
  facilities   hospitals, police stations and fire stations (OpenStreetMap), snapped to the network
  pois         points of interest used in the thesis: alcohol venues, education, traffic and transport points

Everything is in Greek Grid (EPSG:2100), in metres.

Run:  python scripts/01_prepare.py
"""
from pathlib import Path
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import Point, box

warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "athens_thesis.gpkg"
OUT = ROOT / "data" / "processed" / "athens.gpkg"
GGRS = "EPSG:2100"
SNAP_TOL = 10        # metres: the thesis snapped collisions to the nearest road within 10 m
FACILITY_MAX = 1500  # metres: sites just outside the study network still serve it; the gap is added as an approach leg

SEVERITY = {"Ελαφρά τραυματίας": "Slight", "Βαριά τραυματίας": "Serious", "Νεκρός": "Fatal"}

# Point categories, following the thesis (section 4.4)
ALCOHOL = {"bar", "pub", "nightclub", "beverages"}
EDUCATION = {"school", "college", "university", "kindergarten"}
TRAFFIC = {"crossing", "traffic_signals", "stop", "mini_roundabout"}
TRANSPORT = {"bus_stop", "bus_station", "tram_stop", "railway_station", "railway_halt", "taxi"}

# Public hospitals that OpenStreetMap stores as building outlines rather than points, so they are missing
# from the point layer. Each is placed at the bus stop named after it, which sits at the hospital entrance.
# kind: general = public general hospital with an emergency department on the duty rota (εφημερίες)
HOSPITAL_PROXIES = [
    # name (English), name (Greek), stop name in transport_data, kind
    ("Laiko General Hospital", "Γ.Ν. Λαϊκό", "ΛΑΪΚΟ", "general"),
    ("Hippokration General Hospital", "Γ.Ν. Ιπποκράτειο", "ΙΠΠΟΚΡΑΤΕΙΟ", "general"),
    ("Elpis General Hospital", "Γ.Ν. Ελπίς", "ΝΟΣΟΚΟΜΕΙΟ ΕΛΠΙΔΑ", "general"),
    ("Red Cross General Hospital (Korgialenio-Benakio)", "Γ.Ν. Ερυθρός Σταυρός", "ΕΡΥΘΡΟΣ ΣΤΑΥΡΟΣ", "general"),
    ("Children's hospitals (Agia Sofia, Aglaia Kyriakou)", "Νοσοκομεία Παίδων", "ΝΟΣ.ΠΑΙΔΩΝ", "children"),
    ("401 General Military Hospital", "401 Γ.Σ.Ν.Α.", "ΣΤΡΑΤΙΩΤΙΚΟ ΝΟΣΟΚΟΜΕΙΟ", "other"),
]
# OpenStreetMap hospital points: which ones are public general hospitals, and which to leave out
OSM_HOSPITAL_KIND = {
    "ΕΥΑΓΓΕΛΙΣΜΟΣ": ("Evangelismos General Hospital", "general"),
    "Ευαγελισμος": ("Evangelismos General Hospital", "general"),
    "Σωτηρια": ("Sotiria General Hospital", "general"),
    "Αγια Όλγα": ("Agia Olga General Hospital", "general"),
    'Γενικό Νοσοκομείο Αθηνών "Αλεξανδρα"': ("Alexandra General Hospital (maternity)", "other"),
    'Παράρτημα Γεν. Νοσ. Αθηνών "Αλεξάνδρα"': ("Alexandra General Hospital (maternity)", "other"),
}
OSM_HOSPITAL_SKIP = {"ΦΑΡΜΑΚΕΙΟ", "Δρομοκαϊτειο"}  # pharmacies tagged as hospitals; psychiatric hospital


def log(msg=""):
    print(msg, flush=True)


def read(layer):
    return gpd.read_file(RAW, layer=layer)


def main():
    if not RAW.exists():
        raise SystemExit(f"Missing {RAW}. Put the thesis GeoPackage there first.")
    OUT.unlink(missing_ok=True)

    # --- study area -------------------------------------------------------------------------------
    munis = read("athina_poly").to_crs(GGRS)
    boundary = munis[munis["ESYE_ID"] == "99010000"][["geometry"]].copy()   # Municipality of Athens
    boundary["name"] = "Municipality of Athens"
    log(f"Boundary: {boundary.area.sum() / 1e6:.1f} km²")

    # --- network ----------------------------------------------------------------------------------
    roads = read("athensroads_SpatialJoin").to_crs(GGRS)
    roads = roads[roads.geometry.notna() & ~roads.geometry.is_empty].reset_index(drop=True)
    c0 = np.round(shapely.get_coordinates(shapely.get_point(roads.geometry.values, 0)), 1)
    c1 = np.round(shapely.get_coordinates(shapely.get_point(roads.geometry.values, -1)), 1)
    keys, inv = np.unique(np.vstack([c0, c1]), axis=0, return_inverse=True)
    inv = inv.ravel()
    n = len(roads)
    edges = gpd.GeoDataFrame({"u": inv[:n], "v": inv[n:], "length": roads.length.values},
                             geometry=roads.geometry.values, crs=GGRS)
    edges = edges[edges["u"] != edges["v"]].reset_index(drop=True)
    # keep the connected network (drops a few tiny detached pieces)
    import networkx as nx
    G = nx.Graph()
    G.add_edges_from(zip(edges["u"], edges["v"]))
    main_cc = max(nx.connected_components(G), key=len)
    edges = edges[edges["u"].isin(main_cc)].reset_index(drop=True)
    log(f"Network: {len(edges):,} segments, {edges['length'].sum() / 1000:.0f} km, "
        f"{len(main_cc):,} junctions")
    nodes = gpd.GeoDataFrame({"node": np.arange(len(keys))}, geometry=gpd.points_from_xy(keys[:, 0], keys[:, 1]),
                             crs=GGRS)
    nodes = nodes[nodes["node"].isin(main_cc)].reset_index(drop=True)

    # --- collisions -------------------------------------------------------------------------------
    col = read("DATA_THESIS_Project").to_crs(GGRS)
    total = len(col)
    ok = col.geometry.notna() & ~col.geometry.is_empty
    ok &= col.geometry.x.between(400000, 600000)
    missing = col[~ok]
    col = col[ok].copy()
    log(f"Collisions: {total:,} in the data, {len(missing)} without a usable location: "
        + "; ".join(f"{r.FullAddres} ({r.Year_})" for r in missing.itertuples()))
    col = gpd.GeoDataFrame({
        "id": col["OBJECTID"].astype(int).values,
        "year": col["Year_"].astype(int).values,
        "severity": col["Casualty"].map(SEVERITY).values,
        "persons": col["Persons"].astype(int).values,
        "street": col["Address"].str.strip().values,
        "number": col["Number_"].astype(str).values,
    }, geometry=col.geometry.values, crs=GGRS)
    assert col["severity"].notna().all()

    # snap each collision to the nearest street segment
    j = gpd.sjoin_nearest(col, edges[["geometry"]], how="left", distance_col="snap_m")
    j = j[~j.index.duplicated()]
    col["edge"] = j["index_right"].values
    col["snap_m"] = j["snap_m"].values
    seg = edges.geometry.values[col["edge"].values]
    col["frac"] = shapely.line_locate_point(seg, col.geometry.values, normalized=True)
    log(f"Snapping: median {col['snap_m'].median():.1f} m to the nearest street; "
        f"{(col['snap_m'] <= SNAP_TOL).mean():.0%} within {SNAP_TOL} m (the thesis tolerance)")
    by_sev = col["severity"].value_counts().to_dict()
    log(f"  severity: {by_sev}")

    # --- points of interest -----------------------------------------------------------------------
    area = boundary.union_all()          # points inside the municipality, as in the thesis
    frames = []
    for layer in ("points_of_int", "traffic_data", "transport_data"):
        g = read(layer).to_crs(GGRS)
        g = g[g.intersects(area)]
        frames.append(g[["fclass", "name", "geometry"]])
    pois = pd.concat(frames, ignore_index=True)
    pois = gpd.GeoDataFrame(pois, geometry="geometry", crs=GGRS)
    group = np.select([pois.fclass.isin(ALCOHOL), pois.fclass.isin(EDUCATION),
                       pois.fclass.isin(TRAFFIC), pois.fclass.isin(TRANSPORT)],
                      ["Alcohol", "Education", "Traffic", "Transport"], default="")
    pois["group"] = group
    pois = pois[pois["group"] != ""].reset_index(drop=True)
    log("Points of interest: " + ", ".join(f"{k} {v:,}" for k, v in pois["group"].value_counts().items()))

    # --- emergency facilities ---------------------------------------------------------------------
    poi_all = read("points_of_int").to_crs(GGRS)
    near = poi_all[poi_all.intersects(boundary.buffer(3000).union_all())]
    fac = []
    for r in near[near.fclass == "hospital"].itertuples():
        if not isinstance(r.name, str) or r.name in OSM_HOSPITAL_SKIP:
            continue
        name, kind = OSM_HOSPITAL_KIND.get(r.name, (r.name, "other"))
        fac.append(("hospital", kind, name, r.name, "OpenStreetMap", r.geometry))
    stops = read("transport_data").to_crs(GGRS)
    for name, greek, stop, kind in HOSPITAL_PROXIES:
        s = stops[stops["name"] == stop]
        if s.empty:
            log(f"  ! no stop called {stop}")
            continue
        p = Point(s.geometry.x.mean(), s.geometry.y.mean())
        fac.append(("hospital", kind, name, greek, "OpenStreetMap (bus stop at the entrance)", p))
    for r in near[near.fclass == "police"].itertuples():
        nm = r.name if isinstance(r.name, str) else "Police station"
        fac.append(("police", "police", nm, nm, "OpenStreetMap", r.geometry))
    for r in near[near.fclass == "fire_station"].itertuples():
        nm = r.name if isinstance(r.name, str) else "Fire station"
        fac.append(("fire", "fire", nm, nm, "OpenStreetMap", r.geometry))
    fac = gpd.GeoDataFrame(fac, columns=["type", "kind", "name", "name_local", "source", "geometry"],
                           geometry="geometry", crs=GGRS)
    # one point per named site (OSM sometimes has two)
    fac["x"], fac["y"] = fac.geometry.x, fac.geometry.y
    fac = (fac.groupby(["type", "kind", "name"], as_index=False)
              .agg(name_local=("name_local", "first"), source=("source", "first"), x=("x", "mean"), y=("y", "mean")))
    fac = gpd.GeoDataFrame(fac, geometry=gpd.points_from_xy(fac.x, fac.y), crs=GGRS).drop(columns=["x", "y"])
    # snap to the nearest network junction; drop sites outside the study network
    j = gpd.sjoin_nearest(fac, nodes, how="left", distance_col="snap_m")
    j = j[~j.index.duplicated()]
    fac["node"], fac["snap_m"] = j["node"].values, j["snap_m"].values
    dropped = fac[fac["snap_m"] > FACILITY_MAX]
    fac = fac[fac["snap_m"] <= FACILITY_MAX].reset_index(drop=True)
    log(f"Facilities on the Athens network: {fac.groupby('type').size().to_dict()}; "
        f"{len(dropped)} outside the study area left out")
    for r in fac[fac["type"] == "hospital"].sort_values("kind").itertuples():
        log(f"    {r.kind:9s} {r.name}  ({r.snap_m:.0f} m to network)")

    boundary.to_file(OUT, layer="boundary", driver="GPKG")
    edges.to_file(OUT, layer="edges", driver="GPKG")
    nodes.to_file(OUT, layer="nodes", driver="GPKG")
    col.to_file(OUT, layer="collisions", driver="GPKG")
    pois.to_file(OUT, layer="pois", driver="GPKG")
    fac.to_file(OUT, layer="facilities", driver="GPKG")
    log(f"\nSaved {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
