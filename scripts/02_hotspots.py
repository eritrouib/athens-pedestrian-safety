"""
Step 2: where collisions concentrate, and how that changed from 2008 to 2017.

  1. Street hotspots: network kernel density on 50 m street pieces ("lixels"), quartic kernel, 250 m bandwidth,
     for all collisions and for killed or seriously injured (KSI). Density follows the streets instead of
     spreading across buildings, so it fixes the main weakness of planar kernel density in a dense city.
  2. Area hotspots: Getis-Ord Gi* on a 200 m hexagon grid (the thesis's space-time cube cell size),
     999 permutations, with a false discovery rate check.
  3. Clusters and outliers: Anselin Local Moran's I on the same grid.
  4. Emerging hotspots: the ArcGIS "Emerging Hot Spot Analysis" logic rebuilt in Python: Gi* for every year
     (each year plus the one before, like a one-step time neighbourhood), a Mann-Kendall trend test on each
     cell's z-scores, and the same categories (new, consecutive, intensifying, persistent, diminishing,
     sporadic, oscillating, historical).
  5. Trend and centre: a Mann-Kendall test on yearly totals, and the mean centre for each year.

Run:  python scripts/02_hotspots.py
"""
from pathlib import Path
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from scipy import sparse, stats

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "athens.gpkg"
OUT = ROOT / "data" / "processed" / "hotspots.gpkg"
GGRS = "EPSG:2100"
STEP, BANDWIDTH = 50, 250
HEX = 200            # hexagon size (centre to opposite side, metres), as in the thesis
PERMUTATIONS, SEED = 999, 42
YEARS = list(range(2008, 2018))


def log(msg=""):
    print(msg, flush=True)


# ---------------------------------------------------------------------------------------------
# network kernel density
# ---------------------------------------------------------------------------------------------
def lixelize(edges, step):
    lengths = edges.length.values
    n = np.maximum(1, np.ceil(lengths / step).astype(int))
    idx = np.repeat(np.arange(len(edges)), n)
    k = np.arange(len(idx)) - np.repeat(np.cumsum(n) - n, n)
    f0, f1 = k / n[idx], (k + 1) / n[idx]
    geoms = edges.geometry.values[idx]
    c0 = shapely.get_coordinates(shapely.line_interpolate_point(geoms, f0, normalized=True))
    c1 = shapely.get_coordinates(shapely.line_interpolate_point(geoms, f1, normalized=True))
    lix = gpd.GeoDataFrame({"edge": idx, "f0": f0, "f1": f1},
                           geometry=shapely.linestrings(np.stack([c0, c1], axis=1)), crs=GGRS)
    return lix, np.round(c0, 1), np.round(c1, 1)


def adjacency(c0, c1):
    keys, inverse = np.unique(np.vstack([c0, c1]), axis=0, return_inverse=True)
    inverse = inverse.ravel()
    n = len(c0)
    rows = np.concatenate([np.arange(n), np.arange(n)])
    B = sparse.csr_matrix((np.ones(2 * n, dtype=np.float32), (rows, inverse)), shape=(n, len(keys)))
    A = (B @ B.T).tocsr()
    A.setdiag(0)
    A.eliminate_zeros()
    A.data[:] = 1
    return A


def network_kde(A, counts, step, bandwidth):
    hops = int(bandwidth // step)
    weight = lambda d: (1 - (d / bandwidth) ** 2) ** 2 if d < bandwidth else 0.0
    n = A.shape[0]
    one_step = (sparse.identity(n, format="csr", dtype=np.float32) + A).tocsr()
    reach = sparse.identity(n, format="csr", dtype=np.float32)
    density = {k: weight(0) * v for k, v in counts.items()}
    for h in range(1, hops + 1):
        w = weight(h * step)
        if w <= 0:
            break
        new = reach @ one_step
        new.data[:] = 1
        exact = (new - reach).tocsr()
        exact.eliminate_zeros()
        for k, v in counts.items():
            density[k] = density[k] + w * (exact @ v)
        reach = new
    return density


def top_locations(lix, column, col, n=10, gap=600):
    cand = lix[lix[column] > 0].sort_values(column, ascending=False)
    cents = cand.geometry.centroid
    chosen = []
    for i in cand.index:
        p = cents.loc[i]
        if all(p.distance(cents.loc[j]) >= gap for j in chosen):
            chosen.append(i)
        if len(chosen) == n:
            break
    top = lix.loc[chosen].copy()
    # label each place with the streets of the collisions around it (thesis addresses)
    labels, counts = [], []
    for g in top.geometry:
        near = col[col.distance(g.centroid) <= 150]
        st = near["street"].value_counts()
        names = [s.title() for s in st.index[:2]]
        labels.append(" / ".join(names) if names else "")
        counts.append(len(near))
    top["label"] = labels
    top["collisions_150m"] = counts
    return top


# ---------------------------------------------------------------------------------------------
# hexagon grid and spatial statistics
# ---------------------------------------------------------------------------------------------
def hex_grid(area, size):
    """Flat-topped hexagons; `size` is the distance between opposite sides (the cell 'width' used by ArcGIS)."""
    r = size / np.sqrt(3)                       # centre to corner
    dx, dy = 1.5 * r, size
    xmin, ymin, xmax, ymax = area.bounds
    cells = []
    for i, x in enumerate(np.arange(xmin - r, xmax + r, dx)):
        off = 0 if i % 2 == 0 else size / 2
        for y in np.arange(ymin - size + off, ymax + size, dy):
            ang = np.deg2rad(np.arange(0, 360, 60))
            cells.append(shapely.Polygon(np.c_[x + r * np.cos(ang), y + r * np.sin(ang)]))
    g = gpd.GeoDataFrame(geometry=cells, crs=GGRS)
    return g[g.intersects(area)].reset_index(drop=True)


def weights(hexes):
    from libpysal.weights import Queen
    return Queen.from_dataframe(hexes, use_index=False, silence_warnings=True)


def gi_star(values, w, permutations=PERMUTATIONS):
    from esda.getisord import G_Local
    g = G_Local(np.asarray(values, float), w, transform="B", star=True,
                permutations=permutations, seed=SEED, keep_simulations=False)
    p = g.p_sim if permutations else g.p_norm
    return g.Zs, p


def classify_gi(z, p):
    cls = np.full(len(z), "Not significant", dtype=object)
    for conf, cut in ((90, 0.10), (95, 0.05), (99, 0.01)):
        cls[(p <= cut) & (z > 0)] = f"Hot spot {conf}%"
        cls[(p <= cut) & (z < 0)] = f"Cold spot {conf}%"
    return cls


def mann_kendall(series):
    t = np.arange(len(series))
    if np.all(series == series[0]):
        return 0.0, 1.0
    tau, p = stats.kendalltau(t, series)
    return tau, p


def emerging_hotspots(counts, w):
    """counts: cells x years. Returns category per cell (ArcGIS Emerging Hot Spot Analysis definitions)."""
    n, T = counts.shape
    Z = np.zeros((n, T))
    P = np.ones((n, T))
    for t in range(T):
        x = counts[:, t] + (counts[:, t - 1] if t > 0 else 0)   # this year plus the previous one
        if x.sum() == 0:
            continue
        Z[:, t], P[:, t] = gi_star(x, w, permutations=0)
    hot = (Z > 0) & (P <= 0.10)
    cold = (Z < 0) & (P <= 0.10)
    tau = np.zeros(n)
    tp = np.ones(n)
    for i in range(n):
        tau[i], tp[i] = mann_kendall(Z[i])
    cats = np.full(n, "No pattern detected", dtype=object)
    share = hot.mean(axis=1)
    last_hot = hot[:, -1]
    for i in range(n):
        h = hot[i]
        if not h.any():
            continue
        up, down = tp[i] <= 0.05 and tau[i] > 0, tp[i] <= 0.05 and tau[i] < 0
        if share[i] >= 0.9 and last_hot[i]:
            cats[i] = "Intensifying" if up else "Diminishing" if down else "Persistent"
        elif share[i] >= 0.9 and not last_hot[i]:
            cats[i] = "Historical"
        elif last_hot[i]:
            first = np.argmax(h)
            if h.sum() == 1:
                cats[i] = "New"
            elif h[first:].all() and first > 0 and (T - first) < 0.9 * T:
                cats[i] = "Consecutive"
            elif cold[i].any():
                cats[i] = "Oscillating"
            else:
                cats[i] = "Sporadic"
        # hot in some years but not the last, and under 90% of years: no pattern (as in ArcGIS)
    return cats, Z, tau, tp


def local_moran(values, w):
    from esda.moran import Moran_Local
    m = Moran_Local(np.asarray(values, float), w, transformation="r", permutations=PERMUTATIONS, seed=SEED)
    lab = np.array(["Not significant", "High-High", "Low-High", "Low-Low", "High-Low"], dtype=object)
    q = np.where(m.p_sim <= 0.05, m.q, 0)
    return lab[q], m.Is, m.p_sim


def main():
    edges = gpd.read_file(DATA, layer="edges")
    col = gpd.read_file(DATA, layer="collisions")
    boundary = gpd.read_file(DATA, layer="boundary")
    OUT.unlink(missing_ok=True)
    col["ksi"] = col["severity"].isin(["Fatal", "Serious"])

    # ---- 1. network kernel density ----
    lix, c0, c1 = lixelize(edges, STEP)
    A = adjacency(c0, c1)
    # place each collision on its lixel
    nper = lix.groupby("edge").size()
    first = np.concatenate([[0], np.cumsum(nper.values)[:-1]])
    start = pd.Series(first, index=nper.index)
    k = np.minimum((col["frac"].values * nper.loc[col["edge"]].values).astype(int), nper.loc[col["edge"]].values - 1)
    col["lixel"] = start.loc[col["edge"]].values + k
    counts = {"all": np.bincount(col["lixel"], minlength=len(lix)).astype(float),
              "ksi": np.bincount(col.loc[col.ksi, "lixel"], minlength=len(lix)).astype(float)}
    dens = network_kde(A, counts, STEP, BANDWIDTH)
    years = len(YEARS)
    for kk in counts:
        lix[f"n_{kk}"] = counts[kk].astype(int)
        lix[f"kde_{kk}"] = dens[kk] / (BANDWIDTH / 1000) / years    # weighted collisions per km of street per year
    log(f"Lixels: {len(lix):,} pieces of ~{STEP} m; peak density {lix.kde_all.max():.1f} collisions/km/year")
    top_all = top_locations(lix, "kde_all", col)
    top_ksi = top_locations(lix, "kde_ksi", col[col.ksi])
    log("Top street hotspots (all collisions):")
    for r in top_all.itertuples():
        log(f"   {r.kde_all:5.1f}  {r.label}  ({r.collisions_150m} collisions within 150 m)")

    # ---- 2-4. hexagon grid ----
    area = boundary.union_all()
    hexes = hex_grid(area, HEX)
    j = gpd.sjoin(col[["year", "ksi", "geometry"]], hexes, predicate="within", how="inner")
    tab = pd.crosstab(j["index_right"], j["year"]).reindex(index=hexes.index, columns=YEARS, fill_value=0)
    hexes["n"] = tab.sum(axis=1).values
    hexes["n_ksi"] = j[j.ksi].groupby("index_right").size().reindex(hexes.index, fill_value=0).values
    for y in YEARS:
        hexes[f"y{y}"] = tab[y].values
    w = weights(hexes)
    z, p = gi_star(hexes["n"].values, w)
    import esda
    hexes["gi_z"], hexes["gi_p"] = z, p
    hexes["gi_class"] = classify_gi(z, p)
    hexes["gi_fdr"] = p <= esda.fdr(p, 0.05)
    log(f"Gi*: {(hexes.gi_class.str.startswith('Hot')).sum()} hot cells at 90%+, "
        f"{(hexes.gi_fdr & (hexes.gi_z > 0)).sum()} of them pass the false discovery rate check")
    hexes["lisa"], hexes["lisa_I"], hexes["lisa_p"] = local_moran(hexes["n"].values, w)
    log("Local Moran's I: " + ", ".join(f"{k} {v}" for k, v in hexes.lisa.value_counts().items()))
    cats, Z, tau, tp = emerging_hotspots(tab.values.astype(float), w)
    hexes["emerging"], hexes["trend_tau"], hexes["trend_p"] = cats, tau, tp
    log("Emerging hotspots: " + ", ".join(f"{k} {v}" for k, v in pd.Series(cats).value_counts().items()))

    # ---- 5. overall trend and mean centres ----
    yearly = col.groupby("year").size().reindex(YEARS, fill_value=0)
    tau_all, p_all = mann_kendall(yearly.values)
    log(f"Yearly totals: {yearly.to_dict()}  Mann-Kendall tau={tau_all:.2f}, p={p_all:.4f}")
    mc = col.groupby("year").apply(lambda d: pd.Series({"x": d.geometry.x.mean(), "y": d.geometry.y.mean(), "n": len(d)}))
    centres = gpd.GeoDataFrame(mc.reset_index(), geometry=gpd.points_from_xy(mc.x, mc.y), crs=GGRS)

    lix.to_file(OUT, layer="lixels", driver="GPKG")
    hexes.to_file(OUT, layer="hexes", driver="GPKG")
    top_all.to_file(OUT, layer="top_all", driver="GPKG")
    top_ksi.to_file(OUT, layer="top_ksi", driver="GPKG")
    centres.to_file(OUT, layer="mean_centres", driver="GPKG")
    col[["id", "lixel"]].to_csv(ROOT / "data" / "processed" / "collision_lixel.csv", index=False)
    pd.DataFrame({"tau": [tau_all], "p": [p_all]}).to_csv(ROOT / "data" / "processed" / "trend.csv", index=False)
    log(f"\nSaved {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
