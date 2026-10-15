"""
Step 4: the built environment and collisions (thesis section 4.4, with stronger methods).

Unit: the 200 m hexagons from step 2 (cells with at least 300 m of street).
Outcome: collisions 2008-2017 in each cell.
Exposure: km of street in the cell (an offset), so a cell isn't "dangerous" just because it has more road.
Explanatory counts per cell, as grouped in the thesis: alcohol venues, education, traffic points
(crossings, signals, stops) and transport points (bus, tram, metro, taxi).

  1. Negative binomial regression (handles over-dispersed counts), reported as rate ratios per
     standard deviation, with 95% confidence intervals, plus Moran's I on the residuals.
  2. Each group on its own (like the thesis's bivariate maps), for comparison.
  3. Geographically Weighted Poisson Regression (the thesis used GWR), adaptive bisquare kernel,
     bandwidth chosen by AICc, giving a local coefficient for each group in each cell.

Run:  python scripts/04_model.py
"""
from pathlib import Path
import json
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "athens.gpkg"
HOT = ROOT / "data" / "processed" / "hotspots.gpkg"
OUT = ROOT / "data" / "processed" / "model.gpkg"
GROUPS = ["Alcohol", "Education", "Traffic", "Transport"]
MIN_ROAD_KM = 0.3


def log(msg=""):
    print(msg, flush=True)


def main():
    import statsmodels.api as sm
    hexes = gpd.read_file(HOT, layer="hexes")
    edges = gpd.read_file(DATA, layer="edges")
    pois = gpd.read_file(DATA, layer="pois")
    OUT.unlink(missing_ok=True)

    # street km per cell
    inter = gpd.overlay(edges[["geometry"]], hexes[["geometry"]].reset_index(), how="intersection",
                        keep_geom_type=True)
    hexes["road_km"] = inter.assign(L=inter.length / 1000).groupby("index")["L"].sum().reindex(
        hexes.index, fill_value=0).values
    j = gpd.sjoin(pois[["group", "geometry"]], hexes[["geometry"]], predicate="within")
    tab = pd.crosstab(j["index_right"], j["group"]).reindex(index=hexes.index, columns=GROUPS, fill_value=0)
    for g in GROUPS:
        hexes[g] = tab[g].values
    d = hexes[hexes["road_km"] >= MIN_ROAD_KM].copy()
    log(f"Cells used: {len(d)} of {len(hexes)} (at least {MIN_ROAD_KM} km of street); "
        f"{int(d['n'].sum())} collisions")

    X = d[GROUPS].astype(float)
    sd = X.std()
    Xs = (X - X.mean()) / sd
    off = np.log(d["road_km"].values)
    y = d["n"].values

    # 1. negative binomial, all groups together
    nb = sm.NegativeBinomial(y, sm.add_constant(Xs), offset=off).fit(disp=0, maxiter=200)
    params, ci = nb.params, nb.conf_int()
    alpha = params["alpha"]
    rows = []
    for g in GROUPS:
        rows.append({"variable": g, "irr_per_sd": float(np.exp(params[g])), "lo": float(np.exp(ci.loc[g, 0])),
                     "hi": float(np.exp(ci.loc[g, 1])), "p": float(nb.pvalues[g]), "sd": float(sd[g]),
                     "mean": float(X[g].mean())})
    res = pd.DataFrame(rows)
    log(f"\nNegative binomial (alpha={alpha:.2f}), rate ratio per standard deviation, exposure = street km:")
    for r in res.itertuples():
        log(f"  {r.variable:9s} {r.irr_per_sd:.2f} ({r.lo:.2f}-{r.hi:.2f}) p={r.p:.4f}   1 SD = {r.sd:.1f} points")
    pois_fit = sm.GLM(y, sm.add_constant(Xs), family=sm.families.Poisson(), offset=off).fit()
    log(f"  overdispersion check: Poisson deviance/df = {pois_fit.deviance / pois_fit.df_resid:.1f}")

    # residual spatial autocorrelation
    from libpysal.weights import Queen
    from esda.moran import Moran
    w = Queen.from_dataframe(d, use_index=False, silence_warnings=True)
    mu = nb.predict(sm.add_constant(Xs), offset=off)
    resid = (y - mu) / np.sqrt(mu + alpha * mu ** 2)
    mi = Moran(resid, w, permutations=999)
    log(f"  Moran's I on residuals: {mi.I:.3f} (p={mi.p_sim:.3f})")

    # 2. each group on its own
    single = []
    for g in GROUPS:
        m = sm.NegativeBinomial(y, sm.add_constant(Xs[[g]]), offset=off).fit(disp=0, maxiter=200)
        single.append({"variable": g, "irr_per_sd": float(np.exp(m.params[g])),
                       "lo": float(np.exp(m.conf_int().loc[g, 0])), "hi": float(np.exp(m.conf_int().loc[g, 1])),
                       "p": float(m.pvalues[g])})
    single = pd.DataFrame(single)
    log("  each on its own: " + ", ".join(f"{r.variable} {r.irr_per_sd:.2f}" for r in single.itertuples()))

    # 3. geographically weighted Poisson regression
    gwr_info = {}
    try:
        from mgwr.gwr import GWR
        from mgwr.sel_bw import Sel_BW
        from spglm.family import Poisson
        cents = np.c_[d.geometry.centroid.x, d.geometry.centroid.y]
        Y = y.reshape(-1, 1).astype(float)
        Xg = Xs.values
        o = d["road_km"].values.reshape(-1, 1)
        bw = Sel_BW(cents, Y, Xg, family=Poisson(), offset=o, kernel="bisquare", fixed=False).search(
            criterion="AICc", bw_min=120, bw_max=len(d) - 1)   # at least 120 cells, so every neighbourhood has some of each kind
        g = GWR(cents, Y, Xg, bw=bw, family=Poisson(), offset=o, kernel="bisquare", fixed=False).fit()
        tv = g.filter_tvals()
        for k, name in enumerate(GROUPS, start=1):
            d[f"gwr_{name}"] = np.exp(g.params[:, k])
            d[f"gwr_{name}_sig"] = tv[:, k] != 0
        gwr_info = {"bandwidth_neighbours": int(bw), "aicc": float(g.aicc)}
        log(f"\nGWR (Poisson): adaptive bandwidth {int(bw)} neighbouring cells, AICc {g.aicc:.0f}")
        for name in GROUPS:
            v = d[f"gwr_{name}"]
            log(f"  {name:9s} local rate ratio per SD: {v.quantile(.1):.2f} to {v.quantile(.9):.2f} "
                f"(10th-90th pct); significant in {d[f'gwr_{name}_sig'].mean():.0%} of cells")
    except Exception as e:  # GWR is optional
        log(f"GWR skipped: {e}")

    d["expected"] = mu
    d["resid"] = resid
    d.to_file(OUT, layer="cells", driver="GPKG")
    summary = {"nb": res.to_dict("records"), "single": single.to_dict("records"), "alpha": float(alpha),
               "cells": int(len(d)), "collisions": int(d["n"].sum()), "moran_resid": float(mi.I),
               "moran_resid_p": float(mi.p_sim), "gwr": gwr_info}
    (ROOT / "data" / "processed" / "model.json").write_text(json.dumps(summary, indent=1))
    log(f"\nSaved {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
