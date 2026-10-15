# Pedestrian collisions in Athens, 2008–2017

**[Open the emergency services dashboard](https://etymo.co.uk/athens-pedestrian-safety/)** · **[Read the story](https://etymo.co.uk/athens-pedestrian-safety/story.html)**

My MSc thesis in Geoinformatics (National Technical University of Athens, 2021), *Spatial-temporal analysis of traffic accidents in Athens: the case of pedestrians*, re-analysed in Python in 2026. The thesis used ArcGIS Pro, ArcMap and an ArcGIS Online dashboard. This version is open, reproducible and runs in any browser.

## What's in it

**The dashboard** (`docs/index.html`) is built for emergency services, like the thesis dashboard:

- Every collision, coloured by severity, with a year filter, a year-by-year playback, a chart by year and a severity split.
- Hospitals, police stations and fire stations.
- Streets coloured by the drive to the nearest hospital, police or fire station, at 20, 30 or 40 km/h.
- **On duty today:** Athens public hospitals take turns on emergency duty (εφημερίες). Tick the ones on duty and every time recalculates.
- **Click anywhere** for the nearest on-duty hospitals, police and fire stations with times and distances, and the collision history within 150 m.
- Street hotspots, space-time hotspot areas, and optional live traffic (see below).

**The story** (`docs/story.html`) walks through the analysis with scrolling maps and charts.

## Main findings

- 1,318 collisions: 30 pedestrians killed, 80 seriously injured, 1,208 slightly injured. Numbers fell by 55% from 2008 to 2017 (Mann-Kendall tau = -0.98, p < 0.001).
- The densest street is Patision near Omonia and Aiolou (25 collisions within 150 m). Killed or seriously injured collisions follow the wide avenues: Alexandras, Vasilissis Sofias and Galatsiou.
- 20 persistent hotspot cells (hot in at least 9 of 10 years), 13 of them around Omonia, matching the thesis's space-time cube. Most new and consecutive hotspots are in the northern districts.
- Once busyness is accounted for, alcohol venues and schools no longer explain where collisions happen; crossings, signals and stops do. Negative binomial rate ratios per standard deviation: traffic points 1.46 (1.32–1.60), transport stops 1.32 (1.20–1.45), alcohol venues 1.06 (0.98–1.14), education 1.04 (0.95–1.13).
- Median drive to the nearest public general hospital: 5.3 minutes at 30 km/h without traffic, from under 1 minute in the east to about 8 minutes in the far west.

## Methods

| Step | Script | What it does |
|---|---|---|
| 1 | `01_prepare.py` | Reads the thesis GeoPackage; cleans the collisions (3 of 1,318 have no location); builds a routable network from the thesis street layer; assembles hospitals, police and fire stations from OpenStreetMap |
| 2 | `02_hotspots.py` | Network kernel density on 50 m street pieces (quartic, 250 m); Getis-Ord Gi* and Local Moran's I on 200 m hexagons (999 permutations); emerging hotspots rebuilt in Python (Gi* per year with a one-year time window, Mann-Kendall trend, ArcGIS category rules); yearly mean centres |
| 3 | `03_access.py` | Drive distance along the network from every street and collision to each hospital and to the nearest police and fire station; walking distance from points of interest to the nearest collision (the thesis's 100/250/500 m zones) |
| 4 | `04_model.py` | Negative binomial regression on hexagons, with km of street as exposure; Moran's I on residuals; Geographically Weighted Poisson Regression (adaptive bisquare, AICc) |
| 5 | `05_build_site.py` | Writes the data files and copies the dashboard and story into `docs/` |

```
python -m pip install -r requirements.txt
python scripts/01_prepare.py
python scripts/02_hotspots.py
python scripts/03_access.py
python scripts/04_model.py
python scripts/05_build_site.py
```

The thesis data goes in `data/raw/athens_thesis.gpkg`. Everything runs in under a minute.

## Live traffic (optional)

Esri's live traffic layer, used in the thesis dashboard, needs an ArcGIS subscription. The dashboard can instead show TomTom's free traffic flow tiles:

1. Create a free account at [developer.tomtom.com](https://developer.tomtom.com) and copy your API key.
2. In the key's settings, restrict it to your website's address, because the key is visible in the page.
3. Run once: `python scripts/05_build_site.py --traffic-key YOUR_KEY` (saved in `config/traffic_key.txt`), then commit and push.

## Limits

- **Times are best-case.** They assume an average speed with no traffic. The thesis street network has no road types or one-way information, so routes may use streets in either direction.
- **Facilities come from OpenStreetMap.** Public hospitals that OpenStreetMap stores only as building outlines (Laiko, Ippokrateio, Elpis, the Red Cross, the children's hospitals, 401) are placed at the bus stop named after them, by the entrance. OpenStreetMap lists only two fire stations in or near the municipality, so fire times are an upper limit. Sites just outside the study network are joined by a straight-line approach leg (×1.3).
- **The data has only the year** of each collision: no day, hour, age or vehicle type.
- The geographically weighted regression settles on its smallest allowed neighbourhood (120 cells), so the local coefficients are noisy; read them as broad patterns.

## Data and licences

Pedestrian collisions 2008–2017: Hellenic Statistical Authority (ELSTAT), from Hellenic Police records, as published with the thesis. Streets, points of interest, hospitals, police and fire stations: © OpenStreetMap contributors (ODbL). Basemaps © Esri.

---

Made by [Eriola Trungu Impersimi](https://etymo.co.uk) · @ET
