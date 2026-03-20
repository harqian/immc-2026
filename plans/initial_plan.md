# IMMC Risk Map — Coding Agent Plan

## Goal
Produce a spatial risk heatmap over Etosha National Park (22,935 km²) with composite risk scores per grid cell, broken down by threat type and species. Output: a georeferenced risk tensor `R[zone, species, threat]` and a visual heatmap.

---

## Step 0: Setup

```
pip install geopandas matplotlib shapely rasterio numpy pandas requests
```

Working directory: `immc/risk_map/`

Coordinate system: EPSG:4326 (WGS84). Etosha bounding box (approximate):
- Lat: -19.35 to -18.45
- Lon: 15.4 to 17.15

---

## Step 1: Base Map + Grid Discretization

**Input:** Etosha park boundary (digitize from problem statement map or use OpenStreetMap relation)

**Task:**
1. Download Etosha boundary from OSM (relation ID: 6031845) via Overpass API, OR manually define polygon from the bounding coords
2. Create a grid overlay. Recommended: **5km × 5km cells** → roughly 918 cells covering the park. This is coarse enough to be tractable for the optimizer later but fine enough to capture spatial variation
3. Clip grid to park boundary
4. Tag each cell with terrain type: `pan` (Etosha Pan ~4800 km²), `savanna`, `grassland`, `woodland` — use satellite imagery or the provided map to classify. Simplest approach: manually define the pan polygon (it's a single large feature), everything else is `non-pan`
5. Tag each cell with infrastructure: nearest road (from OSM road network), nearest gate, nearest waterhole (86 waterholes from problem data), nearest camp

**Output:** `grid.geojson` with columns: `cell_id, geometry, terrain, dist_to_road, dist_to_gate, dist_to_waterhole, dist_to_fence`

```python
# Pseudocode
import geopandas as gpd
from shapely.geometry import box
import numpy as np

# Define grid
x_min, x_max = 15.4, 17.15
y_min, y_max = -19.35, -18.45
cell_size = 0.045  # ~5km at this latitude

cells = []
x = x_min
while x < x_max:
    y = y_min
    while y < y_max:
        cells.append(box(x, y, x + cell_size, y + cell_size))
        y += cell_size
    x += cell_size

grid = gpd.GeoDataFrame(geometry=cells, crs="EPSG:4326")
# Clip to park boundary
grid = gpd.overlay(grid, park_boundary, how='intersection')
```

---

## Step 2: Poaching Risk Layer

**Data available:**
- Aggregate stats: 46 rhinos poached in Etosha in 2022, 19 in Q1 2024 (found during dehorning ops), insider threat confirmed
- Spatial proxies (no exact GPS coordinates of poaching incidents are public):
  - Proximity to park gates (poachers need access/egress)
  - Proximity to fence boundary (fence breaches)
  - Proximity to roads (vehicle access for horn extraction)
  - Distance from ranger camps (surveillance gap)
  - Rhino density (target density — rhinos cluster at waterholes, especially at night)

**Task:**
1. Compute per-cell poaching risk as weighted combination of spatial proxies:
   ```
   P(cell) = w1 * inv_dist_gate + w2 * inv_dist_fence + w3 * inv_dist_road
            + w4 * rhino_density(cell) + w5 * inv_dist_ranger_camp
   ```
   Where `inv_dist_X = 1 / (1 + dist_to_X / scale)` (logistic decay)
2. Normalize to [0, 1]
3. Weight calibration: southern fence near Anderson/Galton gates should be highest risk (this is where most access occurs). Use the known fact that poaching is concentrated there.

**Important assumption to document:** We don't have incident-level poaching GPS data. This is a proxy model. Sensitivity analysis in Pillar 4 should vary these weights.

**Output:** `poaching_risk` column added to grid

---

## Step 3: Wildfire Risk Layer

**Data available:**
- Copernicus Sentinel-2 wildfire data (the link you have)
- Wikipedia: ~1/3 of park burned in September 2025 fire
- Terrain type is the main driver: grassland/savanna burns, pan/bare ground doesn't

**Task:**
1. If Copernicus API is accessible: pull MODIS/VIIRS active fire data for Etosha bbox over last 3 years via the Copernicus Data Space API. Extract burn scar polygons
2. Fallback (more likely for comp timeline): use terrain type as primary predictor + distance from roads (fire spread stops at roads/firebreaks)
   ```
   F(cell) = terrain_flammability * (1 - road_proximity_factor) * seasonal_weight
   ```
   Where `terrain_flammability`: pan=0, savanna=0.7, grassland=1.0, woodland=0.9
3. If you got the Copernicus data: overlay historical burn scars to create empirical fire frequency per cell

**Output:** `fire_risk` column added to grid

---

## Step 4: Animal Distribution Layer

**Data available:**
- **Elephants**: Movebank dataset (Tsalyuk et al. 2018) — GPS tracking of elephants in Etosha. ~283K GPS points from 2007-2009. Download as CSV from Movebank (need to agree to terms). Columns: `timestamp, location-long, location-lat`
- **Elephant population**: ~2,500 in Etosha (problem statement)
- **Black rhino**: Exact locations NOT public (security). Known to cluster at waterholes, especially at night. Use waterhole locations as proxy density centers
- **Lions**: No downloadable GPS data. ORC.eco has map of carnivore distribution in Greater Etosha. Manually digitize approximate zones, or use published literature estimates (~300-500 lions in Etosha)
- **Herbivores** (zebra, springbok, oryx, kudu): Broadly distributed across savanna/grassland. Use terrain as proxy

**Task:**
1. **Elephants**: Load Movebank CSV → kernel density estimation (KDE) over grid
   ```python
   from scipy.stats import gaussian_kde
   # Create KDE from GPS points
   coords = np.vstack([lons, lats])
   kde = gaussian_kde(coords, bw_method=0.05)
   # Evaluate at grid cell centroids
   ```
2. **Rhinos**: Create waterhole-centered Gaussian blobs (σ ≈ 5-10km), weighted by waterhole type (natural > man-made, perennial > seasonal). This is a PROXY — document it clearly
3. **Lions/predators**: Assign broad density zones from literature. Western Etosha is known lion territory. Uniform-ish elsewhere except pan (zero)
4. **Herbivores**: Terrain-based. `savanna=high, grassland=high, woodland=medium, pan=zero`

**Output:** Per-cell density estimates for each species group: `elephant_density, rhino_density, lion_density, herbivore_density`

---

## Step 5: Tourism/Human-Wildlife Interaction Risk

**Data available:**
- ~200,000 annual visitors (problem statement)
- Tourism is concentrated on road network between camps (Okaukuejo, Halali, Namutoni, Dolomite, Onkoshi)
- Waterholes along tourist roads are high-traffic

**Task:**
1. Compute per-cell tourism pressure as function of distance to tourist roads and camps
   ```
   T(cell) = visitor_density_proxy * (1 / (1 + dist_to_tourist_road / 5km))
   ```
2. Human-wildlife conflict risk = overlap of tourism pressure × animal density

**Output:** `tourism_risk` column added to grid

---

## Step 6: Composite Risk Score

**Task:**
Combine all layers into composite risk tensor:

```python
# Per species group, per threat type
# R[cell, species, threat] = threat_intensity(cell) * species_density(cell) * species_vulnerability(species, threat)

vulnerability = {
    ('rhino', 'poaching'): 1.0,      # highest
    ('elephant', 'poaching'): 0.3,    # declining, but still exists
    ('rhino', 'fire'): 0.2,           # mobile, can flee
    ('herbivore', 'fire'): 0.4,       # some mortality
    ('lion', 'tourism'): 0.3,         # disturbance
    ('elephant', 'tourism'): 0.5,     # crop raiding near boundaries
    # ... fill matrix
}

# Aggregate: weighted sum across species and threats
# Species weights by conservation priority:
#   black_rhino = 1.0 (endangered, flagship)
#   elephant = 0.6
#   lion = 0.5
#   herbivore = 0.2
```

**Output:** `composite_risk` column + full tensor saved as `risk_tensor.npy`

---

## Step 7: Visualization

**Task:**
1. Plot choropleth heatmap of composite risk over Etosha base map
2. Separate heatmaps per threat type (poaching, fire, tourism)
3. Overlay: waterholes, gates, roads, camps
4. Export as PNG for paper + interactive HTML (folium) for team reference

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
for ax, threat in zip(axes, ['poaching', 'fire', 'tourism']):
    grid.plot(column=f'{threat}_risk', ax=ax, cmap='YlOrRd',
              legend=True, edgecolor='none')
    ax.set_title(f'{threat.title()} risk')
plt.tight_layout()
plt.savefig('risk_heatmaps.png', dpi=300)
```

---

## Data Acquisition Checklist

| Data | Source | Format | Status | Fallback |
|------|--------|--------|--------|----------|
| Park boundary | OSM / problem map | polygon | Need to download | Manual bbox |
| Waterhole locations | Problem statement (86) | lat/lon list | Need to digitize from map | Approximate from map |
| Road network | OSM | linestrings | Need to download | Digitize main roads |
| Elephant GPS tracks | Movebank (Tsalyuk 2018) | CSV | Need to register + download | Use waterhole proximity |
| Rhino locations | NOT PUBLIC | — | Unavailable | Waterhole-based proxy |
| Lion distribution | ORC.eco map | image (no data) | Manual digitize | Literature zones |
| Poaching stats | MEFT reports, news | aggregate counts | Collected (see above) | Already have |
| Fire history | Copernicus / MODIS | raster | Need API access | Terrain-based proxy |
| Gate locations | Problem map | lat/lon | Digitize from map | — |
| Camp locations | Problem map | lat/lon | Digitize from map | — |
| Fence perimeter | Problem statement (~850km) | polygon | Same as park boundary | — |

---

## Key Assumptions to Document in Paper

1. Poaching risk is modeled via spatial proxies (access, surveillance gaps), not incident data
2. Rhino distribution proxied by waterhole proximity — real locations are classified
3. Fire risk is terrain-driven; we don't have multi-year burn scar data unless Copernicus API works
4. Tourism pressure is road-proximity based; no actual visitor tracking data
5. Vulnerability matrix weights are literature-informed estimates, not calibrated
6. Grid resolution (5km) trades spatial precision for computational tractability

---

## Estimated Timeline

| Task | Time |
|------|------|
| Step 0-1: Setup + grid | 1 hr |
| Step 2: Poaching layer | 1.5 hr |
| Step 3: Fire layer | 1 hr |
| Step 4: Animal distributions | 2 hr (depends on Movebank download) |
| Step 5: Tourism layer | 0.5 hr |
| Step 6: Composite | 0.5 hr |
| Step 7: Visualization | 1 hr |
| **Total** | **~7.5 hr** |

---

## File Structure

```
immc/
├── risk_map/
│   ├── data/
│   │   ├── etosha_boundary.geojson
│   │   ├── waterholes.csv          # lat, lon, type, name
│   │   ├── gates.csv               # lat, lon, name
│   │   ├── camps.csv               # lat, lon, name
│   │   ├── elephant_tracks.csv     # from Movebank
│   │   └── roads.geojson           # from OSM
│   ├── scripts/
│   │   ├── 01_build_grid.py
│   │   ├── 02_poaching_risk.py
│   │   ├── 03_fire_risk.py
│   │   ├── 04_animal_density.py
│   │   ├── 05_tourism_risk.py
│   │   ├── 06_composite_risk.py
│   │   └── 07_visualize.py
│   ├── outputs/
│   │   ├── grid.geojson
│   │   ├── risk_tensor.npy
│   │   ├── risk_heatmaps.png
│   │   └── interactive_map.html
│   └── README.md
```
