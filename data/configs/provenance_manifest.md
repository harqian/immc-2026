# config provenance manifest

this file inventories the numeric values in `data/configs/` that most affect the optimization and records whether they are:

- directly source-backed
- source-informed but still derived
- unresolved model assumptions

it does not modify the configs. it is a provenance and replacement guide.

## adopted basis

the configs now use **annualized USD** for cost-like fields.

- `asset_types.*.unit_cost` is interpreted as annualized USD per asset-year
- `daily_asset_availability.budget_total` is interpreted as annualized USD available to spend per scenario-year
- `artificial_waterhole_interventions.capital_cost` and `tourism_cost` are interpreted as annualized USD scenario costs

where a direct annual cost was unavailable, this manifest uses a simple annualization rule and states it explicitly.

## scope

this pass focuses on the highest-impact numeric fields in:

- `asset_types.yaml`
- `daily_asset_availability.yaml`
- `optimization_scenarios.yaml`

before this update, some values in these files could not be responsibly replaced because the config schema did not define a clear **currency basis** or **time basis** for `unit_cost` and `budget_total`. the chosen basis is now:

- annualized USD planning costs

this still leaves some unresolved modeling assumptions, especially around useful life and operating-cost add-ons, but it removes the biggest comparability problem.

## source inventory used in this manifest

1. raw-source manifest for geospatial inputs:
   - [`data/raw/manifest.csv`](/Users/hq/code/immc-risk-intervention/data/raw/manifest.csv)
2. Etosha fire equipment planning:
   - Namibia MEFT fire management strategy, Table 4 for Etosha NP vehicle fire management units
   - https://www.meft.gov.na/files/downloads/66c_Fire%20Management_Strategy%20Final%20Version.pdf
3. Etosha staffing / compensation proxy:
   - MEFT vacancy notice for Etosha control warden and chief warden salary scales
   - https://www.meft.gov.na/files/files/Meft%20integr%20hum.pdf
   - https://www.meft.gov.na/files/files/Ministry-of-Environment_Vacancies.pdf
4. Etosha waterhole / tourism / road ecology:
   - Patterson et al. 2025, Etosha lions and hyenas habitat selection
   - https://the-eis.com/elibrary/sites/default/files/downloads/literature/Effects%20of%20tourism%20on%20seasonal%20movements%20and%20fine_scale%20habitat%20selection%20of%20African%20lions%20and%20spotted%20hyenas%20in%20Etosha%20National%20Park_1.pdf
   - Scott-Hayward et al. 2025, elephant carcass intensity near waterholes and roads
   - https://nje.org.na/index.php/nje/article/view/volume10-scott-hayward
   - NWR live waterhole camera announcement
   - https://www.nwr.com.na/nwr-launches-live-waterhole-camera-in-etosha/
5. primary hardware references:
   - DJI Matrice 4 specs / release / store pages
   - https://enterprise.dji.com/kr/mobile/matrice-4-series/specs
   - https://enterprise.dji.com/mobile/news/detail/matrice-4-series-release
   - https://store.dji.com/hr/product/dji-matrice-4t-plus-combo
   - RECONYX HyperFire 4K camera pricing
   - https://ww4.dev.reconyx.com/product/hyperfire-4k-professional-camera
   - RECONYX cellular plan pricing
   - https://www.reconyx.com/page/cellplan
6. vehicle and infrastructure capital proxies:
   - Pupkewitz Toyota Hilux pricing
   - https://www.pupkewitz-motors.com/toyota/new-cars/hilux-double-cab/
   - Namibia environment ministry borehole handover announcement
   - https://www.meft.gov.na/news/294/environment-ministry-hands-over-20-boreholes-worth-n-141million/
7. exchange-rate references used for this pass:
   - USD/NAD reference
   - https://www.exchange-rates.org/exchange-rate-history/usd-nad-2026
   - AUD/USD reference
   - https://www.tradingeconomics.com/australia/currency
8. 2025/26 ministry budget speech:
   - The Namibian reproduction of Vote 18 speech with programme allocations
   - https://www.namibian.com.na/namibia-allocates-n797-million-for-environment-forestry-and-tourism-in-2025-26-budget/

## decision rules

- `source-backed replacement`: there is a reasonably direct external source for the value or bound
- `source-informed derivation`: there is a source for the physical or institutional basis, but the exact config value is still a modeling derivation
- `model assumption`: no defensible external replacement was found in this pass

## manifest

| file / key | current value | proposed defensible value | status | rationale | confidence |
| --- | --- | --- | --- | --- | --- |
| `asset_types.yaml :: person.unit_cost` | `570.0` | `22470 USD/year` | `source-informed derivation` | MEFT Etosha control warden compensation is listed at `N$337,984–403,922` per annum plus `N$8,760` transport, `N$14,520` housing, and `N$21,000` remoteness allowances for Etosha. using the low end plus allowances gives `N$382,264/year`. converted at `17.0142 NAD/USD` gives about `22467 USD/year`, rounded to `22470`. | medium |
| `asset_types.yaml :: car.unit_cost` | `450.0` | `7810 USD/year` | `source-informed derivation` | Pupkewitz lists a Toyota Hilux 2.4 GD-6 4x4 SR MT from `N$664,500`. converted at `17.0142 NAD/USD` this is about `39056 USD` capital cost. annualizing over a 5-year service life gives about `7811 USD/year`, rounded to `7810`. this is capital recovery only, not fuel or maintenance. | medium |
| `asset_types.yaml :: drone.unit_cost` | `800.0` | `1580 USD/year` | `source-informed derivation` | DJI lists the Matrice 4T Plus Combo at `A$11,248`, and an AUD/USD market rate around `0.70371` on March 6, 2026 implies about `7915 USD` capital cost. annualizing over a 5-year service life gives about `1583 USD/year`, rounded to `1580`. | low |
| `asset_types.yaml :: camera.unit_cost` | `90.0` | `160 USD/year` | `source-informed derivation` | RECONYX HyperFire 4K is listed at `$449.99`, a security enclosure at `$49.99`, and cellular access starts at `$5/month`. using capital cost plus the lowest annual service cost gives `((449.99 + 49.99) / 5) + 60 = 160 USD/year` under a 5-year service life. | medium |
| `asset_types.yaml :: person.response_speed_kmh` | `28.0` | `retain pending better patrol movement data` | `model assumption` | no Etosha-specific patrol-on-foot or patrol-on-vehicle speed study was found. note that `person` currently uses the `car` terrain profile, which is itself a structural modeling choice that should be revisited separately. | low |
| `asset_types.yaml :: car.response_speed_kmh` | `28.0` | `retain or replace with a sourced park-operational travel speed after field validation` | `model assumption` | Etosha road-access sources confirm substantial road infrastructure and staff-only tertiary roads, but no authoritative average operational speed for park patrol vehicles was found in this pass. | low |
| `asset_types.yaml :: drone.response_speed_kmh` | `55.0` | `55-60 km/h` | `source-informed derivation` | DJI Matrice 4 max horizontal speed is `21 m/s` (`~75 km/h`). current `55` is defensible as a conservative below-max operational speed for patrol/search tasks rather than straight-line platform maximum. | medium |
| `asset_types.yaml :: camera.coverage_radius_m` | `75.0` | `30-45 m` | `source-backed replacement` | RECONYX HyperFire 4K lists PIR detect range up to `100 ft` (`~30 m`) and illumination range up to `150 ft` (`~45 m`). current `75 m` looks optimistic for a single fixed camera footprint. | high |
| `asset_types.yaml :: camera_bundle_size` | `5` | `1` if a unit means a single installed waterhole camera; otherwise document explicitly as a multi-camera pod assumption | `source-informed derivation` | NWR’s official Etosha live-camera announcement documents a single live webcam at Okaukuejo and says other waterholes will be covered later. that supports a single-camera interpretation much better than an unexplained bundle of 5. | medium |
| `asset_types.yaml :: camera.risk_suppression_factor` | `0.35` | `retain as calibrated assumption until empirical deterrence evidence is added` | `model assumption` | the Etosha sources found here support that waterholes matter for wildlife aggregation and tourism, but they do not quantify a direct anti-poaching or risk-suppression effect from cameras. | low |
| `daily_asset_availability.yaml :: max_cars` | `100` | `retain 100 by user instruction` | `model assumption` | the MEFT fire management strategy supports a documented minimum baseline of `12` mobile vehicle fire management units, but the config intentionally keeps `100` as a scenario cap rather than a provenance-backed inventory figure. | medium |
| `daily_asset_availability.yaml :: included_cars` | `100` | `retain 100 by user instruction` | `model assumption` | the same caveat applies here: `100` is not source-backed inventory, but it is an explicit scenario decision for now. | medium |
| `daily_asset_availability.yaml :: max_drones` | `3` | `retain until a fleet inventory or procurement source is found` | `model assumption` | no Etosha-specific official drone fleet count or procurement record was found in this pass. | low |
| `daily_asset_availability.yaml :: included_drones` | `0` | `retain unless an existing drone program can be documented` | `model assumption` | no park-specific existing drone inventory source was found in this pass. | low |
| `daily_asset_availability.yaml :: max_cameras` | `20` | `5` if limited to camp/tourism waterholes; `59` as a park-wide upper bound if all artificial waterholes are eligible; current `20` is not source-backed | `source-informed derivation` | Patterson et al. 2025 describes `59 artificial waterholes` in the park and notes that `six tourist camps` exist, with `four` having large lit waterholes, `one` having an unlit waterhole, and `one` having no waterhole. current `20` may be workable as a scenario choice, but it is not obviously tied to the known Etosha waterhole inventory. | medium |
| `daily_asset_availability.yaml :: included_cameras` | `0` | `1` if you want to anchor to the documented Okaukuejo live camera; otherwise retain `0` and label as “no documented baseline inventory” | `source-informed derivation` | NWR officially documents at least one installed live waterhole camera at Okaukuejo. that makes `0` conservative but not strictly reflective of known current park practice. | low |
| `daily_asset_availability.yaml :: budget_total` | `12000.0` | `2542700 USD/year` | `source-informed derivation` | The Namibian's reproduction of the 2025/26 Vote 18 speech reports `N$346,095,000` for the ministry's `Wildlife and protected area management` programme. treating Etosha as roughly `1/8` of that programme gives `N$43,261,875`. converted at `17.0142 NAD/USD`, that is about `2,542,692 USD/year`, rounded to `2,542,700`. this is a top-down Etosha programme-share proxy, not a published Etosha line-item budget. | medium |
| `daily_asset_availability.yaml :: tau_fire_min` | `45.0` | `retain as scenario parameter unless a response-to-suppression standard is adopted` | `model assumption` | the MEFT fire strategy supports vehicle/fire-management needs but does not provide a clean park-wide wildfire response threshold equivalent to this optimization parameter. | low |
| `daily_asset_availability.yaml :: beta_fire` | `0.09` | `retain as calibrated assumption` | `model assumption` | no source in this pass supports a direct exponential fire-delay penalty slope for Etosha. | low |
| `daily_asset_availability.yaml :: lambda_fire` | `2.5` | `retain as calibrated assumption` | `model assumption` | no source in this pass supports a direct response-objective weighting for wildfire penalty relative to travel time. | low |
| `optimization_scenarios.yaml :: top_site_count` | `60` | `retain as scenario-size assumption` | `model assumption` | this is a solver tractability / scenario design knob, not a source-backed park constant. | high |
| `optimization_scenarios.yaml :: merge_distance_m` | `1500.0` | `retain unless field placement standards are defined` | `model assumption` | no field deployment standard for “duplicate nearby candidate sites” was found in this pass. | medium |
| `optimization_scenarios.yaml :: waterhole_influence_radius_m` | `5000.0` | `2500-3000 m` | `source-informed derivation` | Scott-Hayward et al. 2025 found high elephant carcass intensity close to waterholes at distances `<2.5 km`. the current `5 km` is probably too broad if the goal is to model concentrated waterhole effects rather than general water-access landscape influence. | medium |
| `optimization_scenarios.yaml :: protection_benefit.camera_gain_factor` | `0.5` | `retain as calibrated assumption until direct camera-effect evidence is introduced` | `model assumption` | current sources support that waterholes attract wildlife and tourists, but they do not quantify a direct additive protection gain from adding a camera. | low |
| `optimization_scenarios.yaml :: artificial_waterhole_interventions.capital_cost` | `2500.0` | `4150 USD/year` | `source-informed derivation` | Namibia's environment ministry reported `N$14.1 million` for `20` boreholes, implying about `N$705,000` per borehole. converted at `17.0142 NAD/USD` this is roughly `41435 USD` capital cost; annualized over 10 years it is about `4144 USD/year`, rounded to `4150`. this is a rough water-point proxy, not a direct Etosha intervention quote. | medium |
| `optimization_scenarios.yaml :: artificial_waterhole_interventions.tourism_cost` | `400.0` | `retain as modeled penalty unless visitor-revenue or amenity evidence is sourced` | `model assumption` | current sources show waterholes have tourism value, but do not provide a monetized tourism-loss estimate for intervention sites. | low |
| `optimization_scenarios.yaml :: artificial_waterhole_interventions.expected_density_dispersion_benefit` | `0.15` | `retain as ecological scenario assumption` | `model assumption` | published sources show that artificial water provision changes wildlife aggregation, but this pass did not find a clean Etosha-specific percentage effect suitable for direct substitution. | low |

## practical recommendations

if the goal is to improve defensibility without rewriting the model yet, the most supportable next edits would be:

1. keep the annualized USD basis explicit
   - the configs should continue to state that `unit_cost`, `budget_total`, and intervention costs are annualized USD
2. separate scenario choice from provenance
   - `max_cars: 100` and `included_cars: 100` are now intentional scenario values, not source-backed inventory values
   - the closest documented operational baseline found here is still `12` fire-management vehicles
3. reduce camera detection radius to a sourced hardware range
   - replace `camera.coverage_radius_m: 75.0` with something in the `30-45` m range
4. revisit camera bundle semantics
   - if one camera means one installed waterhole camera, use `camera_bundle_size: 1`
   - if the bundle is intentional, document exactly what the 5-camera pod represents
5. tighten waterhole influence distance
   - replace `waterhole_influence_radius_m: 5000.0` with `2500-3000.0` if you want the parameter to reflect concentrated Etosha waterhole effects

## unresolved gaps

the following values still need better external evidence before they can be called defensible:

- full park patrol/staffing headcount for Etosha
- surveillance or anti-poaching vehicle fleet inventory beyond fire-management units
- drone fleet inventory and drone operating-cost basis
- borehole / artificial waterhole intervention capex for Etosha or comparable Namibian parks
- monetized tourism penalty from modifying or adding waterhole infrastructure
- empirical anti-poaching or risk-suppression effect size for cameras
- explicit calibration basis for `tau_fire_min`, `beta_fire`, and `lambda_fire`

until those are sourced, they should remain clearly labeled as scenario assumptions, not inferred provenance.
