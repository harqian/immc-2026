# Raw data contract

`data/raw/` holds source artifacts exactly as acquired or manually curated for the Etosha MVP.
normalization scripts are responsible for converting these inputs into stable layers under
`data/processed/`; they should not rewrite the originals in place.

## required structure

each dataset gets its own subdirectory:

- `etosha_boundary/`
- `roads/`
- `waterholes/`
- `gates/`
- `camps/`
- `elephants/`
- `rhino_reference/`
- `carnivore_reference/`
- `wildfires/`

keep the original download, export, screenshot, map extract, or manually curated table inside
the matching directory. if a source requires a one-time manual step, store the resulting artifact
here and record the exact method in `manifest.csv`.

if a source is image-only or figure-only, keep both:

- the original image or document extract
- the GCP table used to georeference it
- the digitized derivative used downstream, such as a csv of approximate detections

the digitized derivative is still a raw artifact here because it is a source-specific extraction,
not a normalized analysis layer.

## manifest rules

`manifest.csv` is the provenance index for this folder. every MVP dataset must have one row with:

- `dataset_id`: stable short identifier used by scripts
- `source_url`: original source page or download endpoint
- `acquisition_method`: `scripted_download`, `manual_download`, `manual_digitization`,
  `manual_curation`, or another explicit method
- `local_path`: dataset directory or file path under `data/raw/`
- `license_or_terms`: reuse constraints or access terms
- `date_acquired`: `YYYY-MM-DD` when the artifact was collected; leave blank until acquired
- `scripted`: `true` or `false`
- `manual_steps`: exact human actions needed to obtain or refresh the artifact
- `notes`: scope, caveats, or downstream interpretation notes

## acquisition policy

- do not invent substitute data silently; if a source is missing, leave the raw directory empty and
  keep the manifest row explicit about what is still needed
- if licensing prevents checking in the raw file, keep the manifest row and document the retrieval
  steps precisely enough for a teammate to reproduce them
- prefer keeping dataset-level provenance here and file-level transformation logic in scripts
- preserve source names, timestamps, and any accompanying metadata files when available
- when a map image is digitized, record the georeferencing method and error caveats in the
  manifest notes instead of presenting the resulting coordinates as survey-grade truth

## phase 1 status

at the end of phase 1, the expected outcome is scaffolding plus provenance definitions. most raw
directories may still be empty until source acquisition is completed in phase 2.
