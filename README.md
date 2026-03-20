# Etosha actual-data mvp

this repo contains a staged geospatial pipeline for building an Etosha risk-analysis foundation
from real-source inputs. phase 1 establishes the project scaffold, raw-data contract, manifest,
and environment checks; later phases will add normalization, feature engineering, modeling, and
visualization scripts.

## bootstrap

this machine uses an externally managed system python, so install dependencies in a local virtual
environment instead of the global interpreter:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/01_build_manifest.py
python scripts/00_validate_environment.py
python scripts/04_digitize_reference_maps.py
python scripts/01_build_manifest.py --check
```

## current repo contract

- raw inputs live under `data/raw/` and keep original provenance
- processed normalized layers will be written under `data/processed/`
- reusable model outputs will be written under `outputs/`
- `data/raw/manifest.csv` is generated from `scripts/01_build_manifest.py` and should not be
  hand-edited without updating the script definition
- image-only wildlife references are first digitized into raw csv artifacts by
  `scripts/04_digitize_reference_maps.py`, then normalized into buffered support layers

phase 1 does not require the actual source artifacts to be present yet; empty raw dataset
directories are expected until source acquisition begins.
