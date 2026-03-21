#!/usr/bin/env python3
from __future__ import annotations

import argparse

from _optimization_common import summarize_bundle, validate_config_bundle, validate_phase1_input_contract


def main() -> int:
    parser = argparse.ArgumentParser(description="build and validate surveillance optimization inputs")
    parser.add_argument(
        "--scenario-id",
        default="etosha_placeholder_baseline",
        help="scenario id to validate against the temporary optimization config set",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate the temporary optimization configs and phase 1 input contract without solving",
    )
    args = parser.parse_args()

    bundle = validate_config_bundle()
    validate_phase1_input_contract(bundle, args.scenario_id)
    if args.validate_only:
        print(f"{summarize_bundle(bundle)}; phase 1 input contract passed for {args.scenario_id}")
        return 0
    raise NotImplementedError("optimization solve implementation starts in later phases; use --validate-only for phase 1")


if __name__ == "__main__":
    raise SystemExit(main())
