#!/usr/bin/env python3
from __future__ import annotations

import argparse

from _optimization_common import summarize_bundle, validate_config_bundle, validate_phase1_input_contract


def main() -> int:
    parser = argparse.ArgumentParser(description="validate optimization config and upstream inputs")
    parser.add_argument(
        "--scenario-id",
        default="etosha_placeholder_baseline",
        help="scenario id to validate against the temporary optimization config set",
    )
    args = parser.parse_args()

    bundle = validate_config_bundle()
    validate_phase1_input_contract(bundle, args.scenario_id)
    print(f"{summarize_bundle(bundle)}; upstream optimization inputs passed for {args.scenario_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
