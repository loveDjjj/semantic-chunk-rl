"""Prepare and train the semantic planner from reviewed teacher plans."""

import argparse
import json
from pathlib import Path


def recipe_args(parser, argv, section, fields):
    from scrl.config import DEFAULT_CONFIGS

    default = DEFAULT_CONFIGS / "sft.json"
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument("--config", type=Path, default=default)
    chosen, _ = selector.parse_known_args(argv)
    parser.add_argument("--config", type=Path, default=default, help="SFT recipe (paths are supplied separately)")
    try:
        options = json.loads(chosen.config.read_bytes())[section]
        if not isinstance(options, dict) or set(options) != set(fields):
            raise ValueError(f"Expected exactly {', '.join(fields)} in the {section} section")
        for key, value in options.items():
            expected = fields[key]
            if type(value) is not expected and not (expected is float and type(value) is int):
                raise ValueError(f"Invalid type for {section}.{key}")
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    parser.set_defaults(**options)
    return parser.parse_args(argv)
