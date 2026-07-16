"""Explicitly load a dotenv file as data, then replace this process.

This helper exists for ``start.sh``.  It deliberately disables dotenv
interpolation so credential characters such as ``$`` are never interpreted as
shell variables.  Values are passed only in the child environment and are not
printed or placed on the command line.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import sys
from typing import Mapping, Sequence

from dotenv import dotenv_values


_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class LaunchEnvironmentError(ValueError):
    """The explicitly selected dotenv file cannot be loaded safely."""


def environment_from_dotenv(
    path: str | Path, *, base_environment: Mapping[str, str] | None = None
) -> dict[str, str]:
    """Return a child environment with literal, non-interpolated dotenv values."""
    env_path = Path(path)
    if not env_path.is_file() or env_path.is_symlink():
        raise LaunchEnvironmentError("dotenv path must be a regular file")
    try:
        parsed = dotenv_values(dotenv_path=env_path, interpolate=False)
    except (OSError, UnicodeError) as exc:
        raise LaunchEnvironmentError("dotenv file could not be read") from exc
    child = dict(os.environ if base_environment is None else base_environment)
    for name, value in parsed.items():
        if _ENV_NAME.fullmatch(name) is None:
            raise LaunchEnvironmentError("dotenv contains an invalid variable name")
        if value is None:
            raise LaunchEnvironmentError(f"dotenv variable {name} has no value")
        if "\x00" in value:
            raise LaunchEnvironmentError(f"dotenv variable {name} contains a NUL byte")
        child[name] = value
    return child


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--env-file", required=True)
    parser.add_argument("--script", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    script = Path(args.script).resolve()
    if not script.is_file() or script.is_symlink():
        print("ERROR: launcher script must be a regular file", file=sys.stderr)
        return 1
    try:
        environment = environment_from_dotenv(args.env_file)
    except LaunchEnvironmentError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Loaded dotenv configuration from {Path(args.env_file).resolve()}")
    os.execve(sys.executable, [sys.executable, str(script)], environment)
    return 1  # pragma: no cover - os.execve replaces the process


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

