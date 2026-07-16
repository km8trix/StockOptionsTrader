from __future__ import annotations

from pathlib import Path

import pytest

from scripts.launch_with_env import LaunchEnvironmentError, environment_from_dotenv


def test_dotenv_values_are_literal_and_do_not_expand_dollar_names(tmp_path: Path):
    path = tmp_path / ".env"
    path.write_text(
        "APP_AUTH_USERNAME=operator\n"
        "APP_AUTH_PASSWORD='correct$wiper$HOME'\n"
        "EMPTY=\n",
        encoding="utf-8",
    )

    result = environment_from_dotenv(path, base_environment={"HOME": "/secret/home"})

    assert result == {
        "HOME": "/secret/home",
        "APP_AUTH_USERNAME": "operator",
        "APP_AUTH_PASSWORD": "correct$wiper$HOME",
        "EMPTY": "",
    }


def test_dotenv_overrides_only_declared_child_values(tmp_path: Path):
    path = tmp_path / ".env"
    path.write_text("A=new\n", encoding="utf-8")
    assert environment_from_dotenv(path, base_environment={"A": "old", "B": "kept"}) == {
        "A": "new",
        "B": "kept",
    }


def test_bare_name_without_value_is_rejected(tmp_path: Path):
    path = tmp_path / ".env"
    path.write_text("MISSING\n", encoding="utf-8")
    with pytest.raises(LaunchEnvironmentError, match="has no value"):
        environment_from_dotenv(path, base_environment={})


def test_symlinked_dotenv_is_rejected(tmp_path: Path):
    target = tmp_path / "target"
    target.write_text("A=value\n", encoding="utf-8")
    link = tmp_path / ".env"
    link.symlink_to(target)
    with pytest.raises(LaunchEnvironmentError, match="regular file"):
        environment_from_dotenv(link, base_environment={})
