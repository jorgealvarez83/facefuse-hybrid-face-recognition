"""CLI surface and configuration defaults."""

import pytest

from facefuse import config
from facefuse.cli import COMMANDS, main
from facefuse.recognition import resolve_ctx_id


def test_bare_invocation_lists_the_commands(capsys):
    assert main([]) == 0
    out = capsys.readouterr().out
    for command in COMMANDS:
        assert command in out


def test_version_flag(capsys):
    assert main(["--version"]) == 0
    assert "facefuse" in capsys.readouterr().out


def test_unknown_command_exits_nonzero(capsys):
    assert main(["nope"]) == 2
    assert "unknown command" in capsys.readouterr().err


@pytest.mark.parametrize("module_name", [target for _, target in COMMANDS.values()])
def test_every_subcommand_module_exposes_a_parser_and_main(module_name):
    module = __import__(module_name, fromlist=["main", "build_parser"])
    assert callable(module.main)
    parser = module.build_parser()
    assert parser.prog.startswith("facefuse ")
    assert "--help" in parser.format_help()


def test_pipeline_parser_defaults_and_flags():
    from facefuse.pipeline import build_parser

    args = build_parser().parse_args([])
    assert args.optimize is True
    assert args.device == "auto"
    assert build_parser().parse_args(["--no-optimize"]).optimize is False
    assert build_parser().parse_args(["--no-show"]).no_show is True


def test_config_paths_are_absolute_and_under_the_repo():
    for path in (config.WEIGHTS_PATH, config.GALLERY_DIR, config.PROBE_DIR):
        assert path.is_absolute()
    assert config.GALLERY_DIR.parent == config.ROOT / "data"


def test_ga_bounds_are_well_formed():
    assert len(config.GA_BOUNDS) == 5
    for low, high in config.GA_BOUNDS:
        assert low < high


@pytest.mark.parametrize("device,expected", [("cpu", -1), ("gpu", 0), ("cuda", 0), (3, 3)])
def test_device_resolution(device, expected):
    assert resolve_ctx_id(device) == expected


def test_auto_device_resolution_returns_a_valid_ctx_id():
    assert resolve_ctx_id("auto") in (-1, 0)


def test_core_modules_import_without_the_heavy_model_stack():
    """torch / ultralytics / insightface must stay optional at import time.

    Keeps CI light and lets `pad_image`, the metrics and the fusion rule be used
    without a multi-gigabyte install.
    """
    import importlib
    import sys

    heavy = {"torch", "ultralytics", "insightface"}
    already_loaded = heavy & set(sys.modules)

    for name in ["facefuse.face_detection", "facefuse.hybrid_recognition",
                 "facefuse.recognition", "facefuse.pipeline", "facefuse.annotate",
                 "facefuse.realtime", "facefuse.evaluation"]:
        importlib.import_module(name)

    newly_loaded = (heavy & set(sys.modules)) - already_loaded
    assert not newly_loaded, f"importing facefuse pulled in {sorted(newly_loaded)}"
