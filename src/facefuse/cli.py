"""``facefuse`` command-line entry point."""

from __future__ import annotations

import sys

from . import __version__

COMMANDS = {
    "annotate": ("recognise faces in an image and save an annotated copy",
                 "facefuse.annotate"),
    "webcam": ("run live recognition on a webcam or video file",
               "facefuse.realtime"),
    "benchmark": ("enrol, tune and benchmark hybrid vs deep-only",
                  "facefuse.pipeline"),
}


def _usage() -> str:
    lines = ["usage: facefuse <command> [options]", "", "commands:"]
    lines += [f"  {name:<11} {help_text}" for name, (help_text, _) in COMMANDS.items()]
    lines += ["", "Run 'facefuse <command> --help' for command options."]
    return "\n".join(lines)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)

    if not argv or argv[0] in ("-h", "--help", "help"):
        print(_usage())
        return 0
    if argv[0] in ("-V", "--version"):
        print(f"facefuse {__version__}")
        return 0

    command = argv[0]
    if command not in COMMANDS:
        print(f"facefuse: unknown command {command!r}\n", file=sys.stderr)
        print(_usage(), file=sys.stderr)
        return 2

    module_name = COMMANDS[command][1]
    module = __import__(module_name, fromlist=["main"])
    module.main(argv[1:])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
