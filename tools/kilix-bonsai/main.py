"""kilix-bonsai — the command, and the TUI it opens with no arguments.

Everything the TUI can do is also a subcommand, for the same reason
`kilix-settings` has `--set`: provisioning a machine, a Makefile, and an SSH
session all want the non-interactive path, and a UI that is the only way to
reach a behaviour makes that behaviour untestable.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "src"))

from kilix_bonsai import catalog, paths, provision, store, tui  # noqa: E402


def _model(model_id: str):
    try:
        return catalog.find(model_id)
    except catalog.CatalogError as error:
        print(f"kilix-bonsai: {error}", file=sys.stderr)
        raise SystemExit(2) from error


def cmd_list(args: argparse.Namespace) -> int:
    models = catalog.load()
    width = max((len(model.id) for model in models), default=10)
    for model in models:
        variant = model.default_variant
        state = store.variant_state(model, variant)
        print(f"{model.id:<{width}}  {store.human_bytes(variant.bytes):>7}  "
              f"{state.state:<8}  {model.task:<16}  {model.title}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    models = [_model(args.model)] if args.model else catalog.load()
    for model in models:
        print(f"{model.title} ({model.id})")
        print(f"  store        {model.store}")
        if model.shared_with:
            print(f"  shared with  {model.shared_with}")
        deps = store.deps_state(model)
        print("  dependencies " + ("installed" if deps else "not installed"))
        for variant in model.variants:
            state = store.variant_state(model, variant)
            detail = state.state
            if state.state == store.PARTIAL:
                detail = (f"partial "
                          f"{store.human_bytes(state.present_bytes)}/"
                          f"{store.human_bytes(variant.bytes)}, "
                          f"{len(state.missing)} files outstanding")
            print(f"  {variant.id:<18} {store.human_bytes(variant.bytes):>7}  "
                  f"{detail}")
    return 0


def cmd_path(args: argparse.Namespace) -> int:
    model = _model(args.model)
    print(model.variant(args.variant).directory(model.store))
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    """Emit one variant as tab-separated lines, for the shell scripts to read.

    This exists so `pull.sh` and `install-deps.sh` never parse MODEL.json or
    re-derive a store path themselves. The catalog and the path rules have one
    implementation; the scripts own the transfer and the packages, which is the
    part a shell is genuinely better at.
    """
    model = _model(args.model)
    variant = model.variant(args.variant)
    directory = variant.directory(model.store)
    out = sys.stdout
    print(f"MODEL\t{model.id}", file=out)
    print(f"TITLE\t{model.title}", file=out)
    print(f"VARIANT\t{variant.id}", file=out)
    print(f"DIR\t{directory}", file=out)
    print(f"STORE\t{model.store}", file=out)
    print(f"VENV\t{paths.venv_dir(model.id)}", file=out)
    print(f"BYTES\t{variant.bytes}", file=out)
    for package in model.deps.get("apt") or ():
        print(f"APT\t{package}", file=out)
    for package in model.deps.get("pip") or ():
        print(f"PIP\t{package}", file=out)
    for item in variant.files:
        print(f"FILE\t{item.path}\t{item.size}\t{item.sha256 or '-'}\t"
              f"{item.url}", file=out)
    return 0


def cmd_pull(args: argparse.Namespace) -> int:
    model = _model(args.model)
    variant = model.variant(args.variant)
    return provision.run(provision.pull_argv(
        model, variant, force=args.force, dry_run=args.dry_run,
        source=args.source))


def cmd_deps(args: argparse.Namespace) -> int:
    model = _model(args.model)
    return provision.run(provision.deps_argv(model, apt=args.apt))


def cmd_verify(args: argparse.Namespace) -> int:
    model = _model(args.model)
    variant = model.variant(args.variant)
    ok = store.verify(model, variant, report=lambda path, good, detail: print(
        f"{'ok  ' if good else 'FAIL'} {path} — {detail}"))
    return 0 if ok else 1


def cmd_doctor(args: argparse.Namespace) -> int:
    """Report every prerequisite a download depends on, before it costs time."""
    ok = True

    def check(label: str, good: bool, detail: str) -> None:
        nonlocal ok
        ok = ok and good
        print(f"{'ok  ' if good else 'FAIL'} {label:<22} {detail}")

    downloader = shutil.which("curl") or shutil.which("wget")
    check("downloader", bool(downloader), downloader or
          "neither curl nor wget is installed; pull.sh needs one")
    check("sha256sum", bool(shutil.which("sha256sum")),
          shutil.which("sha256sum") or
          "no sha256sum; downloads cannot be verified")
    for name, value in sorted(paths.variables().items()):
        print(f"     {name:<22} {value}")
    for model in catalog.load():
        root = model.store
        while root and not os.path.isdir(root):
            parent = os.path.dirname(root)
            if parent == root:
                break
            root = parent
        writable = bool(root) and os.access(root, os.W_OK)
        free = shutil.disk_usage(root).free if root else 0
        needed = model.default_variant.bytes
        state = store.variant_state(model, model.default_variant)
        if state.state == store.PRESENT:
            check(model.id, True, f"ready in {model.store}")
            continue
        outstanding = needed - state.present_bytes
        check(model.id, writable and free > outstanding,
              f"{store.human_bytes(outstanding)} to fetch, "
              f"{store.human_bytes(free)} free under {root}"
              + ("" if writable else " (NOT WRITABLE)"))
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kilix-bonsai",
        description="BitNet models for Kilix: inspect, download, verify.")
    parser.add_argument("--screenshot", metavar="PATH",
                        help="render one frame of the TUI to PATH and exit")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("list", help="one line per model").set_defaults(
        func=cmd_list)

    status = sub.add_parser("status", help="what is on disk")
    status.add_argument("model", nargs="?")
    status.set_defaults(func=cmd_status)

    path = sub.add_parser("path", help="print where a variant's weights live")
    path.add_argument("model")
    path.add_argument("--variant")
    path.set_defaults(func=cmd_path)

    plan = sub.add_parser(
        "plan", help="tab-separated variant description, read by the scripts")
    plan.add_argument("model")
    plan.add_argument("--variant")
    plan.set_defaults(func=cmd_plan)

    pull = sub.add_parser("pull", help="download a model's weights")
    pull.add_argument("model")
    pull.add_argument("--variant")
    pull.add_argument("--from", dest="source", metavar="DIR",
                      help="adopt verified files from a copy already on this "
                           "machine instead of downloading them")
    pull.add_argument("--force", action="store_true")
    pull.add_argument("--dry-run", action="store_true")
    pull.set_defaults(func=cmd_pull)

    deps = sub.add_parser("deps", help="install a model's dependencies")
    deps.add_argument("model")
    deps.add_argument("--apt", action="store_true",
                      help="install system packages too (uses sudo)")
    deps.set_defaults(func=cmd_deps)

    verify = sub.add_parser("verify", help="hash a variant against its digests")
    verify.add_argument("model")
    verify.add_argument("--variant")
    verify.set_defaults(func=cmd_verify)

    sub.add_parser("doctor",
                   help="check downloader, space, and store permissions"
                   ).set_defaults(func=cmd_doctor)
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(argv)
    if args.command is None:
        return tui.main(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
