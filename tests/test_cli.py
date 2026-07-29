#!/usr/bin/env python3
"""The CLI's promises that do not need a model or a runtime: resolution
order, refusals with a next step in them, and the doctor not crashing."""

import os
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLI = os.path.join(REPO, "bin", "bonsai-cpu")

SIZE_8B = 1158654496

failures = []


def check(cond, msg):
    if not cond:
        failures.append(msg)


def run(args, env_extra):
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("KILIX_BONSAI_", "BONSAI_CPU_", "GPU_TERMINAL_"))}
    env.update(env_extra)
    return subprocess.run([sys.executable, CLI] + args,
                          capture_output=True, text=True, env=env)


def main():
    with tempfile.TemporaryDirectory() as tmp:
        store = os.path.join(tmp, "store")
        override = os.path.join(tmp, "override")
        runtime = os.path.join(tmp, "runtime")
        os.makedirs(os.path.join(store, "bonsai-8b"))
        os.makedirs(override)
        base = {"KILIX_BONSAI_MODELS_DIR": store,
                "BONSAI_CPU_RUNTIME_DIR": runtime}

        # A store file of the pinned size (sparse — content is not the point).
        gguf = os.path.join(store, "bonsai-8b", "Bonsai-8B-Q1_0.gguf")
        with open(gguf, "wb") as fh:
            fh.truncate(SIZE_8B)

        r = run(["path"], base)
        check(r.returncode == 0 and r.stdout.strip() == gguf,
              f"path resolves via KILIX_BONSAI_MODELS_DIR: {r.stdout!r} "
              f"{r.stderr!r}")

        # The per-model env var wins over the store root.
        ov_gguf = os.path.join(override, "Bonsai-8B-Q1_0.gguf")
        with open(ov_gguf, "wb") as fh:
            fh.truncate(SIZE_8B)
        r = run(["path"], {**base, "KILIX_BONSAI_BONSAI_8B_DIR": override})
        check(r.returncode == 0 and r.stdout.strip() == ov_gguf,
              "per-model dir override wins")

        # --model bypasses the store entirely, even a nonsense one.
        r = run(["path", "--model", "/nonexistent/x.gguf"],
                {**base, "KILIX_BONSAI_MODELS_DIR": "/nonexistent"})
        check(r.returncode == 0 and r.stdout.strip() == "/nonexistent/x.gguf",
              "--model bypasses store resolution")

        # A missing model names the command that fixes it.
        os.remove(gguf)
        r = run(["path"], base)
        check(r.returncode == 2 and "kilix-bonsai pull bonsai-8b" in r.stderr,
              f"missing model points at the store: {r.stderr!r}")

        # A short file reads as an interrupted download, not a mystery.
        with open(gguf, "wb") as fh:
            fh.truncate(1234)
        r = run(["path"], base)
        check(r.returncode == 2 and "interrupted" in r.stderr,
              f"truncated model is diagnosed: {r.stderr!r}")

        # 27B is runnable since its graph was verified under the pinned
        # runtime (2026-07-29); a missing download is still refused with the
        # command that fixes it.
        r = run(["run", "--model-id", "bonsai-27b", "hi"], base)
        check(r.returncode == 2 and "kilix-bonsai pull bonsai-27b" in r.stderr,
              f"missing 27b points at the store: {r.stderr!r}")

        os.makedirs(os.path.join(store, "bonsai-27b"), exist_ok=True)
        gguf27 = os.path.join(store, "bonsai-27b", "Bonsai-27B-Q1_0.gguf")
        with open(gguf27, "wb") as fh:
            fh.truncate(3803452480)
        r = run(["path", "--model-id", "bonsai-27b"], base)
        check(r.returncode == 0 and r.stdout.strip() == gguf27,
              f"path resolves 27b: {r.stderr!r}")

        # With the model present the 27B passes the gate and fails only on
        # the absent runtime — proving supported models reach execution.
        r = run(["run", "--model-id", "bonsai-27b", "hi"], base)
        check(r.returncode == 2 and "bonsai-cpu build" in r.stderr,
              f"27b passes the gate to the runtime check: {r.stderr!r}")

        # verify: a wrong-content file must fail the digest.
        small = os.path.join(tmp, "wrong.gguf")
        with open(small, "wb") as fh:
            fh.write(b"not a model")
        r = run(["verify", "--model", small], base)
        check(r.returncode == 1 and "mismatch" in r.stderr,
              f"verify fails on wrong bytes: {r.stderr!r}")

        # run/chat without a runtime say how to get one.
        with open(gguf, "wb") as fh:
            fh.truncate(SIZE_8B)
        r = run(["run", "hello"], base)
        check(r.returncode == 2 and "bonsai-cpu build" in r.stderr,
              f"missing runtime points at build: {r.stderr!r}")

        # doctor reports rather than crashes, whatever the machine has.
        r = run(["doctor"], base)
        check(r.returncode in (0, 1) and "doctor" in r.stdout,
              f"doctor runs: rc={r.returncode} {r.stderr!r}")

    for msg in failures:
        print(f"FAIL: {msg}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
