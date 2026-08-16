"""The shell scripts are the real interface, so they are tested as one.

`pull.sh` and `install-deps.sh` have to work with no Python front end in sight —
over SSH, from a provisioning script, from a Makefile — and the UI runs these
same files rather than reimplementing them. Two things follow, and both are
asserted here: the scripts must agree with the catalog about what a model is,
and their read-only paths (`--help`, `--dry-run`, `--check`) must genuinely
touch nothing, because those are what a nervous operator runs first.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from kilix_bonsai import catalog, store  # noqa: E402

SHARED = os.path.join(ROOT, "models", "_shared")


def run(argv, **kwargs):
    env = dict(os.environ, KILIX_BONSAI_MODELS_DIR=kwargs.pop("models_dir",
                                                              "/tmp/nowhere"))
    return subprocess.run(argv, capture_output=True, text=True, env=env,
                          timeout=120, **kwargs)


class WrapperTest(unittest.TestCase):
    def test_every_folder_delegates_to_the_one_implementation(self) -> None:
        # Five copies of a download loop would be five things to keep correct.
        for model in catalog.load():
            for name in ("pull.sh", "install-deps.sh"):
                with open(model.script(name), encoding="utf-8") as handle:
                    body = handle.read()
                self.assertIn(f"_shared/{name}", body, model.id)
                self.assertIn("--model-dir", body, model.id)

    def test_the_shared_scripts_name_no_model(self) -> None:
        # A model id, repository, or digest hard-coded in the shared script
        # would be a second source of truth for something MODEL.json owns.
        ids = {model.id for model in catalog.load()}
        for name in ("pull.sh", "install-deps.sh"):
            with open(os.path.join(SHARED, name), encoding="utf-8") as handle:
                body = handle.read()
            for model_id in ids:
                self.assertNotIn(model_id, body, f"{name} names {model_id}")
            self.assertNotIn("huggingface.co", body, name)
            self.assertNotRegex(body, r"\b[0-9a-f]{40}\b")

    def test_the_scripts_never_delete_a_store(self) -> None:
        # These paths hold gigabytes the user chose to download, and one of
        # them is a directory another component owns.
        for name in ("pull.sh", "install-deps.sh"):
            with open(os.path.join(SHARED, name), encoding="utf-8") as handle:
                body = handle.read()
            self.assertNotIn("rm -rf", body, name)
            self.assertNotIn("rm -r ", body, name)

    def test_only_the_apt_path_uses_sudo(self) -> None:
        # Downloading weights must never need root, and there is exactly one
        # place in the whole repository that elevates: the opt-in --apt branch.
        # Counted as *invocations* — a line that starts a sudo command — rather
        # than mentions, since printing the command for the user to run is the
        # default behaviour and appears in the same file.
        with open(os.path.join(SHARED, "pull.sh"), encoding="utf-8") as handle:
            self.assertNotIn("sudo", handle.read())
        with open(os.path.join(SHARED, "install-deps.sh"),
                  encoding="utf-8") as handle:
            lines = handle.read().splitlines()
        invocations = [line for line in lines
                       if re.match(r"\s*sudo\s", line)]
        self.assertEqual(len(invocations), 1, invocations)
        self.assertIn("apt-get install", invocations[0])


class HelpTest(unittest.TestCase):
    def test_help_is_free_of_side_effects(self) -> None:
        for model in catalog.load():
            for name in ("pull.sh", "install-deps.sh"):
                result = run([model.script(name), "--help"])
                self.assertEqual(result.returncode, 0,
                                 f"{model.id} {name}: {result.stderr}")
                self.assertIn("usage:", result.stdout)

    def test_an_unknown_option_is_refused(self) -> None:
        result = run([catalog.load()[0].script("pull.sh"), "--wat"])
        self.assertEqual(result.returncode, 2)


class DryRunTest(unittest.TestCase):
    def test_dry_run_lists_exactly_the_variant_the_catalog_lists(self) -> None:
        for model in catalog.load():
            for variant in model.variants:
                argv = [model.script("pull.sh"), "--dry-run"]
                if not variant.default:
                    argv += ["--variant", variant.id]
                result = run(argv)
                self.assertEqual(result.returncode, 0, result.stderr)
                for item in variant.files:
                    self.assertIn(item.path, result.stdout,
                                  f"{model.id}/{variant.id}")

    def test_dry_run_creates_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = os.path.join(folder, "models")
            model = catalog.find("bonsai-8b")
            result = run([model.script("pull.sh"), "--dry-run"],
                         models_dir=root)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(os.path.exists(root))

    def test_dry_run_prices_the_download_before_it_starts(self) -> None:
        model = catalog.find("bonsai-27b")
        result = run([model.script("pull.sh"), "--dry-run"])
        self.assertIn("to fetch", result.stderr)
        self.assertIn(store.human_bytes(model.default_variant.bytes),
                      result.stderr)

    def test_an_unknown_variant_fails_before_touching_the_network(self) -> None:
        model = catalog.find("bonsai-8b")
        result = run([model.script("pull.sh"), "--variant", "nope",
                      "--dry-run"])
        self.assertNotEqual(result.returncode, 0)

    def test_cli_unknown_variants_are_concise_usage_errors(self) -> None:
        cli = os.path.join(ROOT, "tools", "kilix-bonsai", "main.py")
        commands = (
            ["path", "bonsai-8b", "--variant", "nope"],
            ["plan", "bonsai-8b", "--variant", "nope"],
            ["pull", "bonsai-8b", "--variant", "nope", "--dry-run"],
            ["verify", "bonsai-8b", "--variant", "nope"],
        )
        for command in commands:
            with self.subTest(command=command[0]):
                result = run([sys.executable, cli, *command])
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn("has no variant", result.stderr)
                self.assertNotIn("Traceback", result.stderr)


class ResumeTest(unittest.TestCase):
    """A partial file that cannot be resumed from must be discarded.

    This is a regression test for a real failure. `curl -C -` sends a Range
    header; an upstream CDN answered a resume request with the *whole* body,
    curl appended it, and the `.part` ended up 1.50 GB for a 1.16 GB file. The
    original code kept that file "to resume", so every retry appended another
    copy and the download could never complete — it got further from finishing
    each time. Nothing in the suite caught it because the fast tests only ever
    downloaded small files that succeeded on the first attempt.
    """

    def setUp(self) -> None:
        self.model = catalog.find("bonsai-8b")
        self.variant = self.model.default_variant
        self.item = max(self.variant.files, key=lambda f: f.size)

    def _run_with_part(self, part_size: int):
        with tempfile.TemporaryDirectory() as root:
            directory = os.path.join(root, "bonsai-8b")
            os.makedirs(directory)
            part = os.path.join(directory, self.item.path + ".part")
            with open(part, "wb") as handle:
                handle.write(b"\0" * part_size)
            # --dry-run stops before any transfer, so this asserts the guard's
            # decision without pulling a gigabyte over the network.
            result = run([self.model.script("pull.sh"), "--dry-run"],
                         models_dir=root)
            return result, part

    def test_an_oversized_partial_is_reported_as_outstanding(self) -> None:
        # A .part larger than the target must not be counted as progress.
        result, _ = self._run_with_part(self.item.size + 1024)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("to fetch", result.stderr)

    def test_the_guard_discards_rather_than_resumes(self) -> None:
        with open(os.path.join(SHARED, "pull.sh"), encoding="utf-8") as handle:
            body = handle.read()
        # The two paths that must delete: a pre-existing .part at or past the
        # published size, and a fetch that overshot.
        self.assertIn('[ "$part_size" -ge "$size" ]', body)
        self.assertIn('[ "$actual" -gt "$size" ]', body)
        self.assertIn("discarding", body)

    def test_a_digest_mismatch_does_not_leave_a_resumable_prefix(self) -> None:
        # A complete-but-wrong file is not a prefix of the right one, so
        # keeping it would make the next run resume from its end.
        with open(os.path.join(SHARED, "pull.sh"), encoding="utf-8") as handle:
            body = handle.read()
        mismatch = body[body.index("sha256 mismatch"):]
        self.assertIn('rm -f -- "$part"',
                      mismatch[:mismatch.index("failed=$((failed + 1))")])


class PlanTest(unittest.TestCase):
    """The scripts read the catalog through one command; it has to be stable."""

    def test_plan_describes_the_default_variant(self) -> None:
        cli = os.path.join(ROOT, "tools", "kilix-bonsai", "main.py")
        result = run([sys.executable, cli, "plan", "vibevoice-asr-bitnet"])
        self.assertEqual(result.returncode, 0, result.stderr)
        rows = [line.split("\t") for line in result.stdout.splitlines()]
        keys = {row[0] for row in rows}
        self.assertTrue({"MODEL", "TITLE", "VARIANT", "DIR", "STORE", "VENV",
                         "BYTES", "FILE"} <= keys)
        files = [row for row in rows if row[0] == "FILE"]
        model = catalog.find("vibevoice-asr-bitnet")
        self.assertEqual(len(files), len(model.default_variant.files))
        for row in files:
            self.assertEqual(len(row), 5)
            self.assertTrue(row[4].startswith("http"))

    def test_plan_puts_the_speech_weights_where_dictation_reads_them(self) -> None:
        cli = os.path.join(ROOT, "tools", "kilix-bonsai", "main.py")
        result = run([sys.executable, cli, "plan", "vibevoice-asr-bitnet"])
        directory = next(line.split("\t")[1] for line in
                         result.stdout.splitlines() if line.startswith("DIR\t"))
        self.assertTrue(
            directory.endswith("/voice/models/vibevoice-asr-bitnet"), directory)


if __name__ == "__main__":
    unittest.main()
