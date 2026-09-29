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


class LicenceGateTest(unittest.TestCase):
    """A model whose MODEL.json names a licence_gate downloads nothing without a receipt."""

    def pull(self, model_id, *args, check_exit=None, prepare=None, installed=None):
        with tempfile.TemporaryDirectory() as home:
            if prepare is not None:
                args = tuple(prepare(home)) + args
            bin_dir = os.path.join(home, "bin")
            os.mkdir(bin_dir)
            calls = os.path.join(home, "calls")
            for tool in ("curl", "wget"):
                path = os.path.join(bin_dir, tool)
                with open(path, "w", encoding="utf-8") as handle:
                    handle.write(f"#!/bin/sh\necho {tool} >>{calls!r}\nexit 1\n")
                os.chmod(path, 0o755)
            if check_exit is not None:
                path = os.path.join(bin_dir, "kilix-stt")
                with open(path, "w", encoding="utf-8") as handle:
                    handle.write("#!/bin/sh\n"
                                 f"echo \"$*\" >>{calls!r}\n"
                                 f"[ {check_exit} = 0 ] || echo 'Run: kilix models install '\"$2\" >&2\n"
                                 f"exit {check_exit}\n")
                os.chmod(path, 0o755)
            env = dict(os.environ, HOME=home, PATH=f"{bin_dir}:/usr/bin:/bin",
                       KILIX_BONSAI_MODELS_DIR=os.path.join(home, "models"),
                       KILIX_DATA_HOME=os.path.join(home, "data"),
                       KILIX_BONSAI_VIBEVOICE_DIR=os.path.join(home, "vibevoice"))
            script = os.path.join(ROOT, "models", model_id, "pull.sh")
            result = subprocess.run([script, *args], capture_output=True, text=True,
                                    env=env, timeout=120)
            called = open(calls).read().split("\n") if os.path.exists(calls) else []
            if installed is not None:
                target = os.path.join(home, "vibevoice")
                for name in (os.listdir(target) if os.path.isdir(target) else ()):
                    with open(os.path.join(target, name), "rb") as handle:
                        installed[name] = handle.read()
            return result, [line for line in called if line]

    def test_there_is_no_way_to_skip_verification(self) -> None:
        result, calls = self.pull("vibevoice-asr-bitnet", "--no-verify", check_exit=0)
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertEqual(calls, [])

    def test_every_member_of_a_gated_model_has_a_pinned_digest(self) -> None:
        import json
        for model in catalog.load():
            if not model.licence_gate:
                continue
            with open(os.path.join(ROOT, "models", model.id, "MODEL.json")) as handle:
                document = json.load(handle)
            for variant in document["variants"]:
                for source in variant["sources"]:
                    for member in source["files"]:
                        with self.subTest(model=model.id, variant=variant["id"],
                                          path=member["path"]):
                            self.assertRegex(member.get("sha256") or "", r"^[0-9a-f]{64}$")

    def test_a_gated_member_without_a_digest_stops_the_pull(self) -> None:
        import json
        import shutil
        with tempfile.TemporaryDirectory() as root:
            copy = os.path.join(root, "repo")
            shutil.copytree(ROOT, copy, ignore=shutil.ignore_patterns(".git", "__pycache__"))
            manifest = os.path.join(copy, "models", "vibevoice-asr-bitnet", "MODEL.json")
            with open(manifest) as handle:
                document = json.load(handle)
            document["variants"][0]["sources"][0]["files"][-1]["sha256"] = None
            with open(manifest, "w") as handle:
                json.dump(document, handle)
            home = os.path.join(root, "home")
            bin_dir = os.path.join(home, "bin")
            os.makedirs(bin_dir)
            calls = os.path.join(home, "calls")
            for tool, body in (("curl", "exit 1"), ("wget", "exit 1"), ("kilix-stt", "exit 0")):
                path = os.path.join(bin_dir, tool)
                with open(path, "w") as handle:
                    handle.write(f"#!/bin/sh\necho {tool} >>{calls!r}\n{body}\n")
                os.chmod(path, 0o755)
            env = dict(os.environ, HOME=home, PATH=f"{bin_dir}:/usr/bin:/bin",
                       KILIX_DATA_HOME=os.path.join(home, "data"),
                       KILIX_BONSAI_VIBEVOICE_DIR=os.path.join(home, "vibevoice"))
            result = subprocess.run(
                [os.path.join(copy, "models", "vibevoice-asr-bitnet", "pull.sh")],
                capture_output=True, text=True, env=env, timeout=120)
            called = open(calls).read().split() if os.path.exists(calls) else []
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("has no pinned sha256", result.stderr)
        self.assertNotIn("curl", called)
        self.assertNotIn("wget", called)

    def test_a_tampered_installed_file_is_not_called_complete(self) -> None:
        # Seat 2 round 2 (0.2.2 RC3): with every file already the right size,
        # pull.sh said "already complete" without hashing anything.
        import json
        with open(os.path.join(ROOT, "models", "vibevoice-asr-bitnet", "MODEL.json")) as handle:
            members = [m for v in json.load(handle)["variants"][:1]
                       for s in v["sources"] for m in s["files"]]

        def installed_but_tampered(home):
            target = os.path.join(home, "vibevoice")
            os.makedirs(target)
            for member in members:
                # Sparse, correctly sized files; config.json's bytes are wrong.
                with open(os.path.join(target, member["path"]), "wb") as handle:
                    handle.truncate(member["size"])
            return ()

        result, calls = self.pull("vibevoice-asr-bitnet", check_exit=0,
                                  prepare=installed_but_tampered)
        self.assertNotIn("already complete", result.stderr)
        self.assertNotEqual(result.returncode, 0, result.stderr)
        self.assertIn("curl", calls)                 # it tried to replace them

    def test_a_tampered_local_copy_is_never_adopted(self) -> None:
        # Seat 1 (0.2.2 RC3): same-length edits of config.json, which had a
        # digest, and tokenizer_config.json, which had none.
        import json
        with open(os.path.join(ROOT, "models", "vibevoice-asr-bitnet", "MODEL.json")) as handle:
            members = {m["path"]: m for v in json.load(handle)["variants"][:1]
                       for s in v["sources"] for m in s["files"]}

        def tampered(home):
            source = os.path.join(home, "copy")
            os.mkdir(source)
            for name in ("config.json", "tokenizer_config.json"):
                with open(os.path.join(source, name), "wb") as handle:
                    handle.write(b"x" * members[name]["size"])
            return ("--from", source)

        installed = {}
        result, calls = self.pull("vibevoice-asr-bitnet", check_exit=0,
                                  prepare=tampered, installed=installed)
        self.assertNotEqual(result.returncode, 0, result.stderr)
        self.assertIn("curl", calls)                 # fell through to fetching
        for name in ("config.json", "tokenizer_config.json"):
            self.assertNotEqual(installed.get(name), b"x" * members[name]["size"], name)

    def test_the_speech_model_declares_its_gate(self) -> None:
        gated = {model.id: model.licence_gate for model in catalog.load() if model.licence_gate}
        self.assertEqual(gated, {"vibevoice-asr-bitnet": "vibevoice-asr-bitnet"})

    def test_no_receipt_means_no_download(self) -> None:
        for args in ((), ("--from", "/nonexistent"), ("--force",)):
            with self.subTest(args=args):
                result, calls = self.pull("vibevoice-asr-bitnet", *args, check_exit=3)
                self.assertEqual(result.returncode, 3, result.stderr)
                self.assertIn("Run: kilix models install vibevoice-asr-bitnet", result.stderr)
                self.assertEqual(calls, ["--check-licence vibevoice-asr-bitnet"])

    def test_no_checker_means_no_download(self) -> None:
        result, calls = self.pull("vibevoice-asr-bitnet")
        self.assertEqual(result.returncode, 3, result.stderr)
        self.assertIn("Run: kilix models install vibevoice-asr-bitnet", result.stderr)
        self.assertEqual(calls, [])

    def test_a_receipt_lets_the_download_start(self) -> None:
        result, calls = self.pull("vibevoice-asr-bitnet", check_exit=0)
        self.assertEqual(calls[0], "--check-licence vibevoice-asr-bitnet")
        self.assertIn("curl", calls[1:])

    def test_dry_run_needs_no_receipt(self) -> None:
        result, calls = self.pull("vibevoice-asr-bitnet", "--dry-run", check_exit=3)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(calls, [])

    def test_an_ungated_model_is_not_asked_about(self) -> None:
        result, calls = self.pull("bonsai-8b", check_exit=3)
        self.assertNotIn("--check-licence", " ".join(calls))
        self.assertIn("curl", calls)
