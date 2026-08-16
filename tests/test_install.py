"""One install publishes every command the model stack can launch."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys_path = str(ROOT / "src")
if sys_path not in sys.path:
    sys.path.insert(0, sys_path)

from kilix_bonsai.runtime import chat  # noqa: E402


COMMANDS = (
    "kilix-bonsai",
    "kilix-bonsai-chat",
    "kilix-bonsai-image",
    "kilix-bonsai-speech",
    "bonsai-cpu",
)


class InstallTest(unittest.TestCase):
    def test_installer_publishes_the_bundled_cpu_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            prefix = Path(temporary) / "prefix"
            environment = dict(
                os.environ,
                KILIX_BONSAI_PREFIX=str(prefix),
            )
            result = subprocess.run(
                [str(ROOT / "install.sh")],
                capture_output=True,
                text=True,
                env=environment,
                timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("installed 5 commands", result.stdout)

            for command in COMMANDS:
                launcher = prefix / "bin" / command
                self.assertTrue(launcher.is_file(), command)
                self.assertTrue(os.access(launcher, os.X_OK), command)

            cpu_launcher = (prefix / "bin" / "bonsai-cpu").read_text(
                encoding="utf-8"
            )
            self.assertIn(
                str(ROOT / "bonsai-cpu" / "bin" / "bonsai-cpu"),
                cpu_launcher,
            )

            help_result = subprocess.run(
                [str(prefix / "bin" / "bonsai-cpu"), "--help"],
                capture_output=True,
                text=True,
                env=environment,
                timeout=30,
            )
            self.assertEqual(help_result.returncode, 0, help_result.stderr)
            self.assertIn("usage:", help_result.stdout)

    def test_uninstall_removes_only_generated_launchers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            prefix = Path(temporary) / "prefix"
            environment = dict(
                os.environ,
                KILIX_BONSAI_PREFIX=str(prefix),
            )
            install = subprocess.run(
                [str(ROOT / "install.sh")], capture_output=True, text=True,
                env=environment, timeout=30,
            )
            self.assertEqual(install.returncode, 0, install.stderr)
            unrelated = prefix / "bin" / "unrelated-command"
            unrelated.write_text("user owned\n", encoding="utf-8")

            uninstall = subprocess.run(
                [str(ROOT / "install.sh"), "--uninstall"],
                capture_output=True, text=True, env=environment, timeout=30,
            )
            self.assertEqual(uninstall.returncode, 0, uninstall.stderr)
            self.assertIn("removed 5 launchers", uninstall.stdout)
            for command in COMMANDS:
                self.assertFalse((prefix / "bin" / command).exists(), command)
            self.assertEqual(
                unrelated.read_text(encoding="utf-8"), "user owned\n"
            )

            repeated = subprocess.run(
                [str(ROOT / "install.sh"), "--uninstall"],
                capture_output=True, text=True, env=environment, timeout=30,
            )
            self.assertEqual(repeated.returncode, 0, repeated.stderr)
            self.assertIn("removed 0 launchers", repeated.stdout)

    def test_uninstall_retains_a_modified_launcher(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            prefix = Path(temporary) / "prefix"
            environment = dict(
                os.environ,
                KILIX_BONSAI_PREFIX=str(prefix),
            )
            install = subprocess.run(
                [str(ROOT / "install.sh")], capture_output=True, text=True,
                env=environment, timeout=30,
            )
            self.assertEqual(install.returncode, 0, install.stderr)
            modified = prefix / "bin" / "kilix-bonsai"
            modified.write_text("#!/bin/sh\nexit 7\n", encoding="utf-8")

            uninstall = subprocess.run(
                [str(ROOT / "install.sh"), "--uninstall"],
                capture_output=True, text=True, env=environment, timeout=30,
            )
            self.assertNotEqual(uninstall.returncode, 0)
            self.assertTrue(modified.exists())
            self.assertIn("modified launcher", uninstall.stderr)
            for command in COMMANDS[1:]:
                self.assertFalse((prefix / "bin" / command).exists(), command)

    def test_standalone_cpu_uninstall_is_scoped_and_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            prefix = Path(temporary) / "prefix"
            environment = dict(os.environ, BONSAI_CPU_PREFIX=str(prefix))
            installer = ROOT / "bonsai-cpu" / "install.sh"
            install = subprocess.run(
                [str(installer)], capture_output=True, text=True,
                env=environment, timeout=30,
            )
            self.assertEqual(install.returncode, 0, install.stderr)
            launcher = prefix / "bin" / "bonsai-cpu"
            self.assertTrue(launcher.exists())

            for _ in range(2):
                uninstall = subprocess.run(
                    [str(installer), "--uninstall"], capture_output=True,
                    text=True, env=environment, timeout=30,
                )
                self.assertEqual(uninstall.returncode, 0, uninstall.stderr)
                self.assertFalse(launcher.exists())

    def test_coordinated_version_is_reported_consistently(self) -> None:
        self.assertEqual((ROOT / "VERSION").read_text().strip(), "0.2.0")
        self.assertEqual(
            (ROOT / "bonsai-cpu" / "VERSION").read_text().strip(), "0.2.0"
        )
        for command, expected in (
            (ROOT / "tools" / "kilix-bonsai" / "main.py",
             "kilix-bonsai 0.2.0"),
            (ROOT / "bonsai-cpu" / "bin" / "bonsai-cpu",
             "bonsai-cpu 0.2.0"),
        ):
            result = subprocess.run(
                [sys.executable, str(command), "--version"],
                capture_output=True, text=True, timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(), expected)

    def test_launchers_quote_an_arbitrary_checkout_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            linked_root = Path(temporary) / "checkout'with\"quotes"
            linked_root.symlink_to(ROOT, target_is_directory=True)
            prefix = Path(temporary) / "prefix"
            environment = dict(
                os.environ,
                KILIX_BONSAI_PREFIX=str(prefix),
                BONSAI_CPU_PREFIX=str(prefix),
            )

            for installer in (
                linked_root / "install.sh",
                linked_root / "bonsai-cpu" / "install.sh",
            ):
                result = subprocess.run(
                    [str(installer)], capture_output=True, text=True,
                    env=environment, timeout=30,
                )
                self.assertEqual(result.returncode, 0, result.stderr)

            for command in ("kilix-bonsai", "bonsai-cpu"):
                launcher = prefix / "bin" / command
                syntax = subprocess.run(
                    ["sh", "-n", str(launcher)], capture_output=True,
                    text=True, timeout=30,
                )
                self.assertEqual(syntax.returncode, 0, syntax.stderr)
                help_result = subprocess.run(
                    [str(launcher), "--help"], capture_output=True,
                    text=True, env=environment, timeout=30,
                )
                self.assertEqual(
                    help_result.returncode, 0, help_result.stderr
                )
                self.assertIn("usage:", help_result.stdout)

    def test_runtime_falls_back_to_the_bundled_component(self) -> None:
        saved_which = chat.shutil.which
        saved_prefix = os.environ.get("BONSAI_CPU_PREFIX")
        with tempfile.TemporaryDirectory() as prefix:
            chat.shutil.which = lambda name: None
            os.environ["BONSAI_CPU_PREFIX"] = prefix
            try:
                self.assertEqual(
                    chat.cpu_runtime(),
                    str(ROOT / "bonsai-cpu" / "bin" / "bonsai-cpu"),
                )
            finally:
                chat.shutil.which = saved_which
                if saved_prefix is None:
                    os.environ.pop("BONSAI_CPU_PREFIX", None)
                else:
                    os.environ["BONSAI_CPU_PREFIX"] = saved_prefix


if __name__ == "__main__":
    unittest.main()
