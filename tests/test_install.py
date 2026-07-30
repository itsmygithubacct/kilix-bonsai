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
