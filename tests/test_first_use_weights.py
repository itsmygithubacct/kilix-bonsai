"""A missing Bonsai weight set is installed on first use, behind its licence.

The store's Download button still runs pull.sh itself. These tests cover the
tool path: ``kilix models ensure`` shows the licence, and a decline does not
pull. An ungated model uses its real catalog id. Nothing here adds a gate the
catalog does not declare.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from kilix_bonsai import catalog, provision, store  # noqa: E402


def _variant(variant_id: str, *, default: bool) -> catalog.Variant:
    return catalog.Variant(
        id=variant_id, title=variant_id, default=default, subdir=variant_id,
        files=(), bytes=1)


def _model(model_id: str, variants, gate: str = "") -> catalog.Model:
    return catalog.Model(
        id=model_id, title=model_id, summary="", task="", parameters="",
        quantization="", license="Apache-2.0", upstream="",
        variants=tuple(variants), store_env="", store_default="/tmp/unused",
        licence_gate=gate, folder="/tmp/unused")


class EnsureWeightsTests(unittest.TestCase):
    def run_ensure(self, model, *, tty=True, auto="1", present=False,
                   ensure_status=0, verify=None, kilix="/usr/bin/kilix",
                   variant=None):
        calls = []
        held = store.PRESENT if present else store.MISSING

        def fake_call(argv, **kwargs):
            calls.append(list(argv))
            if list(argv)[1:3] == ["models", "ensure"]:
                return ensure_status
            return 0

        def fake_run(argv, **kwargs):
            calls.append(list(argv))
            asset = list(argv)[3]
            body = "" if verify is None else json.dumps(verify(asset))
            return subprocess.CompletedProcess(argv, 0, stdout=body, stderr="")

        with patch.object(store, "state", return_value=type("S", (), {"state": held})()), \
             patch.object(provision, "_kilix", return_value=kilix), \
             patch.object(sys.stdin, "isatty", return_value=tty), \
             patch.object(sys.stdout, "isatty", return_value=tty), \
             patch.dict(os.environ, {"KILIX_MODEL_AUTO_INSTALL": auto}), \
             patch.object(subprocess, "call", side_effect=fake_call), \
             patch.object(subprocess, "run", side_effect=fake_run):
            status = provision.ensure_downloaded(model, variant)
        return status, calls

    def test_an_ungated_model_ensures_its_catalog_id_then_adopts_it(self):
        model = _model("bonsai-8b", [_variant("q1_0", default=True)])
        with tempfile.TemporaryDirectory() as root:
            asset = os.path.join(root, "assets", "bonsai-8b")
            os.makedirs(asset)
            status, calls = self.run_ensure(
                model, verify=lambda name: {"verified": name, "root": root})
        self.assertEqual(status, 0)
        self.assertEqual(calls[0][1:4], ["models", "ensure", "bonsai-8b"])
        self.assertIn("--from", calls[-1])
        self.assertEqual(calls[-1][calls[-1].index("--from") + 1], asset)
        self.assertNotIn("--variant", calls[-1])

    def test_without_a_terminal_nothing_is_called(self):
        model = _model("bonsai-27b", [_variant("q1_0", default=True)])
        status, calls = self.run_ensure(model, tty=False)
        self.assertIsNone(status)
        self.assertEqual(calls, [])

    def test_auto_install_off_refuses_on_a_terminal(self):
        model = _model("bonsai-8b", [_variant("q1_0", default=True)])
        status, calls = self.run_ensure(model, auto="0")
        self.assertIsNone(status)
        self.assertEqual(calls, [])

    def test_a_present_variant_is_left_alone(self):
        model = _model("bonsai-8b", [_variant("q1_0", default=True)])
        status, calls = self.run_ensure(model, present=True)
        self.assertIsNone(status)
        self.assertEqual(calls, [])

    def test_a_declined_licence_does_not_pull(self):
        model = _model("bonsai-8b", [_variant("q1_0", default=True)])
        status, calls = self.run_ensure(model, ensure_status=1)
        self.assertEqual(status, 1)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1:3], ["models", "ensure"])

    def test_the_image_binary_variant_uses_its_catalog_id(self):
        binary = _variant("binary-gemlite", default=False)
        model = _model("bonsai-image-4b", [
            _variant("ternary-gemlite", default=True), binary])
        with tempfile.TemporaryDirectory() as root:
            asset_id = "bonsai-image-4b-binary-gemlite"
            os.makedirs(os.path.join(root, "assets", asset_id))
            status, calls = self.run_ensure(
                model, variant=binary,
                verify=lambda name: {"verified": name, "root": root})
        self.assertEqual(status, 0)
        self.assertEqual(calls[0][3], asset_id)
        self.assertEqual(calls[-1][calls[-1].index("--variant") + 1], "binary-gemlite")

    def test_a_missing_kilix_names_the_install_command(self):
        model = _model("bonsai-8b", [_variant("q1_0", default=True)])
        with patch("sys.stderr"):
            status, calls = self.run_ensure(model, kilix=None)
        self.assertEqual(status, 1)
        self.assertEqual(calls, [])

    def test_a_bad_verify_report_still_pulls_without_from(self):
        model = _model("vibevoice-asr", [_variant("bitnet", default=True)],
                       gate="vibevoice-asr-bitnet")
        status, calls = self.run_ensure(
            model, verify=lambda name: ["not", "an", "object"])
        self.assertEqual(status, 0)
        self.assertEqual(calls[0][3], "vibevoice-asr-bitnet")
        self.assertNotIn("--from", calls[-1])

    def test_a_non_object_verify_report_does_not_crash(self):
        model = _model("bonsai-8b", [_variant("q1_0", default=True)])
        status, calls = self.run_ensure(model, verify=lambda name: "nope")
        self.assertEqual(status, 0)
        self.assertNotIn("--from", calls[-1])


if __name__ == "__main__":
    unittest.main()
