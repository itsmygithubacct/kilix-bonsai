"""The two shared stores resolve to what their owning components actually read.

This is the suite that earns its keep. Two of the five model stores belong to
other components — the speech weights to kilix-voice's catalog, the image
weights to the image scaffold's data directory — and getting either path wrong
does not fail loudly. It downloads several gigabytes to somewhere plausible and
leaves the other component still looking at an empty directory.

So rather than trusting a comment, these tests re-derive each path the way its
owner does and assert the two agree, including under the environment overrides
the stack supports.
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from kilix_bonsai import catalog, paths  # noqa: E402


class EnvironmentTest(unittest.TestCase):
    """Save and restore the variables these tests move."""

    VARIABLES = ("GPU_TERMINAL_HOME", "KILIX_STORAGE_HOME", "KILIX_DATA_HOME",
                 "KILIX_BONSAI_MODELS_DIR", "KILIX_BONSAI_VIBEVOICE_DIR",
                 "KILIX_BONSAI_BONSAI_IMAGE_4B_DIR")

    def setUp(self) -> None:
        self._saved = {name: os.environ.get(name) for name in self.VARIABLES}
        for name in self.VARIABLES:
            os.environ.pop(name, None)

    def tearDown(self) -> None:
        for name, value in self._saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


class SharedStoreTest(EnvironmentTest):
    def test_speech_weights_land_in_the_voice_catalog(self) -> None:
        # kilix-voice builds a speech model directory as
        # KILIX_DATA_HOME/voice/models/<catalog id>, and the catalog id it
        # reserves for this model is 'vibevoice-asr-bitnet'.
        model = catalog.find("vibevoice-asr-bitnet")
        self.assertEqual(model.store,
                         paths.voice_model_dir("vibevoice-asr-bitnet"))

    def test_the_voice_path_follows_the_stack_defaults(self) -> None:
        os.environ["GPU_TERMINAL_HOME"] = "/tmp/gt"
        self.assertEqual(
            catalog.find("vibevoice-asr-bitnet").store,
            "/tmp/gt/kilix/data/voice/models/vibevoice-asr-bitnet")

    def test_the_voice_path_follows_an_overridden_data_home(self) -> None:
        # A machine that relocated Kilix's data directory must not end up with
        # the weights in the old place and dictation reading the new one.
        os.environ["KILIX_DATA_HOME"] = "/tmp/elsewhere"
        self.assertEqual(
            catalog.find("vibevoice-asr-bitnet").store,
            "/tmp/elsewhere/voice/models/vibevoice-asr-bitnet")

    def test_image_weights_land_in_the_image_scaffold_directory(self) -> None:
        os.environ["GPU_TERMINAL_HOME"] = "/tmp/gt"
        model = catalog.find("bonsai-image-4b")
        self.assertEqual(model.store, "/tmp/gt/bonsai_image_generation")

    def test_image_variant_subdirectories_are_the_ones_upstream_looks_for(self):
        os.environ["GPU_TERMINAL_HOME"] = "/tmp/gt"
        model = catalog.find("bonsai-image-4b")
        directories = {v.id: v.directory(model.store) for v in model.variants}
        self.assertEqual(
            directories["ternary-gemlite"],
            "/tmp/gt/bonsai_image_generation/bonsai-image-4B-ternary-gemlite")
        self.assertEqual(
            directories["binary-gemlite"],
            "/tmp/gt/bonsai_image_generation/bonsai-image-4B-binary-gemlite")


class OwnStoreTest(EnvironmentTest):
    def test_models_this_repository_owns_share_one_root(self) -> None:
        os.environ["GPU_TERMINAL_HOME"] = "/tmp/gt"
        for model_id in ("bonsai-8b", "bonsai-27b", "bitnet-b1.58-2b4t"):
            self.assertEqual(catalog.find(model_id).store,
                             f"/tmp/gt/kilix-bonsai/models/{model_id}")

    def test_a_model_env_override_wins_outright(self) -> None:
        os.environ["GPU_TERMINAL_HOME"] = "/tmp/gt"
        os.environ["KILIX_BONSAI_VIBEVOICE_DIR"] = "/mnt/big/vibe"
        self.assertEqual(catalog.find("vibevoice-asr-bitnet").store,
                         "/mnt/big/vibe")

    def test_no_two_models_share_a_store_directory(self) -> None:
        stores = [model.store for model in catalog.load()]
        self.assertEqual(len(stores), len(set(stores)))

    def test_the_virtualenv_is_never_inside_a_store_someone_else_owns(self) -> None:
        # Dropping a Python environment into the directory kilix-voice
        # enumerates as its speech catalog would make it look like a model.
        for model in catalog.load():
            venv = paths.venv_dir(model.id)
            self.assertFalse(venv.startswith(model.store + os.sep), model.id)


class ResolutionTest(EnvironmentTest):
    def test_an_unknown_variable_is_refused_rather_than_left_literal(self) -> None:
        with self.assertRaises(paths.PathError) as caught:
            paths.resolve("$NOT_A_STORE_ROOT/models")
        self.assertIn("$NOT_A_STORE_ROOT", str(caught.exception))

    def test_paths_are_absolute_and_normalised(self) -> None:
        os.environ["GPU_TERMINAL_HOME"] = "~/somewhere/../gt"
        resolved = paths.resolve("$GPU_TERMINAL_HOME/x")
        self.assertTrue(os.path.isabs(resolved))
        self.assertNotIn("..", resolved)
        self.assertNotIn("~", resolved)


if __name__ == "__main__":
    unittest.main()
