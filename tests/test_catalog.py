"""Every MODEL.json is complete, consistent, and says where its weights go.

These files are the single source of truth for four callers — two shell
scripts, a CLI, and a UI — and their sizes and digests are the only thing
standing between a truncated download and a model that loads to garbage. A typo
in one would be discovered as a failed 4 GB transfer, so it is worth asserting
here instead.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from kilix_bonsai import catalog  # noqa: E402

EXPECTED = {"bonsai-8b", "bonsai-27b", "bonsai-image-4b",
            "vibevoice-asr-bitnet", "bitnet-b1.58-2b4t"}


class CatalogTest(unittest.TestCase):
    def setUp(self) -> None:
        self.models = catalog.load()

    def test_every_model_folder_loads(self) -> None:
        self.assertEqual({model.id for model in self.models}, EXPECTED)

    def test_folder_name_matches_the_declared_id(self) -> None:
        # The scripts derive the model id from their own directory name, so a
        # folder called one thing and declaring another would send pull.sh
        # looking for a model that does not exist.
        for model in self.models:
            self.assertEqual(os.path.basename(model.folder), model.id)

    def test_each_model_has_exactly_one_default_variant(self) -> None:
        for model in self.models:
            defaults = [v for v in model.variants if v.default]
            self.assertEqual(len(defaults), 1, model.id)

    def test_variant_ids_are_unique_within_a_model(self) -> None:
        for model in self.models:
            ids = [variant.id for variant in model.variants]
            self.assertEqual(len(ids), len(set(ids)), model.id)

    def test_declared_bytes_match_the_sum_of_the_files(self) -> None:
        for model in self.models:
            for variant in model.variants:
                self.assertEqual(
                    variant.bytes, sum(f.size for f in variant.files),
                    f"{model.id}/{variant.id}")

    def test_large_files_all_carry_a_digest(self) -> None:
        # HuggingFace stores anything over its LFS threshold with a sha256; a
        # multi-megabyte file arriving without one means the catalog was hand
        # written rather than read from upstream.
        for model in self.models:
            for variant in model.variants:
                for item in variant.files:
                    if item.size > 10 * 1024 * 1024:
                        self.assertTrue(
                            item.sha256, f"{model.id} {item.path} has no digest")
                        self.assertRegex(item.sha256, r"^[0-9a-f]{64}$")

    def test_no_installable_variant_has_a_null_digest(self) -> None:
        # B1 / R4-058: the two previously-null VibeVoice metadata files, fetched
        # read-only from microsoft/VibeVoice-ASR-BitNet@66e78021 on
        # 2026-09-17T17:39:01Z. The default variant is what Enter downloads.
        model = catalog.find("vibevoice-asr-bitnet")
        pinned = {
            "config.json":
                "4873cb753c97f042886a93adfef478978662c21169755670de20f2ff73fa4ca1",
            "generation_config.json":
                "8c970692323e3ea0e9b8b0a4dca79388d31226e41f83c9fd6014804280ebf6e8",
        }
        sizes = {"config.json": 3527, "generation_config.json": 138}
        self.assertTrue(model.default_variant.default)
        for variant in model.variants:
            found = {}
            for item in variant.files:
                if item.path not in pinned:
                    continue
                self.assertTrue(
                    item.sha256,
                    f"{model.id}/{variant.id} {item.path} has a null digest")
                self.assertRegex(item.sha256, r"^[0-9a-f]{64}$")
                self.assertEqual(item.size, sizes[item.path], item.path)
                found[item.path] = item.sha256
            self.assertEqual(found, pinned, variant.id)
        with tempfile.TemporaryDirectory() as folder:
            with open(os.path.join(model.folder, "MODEL.json"),
                      encoding="utf-8") as handle:
                raw = json.load(handle)
            for variant in raw["variants"]:
                for source in variant.get("sources", ()):
                    for entry in source.get("files", ()):
                        if entry.get("path") == "config.json":
                            entry["sha256"] = None
            planted_path = os.path.join(folder, "MODEL.json")
            with open(planted_path, "w", encoding="utf-8") as handle:
                json.dump(raw, handle)
            planted_model = catalog.load_model(folder)
            planted_found = {
                item.path: item.sha256
                for item in planted_model.default_variant.files
                if item.path in pinned
            }
            self.assertIsNone(planted_found["config.json"])
            with self.assertRaises(AssertionError):
                self.assertEqual(planted_found, pinned)

    def test_revisions_are_full_commit_shas(self) -> None:
        # A branch name here would silently install whatever HEAD happened to
        # be at download time, and would do it differently on every machine.
        for model in self.models:
            for variant in model.variants:
                for item in variant.files:
                    self.assertRegex(item.revision, r"^[0-9a-f]{40}$",
                                     f"{model.id} {item.path}")

    def test_no_file_path_escapes_its_store(self) -> None:
        for model in self.models:
            for variant in model.variants:
                for item in variant.files:
                    self.assertFalse(item.path.startswith("/"), item.path)
                    self.assertNotIn("..", item.path.split("/"))

    def test_urls_point_at_the_pinned_revision(self) -> None:
        for model in self.models:
            item = model.default_variant.files[0]
            self.assertIn(f"/{item.repo}/resolve/{item.revision}/", item.url)

    def test_default_variant_is_never_the_largest_conversion_source(self) -> None:
        # Where a model publishes both a ready-to-use quantization and the
        # unquantized source, the default must be the small one: the default is
        # what a confirmation key press downloads.
        for model in self.models:
            largest = max(model.variants, key=lambda v: v.bytes)
            if len(model.variants) > 1 and largest.bytes > 2 * \
                    model.default_variant.bytes:
                self.assertFalse(largest.default, model.id)

    def test_every_folder_has_its_scripts_and_they_are_executable(self) -> None:
        for model in self.models:
            for name in ("pull.sh", "install-deps.sh"):
                path = model.script(name)
                self.assertTrue(os.path.isfile(path), path)
                self.assertTrue(os.access(path, os.X_OK), path)

    def test_every_folder_has_a_readme(self) -> None:
        for model in self.models:
            self.assertTrue(
                os.path.isfile(os.path.join(model.folder, "README.md")),
                model.id)

    def test_an_unknown_variant_is_a_clear_error(self) -> None:
        model = self.models[0]
        with self.assertRaises(catalog.CatalogError) as caught:
            model.variant("no-such-variant")
        self.assertIn("no-such-variant", str(caught.exception))

    def test_an_unknown_model_names_the_ones_that_exist(self) -> None:
        with self.assertRaises(catalog.CatalogError) as caught:
            catalog.find("nope")
        self.assertIn("bonsai-8b", str(caught.exception))

    def test_a_malformed_document_is_rejected_rather_than_half_read(self) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "MODEL.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump({"id": "x", "title": "X"}, handle)
            with self.assertRaises(catalog.CatalogError):
                catalog.load_model(folder)


if __name__ == "__main__":
    unittest.main()
