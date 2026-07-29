"""What counts as present, and what a cheap check is allowed to claim.

`state()` stats and `verify()` hashes, and the difference matters: a UI that
hashed 3.8 GB per frame would be unusable, and one that called a truncated
download "ready" because the path existed would be worse. So the cheap check
compares sizes — the strongest claim a stat supports, and the one that catches
the failure that actually happens — and never reports a hash it did not compute.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from kilix_bonsai import catalog, store  # noqa: E402

# sha256 of b"hello"
HELLO = "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"


def a_model(folder: str, files: list[dict]) -> catalog.Model:
    document = {
        "id": "probe", "title": "Probe", "task": "text",
        "store": {"env": "PROBE_STORE", "default": "$GPU_TERMINAL_HOME/probe"},
        "variants": [{"id": "only", "title": "Only", "default": True,
                      "subdir": "", "sources": [
                          {"repo": "org/probe", "revision": "0" * 40,
                           "files": files}]}],
    }
    with open(os.path.join(folder, "MODEL.json"), "w", encoding="utf-8") as f:
        json.dump(document, f)
    return catalog.load_model(folder)


class StoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = os.path.join(self.tmp.name, "store")
        os.makedirs(self.store)
        os.environ["PROBE_STORE"] = self.store
        self.addCleanup(os.environ.pop, "PROBE_STORE", None)
        self.model = a_model(self.tmp.name, [
            {"path": "weights.bin", "size": 5, "sha256": HELLO},
            {"path": "sub/config.json", "size": 2, "sha256": None},
        ])
        self.variant = self.model.default_variant

    def write(self, name: str, data: bytes) -> None:
        path = os.path.join(self.store, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as handle:
            handle.write(data)

    def test_nothing_on_disk_is_missing(self) -> None:
        self.assertEqual(store.state(self.model).state, store.MISSING)

    def test_some_files_on_disk_is_partial(self) -> None:
        self.write("weights.bin", b"hello")
        self.assertEqual(store.state(self.model).state, store.PARTIAL)

    def test_every_file_at_the_right_size_is_present(self) -> None:
        self.write("weights.bin", b"hello")
        self.write("sub/config.json", b"{}")
        self.assertEqual(store.state(self.model).state, store.PRESENT)

    def test_a_truncated_file_is_not_present(self) -> None:
        # The failure that actually happens: an interrupted transfer leaves a
        # file that exists and is the wrong length.
        self.write("weights.bin", b"hel")
        self.write("sub/config.json", b"{}")
        self.assertEqual(store.state(self.model).state, store.PARTIAL)

    def test_partial_reports_how_much_arrived(self) -> None:
        self.write("weights.bin", b"hello")
        self.assertEqual(store.state(self.model).present_bytes, 5)

    def test_verify_accepts_a_matching_digest(self) -> None:
        self.write("weights.bin", b"hello")
        self.write("sub/config.json", b"{}")
        self.assertTrue(store.verify(self.model, self.variant))

    def test_verify_rejects_a_file_with_the_right_size_and_wrong_bytes(self) -> None:
        # Exactly what a size check cannot catch, which is why verify exists.
        self.write("weights.bin", b"HELLO")
        self.write("sub/config.json", b"{}")
        self.assertFalse(store.verify(self.model, self.variant))

    def test_verify_says_which_check_it_ran(self) -> None:
        # A size check must never be presented as a hash check.
        self.write("weights.bin", b"hello")
        self.write("sub/config.json", b"{}")
        seen = {}
        store.verify(self.model, self.variant,
                     report=lambda path, good, detail: seen.update({path: detail}))
        self.assertEqual(seen["weights.bin"], "sha256 ok")
        self.assertIn("no published digest", seen["sub/config.json"])

    def test_deps_state_is_absent_until_the_script_writes_it(self) -> None:
        self.assertIsNone(store.deps_state(self.model))
        with open(os.path.join(self.store, store.DEPS_STAMP), "w",
                  encoding="utf-8") as handle:
            json.dump({"model": "probe"}, handle)
        self.assertEqual(store.deps_state(self.model)["model"], "probe")

    def test_a_corrupt_stamp_is_not_a_crash(self) -> None:
        with open(os.path.join(self.store, store.DEPS_STAMP), "w",
                  encoding="utf-8") as handle:
            handle.write("not json")
        self.assertIsNone(store.deps_state(self.model))

    def test_any_present_ignores_a_partly_downloaded_model(self) -> None:
        self.write("weights.bin", b"hello")
        self.assertFalse(store.any_present([self.model]))
        self.write("sub/config.json", b"{}")
        self.assertTrue(store.any_present([self.model]))


class HumanBytesTest(unittest.TestCase):
    def test_scales(self) -> None:
        self.assertEqual(store.human_bytes(512), "512B")
        self.assertEqual(store.human_bytes(1536), "1.5K")
        self.assertEqual(store.human_bytes(3_803_452_480), "3.5G")

    def test_zero_reads_as_zero(self) -> None:
        self.assertEqual(store.human_bytes(0), "0B")


if __name__ == "__main__":
    unittest.main()
