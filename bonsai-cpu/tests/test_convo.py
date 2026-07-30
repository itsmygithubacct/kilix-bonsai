#!/usr/bin/env python3
"""Persistence promises: roundtrips, atomicity, tolerant listing, and a
reader that creates nothing."""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from bonsai_cpu import convo   # noqa: E402


class ConvoTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def make(self):
        built = convo.new("bonsai-27b", directory=self.tmp.name)
        built.name = "naming things"
        built.think = True
        built.params["temperature"] = 0.2
        built.messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello", "reasoning": "hmm",
             "stats": {"prompt_n": 8, "predicted_n": 2,
                       "predicted_per_second": 1.4}},
        ]
        return built

    def test_roundtrip_keeps_everything(self):
        built = self.make()
        convo.save(built)
        loaded = convo.load(built.path)
        self.assertEqual(loaded.name, "naming things")
        self.assertEqual(loaded.model_id, "bonsai-27b")
        self.assertTrue(loaded.think)
        self.assertEqual(loaded.params["temperature"], 0.2)
        self.assertEqual(loaded.messages, built.messages)

    def test_save_is_atomic(self):
        built = self.make()
        convo.save(built)
        convo.save(built)
        leftovers = [name for name in os.listdir(self.tmp.name)
                     if name.endswith(".tmp")]
        self.assertEqual(leftovers, [])

    def test_unknown_keys_survive(self):
        built = self.make()
        convo.save(built)
        with open(built.path) as fh:
            document = json.load(fh)
        document["favourite_colour"] = "green"
        with open(built.path, "w") as fh:
            json.dump(document, fh)
        loaded = convo.load(built.path)
        convo.save(loaded)
        with open(built.path) as fh:
            self.assertEqual(json.load(fh)["favourite_colour"], "green")

    def test_listing_skips_corrupt_files_and_orders_newest_first(self):
        first = self.make()
        convo.save(first)
        second = convo.new("bonsai-8b", directory=self.tmp.name)
        second.messages = [{"role": "user", "content": "later"}]
        convo.save(second)
        second.updated = "9999-01-01T00:00:00"
        convo.save(second)
        with open(os.path.join(self.tmp.name, "c-broken.json"), "w") as fh:
            fh.write("{not json")
        listed = convo.listing(self.tmp.name)
        self.assertEqual(len(listed), 2)
        self.assertEqual(listed[0].path, second.path)

    def test_listing_creates_nothing(self):
        missing = os.path.join(self.tmp.name, "never-made")
        self.assertEqual(convo.listing(missing), [])
        self.assertFalse(os.path.exists(missing))

    def test_two_conversations_in_one_second_get_distinct_paths(self):
        first = convo.new("bonsai-8b", directory=self.tmp.name)
        convo.save(first)
        second = convo.new("bonsai-8b", directory=self.tmp.name)
        self.assertNotEqual(first.path, second.path)

    def test_default_name_truncates(self):
        name = convo.default_name("  " + "word " * 30)
        self.assertLessEqual(len(name), convo.NAME_LIMIT)
        self.assertTrue(name.startswith("word"))
        self.assertEqual(convo.default_name("   "), "untitled")


if __name__ == "__main__":
    unittest.main()
