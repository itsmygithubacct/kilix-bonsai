"""Model admission uses current consent and immutable verified descriptors."""
import fcntl
import hashlib
import os
from pathlib import Path
import stat
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from kilix_bonsai.runtime.installed_image import (
    InstalledModel, InstalledImageError, content_authority)


class InstalledImageTests(unittest.TestCase):
    def model(self, root):
        source=root/'assets/model';source.mkdir(parents=True)
        payload=b'private declared test member'
        (source/'weights.bin').write_bytes(payload)
        (source/'weights.bin').chmod(0o600)
        model=InstalledModel.__new__(InstalledModel)
        model.root=root;model.relative=('assets','model');model.maximum_bytes=1024
        model._bound=True;model._license=Mock();model.reference=object()
        model.records=object();model.store=object();model._errors=(OSError,ValueError,RuntimeError)
        model._members={'weights.bin':SimpleNamespace(bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest())}
        return model,payload

    def test_declared_model_is_sealed_and_survives_original_removal(self):
        with tempfile.TemporaryDirectory() as scratch:
            model,payload=self.model(Path(scratch))
            with model.view(lambda:None) as view:
                (model.root/'assets/model/weights.bin').unlink()
                self.assertEqual((view/'weights.bin').read_bytes(),payload)
                fd=os.open(view/'weights.bin',os.O_RDONLY)
                try:
                    self.assertEqual(fcntl.fcntl(fd,1034)&15,15)
                finally:
                    os.close(fd)
            self.assertFalse(view.exists())

    def test_missing_or_changed_population_refuses_before_inference(self):
        for damage in ('missing','changed','extra'):
            with self.subTest(damage=damage), tempfile.TemporaryDirectory() as scratch:
                model,_=self.model(Path(scratch));source=model.root/'assets/model/weights.bin'
                if damage=='missing':source.unlink()
                elif damage=='changed':source.write_bytes(b'changed')
                else:(source.parent/'extra.bin').write_bytes(b'undeclared')
                with self.assertRaises(InstalledImageError):
                    with model.view(lambda:None):
                        self.fail('damaged population admitted')

    def test_refused_consent_does_not_open_model_files(self):
        with tempfile.TemporaryDirectory() as scratch:
            model,_=self.model(Path(scratch))
            model._license.require.side_effect=RuntimeError('private refused receipt')
            model._copy_population=Mock()
            with self.assertRaises(InstalledImageError):
                with model.view(lambda:None):
                    self.fail('refused receipt admitted')
            model._copy_population.assert_not_called()


SHA = 'a' * 40
OTHER = 'b' * 40


class ContentCheckoutTests(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory()
        self.root = Path(self.scratch.name)
        self.repo = self.root / 'repo'
        self.repo.mkdir()
        self.bin = self.root / 'bin'
        self.bin.mkdir()
        self.log = self.root / 'git.log'
        self.log.write_text('')
        stub = self.bin / 'git'
        stub.write_text(
            '#!/usr/bin/env python3\n'
            'import os, sys\n'
            'from pathlib import Path\n'
            'log = Path(os.environ["GIT_STUB_LOG"])\n'
            'repo = Path(os.environ["GIT_STUB_REPO"])\n'
            'with log.open("a") as handle:\n'
            '    handle.write(" ".join(sys.argv[1:]) + "\\n")\n'
            'args = sys.argv[1:]\n'
            'if len(args) >= 2 and args[0] == "-C":\n'
            '    args = args[2:]\n'
            'if args[:2] == ["ls-tree", "HEAD"]:\n'
            '    line = os.environ.get("GIT_STUB_LSTREE", "")\n'
            '    if line:\n'
            '        sys.stdout.write(line + "\\n")\n'
            '    raise SystemExit(0)\n'
            'if args == ["rev-parse", "HEAD"]:\n'
            '    if (repo / ".updated").is_file():\n'
            '        sys.stdout.write(os.environ["GIT_STUB_SHA"] + "\\n")\n'
            '        raise SystemExit(0)\n'
            '    wrong = os.environ.get("GIT_STUB_WRONG", "")\n'
            '    if wrong:\n'
            '        sys.stdout.write(wrong + "\\n")\n'
            '        raise SystemExit(0)\n'
            '    if os.environ.get("GIT_STUB_PRESENT") == "1":\n'
            '        sys.stdout.write(os.environ["GIT_STUB_SHA"] + "\\n")\n'
            '        raise SystemExit(0)\n'
            '    raise SystemExit(1)\n'
            'if args[:4] == ["submodule", "update", "--init", "--"]:\n'
            '    if os.environ.get("GIT_STUB_UPDATE_FAIL") == "1":\n'
            '        raise SystemExit(1)\n'
            '    marker = repo / "third_party/kilix-content/src/kilix_content/__init__.py"\n'
            '    marker.parent.mkdir(parents=True, exist_ok=True)\n'
            '    marker.write_text("ok\\n")\n'
            '    (repo / ".updated").write_text("1")\n'
            '    raise SystemExit(0)\n'
            'raise SystemExit(1)\n')
        stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
        self.saved = os.environ.copy()
        os.environ['PATH'] = str(self.bin) + os.pathsep + self.saved.get('PATH', '')
        os.environ['GIT_STUB_LOG'] = str(self.log)
        os.environ['GIT_STUB_REPO'] = str(self.repo)
        os.environ['GIT_STUB_SHA'] = SHA
        os.environ['GIT_STUB_LSTREE'] = (
            '160000 commit ' + SHA + '\tthird_party/kilix-content')
        os.environ.pop('GIT_STUB_PRESENT', None)
        os.environ.pop('GIT_STUB_WRONG', None)
        os.environ.pop('GIT_STUB_UPDATE_FAIL', None)

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self.saved)
        self.scratch.cleanup()

    def marker(self):
        return self.repo / 'third_party/kilix-content/src/kilix_content/__init__.py'

    def test_a_matching_checkout_is_not_fetched_again(self):
        self.marker().parent.mkdir(parents=True)
        self.marker().write_text('present\n')
        os.environ['GIT_STUB_PRESENT'] = '1'
        os.environ['GIT_STUB_UPDATE_FAIL'] = '1'
        found = content_authority(self.repo)
        self.assertEqual(found, self.marker().parents[1])
        self.assertNotIn('submodule', self.log.read_text())

    def test_an_empty_gitlink_is_checked_out(self):
        found = content_authority(self.repo)
        self.assertEqual(found, self.marker().parents[1])
        self.assertEqual(self.marker().read_text(), 'ok\n')
        self.assertEqual(self.log.read_text().count('submodule update --init'), 1)

    def test_a_different_checkout_is_replaced_by_the_pin(self):
        self.marker().parent.mkdir(parents=True)
        self.marker().write_text('wrong\n')
        os.environ['GIT_STUB_WRONG'] = OTHER
        found = content_authority(self.repo)
        self.assertEqual(found, self.marker().parents[1])
        self.assertIn('submodule update --init', self.log.read_text())
        self.assertEqual(self.marker().read_text(), 'ok\n')

    def test_a_failed_checkout_names_the_pin(self):
        os.environ['GIT_STUB_UPDATE_FAIL'] = '1'
        with self.assertRaises(InstalledImageError) as caught:
            content_authority(self.repo)
        self.assertEqual(str(caught.exception),
                         'pinned kilix-content checkout is unavailable')

    def test_a_tree_without_a_gitlink_does_not_fetch(self):
        os.environ['GIT_STUB_LSTREE'] = ''
        found = content_authority(self.repo)
        self.assertEqual(found, self.marker().parents[1])
        self.assertNotIn('submodule', self.log.read_text())


if __name__=='__main__':
    unittest.main()
