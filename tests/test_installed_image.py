"""Model admission uses current consent and immutable verified descriptors."""
import fcntl
import hashlib
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from kilix_bonsai.runtime.installed_image import InstalledModel,InstalledImageError


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


if __name__=='__main__':
    unittest.main()
