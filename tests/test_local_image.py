"""Local image errors are actionable and never allocate a model on a busy GPU."""
import argparse
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from kilix_bonsai.runtime import image, local_image

class LocalImageTest(unittest.TestCase):
    def request(self, **changes):
        values = dict(prompt='a fire kitten', size='512x512', steps=4,
                      seed=1, input_image=None, output='unused.png')
        values.update(changes)
        return argparse.Namespace(**values)

    def test_busy_gpu_reports_retry_before_loading_model(self):
        torch = Mock()
        torch.cuda.mem_get_info.return_value = (400 * 1024**2, 6000 * 1024**2)
        with patch.object(local_image, 'check'), patch.dict(sys.modules, {'torch': torch}):
            with self.assertRaisesRegex(RuntimeError, 'GPU busy: 400 MiB free.*Retry'):
                local_image.generate(self.request())

    def test_reference_is_refused_explicitly(self):
        with self.assertRaisesRegex(ValueError, 'reference images are not supported'):
            local_image.generate(self.request(input_image='reference.png'))

    def test_default_probe_does_not_contact_remote(self):
        done = Mock(returncode=0, stdout='result: ready', stderr='')
        with patch.object(image, 'REMOTE_SUBCOMMAND', 'remote'), patch.object(
                image, 'cli_path', return_value='/example/bonsai'), patch.object(
                image.subprocess, 'run', return_value=done) as run:
            image.probe_backends()
        self.assertEqual(run.call_count, 1)
        self.assertEqual(run.call_args.args[0], ['/example/bonsai', 'doctor'])
