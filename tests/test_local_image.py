"""Local image errors are actionable and never allocate a model on a busy GPU."""
import argparse
from pathlib import Path
import sys
import tempfile
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

    def test_generation_uses_keyword_only_pipeline_contract(self):
        torch = Mock()
        torch.cuda.mem_get_info.return_value = (7000 * 1024**2, 8192 * 1024**2)

        class Pipeline:
            def __init__(self, **kwargs):
                pass

            def prewarm(self):
                pass

            def generate_png(self, *, prompt, seed, steps, width, height):
                self_test.assertEqual((prompt, seed, steps, width, height),
                                      ('a fire kitten', 1, 4, 512, 512))
                return b'generated image'

        self_test = self
        backend = Mock(GpuPipeline=Pipeline)
        with tempfile.TemporaryDirectory() as directory, patch.object(
                local_image, 'check'), patch.object(local_image, 'weights', return_value=Path(directory)), patch.dict(
                sys.modules, {'torch': torch, 'backend_gpu.pipeline_gpu': backend}):
            output = Path(directory) / 'result.png'
            local_image.generate(self.request(output=str(output)))
            self.assertEqual(output.read_bytes(), b'generated image')

    def test_small_gpu_decoder_uses_float32_model_and_inputs(self):
        torch = Mock()
        torch.cuda.mem_get_info.return_value = (5000 * 1024**2, 6000 * 1024**2)
        vae = Mock()
        decoder = vae.decode
        latents = Mock()

        class Pipeline:
            def __init__(self, **kwargs):
                self._vae = vae

            def prewarm(self):
                pass

            def generate_png(self, **kwargs):
                self._vae.decode(latents, return_dict=False)
                return b'generated image'

        with tempfile.TemporaryDirectory() as directory, patch.object(
                local_image, 'check'), patch.object(local_image, 'weights', return_value=Path(directory)), patch.dict(
                sys.modules, {'torch': torch, 'backend_gpu.pipeline_gpu': Mock(GpuPipeline=Pipeline)}):
            local_image.generate(self.request(output=str(Path(directory) / 'result.png')))
        vae.to.assert_called_once_with(device='cpu', dtype=torch.float32)
        latents.float.assert_called_once_with()
        decoder.assert_called_once_with(latents.float.return_value, return_dict=False)

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
