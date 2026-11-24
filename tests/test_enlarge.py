"""bonsai-enlarge: sizes, crops, schedule and the guarded local run."""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from kilix_bonsai.runtime import enlarge, local_image  # noqa: E402

try:
    from PIL import Image
except ImportError:  # pragma: no cover - the image runtime always has Pillow
    Image = None


class PlanTest(unittest.TestCase):
    def test_default_size_puts_the_long_side_at_512_and_keeps_aspect(self):
        self.assertEqual(enlarge.target_size((32, 32)), (512, 512))
        self.assertEqual(enlarge.target_size((72, 48)), (512, 352))
        self.assertEqual(enlarge.target_size((20, 40)), (256, 512))

    def test_scale_multiplies_then_snaps_to_32_within_range(self):
        self.assertEqual(enlarge.target_size((64, 64), scale=8), (512, 512))
        self.assertEqual(enlarge.target_size((50, 30), scale=10), (512, 288))
        self.assertEqual(enlarge.target_size((4, 4), scale=2), (128, 128))
        self.assertEqual(enlarge.target_size((600, 300), scale=4), (2048, 1216))

    def test_explicit_size_is_exact_and_excludes_scale(self):
        self.assertEqual(enlarge.target_size((9, 9), size=(640, 384)), (640, 384))
        with self.assertRaisesRegex(enlarge.EnlargeError, 'not both'):
            enlarge.target_size((9, 9), scale=2, size=(512, 512))
        with self.assertRaisesRegex(enlarge.EnlargeError, 'positive'):
            enlarge.target_size((9, 9), scale=0)

    def test_parse_size_rejects_off_grid_and_out_of_range(self):
        self.assertEqual(enlarge.parse_size('768x512'), (768, 512))
        for bad in ('500x512', '96x96', '4096x512', 'big'):
            with self.subTest(bad=bad), self.assertRaises(enlarge.EnlargeError):
                enlarge.parse_size(bad)

    def test_crop_must_lie_inside_the_source(self):
        self.assertEqual(enlarge.parse_crop('4,8,16,12'), (4, 8, 16, 12))
        self.assertEqual(enlarge.check_crop(None, (40, 30)), (0, 0, 40, 30))
        self.assertEqual(enlarge.check_crop((24, 14, 16, 16), (40, 30)), (24, 14, 16, 16))
        with self.assertRaisesRegex(enlarge.EnlargeError, 'outside'):
            enlarge.check_crop((25, 14, 16, 16), (40, 30))
        for bad in ('1,2,3', '-1,0,4,4', '0,0,0,4', 'a,b,c,d'):
            with self.subTest(bad=bad), self.assertRaises(enlarge.EnlargeError):
                enlarge.parse_crop(bad)

    def test_schedule_runs_round_steps_times_strength_and_at_least_one(self):
        self.assertEqual(enlarge.schedule_start(8, 0.5), 4)
        self.assertEqual(enlarge.schedule_start(8, 1.0), 0)
        self.assertEqual(enlarge.schedule_start(8, 0.3), 6)
        self.assertEqual(enlarge.schedule_start(4, 0.01), 3)
        self.assertEqual(enlarge.schedule_start(1, 0.5), 0)
        for steps, strength in ((0, 0.5), (8, 0.0), (8, 1.5), (8, -0.2)):
            with self.subTest(steps=steps, strength=strength), \
                    self.assertRaises(enlarge.EnlargeError):
                enlarge.schedule_start(steps, strength)

    def test_parse_colour(self):
        self.assertEqual(enlarge.parse_colour('#102030'), (16, 32, 48))
        with self.assertRaises(enlarge.EnlargeError):
            enlarge.parse_colour('fff')


@unittest.skipIf(Image is None, 'Pillow is not installed')
class PrepareTest(unittest.TestCase):
    def test_transparency_is_laid_on_the_background_before_resampling(self):
        source = Image.new('RGBA', (16, 16), (0, 0, 0, 0))
        source.putpixel((8, 8), (255, 0, 0, 255))
        out = enlarge.prepare(source, None, (128, 128), (0, 0, 255))
        self.assertEqual(out.mode, 'RGB')
        self.assertEqual(out.size, (128, 128))
        self.assertEqual(out.getpixel((0, 0)), (0, 0, 255))
        red, _, blue = out.getpixel((68, 68))
        self.assertGreater(red, 200)
        self.assertLess(blue, 60)

    def test_crop_selects_the_detail(self):
        source = Image.new('RGB', (40, 20), (255, 255, 255))
        for x in range(20, 40):
            for y in range(20):
                source.putpixel((x, y), (0, 200, 0))
        out = enlarge.prepare(source, (20, 0, 20, 20), (128, 128))
        self.assertEqual(out.getpixel((64, 64)), (0, 200, 0))
        self.assertEqual(out.getpixel((2, 2)), (0, 200, 0))


@unittest.skipIf(Image is None, 'Pillow is not installed')
class LocalEnlargeTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.source = self.directory / 'emoji.png'
        Image.new('RGBA', (32, 32), (250, 200, 0, 255)).save(self.source)

    def tearDown(self):
        self.temporary.cleanup()

    def request(self, **changes):
        values = dict(input=str(self.source), output=str(self.directory / 'out/big.png'),
                      crop=None, scale=None, size=None, prompt=None, strength=0.5,
                      steps=8, seed=11, background='ffffff',
                      baseline=str(self.directory / 'out/lanczos.png'),
                      variant='ternary')
        values.update(changes)
        return argparse.Namespace(**values)

    @contextmanager
    def view(self, *_):
        yield self.directory

    def modules(self, torch, forward):
        backend = Mock()
        return {'torch': torch, 'backend_gpu': Mock(pipeline_gpu=backend),
                'backend_gpu.pipeline_gpu': backend,
                'kilix_bonsai.runtime.enlarge_forward': forward}

    def test_busy_gpu_refuses_before_any_model_loads(self):
        torch = Mock()
        torch.cuda.mem_get_info.return_value = (3953 * 1024**2, 6144 * 1024**2)
        forward = Mock()
        with patch.object(local_image, 'check'), patch.object(
                local_image, 'load_pipeline') as load, patch.dict(
                sys.modules, self.modules(torch, forward)):
            with self.assertRaisesRegex(RuntimeError, 'GPU busy: 3953 MiB free'):
                local_image.enlarge(self.request())
        load.assert_not_called()
        forward.img2img_forward.assert_not_called()
        self.assertFalse((self.directory / 'out/big.png').exists())

    def test_runs_the_remaining_steps_from_the_lanczos_baseline(self):
        torch = Mock()
        torch.cuda.mem_get_info.return_value = (5000 * 1024**2, 6144 * 1024**2)
        forward = Mock()
        seen = {}

        def img2img(pipe, init_image, prompt, *, num_steps, start, seed):
            seen.update(size=init_image.size, pixel=init_image.getpixel((5, 5)),
                        prompt=prompt, num_steps=num_steps, start=start, seed=seed)
            return Image.new('RGB', init_image.size, (1, 2, 3))

        forward.img2img_forward.side_effect = img2img
        with patch.object(local_image, 'check'), patch.object(
                local_image, 'model_view', side_effect=self.view), patch.object(
                local_image, 'load_pipeline', return_value='pipe'), patch.dict(
                sys.modules, self.modules(torch, forward)):
            local_image.enlarge(self.request(strength=0.25, prompt='a yellow square emoji'))
        self.assertEqual(seen, dict(size=(512, 512), pixel=(250, 200, 0),
                                    prompt='a yellow square emoji', num_steps=8,
                                    start=6, seed=11))
        out = self.directory / 'out/big.png'
        with Image.open(out) as result:
            self.assertEqual(result.getpixel((0, 0)), (1, 2, 3))
        with Image.open(self.directory / 'out/lanczos.png') as baseline:
            self.assertEqual(baseline.size, (512, 512))
        record = json.loads((self.directory / 'out/big.png.json').read_text())
        self.assertEqual(record['schema'], 'kilix.bonsai.enlarge/v1')
        self.assertEqual(record['crop'], [0, 0, 32, 32])
        self.assertEqual((record['steps'], record['steps_run'], record['seed']), (8, 2, 11))

    def test_bad_crop_fails_before_the_gpu_is_touched(self):
        torch = Mock()
        with patch.object(local_image, 'check') as check, patch.dict(
                sys.modules, self.modules(torch, Mock())):
            with self.assertRaisesRegex(ValueError, 'outside'):
                local_image.enlarge(self.request(crop='20,20,16,16'))
        check.assert_not_called()
        torch.cuda.mem_get_info.assert_not_called()

    def test_cli_parses_enlarge_and_reports_errors_as_result_lines(self):
        with patch.object(local_image, 'enlarge') as run:
            self.assertEqual(local_image.main(['enlarge', 'a.png', '--output', 'b.png',
                                               '--strength', '0.4', '--crop', '1,2,3,4']), 0)
        args = run.call_args.args[0]
        self.assertEqual((args.input, args.output, args.strength, args.crop, args.steps),
                         ('a.png', 'b.png', 0.4, '1,2,3,4', 8))


if __name__ == '__main__':
    unittest.main()
