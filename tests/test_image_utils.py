"""``src.image_utils`` 的单元测试。"""

from __future__ import annotations

import io
import unittest

from PIL import Image

from src.image_utils import (
    ImageValidationError,
    check_bbox_quality,
    crop_with_margin,
    estimate_visual_tokens,
    pad_to_square,
    preprocess_image,
    to_data_url,
    validate_image_bytes,
)


def make_image_bytes(width=800, height=600, fmt="PNG", color=(120, 90, 70)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buffer, format=fmt)
    return buffer.getvalue()


class TestValidateImageBytes(unittest.TestCase):
    def test_valid_png(self):
        width, height, fmt = validate_image_bytes(make_image_bytes())
        self.assertEqual((width, height), (800, 600))
        self.assertEqual(fmt, "PNG")

    def test_valid_jpeg(self):
        width, height, fmt = validate_image_bytes(make_image_bytes(fmt="JPEG"))
        self.assertEqual(fmt, "JPEG")

    def test_empty(self):
        with self.assertRaises(ImageValidationError):
            validate_image_bytes(b"")

    def test_not_an_image(self):
        with self.assertRaises(ImageValidationError) as ctx:
            validate_image_bytes(b"%PDF-1.4 not an image", "report.pdf")
        self.assertIn("无法识别", str(ctx.exception))

    def test_too_small(self):
        with self.assertRaises(ImageValidationError) as ctx:
            validate_image_bytes(make_image_bytes(5, 5))
        self.assertIn("尺寸过小", str(ctx.exception))

    def test_extreme_aspect(self):
        with self.assertRaises(ImageValidationError) as ctx:
            validate_image_bytes(make_image_bytes(2010, 10))
        self.assertIn("长宽比", str(ctx.exception))

    def test_oversized_file(self):
        with self.assertRaises(ImageValidationError) as ctx:
            validate_image_bytes(b"x" * (21 * 1024 * 1024))
        self.assertIn("20 MB", str(ctx.exception))


class TestPreprocess(unittest.TestCase):
    def test_downscale_long_side(self):
        result = preprocess_image(make_image_bytes(2400, 1600), max_side=512)
        self.assertEqual(max(result.width, result.height), 512)
        self.assertEqual(result.original_width, 2400)
        self.assertLess(len(result.jpeg_bytes), len(make_image_bytes(2400, 1600, "JPEG", (120, 90, 70))))

    def test_no_upscale(self):
        result = preprocess_image(make_image_bytes(300, 200), max_side=512)
        self.assertEqual((result.width, result.height), (300, 200))

    def test_exif_orientation_applied(self):
        buffer = io.BytesIO()
        image = Image.new("RGB", (400, 200), (10, 20, 30))
        exif = Image.Exif()
        exif[274] = 6  # 顺时针旋转 90°
        image.save(buffer, format="JPEG", exif=exif)
        result = preprocess_image(buffer.getvalue(), max_side=512)
        self.assertEqual((result.width, result.height), (200, 400))

    def test_rgba_flattened(self):
        buffer = io.BytesIO()
        Image.new("RGBA", (100, 100), (255, 0, 0, 0)).save(buffer, format="PNG")
        result = preprocess_image(buffer.getvalue())
        with Image.open(io.BytesIO(result.jpeg_bytes)) as reopened:
            self.assertEqual(reopened.mode, "RGB")

    def test_data_url(self):
        result = preprocess_image(make_image_bytes(64, 64))
        self.assertTrue(result.data_url.startswith("data:image/jpeg;base64,"))

    def test_corrupt(self):
        with self.assertRaises(ImageValidationError):
            preprocess_image(b"\x00\x01\x02")


class TestTokenEstimate(unittest.TestCase):
    def test_512_square(self):
        # 512/32 = 16 -> 16*16 + 2 = 258
        self.assertEqual(estimate_visual_tokens(512, 512), 258)

    def test_minimum(self):
        self.assertGreaterEqual(estimate_visual_tokens(10, 10), 2)


class TestCropAndPad(unittest.TestCase):
    def test_crop_with_margin_inside_bounds(self):
        image = Image.new("RGB", (1000, 800))
        crop = crop_with_margin(image, [300, 200, 200, 150], margin_ratio=0.1)
        self.assertEqual(crop.size, (240, 180))

    def test_crop_clipped_at_border(self):
        image = Image.new("RGB", (100, 100))
        crop = crop_with_margin(image, [90, 90, 20, 20], margin_ratio=0.5)
        self.assertLessEqual(crop.width, 100)
        self.assertLessEqual(crop.height, 100)

    def test_crop_invalid_bbox(self):
        image = Image.new("RGB", (100, 100))
        with self.assertRaises(ImageValidationError):
            crop_with_margin(image, [500, 500, 10, 10])

    def test_pad_to_square(self):
        padded = pad_to_square(Image.new("RGB", (300, 100)))
        self.assertEqual(padded.size, (300, 300))


class TestBboxQuality(unittest.TestCase):
    def test_pass(self):
        self.assertIsNone(check_bbox_quality(300, 280))

    def test_too_small_side(self):
        self.assertIn("min_side", check_bbox_quality(50, 400))

    def test_too_small_area(self):
        self.assertIn("area", check_bbox_quality(70, 70))

    def test_extreme_aspect(self):
        self.assertIn("aspect", check_bbox_quality(600, 70))


if __name__ == "__main__":
    unittest.main()
