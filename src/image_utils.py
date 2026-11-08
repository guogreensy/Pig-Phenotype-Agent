"""图片校验、预处理与编码。

对外功能：
* :func:`validate_image_bytes` —— 上传图片的可用性检查（中文报错）
* :func:`preprocess_image`     —— EXIF 纠正 → 限长边缩放 → JPEG 重编码
* :func:`to_data_url`          —— 转为视觉模型 API 需要的 Base64 Data URI
* :func:`crop_with_margin`     —— 按 bbox 裁出单只猪（构建数据集用）

模型侧限制（依据阿里云百炼视觉理解文档）：
宽高均须 > 10 像素、长宽比 ≤ 200:1、Base64 编码前单图 ≤ 20MB。
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

from PIL import Image, ImageOps, UnidentifiedImageError

#: Pillow 能打开、模型也接受的格式。
SUPPORTED_FORMATS = {"JPEG", "PNG", "BMP", "GIF", "WEBP", "TIFF"}

#: 允许的扩展名（与格式一起用于给出更友好的提示）。
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff"}

MAX_INPUT_BYTES = 20 * 1024 * 1024
MIN_SIDE_PX = 10
MAX_ASPECT_RATIO = 200.0


class ImageValidationError(ValueError):
    """图片不可用，消息为面向用户的中文提示。"""


@dataclass
class ProcessedImage:
    """预处理结果。"""

    jpeg_bytes: bytes
    width: int
    height: int
    original_width: int
    original_height: int
    source_format: str

    @property
    def data_url(self) -> str:
        return to_data_url(self.jpeg_bytes)


def validate_image_bytes(raw: bytes, filename: str = "") -> Tuple[int, int, str]:
    """校验图片是否可用，返回 ``(width, height, format)``。

    Raises:
        ImageValidationError: 文件为空、超大、格式不支持或无法解码。
    """
    if not raw:
        raise ImageValidationError("文件为空，请重新选择图片。")
    if len(raw) > MAX_INPUT_BYTES:
        raise ImageValidationError(
            "图片体积 {0:.1f} MB 超过 20 MB 上限，请压缩后再上传。".format(len(raw) / 1048576)
        )
    try:
        with Image.open(io.BytesIO(raw)) as probe:
            probe.verify()
        with Image.open(io.BytesIO(raw)) as image:
            fmt = (image.format or "").upper()
            width, height = image.size
    except UnidentifiedImageError as exc:
        raise ImageValidationError(
            "无法识别该文件为图片（文件：{0}）。支持 JPG、PNG、BMP、GIF、WEBP、TIFF。".format(
                filename or "未命名"
            )
        ) from exc
    except Exception as exc:  # Pillow 对损坏文件会抛各种异常
        raise ImageValidationError(
            "图片已损坏或格式不被支持（{0}）。请换一张图片重试。".format(filename or "未命名")
        ) from exc

    if fmt not in SUPPORTED_FORMATS:
        raise ImageValidationError(
            "不支持的图片格式：{0}。请使用 JPG、PNG、BMP 等常见格式。".format(fmt or "未知")
        )
    if min(width, height) < MIN_SIDE_PX:
        raise ImageValidationError(
            "图片尺寸过小（{0}×{1}），宽高均需大于 {2} 像素。".format(width, height, MIN_SIDE_PX)
        )
    if max(width, height) / max(1, min(width, height)) > MAX_ASPECT_RATIO:
        raise ImageValidationError(
            "图片长宽比超过 {0}:1（当前 {1}×{2}），请裁剪后重试。".format(
                int(MAX_ASPECT_RATIO), width, height
            )
        )
    return width, height, fmt


def _normalize_mode(image: Image.Image) -> Image.Image:
    """统一到 RGB；带透明通道时铺白底，避免变成黑块。"""
    if image.mode in ("RGBA", "LA", "P"):
        rgba = image.convert("RGBA")
        canvas = Image.new("RGB", rgba.size, (255, 255, 255))
        canvas.paste(rgba, mask=rgba.split()[-1])
        return canvas
    if image.mode != "RGB":
        return image.convert("RGB")
    return image


def preprocess_image(
    raw: bytes, max_side: int = 512, jpeg_quality: int = 90
) -> ProcessedImage:
    """EXIF 纠正 → 长边缩放到 ``max_side``（不放大）→ JPEG 重编码。"""
    width, height, fmt = validate_image_bytes(raw)
    with Image.open(io.BytesIO(raw)) as image:
        image = ImageOps.exif_transpose(image)
        image = _normalize_mode(image)
        scale = min(1.0, float(max_side) / float(max(image.size)))
        if scale < 1.0:
            new_size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
            image = image.resize(new_size, Image.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=jpeg_quality, optimize=True)
        return ProcessedImage(
            jpeg_bytes=buffer.getvalue(),
            width=image.width,
            height=image.height,
            original_width=width,
            original_height=height,
            source_format=fmt,
        )


def to_data_url(jpeg_bytes: bytes, mime: str = "image/jpeg") -> str:
    """编码为 ``data:<mime>;base64,<...>``。"""
    return "data:{0};base64,{1}".format(mime, base64.b64encode(jpeg_bytes).decode("ascii"))


def estimate_visual_tokens(width: int, height: int, patch: int = 32) -> int:
    """按 ``h*w/(patch*patch)+2`` 估算视觉 Token（Qwen3-VL 口径，仅作成本参考）。"""
    bars_w = max(1, width // patch)
    bars_h = max(1, height // patch)
    return bars_w * bars_h + 2


def crop_with_margin(
    image: Image.Image, bbox: Iterable[float], margin_ratio: float = 0.08
) -> Image.Image:
    """按 bbox 裁出目标，并向外扩 ``margin_ratio`` 的边距（构建数据集用）。"""
    x, y, w, h = [float(v) for v in bbox]
    mx = w * margin_ratio
    my = h * margin_ratio
    left = max(0, int(round(x - mx)))
    top = max(0, int(round(y - my)))
    right = min(image.width, int(round(x + w + mx)))
    bottom = min(image.height, int(round(y + h + my)))
    if right <= left or bottom <= top:
        raise ImageValidationError("裁剪框超出图片范围：{0}".format(list(bbox)))
    return image.crop((left, top, right, bottom))


def pad_to_square(image: Image.Image, fill: Tuple[int, int, int] = (114, 114, 114)) -> Image.Image:
    """补边到正方形，避免模型因非正方形缩放而变形。"""
    side = max(image.size)
    canvas = Image.new("RGB", (side, side), fill)
    canvas.paste(image, ((side - image.width) // 2, (side - image.height) // 2))
    return canvas


def open_image(path) -> Image.Image:
    """读取图片并纠正 EXIF 方向。"""
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def save_jpeg(image: Image.Image, path, quality: int = 92) -> None:
    """保存为 JPEG（自动建目录）。"""
    from pathlib import Path

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    image.save(target, format="JPEG", quality=quality, optimize=True)


def check_bbox_quality(
    width: int, height: int, min_side: int = 64, min_area: int = 8192, max_aspect: float = 4.0
) -> Optional[str]:
    """返回不满足质量要求的原因；合格则返回 None。"""
    if min(width, height) < min_side:
        return "min_side<{0}px".format(min_side)
    if width * height < min_area:
        return "area<{0}px2".format(min_area)
    aspect = max(width, height) / max(1.0, float(min(width, height)))
    if aspect > max_aspect:
        return "aspect>{0:g}".format(max_aspect)
    return None
