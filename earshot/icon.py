from __future__ import annotations

from PIL import Image, ImageDraw


def _ear(
    color: tuple[int, int, int, int], cx: int, width: int, tilt: float
) -> Image.Image:
    ear = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    ImageDraw.Draw(ear).ellipse([cx - width // 2, 0, cx + width // 2, 34], fill=color)
    return ear.rotate(tilt, center=(cx, 17), resample=Image.Resampling.BICUBIC)


def make_icon(color: tuple[int, int, int, int]) -> Image.Image:
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    img.alpha_composite(_ear(color, 19, 15, 25))
    img.alpha_composite(_ear(color, 45, 15, -25))
    draw = ImageDraw.Draw(img)
    draw.ellipse([21, 24, 43, 48], fill=color)
    draw.ellipse([28, 42, 36, 52], fill=color)
    return img
