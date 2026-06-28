import os
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageFilter


def generate_letter_image(letter, angle=0.0, output_path=None,
                          font_size=50, bg_color=255, fg_color=0,
                          image_size=32):
    if len(letter) != 1:
        raise ValueError(f"Expected a single character, got: {repr(letter)}")

    pad = image_size * 2
    canvas = pad + image_size
    big = Image.new("L", (canvas, canvas), color=bg_color)
    draw = ImageDraw.Draw(big)

    font = None
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeMono.ttf",
        "/System/Library/Fonts/Courier.ttc",
        "C:/Windows/Fonts/cour.ttf",
    ]:
        if os.path.exists(path):
            try:
                font = ImageFont.truetype(path, size=font_size)
                break
            except Exception:
                pass
    if font is None:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), letter, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = (canvas - text_w) / 2 - bbox[0]
    y = (canvas - text_h) / 2 - bbox[1]
    draw.text((x, y), letter, fill=fg_color, font=font)

    big = big.rotate(angle, resample=Image.BICUBIC,
                     center=(canvas / 2, canvas / 2))
    left = pad // 2
    top = pad // 2
    img = big.crop((left, top, left + image_size, top + image_size))

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        img.save(output_path, format="PNG")

    return img


def letter_to_tensor(letter, angle=0.0, image_size=32, invert=True):
    img = generate_letter_image(letter, angle=angle, image_size=image_size)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    if invert:
        arr = 1.0 - arr
    return torch.from_numpy(arr).float()


def load_image_tensor(image_path, image_size=32, invert=True, deblur=True):
    img = Image.open(image_path).convert("L")
    img = img.resize((image_size, image_size), Image.LANCZOS)
    if deblur:
        img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=2))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    if invert:
        arr = 1.0 - arr
    return torch.from_numpy(arr).float()
