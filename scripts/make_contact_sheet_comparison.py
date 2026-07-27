"""Combine experiment contact sheets into labeled comparison grids."""

from __future__ import annotations

import argparse
import os

from PIL import Image, ImageDraw, ImageFont


CATEGORIES = {
    "rgb": ("eval_rgb_contact_sheet.png",),
    "normal": (
        "eval_photometric_normal_contact_sheet.png",
        "eval_normals_contact_sheet.png",
        "normals_contact_sheet.png",
    ),
    "alpha": ("alpha_render_contact_sheet.png",),
    "albedo": ("eval_albedo_contact_sheet.png",),
    "shading": ("eval_shading_contact_sheet.png",),
    "ndotl": ("eval_ndotl_contact_sheet.png",),
}


def _font(size):
    path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    if os.path.isfile(path):
        return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _parse_run(value):
    label, separator, path = value.partition("=")
    if not separator or not label or not path:
        raise argparse.ArgumentTypeError("Run must use LABEL=EXPERIMENT_PATH.")
    return label, path


def _find_contact_sheet(path, names):
    for name in names:
        candidate = os.path.join(path, name)
        if os.path.isfile(candidate):
            return candidate
    return None


def build_grid(runs, category, output_path, row_height=180, label_width=310):
    names = CATEGORIES[category]
    rows = []
    missing = []
    for label, path in runs:
        image_path = _find_contact_sheet(path, names)
        if image_path is None:
            missing.append(label)
            continue
        image = Image.open(image_path).convert("RGB")
        width = round(image.width * row_height / image.height)
        image = image.resize((width, row_height), Image.Resampling.LANCZOS)
        rows.append((label, image))
    if missing:
        raise FileNotFoundError(
            f"{category} contact sheet missing for: {', '.join(missing)}."
        )

    header_height = 48
    image_width = max(image.width for _, image in rows)
    canvas = Image.new(
        "RGB",
        (label_width + image_width, header_height + row_height * len(rows)),
        (16, 16, 16),
    )
    draw = ImageDraw.Draw(canvas)
    title_font = _font(24)
    label_font = _font(20)
    draw.text((12, 10), category.upper(), font=title_font, fill=(255, 255, 255))
    for index, timestep in enumerate(("t0", "t59", "t119")):
        x = label_width + round((index + 0.5) * image_width / 3)
        box = draw.textbbox((0, 0), timestep, font=title_font)
        draw.text(
            (x - (box[2] - box[0]) / 2, 10),
            timestep,
            font=title_font,
            fill=(255, 255, 255),
        )

    for row, (label, image) in enumerate(rows):
        y = header_height + row * row_height
        canvas.paste(image, (label_width, y))
        draw.line(
            (0, y, label_width + image_width, y),
            fill=(70, 70, 70),
            width=1,
        )
        draw.multiline_text(
            (12, y + 16),
            label.replace("\\n", "\n"),
            font=label_font,
            fill=(240, 240, 240),
            spacing=6,
        )
    canvas.save(output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--run",
        action="append",
        type=_parse_run,
        required=True,
        help="LABEL=EXPERIMENT_PATH; repeat in desired row order.",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        choices=sorted(CATEGORIES),
        default=["rgb", "normal", "alpha"],
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    for category in args.categories:
        build_grid(
            args.run,
            category,
            os.path.join(args.output_dir, f"{category}_{len(args.run)}way_contact_sheet.png"),
        )


if __name__ == "__main__":
    main()
