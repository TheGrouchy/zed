"""Capture the locked 8x8 Chromium pixel-avatar sampling evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

from PIL import Image


HERE = Path(__file__).resolve().parent
HTML = HERE / "image_sampling_pixelated_oracle.html"
FIXTURE = HERE / "image_sampling_pixelated_chromium.json"
RGBA_FIXTURE = HERE / "image_sampling_pixelated_chromium.rgba"
CASES = (("avatar22", 0, 22), ("avatar26", 30, 26), ("avatar30", 66, 30))


def chromium_version(chrome: Path) -> str:
    if os.name == "nt":
        command = f"(Get-Item -LiteralPath '{chrome}').VersionInfo.ProductVersion"
        return subprocess.check_output(
            ["powershell", "-NoProfile", "-Command", command], text=True
        ).strip()
    return subprocess.check_output([str(chrome), "--version"], text=True).strip().rsplit(" ", 1)[-1]


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def pixelated(source: Image.Image, output_size: int) -> bytes:
    source_bytes = source.tobytes()
    output = bytearray()
    for y in range(output_size):
        source_y = min(7, (((2 * y + 1) * 8) - 1) // (2 * output_size))
        for x in range(output_size):
            source_x = min(7, (((2 * x + 1) * 8) - 1) // (2 * output_size))
            offset = (source_y * 8 + source_x) * 4
            output.extend(source_bytes[offset : offset + 4])
    return bytes(output)


def capture(chrome: Path) -> tuple[dict[str, object], bytes]:
    with tempfile.TemporaryDirectory(prefix="gpui-image-sampling-oracle-") as directory:
        screenshot = Path(directory) / "oracle.png"
        subprocess.run(
            [
                str(chrome),
                "--headless=new",
                "--disable-gpu",
                "--hide-scrollbars",
                "--force-device-scale-factor=1",
                "--default-background-color=00000000",
                "--window-size=96,30",
                f"--screenshot={screenshot}",
                HTML.as_uri(),
            ],
            check=True,
        )
        for _ in range(100):
            if screenshot.exists():
                break
            time.sleep(0.01)
        png = screenshot.read_bytes()
        image = Image.open(screenshot).convert("RGBA")

    source = image.crop((0, 0, 8, 8))
    # The upper-left 8 output pixels are not the original source, so recreate
    # the locked canvas directly by rendering it once at its intrinsic size.
    with tempfile.TemporaryDirectory(prefix="gpui-image-source-oracle-") as directory:
        source_html = Path(directory) / "source.html"
        source_png = Path(directory) / "source.png"
        html = HTML.read_text(encoding="utf-8").replace(
            "#avatar22 { left: 0; width: 22px; height: 22px; }",
            "#avatar22 { left: 0; width: 8px; height: 8px; }",
        )
        source_html.write_text(html, encoding="utf-8")
        subprocess.run(
            [
                str(chrome), "--headless=new", "--disable-gpu", "--hide-scrollbars",
                "--force-device-scale-factor=1", "--default-background-color=00000000",
                "--window-size=96,30", f"--screenshot={source_png}", source_html.as_uri(),
            ],
            check=True,
        )
        source = Image.open(source_png).convert("RGBA").crop((0, 0, 8, 8))

    planes = bytearray()
    cases: dict[str, object] = {}
    for name, left, output_size in CASES:
        rgba = image.crop((left, 0, left + output_size, output_size)).tobytes()
        expected = pixelated(source, output_size)
        if rgba != expected:
            differences = [
                (index // 4 % output_size, index // 4 // output_size,
                 tuple(rgba[index : index + 4]), tuple(expected[index : index + 4]))
                for index in range(0, len(rgba), 4)
                if rgba[index : index + 4] != expected[index : index + 4]
            ]
            raise SystemExit(
                f"Chromium {name} differs from the pixelated model at {len(differences)} pixels: "
                f"{differences[:12]}; actual first row "
                f"{[rgba[x * 4] for x in range(output_size)]}; expected first row "
                f"{[expected[x * 4] for x in range(output_size)]}"
            )
        offset = len(planes)
        planes.extend(rgba)
        cases[name] = {
            "sourceSize": [8, 8],
            "outputSize": [output_size, output_size],
            "rgbaSha256": digest(rgba),
            "rgbaOffset": offset,
            "rgbaLength": len(rgba),
        }

    source_rgba = source.tobytes()
    source_offset = len(planes)
    planes.extend(source_rgba)
    result = {
        "oracle": "Chromium headless transparent PNG",
        "chromiumVersion": chromium_version(chrome),
        "deviceScaleFactor": 1,
        "source": "platform/v2/views.jsx PixelAvatar seed=3",
        "sourceSize": [8, 8],
        "sourceRgbaSha256": digest(source_rgba),
        "sourceRgbaOffset": source_offset,
        "sourceRgbaLength": len(source_rgba),
        "lockedCss": "image-rendering: pixelated",
        "lockedRuntimeSizes": [22, 26, 30],
        "pngSha256": digest(png),
        "rgbaPlaneSha256": digest(bytes(planes)),
        "cases": cases,
    }
    return result, bytes(planes)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chrome", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    actual, rgba_planes = capture(args.chrome.resolve())
    if args.verify and args.write:
        parser.error("--verify and --write are mutually exclusive")
    if args.verify:
        expected = json.loads(FIXTURE.read_text(encoding="utf-8"))
        expected_rgba = RGBA_FIXTURE.read_bytes()
        if actual != expected or rgba_planes != expected_rgba:
            raise SystemExit("Chromium image-sampling oracle drifted:\n" + json.dumps(actual, indent=2))
        print("image-sampling Chromium oracle matches")
    elif args.write:
        FIXTURE.write_text(json.dumps(actual, indent=2) + "\n", encoding="utf-8")
        RGBA_FIXTURE.write_bytes(rgba_planes)
        print(f"wrote {FIXTURE}")
    else:
        print(json.dumps(actual, indent=2))


if __name__ == "__main__":
    main()
