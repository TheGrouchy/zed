"""Capture and verify Chromium's locked CSS image `invert(1)` output."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path

from PIL import Image


HERE = Path(__file__).resolve().parent
HTML = HERE / "image_filter_invert_oracle.html"
FIXTURE = HERE / "image_filter_invert_chromium.json"
RGBA_FIXTURE = HERE / "image_filter_invert_chromium.rgba"
SOURCE_PNG_SHA256 = "4936A82C01E3E31DBDCD175323551B3E331838A90719B91A4CDAA931132D579D"


def chromium_version(chrome: Path) -> str:
    if os.name == "nt":
        command = f"(Get-Item -LiteralPath '{chrome}').VersionInfo.ProductVersion"
        return subprocess.check_output(
            ["powershell", "-NoProfile", "-Command", command], text=True
        ).strip()
    return subprocess.check_output([str(chrome), "--version"], text=True).strip().rsplit(" ", 1)[-1]


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def source_image() -> tuple[bytes, Image.Image]:
    html = HTML.read_text(encoding="utf-8")
    encoded = re.search(r'id="source" src="data:image/png;base64,([^"]+)"', html)
    if encoded is None:
        raise SystemExit("locked mParticle data URL is missing")
    png = base64.b64decode(encoded.group(1))
    if digest(png) != SOURCE_PNG_SHA256:
        raise SystemExit("locked mParticle PNG hash changed")
    return png, Image.open(io.BytesIO(png)).convert("RGBA")


def straight_invert(source: bytes) -> bytes:
    output = bytearray()
    for red, green, blue, alpha in zip(*[iter(source)] * 4):
        if alpha == 0:
            output.extend((0, 0, 0, 0))
        else:
            # Shader contract: premultiplied_out = alpha - premultiplied_in.
            # Chromium's 8-bit canvas path rounds on both conversions.
            channels = []
            for channel in (red, green, blue):
                premultiplied = (channel * alpha + 127) // 255
                inverted_premultiplied = alpha - premultiplied
                channels.append((inverted_premultiplied * 255 + alpha // 2) // alpha)
            output.extend((*channels, alpha))
    return bytes(output)


def capture(chrome: Path) -> tuple[dict[str, object], bytes]:
    source_png, source = source_image()
    with tempfile.TemporaryDirectory(prefix="gpui-image-filter-oracle-") as directory:
        screenshot = Path(directory) / "oracle.png"
        subprocess.run(
            [
                str(chrome),
                "--headless=new",
                "--disable-gpu",
                "--hide-scrollbars",
                "--force-device-scale-factor=1",
                "--default-background-color=00000000",
                "--window-size=35,16",
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
        screenshot_image = Image.open(screenshot).convert("RGBA")

    normal = screenshot_image.crop((0, 0, 16, 16)).tobytes()
    inverted = screenshot_image.crop((16, 0, 32, 16)).tobytes()
    edge_output = screenshot_image.crop((32, 0, 35, 1)).tobytes()
    source_rgba = source.tobytes()
    if source.size != (48, 48):
        raise SystemExit(f"locked mParticle intrinsic size changed: {source.size}")
    expected_inverted = straight_invert(normal)
    if inverted != expected_inverted:
        mismatch = next(
            index // 4
            for index in range(0, len(inverted), 4)
            if inverted[index : index + 4] != expected_inverted[index : index + 4]
        )
        raise SystemExit(f"Chromium mParticle invert differs at pixel {mismatch}")
    expected_edges = straight_invert(
        bytes((17, 33, 49, 0, 64, 128, 192, 128, 10, 20, 30, 255))
    )
    if edge_output != expected_edges:
        raise SystemExit(f"Chromium edge pixels differ: {list(edge_output)}")

    planes = normal + inverted + edge_output
    normal_length = len(normal)
    inverted_length = len(inverted)
    metadata = {
        "oracle": "Chromium headless transparent PNG",
        "chromiumVersion": chromium_version(chrome),
        "deviceScaleFactor": 1,
        "lockedSource": "platform/v2/routes.css img.int-logo.inv { filter: invert(1) }",
        "lockedAsset": "platform/integration-logos/mparticle-favicon.png",
        "sourcePngSha256": digest(source_png),
        "sourceRgbaSha256": digest(source_rgba),
        "sourceSize": [48, 48],
        "lockedRuntimeSize": [16, 16],
        "sourceAlphaCounts": {
            "transparent": sum(source_rgba[index + 3] == 0 for index in range(0, len(source_rgba), 4)),
            "partial": sum(0 < source_rgba[index + 3] < 255 for index in range(0, len(source_rgba), 4)),
            "opaque": sum(source_rgba[index + 3] == 255 for index in range(0, len(source_rgba), 4)),
        },
        "premultipliedFormula": "rgb_out = alpha - rgb_in; alpha_out = alpha",
        "pngSha256": digest(png),
        "rgbaPlaneSha256": digest(planes),
        "cases": {
            "mparticleNormal": {"rgbaOffset": 0, "rgbaLength": normal_length, "rgbaSha256": digest(normal)},
            "mparticleInvert": {"rgbaOffset": normal_length, "rgbaLength": inverted_length, "rgbaSha256": digest(inverted)},
            "alphaEdgesInvert": {
                "sourceRgba": [17, 33, 49, 0, 64, 128, 192, 128, 10, 20, 30, 255],
                "rgbaOffset": normal_length + inverted_length,
                "rgbaLength": len(edge_output),
                "rgbaSha256": digest(edge_output),
            },
        },
    }
    return metadata, planes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chrome", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    if args.verify and args.write:
        parser.error("--verify and --write are mutually exclusive")
    actual, planes = capture(args.chrome.resolve())
    if args.verify:
        expected = json.loads(FIXTURE.read_text(encoding="utf-8"))
        if actual != expected or planes != RGBA_FIXTURE.read_bytes():
            raise SystemExit("Chromium image-filter oracle drifted:\n" + json.dumps(actual, indent=2))
        print("image-filter Chromium oracle matches")
    elif args.write:
        FIXTURE.write_text(json.dumps(actual, indent=2) + "\n", encoding="utf-8")
        RGBA_FIXTURE.write_bytes(planes)
        print(f"wrote {FIXTURE}")
    else:
        print(json.dumps(actual, indent=2))


if __name__ == "__main__":
    main()
