"""Capture deterministic Chromium text-shadow alpha/color evidence."""

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
HTML = HERE / "text_shadow_alpha_oracle.html"
FIXTURE = HERE / "text_shadow_alpha_chromium.json"
ALPHA_FIXTURE = HERE / "text_shadow_alpha_chromium.a8"
WIDTH = 128
HEIGHT = 96
NAMES = ("hard", "blur3", "blur4", "blur9", "locked")


def chromium_version(chrome: Path) -> str:
    if os.name == "nt":
        command = f"(Get-Item -LiteralPath '{chrome}').VersionInfo.ProductVersion"
        return subprocess.check_output(
            ["powershell", "-NoProfile", "-Command", command], text=True
        ).strip()
    return subprocess.check_output([str(chrome), "--version"], text=True).strip().rsplit(" ", 1)[-1]


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def capture(chrome: Path) -> tuple[dict[str, object], bytes]:
    with tempfile.TemporaryDirectory(prefix="gpui-text-shadow-oracle-") as directory:
        screenshot = Path(directory) / "oracle.png"
        subprocess.run(
            [
                str(chrome),
                "--headless=new",
                "--disable-gpu",
                "--hide-scrollbars",
                "--force-device-scale-factor=1",
                "--default-background-color=00000000",
                "--window-size=640,96",
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

    cells: dict[str, object] = {}
    alpha_planes = bytearray()
    for index, name in enumerate(NAMES):
        crop = image.crop((index * WIDTH, 0, (index + 1) * WIDTH, HEIGHT))
        rgba = crop.tobytes()
        alpha = bytes(crop.getchannel("A").get_flattened_data())
        alpha_offset = len(alpha_planes)
        alpha_planes.extend(alpha)
        bbox = crop.getchannel("A").getbbox()
        cells[name] = {
            "rgbaSha256": digest(rgba),
            "alphaSha256": digest(alpha),
            "alphaSum": sum(alpha),
            "nonzeroPixels": sum(value != 0 for value in alpha),
            "alphaBounds": list(bbox) if bbox else None,
            # The complete A8 plane lives in the adjacent binary fixture so
            # Rust can compare every pixel without expanding JSON by 180 KiB.
            "alphaOffset": alpha_offset,
            "alphaLength": len(alpha),
        }

    result = {
        "oracle": "Chromium headless transparent PNG",
        "chromiumVersion": chromium_version(chrome),
        "deviceScaleFactor": 1,
        "viewport": [640, 96],
        "cellSize": [WIDTH, HEIGHT],
        "font": "400 48px/96px Arial, sans-serif",
        "lockedDeclarations": [
            "0 0 4px #0a0a0a, 0 0 9px #0a0a0a",
            "0 0 4px #0a0a0a, 0 0 9px #0a0a0a, 0 1px 3px #0a0a0a",
            "0 0 2px var(--bg), 0 0 3px var(--bg), 0 0 5px var(--bg)",
            "0 0 3px var(--bg), 0 0 5px var(--bg)",
            "0 0 3px var(--bg), 0 0 7px var(--bg)",
        ],
        "lockedBackgrounds": [
            "#E8E1D4",
            "#E1E7EF",
            "#E5E4E1",
            "#1B1712",
            "#14181E",
            "#171716",
        ],
        "pngSha256": digest(png),
        "alphaPlaneSha256": digest(bytes(alpha_planes)),
        "cells": cells,
    }
    return result, bytes(alpha_planes)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chrome", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    actual, alpha_planes = capture(args.chrome.resolve())
    if args.verify and args.write:
        parser.error("--verify and --write are mutually exclusive")
    if args.verify:
        expected = json.loads(FIXTURE.read_text(encoding="utf-8"))
        expected_alpha = ALPHA_FIXTURE.read_bytes()
        if actual != expected or alpha_planes != expected_alpha:
            raise SystemExit("Chromium text-shadow oracle drifted:\n" + json.dumps(actual, indent=2))
        print("text-shadow Chromium oracle matches")
    elif args.write:
        FIXTURE.write_text(json.dumps(actual, indent=2) + "\n", encoding="utf-8")
        ALPHA_FIXTURE.write_bytes(alpha_planes)
        print(f"wrote {FIXTURE}")
    else:
        print(json.dumps(actual, indent=2))


if __name__ == "__main__":
    main()
