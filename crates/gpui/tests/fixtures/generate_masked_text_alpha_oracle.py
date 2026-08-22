"""Regenerate/verify Chromium's transparent masked-text alpha fixture.

Requires Chrome/Chromium and Pillow. The command never updates the checked-in
fixture implicitly; `--verify` compares the fresh capture and exits nonzero on
any pixel or metadata drift.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

from PIL import Image


HERE = Path(__file__).resolve().parent
HTML = HERE / "masked_text_alpha_oracle.html"
FIXTURE = HERE / "masked_text_alpha_chromium.json"


def chromium_version(chrome: Path) -> str:
    if os.name == "nt":
        command = f"(Get-Item -LiteralPath '{chrome}').VersionInfo.ProductVersion"
        return subprocess.check_output(
            ["powershell", "-NoProfile", "-Command", command], text=True
        ).strip()
    output = subprocess.check_output([str(chrome), "--version"], text=True).strip()
    return output.rsplit(" ", 1)[-1]


def capture(chrome: Path) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="gpui-mask-oracle-") as directory:
        screenshot = Path(directory) / "oracle.png"
        subprocess.run(
            [
                str(chrome),
                "--headless=new",
                "--disable-gpu",
                "--hide-scrollbars",
                "--force-device-scale-factor=1",
                "--default-background-color=00000000",
                "--window-size=256,64",
                f"--screenshot={screenshot}",
                HTML.as_uri(),
            ],
            check=True,
        )
        # Chrome's Windows launcher may return just before the browser process
        # finishes the file. A bounded poll avoids accepting an absent capture.
        for _ in range(100):
            if screenshot.exists():
                break
            import time

            time.sleep(0.01)
        png = screenshot.read_bytes()
        image = Image.open(screenshot).convert("RGBA")

    width, height = image.size
    alpha = list(image.getchannel("A").get_flattened_data())
    unmasked = [alpha[y * width + x] for y in range(height) for x in range(128)]
    masked = [alpha[y * width + x + 128] for y in range(height) for x in range(128)]
    full_coverage = [
        [x, y, unmasked[y * 128 + x], masked[y * 128 + x]]
        for y in range(height)
        for x in range(128)
        if unmasked[y * 128 + x] >= 200
    ]
    stride = max(1, len(full_coverage) // 12)
    samples = full_coverage[::stride][:12]
    digest = lambda value: hashlib.sha256(value).hexdigest().upper()
    return {
        "oracle": "Chromium headless transparent PNG",
        "chromiumVersion": chromium_version(chrome),
        "deviceScaleFactor": 1,
        "viewport": [width, height],
        "localMaskWidth": 128,
        "text": "H",
        "font": "400 48px/64px Arial, sans-serif",
        "textAlpha": 0.8,
        "mask": "linear-gradient(90deg, transparent 0%, black 100%)",
        "pngSha256": digest(png),
        "alphaSha256": digest(bytes(alpha)),
        "unmaskedAlphaSha256": digest(bytes(unmasked)),
        "maskedAlphaSha256": digest(bytes(masked)),
        "unmaskedNonzeroPixels": sum(value != 0 for value in unmasked),
        "maskedNonzeroPixels": sum(value != 0 for value in masked),
        "unmaskedAlphaSum": sum(unmasked),
        "maskedAlphaSum": sum(masked),
        "samples": samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chrome", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    actual = capture(args.chrome.resolve())
    if args.verify:
        expected = json.loads(FIXTURE.read_text(encoding="utf-8"))
        if actual != expected:
            raise SystemExit(
                "Chromium masked-text alpha oracle drifted:\n"
                + json.dumps(actual, indent=2)
            )
        print("masked-text Chromium oracle matches")
    else:
        print(json.dumps(actual, indent=2))


if __name__ == "__main__":
    main()
