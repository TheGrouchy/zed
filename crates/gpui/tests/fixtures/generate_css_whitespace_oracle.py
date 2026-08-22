"""Capture the locked CSS whitespace modes from Chromium at fractional scale."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import subprocess
from pathlib import Path


HERE = Path(__file__).resolve().parent
HTML = HERE / "css_whitespace_oracle.html"
FIXTURE = HERE / "css_whitespace_chromium.json"
LOCKED_STYLESHEETS = (
    "styles.css",
    "v2/routes.css",
    "v2/agentic.css",
    "v2/skin-soft.css",
)


def locked_declarations(source_root: Path) -> list[list[str]]:
    declarations: list[list[str]] = []
    pattern = re.compile(r"white-space\s*:\s*(normal|nowrap|pre-line|pre-wrap)\b")
    for relative_path in LOCKED_STYLESHEETS:
        source_path = source_root / relative_path
        if not source_path.is_file():
            raise SystemExit(f"locked stylesheet is missing: {source_path}")
        for line_number, line in enumerate(
            source_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            declarations.extend(
                [[f"{relative_path}:{line_number}", match.group(1)] for match in pattern.finditer(line)]
            )
    return declarations


def chromium_version(chrome: Path) -> str:
    if os.name == "nt":
        command = f"(Get-Item -LiteralPath '{chrome}').VersionInfo.ProductVersion"
        return subprocess.check_output(
            ["powershell", "-NoProfile", "-Command", command], text=True
        ).strip()
    return subprocess.check_output([str(chrome), "--version"], text=True).strip().rsplit(" ", 1)[-1]


def capture(chrome: Path) -> dict[str, object]:
    output = subprocess.check_output(
        [
            str(chrome),
            "--headless=new",
            "--disable-gpu",
            "--force-device-scale-factor=1.25",
            "--virtual-time-budget=1000",
            "--dump-dom",
            HTML.as_uri(),
        ],
        text=True,
        encoding="utf-8",
    )
    match = re.search(r'<script id="result" type="application/json">(.*?)</script>', output, re.DOTALL)
    if match is None:
        raise SystemExit("Chromium did not emit the whitespace oracle payload")
    result = json.loads(html.unescape(match.group(1)))
    declarations = result["declarations"]
    if len(declarations) != 26:
        raise SystemExit(f"expected 26 locked declarations, got {len(declarations)}")
    counts = {mode: sum(item[1] == mode for item in declarations) for mode in ("normal", "nowrap", "pre-line", "pre-wrap")}
    if counts != {"normal": 2, "nowrap": 21, "pre-line": 1, "pre-wrap": 2}:
        raise SystemExit(f"locked declaration counts drifted: {counts}")
    result.update(
        {
            "oracle": "Chromium Range.getClientRects and element metrics",
            "chromiumVersion": chromium_version(chrome),
            "forcedDeviceScaleFactor": 1.25,
            "font": '13.5px/18.25px "Courier New", monospace',
            "widthCssPx": 97.25,
            "sourceHtmlSha256": hashlib.sha256(HTML.read_bytes()).hexdigest().upper(),
            "declarationCounts": counts,
        }
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chrome", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--write", action="store_true")
    parser.add_argument(
        "--source-root",
        type=Path,
        help="optional locked platform source root used to prove the 26 declaration inventory",
    )
    args = parser.parse_args()
    if args.verify and args.write:
        parser.error("--verify and --write are mutually exclusive")
    actual = capture(args.chrome.resolve())
    if args.source_root is not None:
        source_declarations = locked_declarations(args.source_root.resolve())
        if source_declarations != actual["declarations"]:
            raise SystemExit(
                "locked CSS whitespace declaration inventory drifted:\n"
                + json.dumps(source_declarations, indent=2)
            )
    if args.verify:
        expected = json.loads(FIXTURE.read_text(encoding="utf-8"))
        if actual != expected:
            raise SystemExit("Chromium whitespace oracle drifted:\n" + json.dumps(actual, indent=2))
        print("CSS whitespace Chromium oracle matches")
    elif args.write:
        FIXTURE.write_text(json.dumps(actual, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {FIXTURE}")
    else:
        print(json.dumps(actual, indent=2))


if __name__ == "__main__":
    main()
