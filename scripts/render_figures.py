"""Render dependency-free SVG evidence figures from the frozen summary."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = json.loads((ROOT / "results/summary.json").read_text())
OUT = ROOT / "reports/iso-merger-reproduction/images"
OUT.mkdir(parents=True, exist_ok=True)

NAVY = "#172554"
BLUE = "#2563eb"
CYAN = "#0891b2"
GOLD = "#d97706"
RED = "#dc2626"
GREEN = "#059669"
GRID = "#dbe4ee"
TEXT = "#172033"
MUTED = "#64748b"


def header(title: str, subtitle: str, width: int = 920, height: int = 500) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="48" y="48" font-family="Inter,Arial,sans-serif" font-size="24" font-weight="700" fill="{TEXT}">{title}</text>',
        f'<text x="48" y="76" font-family="Inter,Arial,sans-serif" font-size="14" fill="{MUTED}">{subtitle}</text>',
    ]


def text(x: float, y: float, value: str, size: int = 13, anchor: str = "start", color: str = TEXT, weight: int = 400) -> str:
    return f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" font-family="Inter,Arial,sans-serif" font-size="{size}" font-weight="{weight}" fill="{color}">{value}</text>'


def save(name: str, parts: list[str]) -> None:
    parts.append("</svg>")
    (OUT / name).write_text("\n".join(parts) + "\n")


def bar_chart() -> None:
    vals = [
        ("Base", 0.545545, 0.0, NAVY),
        ("Average", 0.605856, 0.006120, CYAN),
        ("ISO", 0.620349, 0.005446, BLUE),
        ("Task Arithmetic", 0.628326, 0.006763, GOLD),
    ]
    p = header("Balanced held-out accuracy", "Mean ± sample SD over five independent specialist-training seeds")
    left, top, bottom, right = 90, 110, 410, 875
    lo, hi = 0.52, 0.65
    for tick in [0.52, 0.55, 0.58, 0.61, 0.64]:
        y = bottom - (tick - lo) / (hi - lo) * (bottom - top)
        p += [f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" stroke="{GRID}"/>', text(left - 10, y + 4, f"{tick:.2f}", 12, "end", MUTED)]
    bw = 110
    for i, (label, value, sd, color) in enumerate(vals):
        x = 125 + i * 170
        y = bottom - (value - lo) / (hi - lo) * (bottom - top)
        p.append(f'<rect x="{x}" y="{y:.1f}" width="{bw}" height="{bottom-y:.1f}" rx="5" fill="{color}"/>')
        err = sd / (hi - lo) * (bottom - top)
        p += [
            f'<line x1="{x+bw/2}" y1="{y-err:.1f}" x2="{x+bw/2}" y2="{y+err:.1f}" stroke="{TEXT}" stroke-width="2"/>',
            f'<line x1="{x+bw/2-8}" y1="{y-err:.1f}" x2="{x+bw/2+8}" y2="{y-err:.1f}" stroke="{TEXT}" stroke-width="2"/>',
            text(x + bw / 2, y - err - 10, f"{100*value:.2f}%", 14, "middle", TEXT, 700),
            text(x + bw / 2, 438, label, 13, "middle"),
        ]
    p += [text(48, 478, "ISO improved on averaging (+1.45 points) but trailed Task Arithmetic (−0.80 points).", 14, "start", MUTED)]
    save("primary_accuracy.svg", p)


def retention_chart() -> None:
    vals = [
        ("Average", 0.451168, CYAN),
        ("ISO", 0.598259, BLUE),
        ("No restoration", 0.602815, GREEN),
        ("No mask", 0.614588, RED),
        ("Task Arithmetic", 0.646826, GOLD),
    ]
    p = header("Specialist-gain retention", "Mean of per-task (merged − base) / (specialist − base), five seeds")
    left, top, bottom, right = 90, 110, 410, 875
    for tick in [0.0, 0.2, 0.4, 0.6, 0.8]:
        y = bottom - tick / 0.8 * (bottom - top)
        p += [f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" stroke="{GRID}"/>', text(left - 10, y + 4, f"{tick:.1f}", 12, "end", MUTED)]
    bw = 105
    for i, (label, value, color) in enumerate(vals):
        x = 112 + i * 150
        y = bottom - value / 0.8 * (bottom - top)
        p += [
            f'<rect x="{x}" y="{y:.1f}" width="{bw}" height="{bottom-y:.1f}" rx="5" fill="{color}"/>',
            text(x + bw / 2, y - 10, f"{100*value:.1f}%", 14, "middle", TEXT, 700),
            text(x + bw / 2, 435, label, 12, "middle"),
        ]
    p.append(text(48, 478, "The two ablations increased, rather than decreased, retained gain in this bounded setup.", 14, "start", MUTED))
    save("gain_retention.svg", p)


def spectrum_chart() -> None:
    vals = [
        ("ISO float64 reconstruction", -12.983, BLUE),
        ("ISO float32 checkpoint", -7.793, CYAN),
        ("Float32 tolerance", -5.0, NAVY),
        ("No-restoration drift", -2.938, RED),
    ]
    p = header("Spectrum preservation spans ten orders of magnitude", "Worst relative error over 85 matrix checks; x-axis is log₁₀(relative error)")
    left, right, top = 280, 700, 125
    lo, hi = -14, -2
    for tick in range(-14, -1, 2):
        x = left + (tick - lo) / (hi - lo) * (right - left)
        p += [f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="410" stroke="{GRID}"/>', text(x, 435, str(tick), 12, "middle", MUTED)]
    for i, (label, value, color) in enumerate(vals):
        y = 165 + i * 68
        x = left + (value - lo) / (hi - lo) * (right - left)
        p += [
            text(left - 18, y + 5, label, 13, "end"),
            f'<line x1="{left}" y1="{y}" x2="{x:.1f}" y2="{y}" stroke="{color}" stroke-width="5"/>',
            f'<circle cx="{x:.1f}" cy="{y}" r="8" fill="{color}"/>',
            text(min(x + 14, 670), y + 5, f"10^{value:.2f}", 12, "start", color, 700),
        ]
    p.append(text(48, 478, "Base-spectrum reconstruction passed both preregistered tolerances on every seed and matrix.", 14, "start", MUTED))
    save("spectrum_error.svg", p)


def seed_robustness() -> None:
    rows = DATA["perSeed"]
    p = header("The method ordering held for every seed", "Balanced accuracy difference relative to ISO; positive favors the comparison method")
    left, right, top, bottom = 100, 875, 120, 405
    lo, hi = -0.02, 0.016
    zero = bottom - (0 - lo) / (hi - lo) * (bottom - top)
    p.append(f'<line x1="{left}" y1="{zero:.1f}" x2="{right}" y2="{zero:.1f}" stroke="{TEXT}" stroke-width="2"/>')
    for tick in [-0.02, -0.01, 0.0, 0.01]:
        y = bottom - (tick - lo) / (hi - lo) * (bottom - top)
        p += [f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" stroke="{GRID}"/>', text(left - 10, y + 4, f"{100*tick:+.1f} pp", 12, "end", MUTED)]
    for i, row in enumerate(rows):
        x = 140 + i * 140
        for delta, color, dx in [
            (row["taskArithmetic"] - row["iso"], GOLD, -10),
            (row["average"] - row["iso"], CYAN, 10),
        ]:
            y = bottom - (delta - lo) / (hi - lo) * (bottom - top)
            p.append(f'<circle cx="{x+dx}" cy="{y:.1f}" r="8" fill="{color}"/>')
        p.append(text(x, 433, f"seed {row['seed']}", 12, "middle"))
    p += [
        f'<circle cx="470" cy="468" r="7" fill="{GOLD}"/>', text(485, 473, "Task Arithmetic − ISO", 13),
        f'<circle cx="700" cy="468" r="7" fill="{CYAN}"/>', text(715, 473, "Average − ISO", 13),
    ]
    save("seed_robustness.svg", p)


def ablation_chart() -> None:
    rows = DATA["perSeed"]
    p = header("Ablations did not isolate the proposed benefit", "Balanced accuracy change from full ISO; all observed changes were non-negative")
    left, right, top, bottom = 100, 875, 120, 405
    lo, hi = -0.0005, 0.0036
    for tick in [0.0, 0.001, 0.002, 0.003]:
        y = bottom - (tick - lo) / (hi - lo) * (bottom - top)
        p += [f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" stroke="{GRID}"/>', text(left - 10, y + 4, f"{100*tick:+.2f} pp", 12, "end", MUTED)]
    for i, row in enumerate(rows):
        x = 140 + i * 140
        for delta, color, dx in [
            (row["isoNoRestore"] - row["iso"], GREEN, -10),
            (row["isoNoMask"] - row["iso"], RED, 10),
        ]:
            y = bottom - (delta - lo) / (hi - lo) * (bottom - top)
            p.append(f'<circle cx="{x+dx}" cy="{y:.1f}" r="8" fill="{color}"/>')
        p.append(text(x, 433, f"seed {row['seed']}", 12, "middle"))
    p += [
        f'<circle cx="390" cy="468" r="7" fill="{GREEN}"/>', text(405, 473, "No restoration − ISO", 13),
        f'<circle cx="620" cy="468" r="7" fill="{RED}"/>', text(635, 473, "No mask − ISO", 13),
    ]
    save("ablation_deltas.svg", p)


bar_chart()
retention_chart()
spectrum_chart()
seed_robustness()
ablation_chart()
print(f"Rendered five figures to {OUT}")
