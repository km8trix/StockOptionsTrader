"""Formal WCAG 2.1 contrast audit of the dark-theme palette (UX polish follow-up).

Parses the CSS custom properties straight out of gui/static/css/style.css and
asserts every text/background pairing the UI actually uses clears its WCAG AA
threshold — 4.5:1 for normal reading text, 3:1 for large text / UI components.
This both records the audit result and fails the build if a future palette tweak
drops a colour below AA. Pure stdlib (re + math), no parsing libraries.
"""
from __future__ import annotations

import re
from pathlib import Path

STYLE_CSS = Path(__file__).resolve().parents[1] / 'gui' / 'static' / 'css' / 'style.css'


def _palette() -> dict[str, str]:
    """`--name: #rrggbb;` custom properties declared in style.css."""
    text = STYLE_CSS.read_text()
    return {name: hex_ for name, hex_ in
            re.findall(r'--([\w-]+):\s*(#[0-9a-fA-F]{6})\s*;', text)}


def _relative_luminance(hex_color: str) -> float:
    h = hex_color.lstrip('#')
    chans = []
    for i in (0, 2, 4):
        c = int(h[i:i + 2], 16) / 255
        chans.append(c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4)
    r, g, b = chans
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _contrast(fg: str, bg: str) -> float:
    hi, lo = sorted((_relative_luminance(fg), _relative_luminance(bg)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def test_reading_text_meets_aa_normal():
    """Body text, captions, links, and P&L numbers — WCAG AA normal (4.5:1)."""
    p = _palette()
    pairings = [
        ('text', 'bg'), ('text', 'surface'), ('text', 'surface-2'),
        ('text-muted', 'bg'), ('text-muted', 'surface'),  # subtitles, captions
        ('accent', 'bg'), ('accent', 'surface'),          # links
        ('gain', 'bg'), ('gain', 'surface'),              # P&L green
        ('loss', 'bg'), ('loss', 'surface'),              # P&L red
        ('warn', 'bg'),
    ]
    failures = [f'{fg} on {bg}: {_contrast(p[fg], p[bg]):.2f}:1'
                for fg, bg in pairings if _contrast(p[fg], p[bg]) < 4.5]
    assert not failures, 'Below WCAG AA 4.5:1 — ' + '; '.join(failures)


def test_white_on_semantic_meets_ui_threshold():
    """#fff on a solid semantic fill (badges, kill banner, primary button) is
    large/UI text — WCAG AA 3:1."""
    p = _palette()
    failures = [f'white on {bg}: {_contrast("#ffffff", p[bg]):.2f}:1'
                for bg in ('loss', 'accent') if _contrast('#ffffff', p[bg]) < 3.0]
    assert not failures, 'Below WCAG AA 3:1 — ' + '; '.join(failures)


if __name__ == '__main__':  # ponytail: print the audit table when run directly
    pal = _palette()
    for fg, bg in [('text', 'bg'), ('text-muted', 'bg'), ('text-muted', 'surface'),
                   ('accent', 'bg'), ('gain', 'bg'), ('loss', 'bg'), ('warn', 'bg')]:
        print(f'{fg:>11} on {bg:<8} {_contrast(pal[fg], pal[bg]):5.2f}:1')
    for bg in ('loss', 'accent'):
        print(f'      white on {bg:<8} {_contrast("#ffffff", pal[bg]):5.2f}:1')
