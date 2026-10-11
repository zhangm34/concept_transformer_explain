"""Display helpers for concept_transformer_presentation.ipynb.

Every helper renders inline-styled HTML: JupyterLab and Cursor strip <style> blocks, so each
element carries its own styling. Colours are the weekly-deck paper palette, stepped to keep at
least 3:1 contrast on light and dark themes. Written for the project's Python 3.8 kernel.
"""
from html import escape

from IPython.display import HTML, Image, display

PALETTE = {
    "accent": "#ac543f",  # terracotta: the disjoint rerun, the point being made
    "neg": "#4978a7",  # blue-grey: the repository protocol and baselines
    "pos": "#458554",  # green: verified or positive outcomes
    "warn": "#be7c1c",  # amber: caveats and upstream issues
    "purple": "#8062b0",  # a further category
    "rule": "rgba(128, 128, 128, 0.45)",  # neutral rules
}
SERIF = "font-family: Newsreader, Georgia, serif;"
KICKER = "font-size: 11px; letter-spacing: 0.12em; text-transform: uppercase; opacity: 0.65;"


def _colour(name):
    """A PALETTE key, or any CSS colour passed through unchanged."""
    return PALETTE.get(name, name)


def stat_row(stats):
    """Display computed headline numbers as paper-style tiles, side by side.

    stats: list of dicts with keys
        label  -- short plain text above the number
        value  -- the number already formatted as text, e.g. f"{100 * acc:.2f}%"
        detail -- optional plain text under the number
        color  -- optional PALETTE key or CSS colour for the top rule (default "rule")
    """
    width = 100 // max(len(stats), 1) - 4
    tiles = []
    for i, stat in enumerate(stats):
        gap = "" if i == len(stats) - 1 else " margin-right: 4%;"
        rule = _colour(stat.get("color", "rule"))
        detail = stat.get("detail")
        tiles.append(
            f'<div style="display: inline-block; vertical-align: top; width: {width}%;{gap} '
            f'border-top: 3px solid {rule}; padding-top: 8px;">'
            f'<div style="{KICKER}">{escape(stat["label"])}</div>'
            f'<div style="{SERIF} font-size: 48px; font-weight: 300; line-height: 1.15; '
            f'margin: 4px 0 2px 0;">{escape(stat["value"])}</div>'
            + (f'<div style="font-size: 13.5px; opacity: 0.75;">{escape(detail)}</div>' if detail else "")
            + "</div>"
        )
    display(HTML("<div>" + "".join(tiles) + "</div>"))


def fact_table(rows, title=None):
    """Display (label, value) pairs as a quiet two-column list, e.g. the environment report."""
    lines = []
    if title:
        lines.append(f'<div style="{KICKER} margin-bottom: 6px;">{escape(title)}</div>')
    for label, value in rows:
        lines.append(
            f'<div style="border-top: 1px solid {PALETTE["rule"]}; padding: 5px 0;">'
            '<span style="display: inline-block; width: 32%; vertical-align: top; opacity: 0.7;">'
            f"{escape(str(label))}</span>"
            '<span style="display: inline-block; width: 66%; vertical-align: top;">'
            f"{escape(str(value))}</span></div>"
        )
    display(HTML('<div style="font-size: 14px; max-width: 780px;">' + "".join(lines) + "</div>"))


def caption(text):
    """Caption under a figure. `text` may contain <b> and <i>; start it with <b>Figure N.k</b>."""
    display(HTML(
        '<div style="font-size: 13.5px; line-height: 1.45; opacity: 0.8; margin: 4px 0 10px 0; '
        'max-width: 960px;">' + text + "</div>"
    ))


def show_image(path, caption_text=None, width=None):
    """Embed an image file in the saved notebook (not a link), optionally captioned."""
    display(Image(filename=str(path), width=width))
    if caption_text:
        caption(caption_text)


def use_paper_style():
    """Matplotlib defaults that match the palette: white canvas, quiet axes, light grid."""
    import matplotlib.pyplot as plt  # first, so the inline backend has applied its own defaults
    from cycler import cycler

    plt.rcParams.update({
        "axes.prop_cycle": cycler(color=[PALETTE[k] for k in ("neg", "accent", "pos", "warn", "purple")]),
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.titlesize": 12,
        "axes.titleweight": "semibold",
        "axes.labelsize": 10,
        "legend.frameon": False,
        "figure.facecolor": "white",  # readable in dark themes too, like the saved PNG figures
        "savefig.facecolor": "white",
    })
    try:  # sharper figures on high-resolution screens and screen shares
        from matplotlib_inline.backend_inline import set_matplotlib_formats
        set_matplotlib_formats("retina")
    except (ImportError, AttributeError):
        pass
