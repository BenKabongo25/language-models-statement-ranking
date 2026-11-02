"""Shared figure style and layouts (vector PDF with embedded fonts + 600 dpi PNG).

Style: serif text (Linux Libertine, the ACM font, when installed, else a Times-like font), the validated
categorical palette (fixed order, colour-blind safe on adjacent pairs), recessive axes and grid.

Layouts: a multi-dataset figure is drawn from *units* (one per dataset; a unit is one panel, or k panels such as
zero-shot | fine-tuned) and saved three ways, so the paper can use any format:

* ``<name>_<Dataset>``: one unit, single-column width;
* ``<name>_landscape``: all units side by side, full text width (a k-panel unit becomes a column);
* ``<name>_portrait``: all units stacked, single-column width (a k-panel unit stays a row).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

PALETTE = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
MARKERS = ['o', 's', 'D', '^', 'v', 'P', 'X', '*']
OTHER = '#c3c2b7'
TEXT, MUTED, GRID, SURFACE = '#1f1f1d', '#52514e', '#e4e3de', '#ffffff'
SEQUENTIAL = ['#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec', '#5598e7', '#3987e5', '#2a78d6', '#256abf',
              '#1c5cab', '#184f95', '#104281', '#0d366b']
SINGLE, DOUBLE = 3.33, 7.0            # column / text widths (inches) of two-column templates (ACM)
FONTS = ['Linux Libertine O', 'Linux Libertine', 'Libertinus Serif', 'TeX Gyre Termes', 'Times New Roman',
         'Nimbus Roman', 'DejaVu Serif']
METRIC_NAMES = {'ndcg': 'NDCG', 'recall': 'Recall', 'precision': 'Precision', 'hr': 'HR', 'f1': 'F1'}
LAYOUTS = ('single', 'landscape', 'portrait')


def style(font=None):
    """Apply the paper style; returns ``matplotlib.pyplot``."""
    import matplotlib
    matplotlib.use('Agg', force=False)
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    available = {f.name for f in font_manager.fontManager.ttflist}
    family = next((f for f in ([font] if font else []) + FONTS if f in available), 'serif')
    plt.rcParams.update({
        'font.family': family, 'font.size': 8, 'axes.titlesize': 8.5, 'axes.labelsize': 8, 'xtick.labelsize': 7.5,
        'ytick.labelsize': 7, 'legend.fontsize': 7.5, 'legend.frameon': False, 'axes.titlelocation': 'left',
        'axes.titlepad': 4, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
        'axes.edgecolor': MUTED, 'axes.linewidth': 0.6, 'axes.labelcolor': TEXT, 'text.color': TEXT,
        'xtick.color': TEXT, 'ytick.color': MUTED, 'xtick.major.size': 0, 'ytick.major.size': 0,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'axes.grid.axis': 'y',
        'grid.color': GRID, 'grid.linewidth': 0.5, 'grid.linestyle': '-', 'axes.axisbelow': True,
        'lines.linewidth': 1.4, 'lines.markersize': 4, 'hatch.linewidth': 0.55,
        'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE})
    return plt


def tint(color, amount=0.62):
    """``color`` mixed with white (``amount`` = share of white)."""
    import matplotlib.colors as mc
    rgb = np.array(mc.to_rgb(color))
    return tuple(rgb + (1 - rgb) * amount)


def metric_label(metric):
    """'ndcg@5' -> 'NDCG@5'."""
    name, _, k = metric.partition('@')
    pretty = METRIC_NAMES.get(name, name)
    return f'{pretty}@{k}' if k else pretty


def save(fig, path, dpi=600):
    """Save ``path`` as PDF and PNG (any extension of ``path`` is replaced); returns the PDF path."""
    import matplotlib.pyplot as plt
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stem = path.with_suffix('') if path.suffix in ('.pdf', '.png', '.svg') else path
    for ext in ('pdf', 'png'):
        fig.savefig(stem.with_name(stem.name + '.' + ext), dpi=dpi, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    return stem.with_name(stem.name + '.pdf')


def legend_top(fig, handles, ncol=None, y=1.0):
    fig.legend(handles=handles, loc='upper center', ncol=ncol or len(handles), frameon=False, handlelength=1.6,
               handletextpad=0.4, columnspacing=1.2, bbox_to_anchor=(0.5, y))


UNIT_LEGEND_COLS = {'single': 3, 'landscape': 2, 'portrait': 3}
LEGEND_ROW = 0.15                        # inches per legend row


def unit_grid(plt, layout, n_units, k, unit_h, legend_h=None):
    """Figure, the k axes of every unit and, when ``legend_h`` (inches per unit) is given, one legend axes per
    unit (below its panels)."""
    if legend_h is None:
        if layout == 'single':
            fig, axes = plt.subplots(1, k, figsize=(SINGLE, unit_h), squeeze=False)
            return fig, [list(axes[0])], None
        if layout == 'landscape':
            fig, axes = plt.subplots(k, n_units, figsize=(DOUBLE, unit_h * k), squeeze=False)
            return fig, [list(axes[:, i]) for i in range(n_units)], None
        fig, axes = plt.subplots(n_units, k, figsize=(SINGLE, unit_h * n_units), squeeze=False)
        return fig, [list(axes[i, :]) for i in range(n_units)], None
    if layout == 'single':
        fig = plt.figure(figsize=(SINGLE, unit_h + legend_h[0]))
        gs = fig.add_gridspec(2, k, height_ratios=[unit_h, legend_h[0]])
        return fig, [[fig.add_subplot(gs[0, c]) for c in range(k)]], [fig.add_subplot(gs[1, :])]
    if layout == 'landscape':
        lh = max(legend_h)
        fig = plt.figure(figsize=(DOUBLE, unit_h * k + lh))
        gs = fig.add_gridspec(k + 1, n_units, height_ratios=[unit_h] * k + [lh])
        units = [[fig.add_subplot(gs[r, i]) for r in range(k)] for i in range(n_units)]
        return fig, units, [fig.add_subplot(gs[k, i]) for i in range(n_units)]
    fig = plt.figure(figsize=(SINGLE, unit_h * n_units + sum(legend_h)))
    ratios = [h for lh in legend_h for h in (unit_h, lh)]
    gs = fig.add_gridspec(2 * n_units, k, height_ratios=ratios)
    units = [[fig.add_subplot(gs[2 * i, c]) for c in range(k)] for i in range(n_units)]
    return fig, units, [fig.add_subplot(gs[2 * i + 1, :]) for i in range(n_units)]


def multi_figure(name, out, units, draw, k=1, unit_h=2.1, panel_names=None, legend=None, layouts=LAYOUTS,
                 font=None, single_name=None, narrow_cols=3):
    """Draw ``units`` (list of ``(dataset, payload)``) with ``draw(axes, payload)`` in every layout.

    ``draw`` fills the k axes of a unit and may return legend handles. ``legend``: ``'shared'`` (one legend above
    the figure, from the first unit; ``narrow_cols`` entries per row at single-column width), ``'unit'`` (one
    legend per unit, below its panels, e.g. dataset-specific categories) or None. ``panel_names`` (k names) label
    the panels of a unit (zero-shot | fine-tuned). ``single_name`` overrides the file name of the single-column
    figures (default ``<name>_<dataset>``). Returns the written PDF paths.
    """
    plt = style(font)
    counts = None
    if legend == 'unit':                     # dry run: number of legend entries of every unit
        counts = []
        for _, payload in units:
            fig, axes, _ = unit_grid(plt, 'single', 1, k, unit_h)
            counts.append(len(draw(axes[0], payload) or []))
            plt.close(fig)
    written = []
    for layout in layouts:
        groups = [[j] for j in range(len(units))] if layout == 'single' else [list(range(len(units)))]
        for group in groups:
            legend_h = None
            if counts is not None:
                ncol = UNIT_LEGEND_COLS[layout]
                legend_h = [LEGEND_ROW * max(1, -(-counts[j] // ncol)) + 0.08 for j in group]
            fig, unit_axes, legend_axes = unit_grid(plt, layout, len(group), k, unit_h, legend_h)
            handles = []
            for i, (j, axes) in enumerate(zip(group, unit_axes)):
                dataset, payload = units[j]
                handles.append(draw(axes, payload) or [])
                _titles(layout, i, len(group), dataset, axes, panel_names, k)
            top = 1.0
            if legend == 'shared' and handles and handles[0]:
                ncol = len(handles[0]) if layout == 'landscape' else min(narrow_cols, len(handles[0]))
                rows = -(-len(handles[0]) // ncol)
                legend_top(fig, handles[0], ncol)
                top = 1 - (0.17 * rows + 0.06) / fig.get_figheight()
            if legend_axes is not None:
                for h, ax in zip(handles, legend_axes):
                    ax.axis('off')
                    if h:
                        ax.legend(handles=h, loc='upper center', ncol=UNIT_LEGEND_COLS[layout], frameon=False,
                                  fontsize=6.5, handletextpad=0.3, columnspacing=0.9, borderaxespad=0.0)
            fig.tight_layout(rect=(0, 0, 1, top), w_pad=1.0, h_pad=0.6)
            label = re.sub(r'[^\w\-]', '', str(units[group[0]][0]))     # file-name safe ('NDCG@5' -> 'NDCG5')
            stem = (single_name or f'{name}_{label}') if layout == 'single' else f'{name}_{layout}'
            written.append(save(fig, Path(out) / stem))
    return written


def _titles(layout, i, n, dataset, axes, panel_names, k):
    if k == 1:
        ax = axes[0]
        ax.set_title(dataset, loc='left')
        if layout == 'landscape' and i > 0:
            ax.set_ylabel('')
        if layout == 'portrait' and i < n - 1:
            ax.set_xlabel('')
        return
    names = panel_names or [''] * k
    if layout == 'single':
        for ax, title in zip(axes, names):
            ax.set_title(f'{dataset} · {title}' if title else dataset, loc='left')
    elif layout == 'landscape':                     # unit = column: dataset on top, panel names on the left
        axes[0].set_title(dataset, loc='left')
        if i == 0:
            for ax, title in zip(axes, names):
                ax.set_ylabel(title)
    else:                                           # unit = row: panel names on top, dataset on the left
        if i == 0:
            for ax, title in zip(axes, names):
                ax.set_title(title, loc='left')
        axes[0].set_ylabel(dataset)
