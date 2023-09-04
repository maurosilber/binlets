from __future__ import annotations

from matplotlib.axes import Axes
from matplotlib.figure import Figure, SubFigure
from matplotlib.lines import Line2D


def label_line(axes: Axes, line: Line2D, *, label: str, offset: float = 0, **kwargs):
    if label == "" or label.startswith("_"):
        return

    kwargs = dict(color=line.get_color()) | kwargs
    y = line.get_ydata()[-1]
    axes.annotate(
        label,
        xy=(1, y + offset),
        xycoords=axes.get_yaxis_transform(),
        verticalalignment="center",
        **kwargs,
    )


def label_lines(axes: Axes, *, labels=None, offsets: list[float] = None, **kwargs):
    lines = axes.lines

    if labels is None:
        labels = [line.get_label() for line in lines]

    if offsets is None:
        offsets = [0 for line in lines]

    for line, label, offset in zip(lines, labels, offsets):
        label_line(axes, line, label=label, offset=offset, **kwargs)


def legend_without_duplicate_labels(fig_or_axes: Figure | Axes, **kwargs):
    if isinstance(fig_or_axes, (Figure, SubFigure)):
        legend = fig_or_axes.legend()
        unique = dict(zip((t.get_text() for t in legend.texts), legend.legend_handles))
        legend.remove()
        fig_or_axes.legend(unique.values(), unique.keys(), **kwargs)
    elif isinstance(fig_or_axes, Axes):
        handles, labels = fig_or_axes.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        fig_or_axes.legend(*zip(unique.values(), unique.keys()), **kwargs)
    else:
        raise NotImplementedError
