"""One still picture of a replay: paths seen from above, and what light delay does.

Reads only the replay. The light-delayed positions come from ``light.emission_time``, the
same rule the simulator uses, applied to the replayed trajectories.
"""

import io
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.layout_engine import ConstrainedLayoutEngine
from matplotlib.ticker import FuncFormatter
from numpy.typing import NDArray

from swarmsim.core.light import emission_time
from swarmsim.core.segments import SegmentStore, Vec3
from swarmsim.replay import Replay

GROUND = "#0E1A2B"
_INK, _MUTED, _ROCK, _GRID = "#E8EEF5", "#A9B7C6", "#7C8CA0", "#24344A"
_PATH_POINTS = 400
_GAP_POINTS = 800
_MM = 1e6  # meters per thousand km


@dataclass(frozen=True)
class _Style:
    color: str
    line: str
    marker: str


# Teams differ by line style and marker as well as color.
TEAM_STYLE = {"blue": _Style("#B8C7D9", "-", "o"), "red": _Style("#F2A33A", "--", "^")}


class _Trajectories:
    """The part of a SegmentStore that the light rules read, served from a replay."""

    def __init__(self, replay: Replay) -> None:
        self._units = replay.header["units"]
        self.state = replay.state

    def birth(self, unit: int) -> int:
        return int(self._units[unit]["birth"])

    def end_time(self, unit: int) -> int | None:
        end = self._units[unit]["end"]
        return None if end is None else int(end)


def _teams(replay: Replay) -> dict[str, list[int]]:
    """Unit ids by team in id order. A team's first unit is its outpost."""
    teams: dict[str, list[int]] = {"blue": [], "red": []}
    for u in replay.header["units"]:
        if u["team"] in teams:
            teams[u["team"]].append(u["id"])
    return teams


def separations(replay: Replay) -> list[tuple[str, int, int]]:
    """(watching team, its outpost, the enemy's first ship) for each side that has both."""
    teams = _teams(replay)
    pairs = (("blue", "red"), ("red", "blue"))
    return [(a, teams[a][0], teams[b][1]) for a, b in pairs if teams[a] and len(teams[b]) > 1]


def light_gap(
    replay: Replay, outpost: int, ship: int, times_us: Sequence[int]
) -> NDArray[np.float64]:
    """True minus light-delayed distance from ``outpost`` to ``ship`` at each time, in meters.

    The delayed distance is to where the ship was when it emitted the light arriving now.
    NaN where the ship does not exist or its light has not arrived. Rocks do not block.
    """
    paths = _Trajectories(replay)
    birth, end = paths.birth(ship), paths.end_time(ship)
    out = np.full(len(times_us), np.nan)
    for i, t in enumerate(times_us):
        if t < birth or (end is not None and t > end):
            continue
        here, _ = replay.state(outpost, t)
        seen = emission_time(cast(SegmentStore, paths), ship, t, here, replay.scenario.c)
        if seen is not None:
            true = float(np.linalg.norm(replay.state(ship, t)[0] - here))
            out[i] = true - float(np.linalg.norm(replay.state(ship, seen)[0] - here))
    return out


def _times(start: int, stop: int, n: int) -> list[int]:
    return [start + (stop - start) * k // (n - 1) for k in range(n)]


def _path(replay: Replay, unit: int, stop: int) -> NDArray[np.float64]:
    u = replay.header["units"][unit]
    last = stop if u["end"] is None else min(stop, u["end"])
    points: list[Vec3] = [
        replay.state(unit, t)[0] for t in _times(u["birth"], max(last, u["birth"]), _PATH_POINTS)
    ]
    return np.array(points) / _MM


def _dress(ax: Axes, title: str, note: str) -> None:
    ax.set_facecolor(GROUND)
    ax.grid(True, color=_GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.tick_params(colors=_MUTED, labelsize=14, length=0)
    for side, spine in ax.spines.items():
        spine.set_visible(side in ("left", "bottom"))
        spine.set_color(_GRID)
    ax.xaxis.label.set(color=_INK, fontsize=15)
    ax.yaxis.label.set(color=_INK, fontsize=15)
    ax.set_title(title, loc="left", color=_INK, fontsize=18, fontweight="bold", pad=32)
    ax.text(0.0, 1.02, note, transform=ax.transAxes, color=_MUTED, fontsize=14, va="bottom")


def _legend(ax: Axes, title: str = "") -> None:
    if ax.get_legend_handles_labels()[0]:
        ax.legend(
            loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0, fontsize=14,
            frameon=False, labelcolor=_INK, handlelength=2.6, labelspacing=0.9,
            title=title, title_fontsize=14, alignment="left",
        ).get_title().set_color(_MUTED)  # fmt: skip


def _tick(value: float, _: object) -> str:
    return f"{value:,.10g}".replace("-", "\N{MINUS SIGN}")


def _draw_paths(ax: Axes, replay: Replay, stop: int) -> None:
    rocks = replay.rocks.centers / _MM
    ax.scatter(rocks[:, 0], rocks[:, 1], s=12, color=_ROCK, linewidths=0, zorder=1, label="Rock")
    for team, ids in _teams(replay).items():
        style, name = TEAM_STYLE[team], team.capitalize()
        for i, unit in enumerate(ids[1:]):
            p = _path(replay, unit, stop)
            ax.plot(
                p[:, 0], p[:, 1], color=style.color, linestyle=style.line, linewidth=2,
                zorder=3, gid="path",
            )  # fmt: skip
            hit = replay.header["units"][unit]["end"] is not None
            ax.plot(
                p[-1:, 0], p[-1:, 1], color=style.color, linestyle="none",
                marker="X" if hit else style.marker, markersize=11, markeredgecolor=GROUND,
                markeredgewidth=1.5, zorder=5,
            )  # fmt: skip
            if i == 0:
                ax.plot(
                    [], [], color=style.color, linestyle=style.line, marker=style.marker,
                    linewidth=2, markersize=9, label=f"{name} ship path and end",
                )  # fmt: skip
        if ids:
            o = replay.state(ids[0], replay.header["units"][ids[0]]["birth"])[0] / _MM
            ax.plot(
                o[0], o[1], markerfacecolor="none", marker="s" if team == "blue" else "D",
                markersize=20, markeredgecolor=style.color, markeredgewidth=2,
                linestyle="none", zorder=4,
                label=f"{name} outpost",
            )  # fmt: skip
    if any(u["end"] is not None for u in replay.header["units"]):
        ax.plot(
            [], [], color=_INK, marker="X", linestyle="none", markersize=10, label="Ship destroyed"
        )  # fmt: skip
    ax.margins(x=0.03, y=0.08)
    ax.set_xlabel("x, thousand km")
    ax.set_ylabel("y, thousand km")
    ax.xaxis.set_major_formatter(FuncFormatter(_tick))


def _draw_gaps(ax: Axes, replay: Replay, stop: int) -> str:
    """Draw the lines; return a note saying how far apart two of them ever are."""
    times = _times(0, stop, _GAP_POINTS)
    hours = np.array(times) / 3.6e9
    units = replay.header["units"]
    pairs = separations(replay)
    gaps = []
    for team, outpost, ship in pairs:
        style = TEAM_STYLE[team]
        target = units[ship]
        gaps.append(light_gap(replay, outpost, ship, times) / 1e3)
        # The first line is wider, so it still shows where the second covers it.
        ax.plot(
            hours, gaps[-1], color=style.color, linestyle=style.line, gid="gap",
            linewidth=2 if len(gaps) == 2 else 4.5,
            label=f"{team.capitalize()} outpost watching\n"
            f"{target['team']} {target['cls']} {target['id']}",
        )  # fmt: skip
    if not pairs:
        ax.text(
            0.5, 0.5, "No enemy ship to watch in this replay", transform=ax.transAxes,
            ha="center", color=_INK, fontsize=16,
        )  # fmt: skip
    ax.set_xlim(0.0, hours[-1])
    ax.set_xlabel("time, hours")
    ax.set_ylabel("true minus light-delayed, km")
    if len(gaps) < 2 or np.isnan(gaps[0] - gaps[1]).all():
        return ""
    meters = np.nanmax(np.abs(gaps[0] - gaps[1])) * 1e3
    return f"Lines at most {meters:,.1f} m apart"


def build_figure(replay: Replay) -> Figure:
    """Both panels. Time runs to a little after the last event, or the whole run if none."""
    horizon = int(replay.header["horizon_us"])
    last = replay.events[-1].t_us if replay.events else horizon
    stop = min(horizon, last + last // 10)
    engine = ConstrainedLayoutEngine(h_pad=0.25, w_pad=0.2, hspace=0.04)
    fig = Figure(figsize=(16, 11), dpi=100, facecolor=GROUND, layout=engine)
    FigureCanvasAgg(fig)
    top, bottom = fig.subplots(2, 1)
    _draw_paths(top, replay, stop)
    apart = _draw_gaps(bottom, replay, stop)

    fig.canvas.draw()
    box = top.get_window_extent()
    (x0, x1), (y0, y1) = top.get_xlim(), top.get_ylim()
    stretch = (box.height / (y1 - y0)) / (box.width / (x1 - x0))
    scale = (
        f"Looking down z. y is stretched {stretch:.1f} times relative to x; "
        "rocks and markers are not to scale"
    )
    _dress(top, f"Paths from above, first {stop / 3.6e9:.1f} hours", scale)
    c = replay.scenario.c
    _dress(
        bottom,
        "True minus light-delayed separation, outpost to one enemy ship",
        f"Below zero: the ship is nearer than it looks. Light: {c:,.0f} m/s. Rocks do not block "
        "the view.",
    )
    _legend(top)
    _legend(bottom, apart)
    return fig


def plot(replay: Replay, out: str | Path) -> None:
    """Write the figure as a PNG. The same replay gives the same bytes."""
    buffer = io.BytesIO()
    build_figure(replay).savefig(buffer, format="png")
    Path(out).write_bytes(buffer.getvalue())
