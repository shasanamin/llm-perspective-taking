from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.models import MODEL_DISPLAY_NAMES


STAT_DISPLAY_NAMES = {
    "mse": "Error",
    "bias": "Bias",
    "var": "Variance",
}


DEFAULT_FIGSIZE = (11.69 * 0.6, 8.27 * 0.5)


RCPARAMS_TICKS = {
    "figure.dpi": 300,
    "xtick.bottom": True,
    "ytick.left": True,
    "legend.loc": "best",
    "legend.fancybox": True,
    "axes.edgecolor": "grey",
    "axes.xmargin": 0.05,
    "text.usetex": False,
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "lines.linewidth": 2.5,
    "lines.markersize": 8,
    "font.size": 12,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.grid": True,
    "grid.linestyle": "--",
    "axes.grid.axis": "y",
    "grid.alpha": 0.3,
    "grid.color": "gray",
}


MAIN_PLOT_SETTINGS = {
    "direct": {"color": "#F2C500", "linestyle": "-", "marker": ".", "hatch": None},
    "perspective": {"color": "#00A087", "linestyle": ":", "marker": "o", "hatch": "-"},
    "perspective_out": {"color": "#91D1C2", "linestyle": ":", "marker": ".", "hatch": "|"},
    "gpt-5.1": {"color": "#E64B35", "linestyle": "--", "marker": "^", "hatch": "xx"},
    "gpt-5.4": {"color": "#0E8A8A", "linestyle": "--", "marker": "D", "hatch": None},
    "gpt-5.4_reasoning=high": {"color": "#0E8A8A", "linestyle": "-.", "marker": "D", "hatch": "//"},
    "gpt-5.4-mini": {"color": "#4C78A8", "linestyle": "--", "marker": "s", "hatch": None},
    "gpt-5.4-mini_reasoning=high": {"color": "#4C78A8", "linestyle": "-.", "marker": "s", "hatch": "//"},
    "gpt-5.4-nano": {"color": "#B65F15", "linestyle": "--", "marker": "X", "hatch": None},
    "gpt-5.4-nano_reasoning=low": {"color": "#D89032", "linestyle": "-.", "marker": "o", "hatch": "//"},
    "gpt-5.4-nano_reasoning=medium": {"color": "#E64B35", "linestyle": "-.", "marker": "^", "hatch": "//"},
    "gpt-5.4-nano_reasoning=high": {"color": "#C44E52", "linestyle": "-.", "marker": "s", "hatch": "//"},
    "gpt-5.4-nano_reasoning=xhigh": {"color": "#7A1F5C", "linestyle": "-.", "marker": "P", "hatch": "//"},
    "gpt-oss:20b": {"color": "#4DBBD5", "linestyle": "--", "marker": "<", "hatch": "//"},
    "gpt-oss:120b": {"color": "#3C5488", "linestyle": "--", "marker": ">", "hatch": ".."},
    "qwen3:1.7b": {"color": "#4DBBD5", "linestyle": "--", "marker": "o", "hatch": None},
    "qwen3:8b": {"color": "#3C5488", "linestyle": "--", "marker": "s", "hatch": None},
    "qwen3:32b": {"color": "#00A087", "linestyle": "--", "marker": "D", "hatch": None},
    "qwen3-r:1.7b": {"color": "#E64B35", "linestyle": "-.", "marker": "^", "hatch": "//"},
    "qwen3-r:8b": {"color": "#F39B7F", "linestyle": "-.", "marker": "v", "hatch": "//"},
    "qwen3-r:32b": {"color": "#DC0000", "linestyle": "-.", "marker": "P", "hatch": "//"},
}


MODEL_FAMILIES = {
    "DeepSeek-R1": ["deepseek-r1:1.5b", "deepseek-r1:7b", "deepseek-r1:32b"],
    "Gemma3": ["gemma3:1b", "gemma3:12b", "gemma3:27b"],
    "GPT": ["gpt-oss:20b", "gpt-oss:120b", "gpt-5.1", "gpt-5.4", "gpt-5.4-mini", "gpt-5.4-nano"],
    "GPT-R": [
        "gpt-5.1_reasoning=low",
        "gpt-5.1_reasoning=medium",
        "gpt-5.1_reasoning=high",
        "gpt-5.4_reasoning=high",
        "gpt-5.4-mini_reasoning=high",
        "gpt-5.4-nano_reasoning=low",
        "gpt-5.4-nano_reasoning=medium",
        "gpt-5.4-nano_reasoning=high",
        "gpt-5.4-nano_reasoning=xhigh",
    ],
    "Qwen3": ["qwen3:1.7b", "qwen3:8b", "qwen3:32b"],
    "Qwen3-R": ["qwen3-r:1.7b", "qwen3-r:8b", "qwen3-r:32b"],
}


MODEL_MIXTURE_FAMILY = {
    "small": ["qwen3:1.7b", "gemma3:1b", "deepseek-r1:1.5b"],
    "mid": ["qwen3:8b", "gemma3:12b", "deepseek-r1:7b"],
    "large": ["qwen3:32b", "gemma3:27b", "deepseek-r1:32b"],
}


MODEL_MIXTURE_SIZE = {
    "gpt": ["gpt-oss:20b", "gpt-oss:120b", "gpt-5.1", "gpt-5.4", "gpt-5.4-mini", "gpt-5.4-nano"],
    "gpt_r": [
        "gpt-5.1_reasoning=low",
        "gpt-5.1_reasoning=medium",
        "gpt-5.1_reasoning=high",
        "gpt-5.4_reasoning=high",
        "gpt-5.4-mini_reasoning=high",
        "gpt-5.4-nano_reasoning=low",
        "gpt-5.4-nano_reasoning=medium",
        "gpt-5.4-nano_reasoning=high",
        "gpt-5.4-nano_reasoning=xhigh",
    ],
    "qwen3-r": ["qwen3-r:1.7b", "qwen3-r:8b", "qwen3-r:32b"],
    "qwen3": ["qwen3:1.7b", "qwen3:8b", "qwen3:32b"],
    "gemma3": ["gemma3:1b", "gemma3:12b", "gemma3:27b"],
    "deepseek": ["deepseek-r1:1.5b", "deepseek-r1:7b", "deepseek-r1:32b"],
}


NATURE_COLORS = [
    "#F2C500",
    "#4DBBD5",
    "#00A087",
    "#E64B35",
    "#3C5488",
    "#8491B4",
    "#91D1C2",
    "#DC0000",
    "#F39B7F",
    "#FFB81C",
    "#E6AB02",
    "#A1CAF1",
]


CUSTOM_MARKERS = ["*", "o", ".", "^", "<", ">"]


def display_name(model_name: str) -> str:
    return MODEL_DISPLAY_NAMES.get(model_name, model_name)


def set_plot_theme() -> None:
    sns.set_theme(style="ticks", rc=RCPARAMS_TICKS)


def save_figure(output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", dpi=300)
    plt.close()
