"""Gráficas del experimento (PNG) con una paleta validada para daltonismo."""

from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": BASELINE, "axes.labelcolor": INK_SECONDARY,
    "axes.titlecolor": INK, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.spines.top": False,
    "axes.spines.right": False, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "font.size": 10, "legend.frameon": False,
})


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def price_distribution(prices: pd.Series, median: float, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 3.6))
    bins = np.logspace(np.log10(prices.min()), np.log10(prices.max()), 45)
    ax.hist(prices[prices <= median], bins=bins, color=SERIES[0], label="estándar (≤ mediana)")
    ax.hist(prices[prices > median], bins=bins, color=SERIES[1], label="alto (> mediana)")
    ax.axvline(median, color=INK_SECONDARY, linewidth=1, linestyle="--")
    ax.annotate(f"mediana de\nentrenamiento:\n€{median:,.0f}", (median, ax.get_ylim()[1] * 0.80),
                xytext=(-8, 0), textcoords="offset points", color=INK, ha="right")
    ax.set_xscale("log")
    ax.set_xlabel("Precio total de la estancia (€, escala log)")
    ax.set_ylabel("Anuncios")
    ax.set_title("Precios de Airbnb en Londres (fin de semana)")
    ax.legend(loc="upper right")
    return _save(fig, path)


def algorithm_comparison(comparison: pd.DataFrame, chosen: str, path: Path) -> Path:
    data = comparison.sort_values("auc_cv")
    fig, ax = plt.subplots(figsize=(7, 3.4))
    colors = [SERIES[0] if a == chosen else BASELINE for a in data["algoritmo"]]
    ax.barh(data["algoritmo"], data["auc_cv"], color=colors, height=0.6)
    for y, value in enumerate(data["auc_cv"]):
        ax.annotate(f"{value:.3f}", (value, y), xytext=(4, 0), textcoords="offset points",
                    va="center", color=INK)
    ax.set_xlim(0.5, 1.0)
    ax.set_xlabel("AUC ROC (validación cruzada, 5 pliegues)")
    ax.set_title("Comparación de algoritmos candidatos")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", colors=INK)
    return _save(fig, path)


def hyperparameter_search(grid: pd.DataFrame, best: Dict[str, object], path: Path) -> Path:
    part = grid[(grid["n_estimators"] == str(best["n_estimators"]))
                & (grid["max_features"] == str(best["max_features"]))]
    order = ["4", "8", "12", "None"]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for color, leaf in zip(SERIES, sorted(part["min_samples_leaf"].unique(), key=int)):
        rows = part[part["min_samples_leaf"] == leaf].set_index("max_depth").reindex(order)
        ax.plot(range(len(order)), rows["score_cv"], color=color, linewidth=2,
                marker="o", markersize=6, label=f"min_samples_leaf = {leaf}")
    best_x = order.index(str(best["max_depth"]))
    ax.annotate(f"elegido: AUC {best['score_cv']:.3f}", (best_x, best["score_cv"]),
                xytext=(-150, 14), textcoords="offset points", color=INK,
                arrowprops={"arrowstyle": "-", "color": MUTED})
    low, high = ax.get_ylim()
    ax.set_ylim(low, high + (high - low) * 0.15)
    ax.set_xticks(range(len(order)), ["4", "8", "12", "sin límite"])
    ax.set_xlabel("max_depth (profundidad máxima de cada árbol)")
    ax.set_ylabel("AUC ROC (validación cruzada)")
    ax.set_title(f"Búsqueda de hiperparámetros (n_estimators={best['n_estimators']}, "
                 f"max_features={best['max_features']})")
    ax.legend(loc="lower right")
    return _save(fig, path)


def confusion_and_roc(metrics: Dict[str, object], path: Path) -> Path:
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4.2))
    matrix = metrics["confusion_matrix"].values
    names = [c.replace("real_", "") for c in metrics["confusion_matrix"].index]
    left.imshow(matrix, cmap=LinearSegmentedColormap.from_list("seq", SEQUENTIAL), vmin=0)
    for i in range(2):
        for j in range(2):
            left.text(j, i, matrix[i, j], ha="center", va="center", fontsize=13,
                      color="#ffffff" if matrix[i, j] > matrix.max() * 0.5 else INK)
    left.set_xticks([0, 1], names)
    left.set_yticks([0, 1], names)
    left.set_xlabel("Predicción")
    left.set_ylabel("Precio real")
    left.set_title("Matriz de confusión (prueba)")
    left.grid(False)
    for spine in left.spines.values():
        spine.set_visible(False)
    roc = metrics["roc"]
    right.plot(roc["fpr"], roc["tpr"], color=SERIES[0], linewidth=2,
               label=f"Random Forest (AUC {metrics['roc_auc']:.3f})")
    right.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle="--", label="azar (AUC 0.5)")
    right.set_xlabel("Tasa de falsos positivos")
    right.set_ylabel("Tasa de verdaderos positivos")
    right.set_title("Curva ROC (prueba)")
    right.legend(loc="lower right")
    fig.tight_layout()
    return _save(fig, path)


def feature_importance(importance: pd.DataFrame, labels: Dict[str, str], path: Path) -> Path:
    data = importance.iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.barh([labels.get(v, v) for v in data["variable"]], data["importancia"],
            xerr=data["desviacion"], color=SERIES[0], height=0.6,
            error_kw={"ecolor": MUTED, "elinewidth": 1})
    ax.set_xlabel("Caída del AUC al desordenar la variable (prueba)")
    ax.set_title("Importancia de cada variable (por permutación)")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", colors=INK)
    return _save(fig, path)


def partial_dependence(dependence: Dict[str, pd.DataFrame], labels: Dict[str, str],
                       path: Path) -> Path:
    n = len(dependence)
    cols = 2
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(9, 3.2 * rows), sharey=True)
    for ax, (feature, data) in zip(np.ravel(axes), dependence.items()):
        ax.plot(data["valor"], data["probabilidad"], color=SERIES[0], linewidth=2)
        ax.set_title(labels.get(feature, feature), fontsize=11)
        ax.set_ylim(0, 1)
        ax.axhline(0.5, color=MUTED, linewidth=1, linestyle="--")
    for ax in np.ravel(axes)[n:]:
        ax.set_visible(False)
    for ax in np.ravel(axes)[::cols]:
        ax.set_ylabel("P(precio alto)")
    fig.suptitle("Efecto de cada variable en la probabilidad de precio alto",
                 x=0.01, ha="left", fontweight="bold", color=INK)
    fig.tight_layout()
    return _save(fig, path)


def category_effects(effects: Dict[str, pd.Series], labels: Dict[str, str],
                     path: Path) -> Path:
    fig, axes = plt.subplots(1, len(effects), figsize=(5 * len(effects), 3.2))
    for ax, (feature, series) in zip(np.atleast_1d(axes), effects.items()):
        data = series.iloc[::-1]
        ax.barh(data.index, data.values, color=SERIES[0], height=0.55)
        for y, value in enumerate(data.values):
            ax.annotate(f"{value:.0%}", (value, y), xytext=(4, 0),
                        textcoords="offset points", va="center", color=INK)
        ax.set_xlim(0, 1.08)
        ax.set_title(labels.get(feature, feature))
        ax.set_xlabel("P(precio alto) promedio")
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", colors=INK)
    fig.tight_layout()
    return _save(fig, path)
