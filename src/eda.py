"""M2 - Analyse exploratoire du dataset (sur df_clean, produit par M1).

Lancer depuis la racine du repo :  python -m src.eda
Les figures sont enregistrees dans report/figures/ et les tableaux affiches dans le terminal.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # pas de fenetre : on enregistre les figures dans des fichiers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import cleaning as c

FIG_DIR = Path(__file__).resolve().parent.parent / "report" / "figures"
BILL_COLS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAY_AMT_COLS = [f"PAY_AMT{i}" for i in range(1, 7)]


# ---------- Calculs (retournent des tableaux, reutilisables dans le rapport) ----------

def target_balance(df: pd.DataFrame) -> pd.DataFrame:
    """Effectifs et proportions de chaque classe de la cible."""
    counts = df[c.TARGET].value_counts().sort_index()
    return pd.DataFrame({"n": counts, "share": (counts / len(df)).round(4)})


def default_rate_by(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Taux de defaut et effectif pour chaque valeur de `col`."""
    g = df.groupby(col)[c.TARGET].agg(n="size", default_rate="mean")
    g["default_rate"] = g["default_rate"].round(4)
    return g


def correlation_with_target(df: pd.DataFrame) -> pd.Series:
    """Correlation de Pearson de chaque variable avec la cible, triee par valeur absolue."""
    corr = df.drop(columns=c.TARGET).corrwith(df[c.TARGET])
    return corr.reindex(corr.abs().sort_values(ascending=False).index).round(3)


def negative_counts(df: pd.DataFrame) -> pd.Series:
    """Nombre de valeurs negatives par colonne de factures et de paiements."""
    return (df[BILL_COLS + PAY_AMT_COLS] < 0).sum()


def skewness(df: pd.DataFrame) -> pd.Series:
    """Asymetrie des variables numeriques continues (grande valeur = queue lourde a droite)."""
    cols = ["LIMIT_BAL", "AGE"] + BILL_COLS + PAY_AMT_COLS
    return df[cols].skew().round(2).sort_values(ascending=False)


# ---------- Figures ----------

def _save(fig, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=150)
    plt.close(fig)


def plot_target_balance(df: pd.DataFrame) -> None:
    tb = target_balance(df)
    fig, ax = plt.subplots(figsize=(4, 3.5))
    ax.bar(["No default (0)", "Default (1)"], tb["share"] * 100)
    for i, share in enumerate(tb["share"] * 100):
        ax.text(i, share + 1, f"{share:.1f}%", ha="center")
    ax.set_ylim(0, 90)
    ax.set_ylabel("% of clients")
    ax.set_title("Target distribution")
    _save(fig, "target_balance.png")


def plot_default_rate_by_pay0(df: pd.DataFrame) -> None:
    g = default_rate_by(df, "PAY_0")
    rates = g["default_rate"] * 100
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.bar(range(len(g)), rates)
    ax.set_xticks(range(len(g)), g.index.astype(str))
    for i, (rate, n) in enumerate(zip(rates, g["n"])):
        ax.text(i, rate + 1.5, f"n={n}", ha="center", fontsize=7)  # effectif : taux fragile si n petit
    ax.axhline(df[c.TARGET].mean() * 100, color="grey", linestyle="--", label="overall rate")
    ax.set_ylim(0, 100)
    ax.set_xlabel("PAY_0 (repayment status, September)")
    ax.set_ylabel("Default rate (%)")
    ax.set_title("Default rate by last repayment status")
    ax.legend(loc="upper left")
    _save(fig, "default_rate_by_pay0.png")


def plot_default_rate_categorical(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), sharey=True)
    for ax, col in zip(axes, c.CATEGORICAL_COLS):
        g = default_rate_by(df, col)
        labels = [f"{code}\n(n={n})" for code, n in zip(g.index, g["n"])]
        ax.bar(range(len(g)), g["default_rate"] * 100)
        ax.set_xticks(range(len(g)), labels)
        ax.axhline(df[c.TARGET].mean() * 100, color="grey", linestyle="--")
        ax.set_title(col)
    axes[0].set_ylabel("Default rate (%)")
    _save(fig, "default_rate_categorical.png")


def plot_bill_correlation(df: pd.DataFrame) -> None:
    corr = df[BILL_COLS].corr()
    fig, ax = plt.subplots(figsize=(5, 4.2))
    im = ax.imshow(corr, vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(BILL_COLS)), BILL_COLS, rotation=45, ha="right")
    ax.set_yticks(range(len(BILL_COLS)), BILL_COLS)
    for i in range(len(BILL_COLS)):
        for j in range(len(BILL_COLS)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", color="white", fontsize=7)
    fig.colorbar(im, ax=ax)
    ax.set_title("Correlation between monthly bills")
    _save(fig, "bill_correlation.png")


def plot_limit_bal(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    ax.boxplot([df.loc[df[c.TARGET] == k, "LIMIT_BAL"] for k in (0, 1)],
               tick_labels=["No default", "Default"])
    ax.set_yscale("log")
    ax.set_ylabel("LIMIT_BAL (log scale)")
    ax.set_title("Credit limit by outcome")
    _save(fig, "limit_bal_by_target.png")


# ---------- Lancement ----------

def run_all(df: pd.DataFrame) -> None:
    pd.set_option("display.width", 120)
    print("== Target balance ==\n", target_balance(df), "\n")
    print("== Missing values (total) ==", int(df.isna().sum().sum()), "\n")
    for col in c.CATEGORICAL_COLS + ["PAY_0"]:
        print(f"== Default rate by {col} ==\n", default_rate_by(df, col), "\n")
    print("== Correlation with target ==\n", correlation_with_target(df).head(10), "\n")
    print("== Negative values per column ==\n", negative_counts(df)[lambda s: s > 0], "\n")
    print("== Skewness ==\n", skewness(df).head(8), "\n")
    print("== Mean correlation between BILL_AMT columns ==",
          round(df[BILL_COLS].corr().values[~np.eye(6, dtype=bool)].mean(), 3), "\n")
    plot_target_balance(df)
    plot_default_rate_by_pay0(df)
    plot_default_rate_categorical(df)
    plot_bill_correlation(df)
    plot_limit_bal(df)
    print("Figures saved in", FIG_DIR)


if __name__ == "__main__":
    data = c.clean(c.load_raw())
    c.validate(data)
    run_all(data)