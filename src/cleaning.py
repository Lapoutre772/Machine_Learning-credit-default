"""M1 - Chargement et nettoyage du dataset "Default of Credit Card Clients" (UCI, id=350).

Contrat d'interface avec les autres modules :
    load_raw()   -> DataFrame brut (colonnes X1..X23 + Y)
    clean(df)    -> DataFrame propre `df_clean` (colonnes nommees, cible `default`)
    CATEGORICAL_COLS, NUMERIC_COLS, TARGET : listes de colonnes utilisees par M3
"""
from pathlib import Path

import pandas as pd

# Le cache est place a la racine du repo (data/), quel que soit le dossier d'execution.
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_CACHE = DATA_DIR / "default_credit_raw.csv"

TARGET = "default"

# Noms d'origine (X1..X23) -> noms de la documentation UCI.
COLUMN_NAMES = {
    "X1": "LIMIT_BAL", "X2": "SEX", "X3": "EDUCATION", "X4": "MARRIAGE", "X5": "AGE",
    "X6": "PAY_0", "X7": "PAY_2", "X8": "PAY_3", "X9": "PAY_4", "X10": "PAY_5", "X11": "PAY_6",
    "X12": "BILL_AMT1", "X13": "BILL_AMT2", "X14": "BILL_AMT3",
    "X15": "BILL_AMT4", "X16": "BILL_AMT5", "X17": "BILL_AMT6",
    "X18": "PAY_AMT1", "X19": "PAY_AMT2", "X20": "PAY_AMT3",
    "X21": "PAY_AMT4", "X22": "PAY_AMT5", "X23": "PAY_AMT6",
}

# Variables qui seront encodees en one-hot par M3 ; toutes les autres sont numeriques.
CATEGORICAL_COLS = ["SEX", "EDUCATION", "MARRIAGE"]
NUMERIC_COLS = [c for c in COLUMN_NAMES.values() if c not in CATEGORICAL_COLS]


def load_raw(use_cache: bool = True) -> pd.DataFrame:
    """Charge le dataset brut depuis le cache local, sinon le telecharge via ucimlrepo."""
    if use_cache and RAW_CACHE.exists():
        return pd.read_csv(RAW_CACHE)
    from ucimlrepo import fetch_ucirepo  # import local : inutile si le cache existe

    dataset = fetch_ucirepo(id=350)
    df = dataset.data.features.join(dataset.data.targets)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_CACHE, index=False)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Renomme les colonnes et regroupe les codes non documentes.

    Toutes les operations sont des correspondances fixes, sans rien apprendre des
    donnees : elles peuvent donc etre faites avant le split sans fuite de donnees.
    """
    df = df.rename(columns={**COLUMN_NAMES, "Y": TARGET}).copy()

    # EDUCATION : 1=postgraduate, 2=university, 3=high school, 4=other.
    # Les codes 0, 5, 6 ne sont pas documentes -> regroupes dans "other" (4).
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})

    # MARRIAGE : 1=married, 2=single, 3=other. Le code 0 n'est pas documente -> 3.
    df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})
        # Doublons : 35 lignes identiques (0,12 %) sont conservees. Ce sont des comptes
    # inactifs (factures et paiements nuls, PAY_x = -2), donc des profils sans
    # variation : sans colonne ID, rien ne prouve que ce soit un meme client.

    return df


def validate(df: pd.DataFrame) -> None:
    """Controles de coherence : leve une AssertionError si quelque chose cloche."""
    assert df.shape == (30000, 24), f"forme inattendue : {df.shape}"
    assert not df.isna().any().any(), "valeurs manquantes inattendues"
    assert set(df[TARGET].unique()) == {0, 1}, "la cible doit etre binaire"
    assert set(df["SEX"].unique()) <= {1, 2}, "SEX hors {1, 2}"
    assert set(df["EDUCATION"].unique()) <= {1, 2, 3, 4}, "EDUCATION hors {1..4}"
    assert set(df["MARRIAGE"].unique()) <= {1, 2, 3}, "MARRIAGE hors {1..3}"
    assert abs(df[TARGET].mean() - 0.2212) < 0.001, "taux de defaut different de 22.12 %"