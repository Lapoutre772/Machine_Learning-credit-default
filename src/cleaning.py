"""M1 - Loading and cleaning the "Default of Credit Card Clients" dataset (UCI, id=350).

Interface used by the other modules:
    load_raw()   -> raw DataFrame (columns X1..X23 + Y)
    clean(df)    -> clean DataFrame `df_clean` (named columns, target `default`)
    CATEGORICAL_COLS, NUMERIC_COLS, TARGET -> column lists used by M3
"""
from pathlib import Path

import pandas as pd

# The cache lives at the repo root (data/), wherever the script is run from.
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_CACHE = DATA_DIR / "default_credit_raw.csv"

TARGET = "default"

# Original names (X1..X23) -> names from the UCI documentation.
COLUMN_NAMES = {
    "X1": "LIMIT_BAL", "X2": "SEX", "X3": "EDUCATION", "X4": "MARRIAGE", "X5": "AGE",
    "X6": "PAY_0", "X7": "PAY_2", "X8": "PAY_3", "X9": "PAY_4", "X10": "PAY_5", "X11": "PAY_6",
    "X12": "BILL_AMT1", "X13": "BILL_AMT2", "X14": "BILL_AMT3",
    "X15": "BILL_AMT4", "X16": "BILL_AMT5", "X17": "BILL_AMT6",
    "X18": "PAY_AMT1", "X19": "PAY_AMT2", "X20": "PAY_AMT3",
    "X21": "PAY_AMT4", "X22": "PAY_AMT5", "X23": "PAY_AMT6",
}

# M3 will one-hot encode these three columns; every other feature is numeric.
CATEGORICAL_COLS = ["SEX", "EDUCATION", "MARRIAGE"]
NUMERIC_COLS = [c for c in COLUMN_NAMES.values() if c not in CATEGORICAL_COLS]


def load_raw(use_cache: bool = True) -> pd.DataFrame:
    """Load the raw dataset from the local cache, or download it with ucimlrepo."""
    if use_cache and RAW_CACHE.exists():
        return pd.read_csv(RAW_CACHE)
    # Imported here so the package is only needed on the first run.
    from ucimlrepo import fetch_ucirepo

    dataset = fetch_ucirepo(id=350)
    df = dataset.data.features.join(dataset.data.targets)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_CACHE, index=False)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Rename the columns and merge the undocumented category codes.

    Every step here is a fixed mapping that learns nothing from the data, so it is
    safe to run before the train/test split (no data leakage).
    """
    df = df.rename(columns={**COLUMN_NAMES, "Y": TARGET}).copy()

    # EDUCATION: 1 = graduate school, 2 = university, 3 = high school, 4 = others.
    # Codes 0, 5 and 6 are not in the UCI documentation. Codes 0 and 5 have default
    # rates close to code 4 (0.0% and 6.4% vs 5.7%), so we merge them into 4.
    # Code 6 is higher (15.7%) but has only 51 clients, too few to keep on its own.
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})

    # MARRIAGE: 1 = married, 2 = single, 3 = others. Code 0 is undocumented,
    # so we treat it as "others".
    df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})

    # The data has 35 duplicated rows (0.12%). We keep them: there is no customer ID,
    # so they are most likely different clients with identical values (such as
    # inactive accounts), and dropping them would be an arbitrary choice.
    return df


def validate(df: pd.DataFrame) -> None:
    """Sanity checks on the cleaned data. Raises an AssertionError if something is off."""
    assert df.shape == (30000, 24), f"unexpected shape: {df.shape}"
    assert not df.isna().any().any(), "unexpected missing values"
    assert set(df[TARGET].unique()) == {0, 1}, "the target must be binary"
    assert set(df["SEX"].unique()) <= {1, 2}, "SEX outside {1, 2}"
    assert set(df["EDUCATION"].unique()) <= {1, 2, 3, 4}, "EDUCATION outside {1..4}"
    assert set(df["MARRIAGE"].unique()) <= {1, 2, 3}, "MARRIAGE outside {1..3}"
    assert abs(df[TARGET].mean() - 0.2212) < 0.001, "default rate differs from 22.12%"