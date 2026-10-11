"""Per-record transformations carried by the fitted credit-risk pipeline."""

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

NUMERIC_FEATURES = [
    "funded_amnt", "interest_rate", "annual_income", "dept_paym_income_ratio",
    "num_30+_delinq_in_2yrs", "num_inq_in_6mths", "num_open_credit_lines",
    "num_derogatory_pub_rec", "total_credit_revolving_bal", "used_credit_share",
    "tot_num_credit_lines", "loan_term_months", "emp_length",
]
CATEGORICAL_FEATURES = [
    "grade", "home_ownership_status", "verification_status", "loan_purpose",
]
POSITIVE_FEATURES = ["funded_amnt", "interest_rate", "annual_income"]
COUNT_FEATURES = [
    "num_30+_delinq_in_2yrs", "num_inq_in_6mths", "num_open_credit_lines",
    "num_derogatory_pub_rec", "tot_num_credit_lines",
]
LOG_FEATURES = ["annual_income", "total_credit_revolving_bal"]


def clean_raw_features(features: pd.DataFrame) -> pd.DataFrame:
    """Return an independent frame; no statistics are learned during cleaning."""
    cleaned = features[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    employment = cleaned["emp_length"].astype("string").str.strip()
    lengths = {"< 1 year": 0.5, "10+ years": 10, "1 year": 1,
               **{f"{year} years": year for year in range(2, 10)}}
    cleaned["emp_length"] = employment.map(lengths).fillna(
        pd.to_numeric(employment, errors="coerce"),
    ).astype(float)
    for column in CATEGORICAL_FEATURES:
        values = cleaned[column].astype("string").str.strip().replace("", pd.NA)
        cleaned[column] = values.to_numpy(dtype=object, na_value=np.nan)
    numeric = cleaned[NUMERIC_FEATURES].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan).astype(float)
    numeric = numeric.mask(numeric < 0)
    numeric[POSITIVE_FEATURES] = numeric[POSITIVE_FEATURES].mask(numeric[POSITIVE_FEATURES] <= 0)
    numeric[COUNT_FEATURES] = numeric[COUNT_FEATURES].mask(numeric[COUNT_FEATURES] % 1 != 0)
    numeric["loan_term_months"] = numeric["loan_term_months"].where(
        numeric["loan_term_months"].isin([36, 60]),
    )
    cleaned[NUMERIC_FEATURES] = numeric
    return cleaned


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("numeric", make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(),
        ), [column for column in NUMERIC_FEATURES if column not in LOG_FEATURES]),
        ("log", make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            FunctionTransformer(np.log1p), StandardScaler(),
        ), LOG_FEATURES),
        ("categorical", make_pipeline(
            SimpleImputer(strategy="constant", fill_value="Sin dato", keep_empty_features=True),
            OneHotEncoder(handle_unknown="ignore"),
        ), CATEGORICAL_FEATURES),
    ])
