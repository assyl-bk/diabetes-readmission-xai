"""Interactive training (Modules 2 and 3): split, preprocessing, models, optional hyperparameter search."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from core.metrics import comparison_table

MODEL_NAMES = ["Logistic Regression", "Decision Tree", "kNN", "Random Forest", "XGBoost"]

# Hyperparameters exposed in the interface: model -> list of controls
UI_PARAMS = {
    "Logistic Regression": [dict(key="C", label="C (inverse of regularisation)", kind="choice", options=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0], default=0.1)],
    "Decision Tree": [dict(key="max_depth", label="max_depth", kind="int", min=2, max=20, default=6, step=1),
                      dict(key="min_samples_leaf", label="min_samples_leaf", kind="int", min=5, max=300, default=100, step=5)],
    "kNN": [dict(key="n_neighbors", label="n_neighbors", kind="int", min=5, max=150, default=50, step=5)],
    "Random Forest": [dict(key="n_estimators", label="n_estimators", kind="int", min=50, max=400, default=150, step=50),
                      dict(key="max_depth", label="max_depth", kind="int", min=3, max=25, default=12, step=1),
                      dict(key="min_samples_leaf", label="min_samples_leaf", kind="int", min=1, max=60, default=20, step=1)],
    "XGBoost": [dict(key="n_estimators", label="n_estimators", kind="int", min=50, max=600, default=250, step=50),
                dict(key="max_depth", label="max_depth", kind="int", min=2, max=10, default=4, step=1),
                dict(key="learning_rate", label="learning_rate", kind="choice", options=[0.01, 0.03, 0.05, 0.1, 0.2, 0.3], default=0.05)],
}

# Random-search ranges (used when "optimise hyperparameters" is ticked)
SEARCH_SPACES = {
    "Logistic Regression": {"clf__C": loguniform(1e-3, 1e2)},
    "Decision Tree": {"clf__max_depth": randint(3, 12), "clf__min_samples_leaf": randint(20, 200)},
    "kNN": {"clf__n_neighbors": randint(15, 100)},
    "Random Forest": {"clf__max_depth": randint(5, 20), "clf__min_samples_leaf": randint(5, 50)},
    "XGBoost": {"clf__max_depth": randint(2, 7), "clf__learning_rate": uniform(0.02, 0.18),
                "clf__min_child_weight": randint(1, 20)},
}


def make_estimator(name: str, params: dict, use_class_weight: bool, pos_weight: float, seed: int):
    cw = "balanced" if use_class_weight else None
    if name == "Logistic Regression":
        return LogisticRegression(max_iter=2000, class_weight=cw, **params)
    if name == "Decision Tree":
        return DecisionTreeClassifier(class_weight=cw, random_state=seed, **params)
    if name == "kNN":
        return KNeighborsClassifier(n_jobs=-1, **params)
    if name == "Random Forest":
        return RandomForestClassifier(class_weight=cw, n_jobs=-1, random_state=seed, **params)
    if name == "XGBoost":
        return XGBClassifier(tree_method="hist", eval_metric="logloss", n_jobs=-1, random_state=seed,
                             scale_pos_weight=pos_weight if use_class_weight else 1.0, **params)
    raise ValueError(name)


def build_preprocessor(num_cols, cat_cols, scale: bool = True) -> ColumnTransformer:
    steps = [("imputer", SimpleImputer(strategy="median"))] + ([("scaler", StandardScaler())] if scale else [])
    numeric = Pipeline(steps)
    categorical = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    pre = ColumnTransformer([("num", numeric, num_cols), ("cat", categorical, cat_cols)])
    pre.set_output(transform="pandas")
    return pre


def split_data(df: pd.DataFrame, y: pd.Series, group: str | None, test_size: float, seed: int):
    """Split by patient when a group column exists (no patient in both sets), otherwise stratified."""
    if group is not None:
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        tr, te = next(gss.split(df, y, groups=df[group]))
        return df.iloc[tr], df.iloc[te], y.iloc[tr], y.iloc[te]
    return train_test_split(df, y, test_size=test_size, random_state=seed, stratify=y)


def train_and_compare(df, target, positive, group, features, selected, params, test_size=0.2,
                      use_class_weight=True, sample_size=None, tune=False, n_iter=8, scale=True, seed=42,
                      progress=None) -> dict:
    """Train the selected models and return the comparison table + test probabilities."""
    df = df.copy()
    y = (df[target].astype(str) == str(positive)).astype(int)
    if sample_size and sample_size < len(df):
        keep = df.sample(sample_size, random_state=seed).index
        df, y = df.loc[keep], y.loc[keep]
    if y.nunique() < 2:
        raise ValueError("The target has a single class after binarisation: choose another positive value.")

    X = df[features]
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in features if c not in num_cols]
    X_tr, X_te, y_tr, y_te = split_data(df, y, group, test_size, seed)
    g_tr = X_tr[group] if group is not None else None
    X_tr, X_te = X_tr[features], X_te[features]
    pos_weight = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))

    probas, times, fitted, chosen = {}, {}, {}, {}
    for i, name in enumerate(selected):
        if progress:
            progress(i / len(selected), f"Training {name}...")
        est = make_estimator(name, params.get(name, {}), use_class_weight, pos_weight, seed)
        pipe = Pipeline([("pre", build_preprocessor(num_cols, cat_cols, scale)), ("clf", est)])
        t0 = time.time()
        if tune:
            cv = GroupKFold(n_splits=3) if group is not None else StratifiedKFold(3, shuffle=True, random_state=seed)
            search = RandomizedSearchCV(pipe, SEARCH_SPACES[name], n_iter=n_iter, scoring="average_precision",
                                        cv=cv, random_state=seed, refit=True, n_jobs=1)
            search.fit(X_tr, y_tr, groups=g_tr) if group is not None else search.fit(X_tr, y_tr)
            pipe = search.best_estimator_
            chosen[name] = {k.replace("clf__", ""): (v.item() if hasattr(v, "item") else v) for k, v in search.best_params_.items()}
        else:
            pipe.fit(X_tr, y_tr)
            chosen[name] = params.get(name, {})
        times[name] = time.time() - t0
        fitted[name] = pipe
        probas[name] = pipe.predict_proba(X_te)[:, 1]
    if progress:
        progress(1.0, "Done")

    table = comparison_table(y_te, probas)
    table["Fit time (s)"] = pd.Series(times)
    return {"y_test": y_te.to_numpy(), "probas": probas, "table": table, "params": chosen,
            "n_train": len(X_tr), "n_test": len(X_te), "positive_rate_test": float(y_te.mean())}
