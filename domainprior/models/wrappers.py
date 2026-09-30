"""One sklearn-style interface per baseline. Missing opt deps -> skipped jobs."""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd


def _prep(X_train, X_test):
    """Median-impute numerics + one-hot categoricals. Fit on train only."""
    Xt, Xe = X_train.copy(), X_test.copy()
    for df in (Xt, Xe):
        for c in df.columns:
            if pd.api.types.is_numeric_dtype(df[c]):
                continue
            df[c] = df[c].astype("category")
    num = Xt.select_dtypes(include=[np.number]).columns
    med = Xt[num].median()
    Xt[num] = Xt[num].fillna(med); Xe[num] = Xe[num].fillna(med)
    Xt = pd.get_dummies(Xt, drop_first=True, dtype="float32")
    Xe = pd.get_dummies(Xe, drop_first=True, dtype="float32")
    Xe = Xe.reindex(columns=Xt.columns, fill_value=0.0)
    return Xt.to_numpy(dtype="float32"), Xe.to_numpy(dtype="float32")


def fit_predict(model_key, X_train, y_train, X_test, seed=0, ctx_cap=2048):
    yt = np.asarray(y_train).astype(int)
    if len(np.unique(yt)) < 2:
        raise ValueError("degenerate train split: single class")
    if model_key == "woe_logreg":
        from sklearn.linear_model import LogisticRegression
        Xtr, Xte = _prep(X_train, X_test)
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(Xtr, yt)
        return clf.predict_proba(Xte)[:, 1]
    if model_key == "hgb":
        from sklearn.ensemble import HistGradientBoostingClassifier
        Xtr, Xte = _prep(X_train, X_test)
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06,
                                             early_stopping=True, random_state=seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(Xtr, yt)
        return clf.predict_proba(Xte)[:, 1]
    if model_key == "lightgbm":
        import lightgbm as lgb
        Xtr, Xte = _prep(X_train, X_test)
        clf = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05,
                                 verbose=-1, random_state=seed)
        clf.fit(Xtr, yt)
        return clf.predict_proba(Xte)[:, 1]
    if model_key == "catboost":
        from catboost import CatBoostClassifier
        Xtr, Xte = _prep(X_train, X_test)
        clf = CatBoostClassifier(depth=6, learning_rate=0.05, iterations=500,
                                 verbose=False, random_seed=seed)
        clf.fit(Xtr, yt)
        return clf.predict_proba(Xte)[:, 1]
    if model_key == "xgboost":
        from xgboost import XGBClassifier
        Xtr, Xte = _prep(X_train, X_test)
        clf = XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=6,
                            subsample=0.8, colsample_bytree=0.8,
                            eval_metric="logloss", random_state=seed)
        clf.fit(Xtr, yt)
        return clf.predict_proba(Xte)[:, 1]
    if model_key == "realmlp":
        from sklearn.neural_network import MLPClassifier
        Xtr, Xte = _prep(X_train, X_test)
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        clf = MLPClassifier(hidden_layer_sizes=(256, 128), activation="relu",
                            early_stopping=True, n_iter_no_change=10,
                            max_iter=200, random_state=seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit((Xtr - mu) / sd, yt)
        return clf.predict_proba((Xte - mu) / sd)[:, 1]
    if model_key == "tabiclv2":
        from tabicl import TabICLClassifier
        Xtr, Xte = _prep(X_train, X_test)
        if len(Xtr) > ctx_cap:  # 6GB GPU guard
            sel = np.random.default_rng(seed).choice(len(Xtr), ctx_cap, replace=False)
            Xtr, ytr = Xtr[sel], yt[sel]
        else:
            ytr = yt
        clf = TabICLClassifier(random_state=seed, n_estimators=4, kv_cache=True, verbose=False)
        clf.fit(Xtr, ytr)
        p = clf.predict_proba(Xte)
        pos = list(clf.classes_).index(1)
        return np.asarray(p[:, pos], dtype=float)
    # placeholders until installed: tabpfn / limix / tabdpt / mitra / autogluon
    raise ImportError(f"backend {model_key!r} not installed in this env")


BASELINES = ["woe_logreg", "hgb", "lightgbm", "catboost", "xgboost", "realmlp",
             "tabiclv2", "tabpfn", "limix", "tabdpt", "mitra", "autogluon"]
