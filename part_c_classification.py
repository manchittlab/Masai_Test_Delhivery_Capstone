"""Stratified delay classification, training-only feature selection and validation thresholding."""
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from src.shipment_prep import FEATURES, NOISE_COLUMNS

def split_data(df):
    X, y = df[FEATURES], df.is_delayed
    Xtr, Xt, ytr, yt = train_test_split(X, y, test_size=.3, random_state=42, stratify=y)
    Xv, Xte, yv, yte = train_test_split(Xt, yt, test_size=.5, random_state=42, stratify=yt)
    return Xtr, Xv, Xte, ytr, yv, yte

def scores(name, y, pred):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(model=name, TN=tn, FP=fp, FN=fn, TP=tp,
                accuracy=accuracy_score(y, pred), precision=precision_score(y, pred, zero_division=0),
                recall=recall_score(y, pred, zero_division=0), f1=f1_score(y, pred, zero_division=0))

df = pd.read_csv("data/modelling_table.csv")
Xtr, Xv, Xte, ytr, yv, yte = split_data(df)
assert (len(Xtr), len(Xv), len(Xte)) == (840, 180, 180)
assert (ytr.sum(), yv.sum(), yte.sum()) == (195, 42, 41)
rows = [scores("blind_all_on_time", yv, np.zeros(len(yv), dtype=int))]
logit = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression())]).fit(Xtr, ytr)
rows.append(scores("logistic_regression", yv, logit.predict(Xv)))
trees = [(d, DecisionTreeClassifier(criterion="gini", max_depth=d, random_state=42).fit(Xtr, ytr))
         for d in [2, 3, 4, 5, 6, 8, 10]]
ranked_trees = sorted(trees, key=lambda pair: (-f1_score(yv, pair[1].predict(Xv)), pair[0]))
d, tree = ranked_trees[0]
rows.append(scores(f"decision_tree_depth_{d}", yv, tree.predict(Xv)))
forest = RandomForestClassifier(n_estimators=100, random_state=42).fit(Xtr, ytr)
rows.append(scores("random_forest_default", yv, forest.predict(Xv)))
pd.DataFrame(rows).round(4).to_csv("outputs/part_c_models.csv", index=False)
selector_all = SelectKBest(f_classif, k="all").fit(Xtr, ytr)
feature_scores = pd.DataFrame(dict(feature=FEATURES, f_score=selector_all.scores_,
                                   rf_importance=forest.feature_importances_,
                                   is_noise=[f in NOISE_COLUMNS for f in FEATURES]))
feature_scores["f_rank"] = feature_scores.f_score.rank(ascending=False, method="min").astype(int)
feature_scores = feature_scores[["feature", "f_score", "f_rank", "rf_importance", "is_noise"]]
feature_scores.round(4).to_csv("outputs/part_c_feature_scores.csv", index=False)
k_rows = []
for k in [5, 7, 10, 12, 17]:
    candidate = Pipeline([("selector", SelectKBest(f_classif, k=k)),
                          ("model", RandomForestClassifier(n_estimators=100, random_state=42))]).fit(Xtr, ytr)
    k_rows.append(dict(k=k, val_f1=f1_score(yv, candidate.predict(Xv))))
pd.DataFrame(k_rows).round(4).to_csv("outputs/part_c_selectk.csv", index=False)
selector = SelectKBest(f_classif, k=7).fit(Xtr, ytr)
tr7, val7 = selector.transform(Xtr), selector.transform(Xv)
params = {"n_estimators": [50, 100, 150], "max_depth": [4, 6, 8], "min_samples_leaf": [1, 5, 10]}
base = RandomForestClassifier(random_state=42)
searches = [
    ("grid_search", GridSearchCV(base, params, cv=5)),
    ("randomized_search", RandomizedSearchCV(base, params, n_iter=9, cv=5, random_state=42)),
]
tuning = []
candidates = {}
for name, search in searches:
    search.fit(tr7, ytr)
    best = search.best_estimator_
    candidates[name] = search.best_params_
    count = 27 if name == "grid_search" else 9
    tuning.append(dict(search=name, candidates=count, cv_folds=5, total_fits=count*5,
                       best_params=json.dumps(search.best_params_, sort_keys=True),
                       best_cv_accuracy=search.best_score_, val_f1=f1_score(yv, best.predict(val7))))
default_params = dict(n_estimators=100, max_depth=None, min_samples_leaf=1)
default = RandomForestClassifier(random_state=42, **default_params).fit(tr7, ytr)
candidates["default_forest_same_7_features"] = default_params
tuning.append(dict(search="default_forest_same_7_features", candidates=1, cv_folds=0,
                   total_fits=1, best_params="defaults", best_cv_accuracy=np.nan,
                   val_f1=f1_score(yv, default.predict(val7))))
pd.DataFrame(tuning).round(4).to_csv("outputs/part_c_tuning.csv", index=False)
priority = {"default_forest_same_7_features": 0, "grid_search": 1, "randomized_search": 2}
chosen = sorted(tuning, key=lambda row: (-row["val_f1"], priority[row["search"]]))[0]["search"]
config = candidates[chosen]
final = Pipeline([("selector", SelectKBest(f_classif, k=7)),
                  ("model", RandomForestClassifier(random_state=42, **config))]).fit(Xtr, ytr)
proba = final.predict_proba(Xv)[:, 1]
thresholds = []
for t in [.1, .2, .3, .4, .5, .6, .7, .8, .9]:
    row = scores("", yv, (proba >= t).astype(int))
    thresholds.append(dict(threshold=t, **{k: v for k, v in row.items() if k != "model"}))
pd.DataFrame(thresholds).round(4).to_csv("outputs/part_c_thresholds.csv", index=False)
t = max((r["threshold"] for r in thresholds if r["recall"] >= .80), default=.1)
test = scores("", yte, (final.predict_proba(Xte)[:, 1] >= t).astype(int))
pd.DataFrame([dict(threshold=t, **{k: v for k, v in test.items() if k != "model"})]).round(4).to_csv("outputs/part_c_final_test.csv", index=False)
pd.DataFrame([dict(selector_k=7, n_estimators=config["n_estimators"],
                   max_depth=config["max_depth"] if config["max_depth"] is not None else "unlimited",
                   min_samples_leaf=config["min_samples_leaf"], threshold=t,
                   chosen_from=chosen)]).to_csv("outputs/part_c_final_config.csv", index=False)
print("Part C:", chosen, "threshold", t, "test F1", round(test["f1"], 4))
