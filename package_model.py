"""Rebuild and persist the selected classifier with training-only fitting."""
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from src.shipment_prep import FEATURES, IMPUTE_COLUMNS, clean_shipments

config = pd.read_csv("outputs/part_c_final_config.csv").iloc[0]
df = pd.read_csv("data/modelling_table.csv")
X, y = df[FEATURES], df.is_delayed
Xtr, Xt, ytr, yt = train_test_split(X, y, test_size=.3, random_state=42, stratify=y)
Xv, Xte, yv, yte = train_test_split(Xt, yt, test_size=.5, random_state=42, stratify=yt)
depth = None if str(config.max_depth) == "unlimited" else int(config.max_depth)
model = Pipeline([("selector", SelectKBest(f_classif, k=int(config.selector_k))),
                  ("model", RandomForestClassifier(n_estimators=int(config.n_estimators), max_depth=depth,
                                                    min_samples_leaf=int(config.min_samples_leaf), random_state=42))])
model.fit(Xtr, ytr)
joblib.dump(model, "models/delay_model.pkl")
pd.DataFrame({"threshold": [config.threshold]}).to_csv("models/delay_threshold.csv", index=False)
pd.DataFrame({"feature": FEATURES}).to_csv("models/feature_order.csv", index=False)
_, medians = clean_shipments(pd.read_csv("data/raw_shipments.csv"))
pd.DataFrame([dict(column=c, median=medians[c]) for c in IMPUTE_COLUMNS]).to_csv("models/impute_values.csv", index=False)
loaded = joblib.load("models/delay_model.pkl")
diff = np.max(np.abs(loaded.predict_proba(Xte)[:, 1] - model.predict_proba(Xte)[:, 1]))
pd.DataFrame([("test_rows_scored", len(Xte)), ("reload_max_abs_probability_diff", diff),
              ("threshold", config.threshold)], columns=["metric", "value"]).round(4).to_csv("outputs/part_d_reload_check.csv", index=False)
print("Packaged model; reload difference", diff)
