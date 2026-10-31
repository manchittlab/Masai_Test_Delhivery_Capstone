"""Compare delivery-hour regressors without using test data for selection."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.shipment_prep import FEATURES

df = pd.read_csv("data/modelling_table.csv")
X, y = df[FEATURES], df.delivery_hours
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=.5, random_state=42)
scaler = StandardScaler().fit(X_train)
checks = dict(train_rows=len(X_train), val_rows=len(X_val), test_rows=len(X_test),
              train_scaled_max_abs_mean=np.abs(scaler.transform(X_train).mean(axis=0)).max(),
              test_scaled_max_abs_mean=np.abs(scaler.transform(X_test).mean(axis=0)).max())
pd.DataFrame(checks.items(), columns=["metric", "value"]).round(4).to_csv("outputs/part_b_checks.csv", index=False)

def metrics(name, split, truth, pred):
    mse = mean_squared_error(truth, pred)
    return dict(model=name, split=split, MAE=mean_absolute_error(truth, pred), MSE=mse,
                RMSE=np.sqrt(mse), R2=r2_score(truth, pred))

rows = [metrics("mean_baseline", "val", y_val, np.full(len(y_val), y_train.mean()))]
depths = [1, 2, 3, 4, 6, 8, 12, None]
depth_rows = []
for depth in depths:
    tree = DecisionTreeRegressor(max_depth=depth, random_state=42).fit(X_train, y_train)
    depth_rows.append(dict(max_depth=depth if depth else "unlimited",
                           train_rmse=np.sqrt(mean_squared_error(y_train, tree.predict(X_train))),
                           val_rmse=np.sqrt(mean_squared_error(y_val, tree.predict(X_val)))))
pd.DataFrame(depth_rows).round(4).to_csv("outputs/part_b_tree_depth.csv", index=False)
best_depth = min(range(len(depths)), key=lambda i: (depth_rows[i]["val_rmse"], float('inf') if depths[i] is None else depths[i]))
models = {
    "linear_regression": Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
    "decision_tree": DecisionTreeRegressor(max_depth=depths[best_depth], random_state=42),
    "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "gradient_boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=.1, random_state=42),
}
for name, model in models.items():
    model.fit(X_train, y_train)
    rows.append(metrics(name, "val", y_val, model.predict(X_val)))
lr_rows = []
for rate in [.01, .1, 1.0]:
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=rate, random_state=42).fit(X_train, y_train)
    lr_rows.append(dict(learning_rate=rate, val_rmse=np.sqrt(mean_squared_error(y_val, model.predict(X_val))),
                        val_r2=r2_score(y_val, model.predict(X_val))))
pd.DataFrame(lr_rows).round(4).to_csv("outputs/part_b_gb_learning_rate.csv", index=False)
ranked = sorted(rows[1:], key=lambda row: row["RMSE"])
X_tv = pd.concat([X_train, X_val]); y_tv = pd.concat([y_train, y_val])
cv = KFold(n_splits=5, shuffle=True, random_state=42)
cv_rows = []
for item in ranked[:2]:
    scores = cross_val_score(models[item["model"]], X_tv, y_tv, cv=cv)
    cv_rows.append(dict(model=item["model"], mean_r2=scores.mean(), std_r2=scores.std()))
pd.DataFrame(cv_rows).round(4).to_csv("outputs/part_b_cv.csv", index=False)
winner = ranked[0]["model"]
rows.append(metrics(winner, "test", y_test, models[winner].predict(X_test)))
pd.DataFrame(rows).round(4).to_csv("outputs/part_b_metrics.csv", index=False)
print("Part B:", winner, "validation RMSE", round(ranked[0]["RMSE"], 4))
