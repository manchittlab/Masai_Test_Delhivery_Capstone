"""Generate the modelling table and descriptive evidence."""
import numpy as np
import pandas as pd
from src.shipment_prep import FEATURES, build_features, clean_shipments

raw = pd.read_csv("data/raw_shipments.csv")
hubs = pd.read_csv("data/hubs.csv")
before = raw.drop_duplicates()
clean, medians = clean_shipments(raw)
table = build_features(clean, hubs)
table["is_delayed"] = (table["delivery_hours"] > table["promised_hours"]).astype(int)
assert len(table) == 1200 and not table[FEATURES + ["delivery_hours", "is_delayed"]].isna().any().any()
table.to_csv("data/modelling_table.csv", index=False)
q1, q3 = table["delivery_hours"].quantile([.25, .75])
lo, hi = q1 - 1.5*(q3-q1), q3 + 1.5*(q3-q1)
checks = {
    "raw_rows": len(raw), "duplicate_rows_removed": len(raw)-len(before), "rows_after_dedup": len(before),
    "missing_distance_km_before": before.distance_km.isna().sum(),
    "missing_parcel_weight_before": before.parcel_weight_kg.isna().sum(),
    "sentinel_weight_rows": (before.parcel_weight_kg > 100).sum(),
    "missing_declared_value_before": before.declared_value.isna().sum(),
    "missing_payment_mode_before": before.payment_mode.isna().sum(),
    "service_type_spellings_before": before.service_type.nunique(),
    "service_type_spellings_after": clean.service_type.nunique(),
    "declared_value_mean_inr": table.declared_value_inr.mean(),
    "declared_value_median_inr": table.declared_value_inr.median(),
    "log_declared_value_mean": table.log_declared_value.mean(),
    "log_declared_value_median": table.log_declared_value.median(),
    "median_distance_km": medians["distance_km"],
    "median_parcel_weight_kg": medians["parcel_weight_kg"],
    "median_declared_value_inr": medians["declared_value_inr"],
    "rows_after_hub_merge": len(table),
    "nulls_in_features_and_target": table[FEATURES + ["delivery_hours", "is_delayed"]].isna().sum().sum(),
    "delayed_count": table.is_delayed.sum(), "delay_rate": table.is_delayed.mean(),
    "hours_q1": q1, "hours_q3": q3, "hours_lower_bound": lo, "hours_upper_bound": hi,
    "hours_outlier_count": ((table.delivery_hours < lo) | (table.delivery_hours > hi)).sum(),
}
pd.DataFrame(checks.items(), columns=["metric", "value"]).round(4).to_csv("outputs/part_a_checks.csv", index=False)
table["distance_band"] = pd.cut(table.distance_km, bins=[0, 200, 400, 600, 2000],
                                labels=["0-200", "201-400", "401-600", "601+"], include_lowest=True)
rows = []
for dim in ["service_type", "hub_tier", "distance_band"]:
    groups = table.groupby(dim, observed=True)
    for group, g in groups:
        rows.append(dict(dimension=dim, group=group, shipments=len(g), delayed=int(g.is_delayed.sum()),
                         delay_rate=g.is_delayed.mean(), mean_hours=g.delivery_hours.mean()))
pd.DataFrame(rows).round(4).to_csv("outputs/part_a_delay_by_group.csv", index=False)
table.select_dtypes(include="number").corr().round(4).to_csv("outputs/part_a_correlations.csv", index=True)
print("Part A:", len(table), "shipments;", checks["delayed_count"], "delayed")
