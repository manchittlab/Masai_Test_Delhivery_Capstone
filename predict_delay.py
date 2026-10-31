"""Score new parcels with the exact preprocessing and trained pipeline."""
import joblib
import numpy as np
import pandas as pd
from src.shipment_prep import clean_shipments, build_features

new = pd.read_csv("data/new_shipments.csv")
hubs = pd.read_csv("data/hubs.csv")
medians = dict(pd.read_csv("models/impute_values.csv").values)
clean, _ = clean_shipments(new, medians=medians)
features = build_features(clean, hubs)
order = pd.read_csv("models/feature_order.csv").feature.tolist()
model = joblib.load("models/delay_model.pkl")
threshold = float(pd.read_csv("models/delay_threshold.csv").threshold.iloc[0])
probability = np.round(model.predict_proba(features[order])[:, 1], 4)
segments = pd.read_csv("outputs/hub_segments.csv")[["hub_id", "segment"]]
result = features[["shipment_id", "hub_id"]].merge(segments, on="hub_id", how="left", validate="many_to_one", sort=False)
result["delay_probability"] = probability
result["flagged"] = (probability >= threshold).astype(int)
result["action"] = np.select([result.flagged.eq(0), result.segment.eq("Escalate")],
                              ["No action", "Escalate to hub manager"], default="Proactive customer call")
assert len(result) == len(new) and not result.isna().any().any()
result.to_csv("outputs/new_shipment_actions.csv", index=False)
print("Scored", len(result), "parcels; flagged", result.flagged.sum())
