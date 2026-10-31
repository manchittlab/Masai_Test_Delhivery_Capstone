"""Shared, deterministic shipment cleaning and dispatch-time features."""
import numpy as np
import pandas as pd

PROMISED_HOURS = {"Express": 36, "Standard": 60, "Economy": 96}
FEATURES = [
    "promised_hours", "distance_km", "parcel_weight_kg", "log_declared_value", "n_stops",
    "rain_flag", "sale_event_flag", "is_cod", "is_weekend", "dispatch_hour",
    "hub_tier", "hub_daily_capacity",
    "label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no",
]
NOISE_COLUMNS = ["label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no"]
IMPUTE_COLUMNS = ["distance_km", "parcel_weight_kg", "declared_value_inr"]


def clean_shipments(raw, medians=None):
    clean = raw.drop_duplicates().copy()
    clean["service_type"] = clean["service_type"].str.strip().str.title()
    clean["payment_mode"] = clean["payment_mode"].fillna("Unknown")
    clean["declared_value_inr"] = pd.to_numeric(
        clean["declared_value"].str.replace("₹", "", regex=False).str.replace(",", "", regex=False)
    )
    clean = clean.drop(columns="declared_value")
    clean["dispatch_time"] = pd.to_datetime(clean["dispatch_time"])
    clean.loc[clean["parcel_weight_kg"] > 100, "parcel_weight_kg"] = np.nan
    if medians is None:
        medians = {col: float(clean[col].median()) for col in IMPUTE_COLUMNS}
    else:
        medians = {col: float(medians[col]) for col in IMPUTE_COLUMNS}
    for col in IMPUTE_COLUMNS:
        clean[col] = clean[col].fillna(medians[col])
    return clean, medians


def build_features(clean, hubs):
    result = clean.merge(hubs[["hub_id", "hub_city", "region", "hub_tier", "hub_daily_capacity"]],
                         how="left", on="hub_id", validate="many_to_one", sort=False)
    result["promised_hours"] = result["service_type"].map(PROMISED_HOURS).astype(int)
    result["is_cod"] = result["payment_mode"].eq("COD").astype(int)
    result["is_weekend"] = result["dispatch_time"].dt.dayofweek.ge(5).astype(int)
    result["dispatch_hour"] = result["dispatch_time"].dt.hour.astype(int)
    result["log_declared_value"] = np.log(result["declared_value_inr"])
    return result
