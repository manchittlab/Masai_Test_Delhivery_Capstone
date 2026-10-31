"""Profile and cluster hubs, then compare operating segments with labels."""
import itertools
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/modelling_table.csv")
profile = df.groupby("hub_id", sort=True).agg(shipments=("shipment_id", "size"),
    mean_delivery_hours=("delivery_hours", "mean"), delay_rate=("is_delayed", "mean"),
    mean_distance_km=("distance_km", "mean"), mean_n_stops=("n_stops", "mean")).reset_index()
profile.round(4).to_csv("outputs/hub_profile.csv", index=False)
cols = ["mean_delivery_hours", "delay_rate", "mean_distance_km", "mean_n_stops"]
X = StandardScaler().fit_transform(profile[cols])
rows = []
selected = None
for k in [2, 3, 4, 5]:
    labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X)
    across, within = [], []
    for i, j in itertools.combinations(range(len(X)), 2):
        (within if labels[i] == labels[j] else across).append(np.linalg.norm(X[i]-X[j]))
    sizes = sorted(np.bincount(labels), reverse=True)
    rows.append(dict(K=k, across_over_within=np.mean(across)/np.mean(within),
                     cluster_sizes="-".join(map(str, sizes))))
    if k == 3:
        selected = labels
pd.DataFrame(rows).round(4).to_csv("outputs/part_d_separation.csv", index=False)
profile["cluster"] = selected
order = profile.groupby("cluster").delay_rate.mean().sort_values().index
names = {c: name for c, name in zip(order, ["Standard", "Watch", "Escalate"])}
profile["segment"] = profile.cluster.map(names)
hubs = pd.read_csv("data/hubs.csv")
segments = profile[["hub_id", "segment"]].merge(hubs[["hub_id", "hub_city", "hub_tier"]], on="hub_id", validate="one_to_one")
segments.to_csv("outputs/hub_segments.csv", index=False)
segment_profile = profile.groupby("segment").agg(hubs=("hub_id", "size"), **{c: (c, "mean") for c in cols}).reset_index()
segment_profile["segment"] = pd.Categorical(segment_profile.segment, ["Standard", "Watch", "Escalate"], ordered=True)
segment_profile.sort_values("segment").round(4).to_csv("outputs/segment_profile.csv", index=False)
expected = segments.hub_tier.map({1: "Standard", 2: "Watch", 3: "Escalate"})
segments.loc[segments.segment != expected].to_csv("outputs/tier_vs_segment_mismatch.csv", index=False)
print("Part D:", rows)
