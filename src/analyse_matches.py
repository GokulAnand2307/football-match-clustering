"""Cluster football matches and identify unusual match profiles."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

FEATURES = ["FTHG", "FTAG", "HS", "AS", "HST", "AST", "HC", "AC", "HY", "AY", "HR", "AR"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    matches = pd.read_csv(args.data)
    missing = sorted(set(FEATURES) - set(matches.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    matrix = matches[FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0)
    scaled = StandardScaler().fit_transform(matrix)
    projection = PCA(n_components=2, random_state=args.seed).fit_transform(scaled)

    kmeans = KMeans(n_clusters=3, n_init=20, random_state=args.seed)
    matches["cluster"] = kmeans.fit_predict(scaled)
    detector = IsolationForest(contamination=0.05, random_state=args.seed)
    matches["anomaly"] = detector.fit_predict(scaled)

    plot = pd.DataFrame({"PC1": projection[:, 0], "PC2": projection[:, 1]})
    plot["Cluster"] = matches["cluster"].astype(str)
    plot["Status"] = matches["anomaly"].map({1: "Typical", -1: "Anomaly"})

    plt.figure(figsize=(9, 6))
    sns.scatterplot(data=plot, x="PC1", y="PC2", hue="Cluster", palette="Set2", s=55, alpha=0.8)
    plt.title("Match profiles grouped with K-means")
    plt.tight_layout()
    plt.savefig(args.output / "kmeans_clusters.svg")
    plt.close()

    plt.figure(figsize=(9, 6))
    sns.scatterplot(data=plot, x="PC1", y="PC2", hue="Status", palette={"Typical": "#3b82f6", "Anomaly": "#ef4444"}, s=55, alpha=0.8)
    plt.title("Unusual match profiles identified by Isolation Forest")
    plt.tight_layout()
    plt.savefig(args.output / "match_anomalies.svg")
    plt.close()

    score = silhouette_score(scaled, matches["cluster"])
    print(f"Silhouette score: {score:.3f}")

    columns = [c for c in ["Date", "HomeTeam", "AwayTeam"] if c in matches.columns]
    matches.loc[matches["anomaly"] == -1, columns + FEATURES].to_csv(
        args.output / "detected_anomalies.csv", index=False
    )


if __name__ == "__main__":
    main()
