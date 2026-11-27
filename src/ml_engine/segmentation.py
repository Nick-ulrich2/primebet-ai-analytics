"""Module industriel de segmentation comportementale des joueurs Primebet (K-Means)."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

# Chemins centralisés depuis la racine du projet
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLEAN_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "primebet_players_clean.csv"
SEGMENTED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "primebet_players_segmented.csv"
PROFILES_JSON_PATH = PROJECT_ROOT / "data" / "processed" / "cluster_profiles.json"
MODEL_PATH = PROJECT_ROOT / "models" / "primebet_kmeans_pipeline.pkl"

FEATURES_ML = [
    "bets_placed_total",
    "avg_bet_amount_eur",
    "total_ggr",
    "days_since_last_activity",
    "gaming_sessions_per_week",
    "casino_share_ratio",
    "loyalty_points",
]

NON_NEGATIVE_FEATURES = [
    "bets_placed_total",
    "avg_bet_amount_eur",
    "days_since_last_activity",
    "gaming_sessions_per_week",
    "loyalty_points",
]


def sanitize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Corrige les anomalies métier sur l'ensemble du DataFrame avant analyse/export."""
    clean = df.copy()
    for col in NON_NEGATIVE_FEATURES + ["ad_spend_eur", "total_stake_eur", "total_gain_eur"]:
        if col in clean.columns:
            clean[col] = clean[col].clip(lower=0)
    if "casino_share_ratio" in clean.columns:
        clean["casino_share_ratio"] = clean["casino_share_ratio"].clip(lower=0, upper=1)
    if "age" in clean.columns:
        clean["age"] = clean["age"].clip(lower=18, upper=95)
    if "satisfaction_score" in clean.columns:
        clean["satisfaction_score"] = clean["satisfaction_score"].clip(lower=0, upper=10)
    return clean


def build_segmentation_pipeline(n_clusters: int = 4, random_state: int = 42) -> Pipeline:
    """Construit un Pipeline scikit-learn autonome incluant les bornes métier."""
    domain_cleaner = ColumnTransformer(
        transformers=[
            (
                "clip_non_neg",
                FunctionTransformer(np.clip, kw_args={"a_min": 0, "a_max": None}),
                NON_NEGATIVE_FEATURES,
            ),
            (
                "clip_ratio",
                FunctionTransformer(np.clip, kw_args={"a_min": 0, "a_max": 1}),
                ["casino_share_ratio"],
            ),
            ("keep_ggr", "passthrough", ["total_ggr"]),
        ],
        remainder="drop",
    )

    return Pipeline(
        steps=[
            ("cleaner", domain_cleaner),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("kmeans", KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)),
        ]
    )


def assign_dynamic_personas(df: pd.DataFrame) -> dict[int, str]:
    """Assigne dynamiquement les personas à partir des statistiques moyennes des clusters."""
    stats = df.groupby("cluster")[["total_ggr", "casino_share_ratio", "loyalty_points"]].mean()
    remaining = set(stats.index.tolist())
    persona_map: dict[int, str] = {}

    # 1. VIP = GGR moyen le plus élevé
    vip_id = int(stats.loc[list(remaining), "total_ggr"].idxmax())
    persona_map[vip_id] = "VIP / High-Rollers à Sécuriser"
    remaining.remove(vip_id)

    # 2. Casino = ratio casino le plus élevé parmi les restants
    casino_id = int(stats.loc[list(remaining), "casino_share_ratio"].idxmax())
    persona_map[casino_id] = "Passionnés Casino & Crash Games"
    remaining.remove(casino_id)

    # 3. Parieurs Sportifs Fidèles = points de fidélité les plus élevés parmi les restants
    loyal_id = int(stats.loc[list(remaining), "loyalty_points"].idxmax())
    persona_map[loyal_id] = "Parieurs Sportifs Fidèles"
    remaining.remove(loyal_id)

    # 4. Parieurs Sportifs Volatils = dernier cluster restant
    volatile_id = int(remaining.pop())
    persona_map[volatile_id] = "Parieurs Sportifs Volatils (À Fidéliser)"

    return persona_map


def build_cluster_profiles(df: pd.DataFrame, persona_map: dict[int, str], sil_score: float) -> dict:
    """Génère le contrat JSON propre et cohérent destiné à l'Agent IA et au Dashboard."""
    profiles: dict = {
        "_metadata": {
            "total_players": int(len(df)),
            "n_clusters": int(len(persona_map)),
            "silhouette_score": round(float(sil_score), 4),
            "features_used": FEATURES_ML,
        },
        "segments": {},
    }

    for cluster_id, persona_name in sorted(persona_map.items()):
        sub = df[df["cluster"] == cluster_id]
        ratio_casino = float(sub["casino_share_ratio"].mean())
        top_sport = str(sub["favourite_sport"].mode()[0])
        top_casino_game = str(sub["game_genre_preference"].mode()[0])

        produit_principal = (
            f"Paris Sportifs ({top_sport})" if ratio_casino < 0.2 else top_casino_game
        )

        profiles["segments"][persona_name] = {
            "cluster_id": cluster_id,
            "nb_joueurs": int(len(sub)),
            "part_base_pct": round(len(sub) / len(df) * 100, 1),
            "mise_moyenne": round(float(sub["avg_bet_amount_eur"].mean()), 2),
            "ggr_moyen": round(float(sub["total_ggr"].mean()), 2),
            "jours_inactivite_moyen": round(float(sub["days_since_last_activity"].mean()), 1),
            "sessions_par_semaine": round(float(sub["gaming_sessions_per_week"].mean()), 1),
            "ratio_casino": round(ratio_casino, 2),
            "points_fidelite_moyen": round(float(sub["loyalty_points"].mean()), 1),
            "taux_churn_pct": round(float(sub["churn"].mean() * 100), 1),
            "produit_principal": produit_principal,
            "jeu_casino_cross_sell": top_casino_game,
            "paiement_prefere_top": str(sub["payment_method"].mode()[0]),
        }

    return profiles


def run_segmentation() -> None:
    """Exécute le pipeline complet : nettoyage, entraînement, validation et exports."""
    if not CLEAN_DATA_PATH.exists():
        raise FileNotFoundError(f"Fichier introuvable : {CLEAN_DATA_PATH}")

    raw_df = pd.read_csv(CLEAN_DATA_PATH)
    df = sanitize_dataframe(raw_df)

    # Vérification stricte de la présence des 7 features
    missing_cols = [c for c in FEATURES_ML if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Colonnes ML manquantes dans le dataset : {missing_cols}")

    pipeline = build_segmentation_pipeline(n_clusters=4, random_state=42)
    df["cluster"] = pipeline.fit_predict(df[FEATURES_ML])

    # Calcul du score de séparation (Silhouette Score)
    X_scaled = pipeline[:-1].transform(df[FEATURES_ML])
    sil_score = silhouette_score(X_scaled, df["cluster"])

    # Assignation dynamique des noms de personas
    persona_map = assign_dynamic_personas(df)
    df["persona"] = df["cluster"].map(persona_map)

    # Sauvegarde des artefacts de production
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    SEGMENTED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, MODEL_PATH)
    df.to_csv(SEGMENTED_DATA_PATH, index=False)

    profiles = build_cluster_profiles(df, persona_map, sil_score)
    with open(PROFILES_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(profiles, f, indent=2, ensure_ascii=False)

    print(f"[OK] Modèle sauvegardé    -> {MODEL_PATH}")
    print(f"[OK] Dataset segmenté     -> {SEGMENTED_DATA_PATH} ({len(df)} lignes)")
    print(f"[OK] Profils JSON IA      -> {PROFILES_JSON_PATH}")
    print(f"[INFO] Silhouette Score   -> {sil_score:.4f}")
    print("\n--- RÉSUMÉ DES SEGMENTS ---")
    for name, data in profiles["segments"].items():
        print(
            f"  - {name:42s} | n={data['nb_joueurs']:4d} ({data['part_base_pct']:4.1f}%) "
            f"| mise={data['mise_moyenne']:6.2f} | GGR={data['ggr_moyen']:7.2f} "
            f"| top={data['produit_principal']}"
        )


if __name__ == "__main__":
    run_segmentation()