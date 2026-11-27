"""Module industriel de prédiction du risque de Churn des joueurs Primebet."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

# Chemins centralisés depuis la racine du projet
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SEGMENTED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "primebet_players_segmented.csv"
PROFILES_JSON_PATH = PROJECT_ROOT / "data" / "processed" / "cluster_profiles.json"
CHURN_REPORT_PATH = PROJECT_ROOT / "data" / "processed" / "churn_model_report.json"
CHURN_MODEL_PATH = PROJECT_ROOT / "models" / "primebet_churn_pipeline.pkl"

CHURN_NON_NEG_NUM = [
    "subscription_tenure_months",
    "bets_placed_total",
    "avg_bet_amount_eur",
    "total_stake_eur",
    "total_gain_eur",
    "avg_betting_odds",
    "betting_win_rate",
    "payment_failures",
    "days_since_last_activity",
    "gaming_sessions_per_week",
    "avg_gaming_session_mins",
    "time_on_site_mins",
    "loyalty_points",
    "support_tickets_raised",
]
CHURN_BOUNDED_NUM = ["age", "satisfaction_score", "casino_share_ratio"]
CHURN_FREE_NUM = ["total_ggr"]
CHURN_CAT = [
    "region",
    "primary_device",
    "favourite_sport",
    "preferred_bet_type",
    "betting_profile",
    "payment_method",
    "acquisition_channel",
]
CHURN_FEATURES = CHURN_NON_NEG_NUM + CHURN_BOUNDED_NUM + CHURN_FREE_NUM + CHURN_CAT

# Colonnes permettant d'écarter les doublons synthétiques avant le split train/test
DEDUP_SIGNATURE = ["total_stake_eur", "total_gain_eur", "loyalty_points"]


def build_churn_pipeline(random_state: int = 42) -> Pipeline:
    """Construit un Pipeline autonome de prédiction de Churn avec garde-fous métier."""
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "non_neg",
                Pipeline(
                    steps=[
                        ("clip", FunctionTransformer(np.clip, kw_args={"a_min": 0, "a_max": None})),
                        ("imp", SimpleImputer(strategy="median")),
                    ]
                ),
                CHURN_NON_NEG_NUM,
            ),
            (
                "age",
                Pipeline(
                    steps=[
                        ("clip", FunctionTransformer(np.clip, kw_args={"a_min": 18, "a_max": 95})),
                        ("imp", SimpleImputer(strategy="median")),
                    ]
                ),
                ["age"],
            ),
            (
                "satisfaction",
                Pipeline(
                    steps=[
                        ("clip", FunctionTransformer(np.clip, kw_args={"a_min": 0, "a_max": 10})),
                        ("imp", SimpleImputer(strategy="median")),
                    ]
                ),
                ["satisfaction_score"],
            ),
            (
                "ratio",
                Pipeline(
                    steps=[
                        ("clip", FunctionTransformer(np.clip, kw_args={"a_min": 0, "a_max": 1})),
                        ("imp", SimpleImputer(strategy="median")),
                    ]
                ),
                ["casino_share_ratio"],
            ),
            ("ggr", SimpleImputer(strategy="median"), CHURN_FREE_NUM),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imp", SimpleImputer(strategy="most_frequent")),
                        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                CHURN_CAT,
            ),
        ],
        remainder="drop",
    )

    classifier = HistGradientBoostingClassifier(
        max_depth=4,
        learning_rate=0.04,
        max_iter=150,
        min_samples_leaf=25,
        l2_regularization=2.0,
        class_weight="balanced",
        random_state=random_state,
    )

    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])


def assign_risk_level(prob_pct: pd.Series) -> pd.Series:
    """Classe chaque joueur en 3 niveaux actionnables pour le marketing."""
    return pd.cut(
        prob_pct,
        bins=[-1.0, 40.0, 70.0, 101.0],
        labels=["Faible", "Modéré", "Critique"],
    ).astype(str)


def run_churn_training() -> None:
    """Entraîne, évalue sans fuite, sauvegarde le modèle et score toute la base."""
    if not SEGMENTED_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {SEGMENTED_DATA_PATH}. "
            "Lance d'abord : python -m src.ml_engine.segmentation"
        )

    df = pd.read_csv(SEGMENTED_DATA_PATH)
    missing_cols = [c for c in CHURN_FEATURES + ["churn"] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Colonnes manquantes pour le modèle Churn : {missing_cols}")

    # 1. Déduplication stricte avant split pour une évaluation honnête sans fuite
    df_unique = df.drop_duplicates(subset=DEDUP_SIGNATURE).copy()
    X_pool = df_unique[CHURN_FEATURES]
    y_pool = df_unique["churn"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X_pool, y_pool, test_size=0.2, random_state=42, stratify=y_pool
    )

    # 2. Entraînement & évaluation sur holdout (20 % jamais vu)
    eval_pipeline = build_churn_pipeline(random_state=42)
    eval_pipeline.fit(X_train, y_train)

    train_probs = eval_pipeline.predict_proba(X_train)[:, 1]
    test_probs = eval_pipeline.predict_proba(X_test)[:, 1]
    test_preds = (test_probs >= 0.5).astype(int)

    train_auc = roc_auc_score(y_train, train_probs)
    test_auc = roc_auc_score(y_test, test_probs)
    test_precision = precision_score(y_test, test_preds, zero_division=0)
    test_recall = recall_score(y_test, test_preds, zero_division=0)
    test_f1 = f1_score(y_test, test_preds, zero_division=0)

    # Validation métier des 3 paliers sur le jeu de test jamais vu
    test_eval_df = pd.DataFrame(
        {
            "churn": y_test.values,
            "risk_level": assign_risk_level(pd.Series(test_probs * 100.0)),
        }
    )
    holdout_buckets = {}
    for level in ["Faible", "Modéré", "Critique"]:
        sub_b = test_eval_df[test_eval_df["risk_level"] == level]
        holdout_buckets[level] = {
            "nb_joueurs_test": int(len(sub_b)),
            "taux_churn_reel_pct": round(float(sub_b["churn"].mean() * 100), 1) if len(sub_b) else 0.0,
        }

    # 3. Facteurs explicatifs principaux (Permutation Importance sur X_test)
    perm = permutation_importance(
        eval_pipeline, X_test, y_test, scoring="roc_auc", n_repeats=5, random_state=42
    )
    top_drivers = (
        pd.Series(perm.importances_mean, index=CHURN_FEATURES)
        .sort_values(ascending=False)
        .head(8)
    )
    top_drivers_dict = {feat: round(float(val), 4) for feat, val in top_drivers.items()}

    # 4. Ré-entraînement final sur 100 % des profils uniques pour la production
    prod_pipeline = build_churn_pipeline(random_state=42)
    prod_pipeline.fit(X_pool, y_pool)

    CHURN_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(prod_pipeline, CHURN_MODEL_PATH)

    # 5. Scoring de toute la base (10 300 joueurs) et mise à jour du CSV segmenté
    all_probs_pct = np.round(prod_pipeline.predict_proba(df[CHURN_FEATURES])[:, 1] * 100.0, 1)
    df["churn_risk_score_pct"] = all_probs_pct
    df["churn_risk_level"] = assign_risk_level(df["churn_risk_score_pct"])
    df.to_csv(SEGMENTED_DATA_PATH, index=False)

    # 6. Synthèse par persona + enrichissement de cluster_profiles.json
    persona_risk_summary = {}
    if "persona" in df.columns:
        for persona_name, sub in df.groupby("persona"):
            n_crit = int((sub["churn_risk_level"] == "Critique").sum())
            n_mod = int((sub["churn_risk_level"] == "Modéré").sum())
            persona_risk_summary[str(persona_name)] = {
                "nb_joueurs": int(len(sub)),
                "risque_churn_moyen_pct": round(float(sub["churn_risk_score_pct"].mean()), 1),
                "joueurs_risque_critique": n_crit,
                "part_risque_critique_pct": round(n_crit / len(sub) * 100, 1),
                "joueurs_risque_modere": n_mod,
            }

        if PROFILES_JSON_PATH.exists():
            with open(PROFILES_JSON_PATH, "r", encoding="utf-8") as f:
                profiles_json = json.load(f)
            for p_name, risk_stats in persona_risk_summary.items():
                if p_name in profiles_json.get("segments", {}):
                    profiles_json["segments"][p_name].update(
                        {
                            "risque_churn_moyen_pct": risk_stats["risque_churn_moyen_pct"],
                            "joueurs_risque_critique": risk_stats["joueurs_risque_critique"],
                            "part_risque_critique_pct": risk_stats["part_risque_critique_pct"],
                        }
                    )
            profiles_json["_churn_metadata"] = {
                "model": "HistGradientBoostingClassifier (balanced)",
                "holdout_roc_auc": round(float(test_auc), 4),
                "top_churn_drivers": list(top_drivers_dict.keys())[:5],
            }
            with open(PROFILES_JSON_PATH, "w", encoding="utf-8") as f:
                json.dump(profiles_json, f, indent=2, ensure_ascii=False)

    # 7. Sauvegarde du rapport complet du modèle de Churn
    report = {
        "dataset": {
            "total_scored_players": int(len(df)),
            "unique_profiles_for_training": int(len(df_unique)),
            "train_size": int(len(X_train)),
            "test_size": int(len(X_test)),
        },
        "metrics_holdout_unseen": {
            "train_roc_auc": round(float(train_auc), 4),
            "test_roc_auc": round(float(test_auc), 4),
            "precision_at_0_5": round(float(test_precision), 4),
            "recall_at_0_5": round(float(test_recall), 4),
            "f1_at_0_5": round(float(test_f1), 4),
        },
        "holdout_risk_calibration": holdout_buckets,
        "top_churn_drivers_auc_drop": top_drivers_dict,
        "risk_by_persona": persona_risk_summary,
    }
    with open(CHURN_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Affichage console
    print(f"[OK] Modèle Churn sauvegardé -> {CHURN_MODEL_PATH}")
    print(f"[OK] CSV enrichi (scores)    -> {SEGMENTED_DATA_PATH}")
    print(f"[OK] Rapport Churn JSON      -> {CHURN_REPORT_PATH}")
    print(f"[OK] Profils IA enrichis     -> {PROFILES_JSON_PATH}")
    print(
        f"\n[EVAL HOLDOUT SANS FUITE] ROC-AUC Test: {test_auc:.4f} (Train: {train_auc:.4f}) "
        f"| Recall: {test_recall:.3f} | Precision: {test_precision:.3f} | F1: {test_f1:.3f}"
    )
    print("\n--- CALIBRATION SUR JEU DE TEST JAMAIS VU (n=1500) ---")
    for lvl, vals in holdout_buckets.items():
        print(f"  - Risque {lvl:8s} : {vals['nb_joueurs_test']:4d} joueurs | Churn réel = {vals['taux_churn_reel_pct']:4.1f}%")
    print("\n--- TOP 5 FACTEURS DU CHURN ---")
    for feat, score in list(top_drivers_dict.items())[:5]:
        print(f"  - {feat:28s} : +{score:.4f} AUC")
    print("\n--- ALERTES CHURN PAR PERSONA (BASE COMPLÈTE) ---")
    for pname, stats in persona_risk_summary.items():
        print(
            f"  - {pname:42s} | Critique={stats['joueurs_risque_critique']:4d} "
            f"({stats['part_risque_critique_pct']:4.1f}%) | Risque moy={stats['risque_churn_moyen_pct']:4.1f}%"
        )


if __name__ == "__main__":
    run_churn_training()