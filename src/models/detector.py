"""
Layer 2, Layer 3 & Layer 4: Real-Time Fraud Risk Detector & Explainable Decision Engine.

Combines:
- Layer 2: Deterministic Hard Guardrail Rules (Syndicate Device Blacklist & Extreme Impossible Travel)
- Layer 3:
  1. Supervised LightGBM Classifier (captures known ATO, Velocity Card Testing, and Fraud Ring patterns)
  2. Unsupervised Isolation Forest (trained on legitimate transactions to flag zero-day anomalies)
- Layer 4:
  3-Tier Cost-Sensitive Decision Engine (APPROVE / CHALLENGE_OTP / BLOCK) + Explainable Risk Reason Codes
  (inspired by Stripe Radar #1, PayPal #193/#233, Grab #72, and Wayfair #257).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import pickle
import time
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from src.config import ARTIFACTS_DIR, DATA_DIR, MERCHANT_CATEGORIES, RISK_TIER_THRESHOLDS
from src.features.pipeline import FRAUD_FEATURE_COLUMNS, StreamingFraudFeatureEngine


@dataclass
class FraudAssessmentResult:
    """Structured real-time risk decision returned by Layer 4 of NusaPay Shield."""

    risk_score_0_100: int
    decision: str  # "APPROVE" | "CHALLENGE_OTP" | "BLOCK"
    decision_label_id: str  # Indonesian user-friendly decision headline
    decision_summary_id: str
    supervised_prob: float
    anomaly_score_norm: float
    triggered_guardrail_rule: str | None
    risk_factors: list[dict[str, Any]] = field(default_factory=list)
    safety_badges: list[str] = field(default_factory=list)
    features_snapshot: dict[str, float] = field(default_factory=dict)
    latency_ms: float = 0.0


class HybridFraudDetector:
    """
    Production Hybrid Supervised + Unsupervised Real-Time Payment Fraud Engine.
    """

    def __init__(
        self,
        n_estimators: int = 160,
        learning_rate: float = 0.04,
        num_leaves: int = 31,
        seed: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.seed = seed

        self.lgbm_model_: lgb.LGBMClassifier | None = None
        self.iso_forest_: IsolationForest | None = None
        self.feature_engine_: StreamingFraudFeatureEngine = StreamingFraudFeatureEngine()
        self.feature_importances_: dict[str, float] = {}
        self.iso_min_score_: float = -0.5
        self.iso_max_score_: float = 0.5

    def fit(
        self,
        users_df: pd.DataFrame,
        train_tx_df: pd.DataFrame,
    ) -> tuple[HybridFraudDetector, pd.DataFrame]:
        """
        Extract causal streaming features on train_tx_df and train both:
        1. LightGBM Supervised Classifier (with class-weight balancing for ~4.9% fraud imbalance)
        2. Isolation Forest Unsupervised Anomaly Detector (fitted on legitimate transactions)
        """
        self.feature_engine_ = StreamingFraudFeatureEngine().initialize_users(users_df)
        X_train = self.feature_engine_.transform_stream_dataframe(train_tx_df)
        y_train = train_tx_df["is_fraud"].to_numpy(dtype=np.int32)

        # 1. Supervised LightGBM Classifier
        pos_count = max(1, int(y_train.sum()))
        neg_count = int((y_train == 0).sum())
        scale_pos = float(np.sqrt(neg_count / pos_count))

        self.lgbm_model_ = lgb.LGBMClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            min_child_samples=20,
            scale_pos_weight=scale_pos,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=self.seed,
            n_jobs=-1,
            importance_type="gain",
            verbosity=-1,
        )
        self.lgbm_model_.fit(X_train, y_train)

        gains = self.lgbm_model_.feature_importances_
        total_gain = max(1e-8, float(gains.sum()))
        self.feature_importances_ = {
            col: round(float(g / total_gain), 4)
            for col, g in sorted(zip(FRAUD_FEATURE_COLUMNS, gains), key=lambda x: x[1], reverse=True)
        }

        # 2. Unsupervised Isolation Forest trained on normal transactions to detect novel out-of-distribution spikes
        X_legit = X_train[y_train == 0]
        self.iso_forest_ = IsolationForest(
            n_estimators=100,
            contamination=0.04,
            random_state=self.seed,
            n_jobs=-1,
        )
        self.iso_forest_.fit(X_legit)

        # Calibrate anomaly score range (more negative decision_function = more anomalous)
        raw_iso = -self.iso_forest_.decision_function(X_train)
        self.iso_min_score_ = float(np.percentile(raw_iso, 2))
        self.iso_max_score_ = float(np.percentile(raw_iso, 99))

        return self, X_train

    def predict_hybrid_probabilities(self, X_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return (hybrid_prob, supervised_prob, normalized_anomaly_score) for a feature DataFrame.
        """
        assert self.lgbm_model_ is not None and self.iso_forest_ is not None

        sup_prob = self.lgbm_model_.predict_proba(X_df)[:, 1]
        raw_iso = -self.iso_forest_.decision_function(X_df)
        denom = max(1e-6, self.iso_max_score_ - self.iso_min_score_)
        anom_norm = np.clip((raw_iso - self.iso_min_score_) / denom, 0.0, 1.0)

        # Apply deterministic guardrail boost when extreme syndicate device or impossible travel is present
        syndicate_mask = X_df["graph_device_unique_users_count"].to_numpy() >= 4.0
        impossible_ato_mask = (
            (X_df["geo_travel_speed_kmh"].to_numpy() >= 500.0)
            & (X_df["device_is_untrusted_for_user"].to_numpy() >= 0.5)
            & (X_df["amount_to_user_mean_ratio"].to_numpy() >= 5.0)
        )
        rapid_burst_mask = X_df["velocity_tx_count_10m"].to_numpy() >= 3.0

        hybrid_prob = 0.84 * sup_prob + 0.16 * anom_norm
        hybrid_prob = np.where(syndicate_mask | impossible_ato_mask | rapid_burst_mask, np.maximum(hybrid_prob, 0.88), hybrid_prob)
        return np.clip(hybrid_prob, 0.0, 1.0), sup_prob, anom_norm

    def assess_single_transaction(
        self,
        tx_event: dict[str, Any],
        update_state: bool = True,
    ) -> FraudAssessmentResult:
        """
        Execute the full 4-Layer Real-Time Risk Assessment on a single incoming transaction
        and generate human-friendly Indonesian risk reason codes.
        """
        t0 = time.perf_counter()
        feat_dict = self.feature_engine_.extract_features_for_event(tx_event, update_state=update_state)
        X_one = pd.DataFrame([feat_dict], columns=FRAUD_FEATURE_COLUMNS)

        hyb_probs, sup_probs, anom_norms = self.predict_hybrid_probabilities(X_one)
        hyb_p = float(hyb_probs[0])
        sup_p = float(sup_probs[0])
        anom_p = float(anom_norms[0])

        # Layer 2 Deterministic Guardrail Check
        guardrail_rule: str | None = None
        if feat_dict["graph_device_unique_users_count"] >= 4.0:
            guardrail_rule = "RULE_SYNDICATE_DEVICE_RING: Perangkat terdeteksi dipakai bersama oleh sindikat banyak akun"
        elif (
            feat_dict["geo_travel_speed_kmh"] >= 500.0
            and feat_dict["device_is_untrusted_for_user"] >= 0.5
            and feat_dict["amount_to_user_mean_ratio"] >= 4.0
        ):
            guardrail_rule = "RULE_IMPOSSIBLE_TRAVEL_ATO: Perpindahan lokasi mustahil (>500 km/jam) dari perangkat asing"
        elif feat_dict["velocity_tx_count_10m"] >= 3.0:
            guardrail_rule = "RULE_VELOCITY_BURST_10M: Serangan transaksi beruntun dalam waktu < 10 menit"

        risk_score = int(round(np.clip(hyb_p * 100.0, 1.0, 99.0)))
        if guardrail_rule is not None:
            risk_score = max(risk_score, 92)

        # Build human-readable Explainable Risk Factors & Safety Badges
        risk_factors: list[dict[str, Any]] = []
        safety_badges: list[str] = []

        # Check Device & IP Graph
        if feat_dict["graph_device_unique_users_count"] >= 3.0:
            cnt = int(feat_dict["graph_device_unique_users_count"])
            risk_factors.append(
                {
                    "code": "SYNDICATE_DEVICE_GRAPH",
                    "severity": "CRITICAL",
                    "title": f"Indikasi Sindikat Fraud Ring ({cnt} Akun di 1 Perangkat)",
                    "detail": f"Perangkat ini tercatat digunakan oleh {cnt} akun berbeda dalam jaringan graf NusaPay.",
                }
            )
        elif feat_dict["device_is_untrusted_for_user"] >= 0.5:
            risk_factors.append(
                {
                    "code": "UNTRUSTED_NEW_DEVICE",
                    "severity": "HIGH",
                    "title": "Login dari Perangkat Baru / Tidak Dikenal",
                    "detail": "Transaksi dikirim dari perangkat yang belum pernah terdaftar pada profil keamanan Anda.",
                }
            )
        else:
            safety_badges.append("Perangkat Terpercaya Milik Anda")

        # Check Geospatial & Impossible Travel
        if feat_dict["geo_travel_speed_kmh"] >= 350.0:
            spd = int(round(feat_dict["geo_travel_speed_kmh"]))
            dist = round(feat_dict["geo_dist_from_prev_tx_km"], 1)
            risk_factors.append(
                {
                    "code": "IMPOSSIBLE_GEO_TRAVEL",
                    "severity": "CRITICAL",
                    "title": f"Perpindahan Lokasi Mustahil ({dist} km • {spd} km/jam)",
                    "detail": f"Lokasi transaksi berpindah sejauh {dist} km dari transaksi sebelumnya dengan kecepatan tak wajar.",
                }
            )
        elif feat_dict["geo_dist_from_home_km"] >= 80.0:
            dist_h = round(feat_dict["geo_dist_from_home_km"], 1)
            risk_factors.append(
                {
                    "code": "OUT_OF_TOWN_LOCATION",
                    "severity": "MEDIUM",
                    "title": f"Lokasi Transaksi di Luar Kota ({dist_h} km dari rumah)",
                    "detail": "Titik transaksi berada jauh di luar area domisili harian akun Anda.",
                }
            )
        else:
            safety_badges.append("Lokasi Sesuai Area Domisili")

        # Check Velocity Counters
        if feat_dict["velocity_tx_count_10m"] >= 2.0:
            v10 = int(feat_dict["velocity_tx_count_10m"]) + 1
            risk_factors.append(
                {
                    "code": "HIGH_VELOCITY_BURST",
                    "severity": "CRITICAL" if v10 >= 4 else "HIGH",
                    "title": f"Frekuensi Transaksi Beruntun ({v10}x dalam 10 Menit)",
                    "detail": "Pola transaksi berkecepatan tinggi menyerupai skrip uji kartu otomatis (Card Testing Bot).",
                }
            )
        else:
            safety_badges.append("Frekuensi Transaksi Normal")

        # Check Spend Amount Anomaly
        ratio = feat_dict["amount_to_user_mean_ratio"]
        if ratio >= 6.0:
            risk_factors.append(
                {
                    "code": "EXTREME_AMOUNT_SPIKE",
                    "severity": "HIGH",
                    "title": f"Lonjakan Nominal Ekstrem ({ratio:.1f}x Lipat Kebiasaan)",
                    "detail": "Nominal pembayaran jauh melampaui rata-rata pengeluaran historis akun ini.",
                }
            )
        elif ratio >= 3.0:
            risk_factors.append(
                {
                    "code": "ELEVATED_AMOUNT",
                    "severity": "MEDIUM",
                    "title": f"Nominal Lebih Besar dari Biasanya ({ratio:.1f}x Rata-rata)",
                    "detail": "Transaksi bernilai cukup tinggi dibanding rata-rata transaksi harian Anda.",
                }
            )
        else:
            safety_badges.append("Nominal Sesuai Kebiasaan Belanja")

        # Check Odd Hour & High Risk Merchant Category
        if feat_dict["temporal_is_odd_hour_00_05"] >= 0.5 and risk_score >= 38:
            risk_factors.append(
                {
                    "code": "ODD_HOUR_TRANSACTION",
                    "severity": "MEDIUM",
                    "title": "Transaksi pada Jam Rawan Dini Hari (00:00 - 05:00)",
                    "detail": "Dilakukan di luar jam aktif normal pemilik akun.",
                }
            )

        if feat_dict["merchant_category_base_risk"] >= 0.65 and risk_score >= 38:
            cat_key = str(tx_event.get("merchant_category", ""))
            cat_label = MERCHANT_CATEGORIES.get(cat_key, {}).get("label", cat_key)
            risk_factors.append(
                {
                    "code": "HIGH_RISK_ASSET_CATEGORY",
                    "severity": "MEDIUM",
                    "title": f"Kategori Rawan Pencairan Cepat ({cat_label})",
                    "detail": "Kategori ini sering menjadi sasaran pencairan dana instan oleh pembajak akun.",
                }
            )

        # Layer 4: 3-Tier Cost-Sensitive Decision
        block_th = RISK_TIER_THRESHOLDS["block_threshold"]
        chal_th = RISK_TIER_THRESHOLDS["challenge_threshold"]

        # Refine risk score so benign transactions with 0 risk factors stay cleanly in green (2 - 22)
        if not risk_factors:
            risk_score = min(risk_score, 18)
        elif len(risk_factors) == 1 and risk_factors[0]["severity"] == "MEDIUM" and guardrail_rule is None:
            risk_score = int(np.clip(risk_score, chal_th, block_th - 8))

        if risk_score >= block_th:
            decision = "BLOCK"
            decision_label = "DIBLOKIR OTOMATIS (Proteksi Saldo Aktif)"
            decision_summary = (
                "NusaPay Shield menahan transaksi ini karena terdeteksi sinyal pembajakan akun atau "
                "sindikat penipuan berisiko tinggi. Saldo dompet Anda tetap aman."
            )
        elif risk_score >= chal_th:
            decision = "CHALLENGE_OTP"
            decision_label = "PERLU VERIFIKASI TAMBAHAN (Step-Up OTP / PIN)"
            decision_summary = (
                "Transaksi ini memiliki perbedaan dari kebiasaan rutin Anda. Untuk memastikan ini benar Anda, "
                "sistem meminta konfirmasi kode OTP/Biometrik sebelum melanjutkan."
            )
        else:
            decision = "APPROVE"
            decision_label = "DISETUJUI INSTAN (Transaksi Aman)"
            decision_summary = (
                "Seluruh pemeriksaan perangkat, lokasi, frekuensi, dan profil nominal dinyatakan aman."
            )

        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 3)

        return FraudAssessmentResult(
            risk_score_0_100=risk_score,
            decision=decision,
            decision_label_id=decision_label,
            decision_summary_id=decision_summary,
            supervised_prob=round(sup_p, 4),
            anomaly_score_norm=round(anom_p, 4),
            triggered_guardrail_rule=guardrail_rule,
            risk_factors=risk_factors,
            safety_badges=safety_badges,
            features_snapshot=feat_dict,
            latency_ms=elapsed_ms,
        )

    def save(self, path: Path = ARTIFACTS_DIR / "hybrid_fraud_detector.pkl") -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path = ARTIFACTS_DIR / "hybrid_fraud_detector.pkl") -> HybridFraudDetector:
        with open(path, "rb") as f:
            return pickle.load(f)


def evaluate_fraud_detection_pipeline(
    users_df: pd.DataFrame,
    train_tx_df: pd.DataFrame,
    test_tx_df: pd.DataFrame,
) -> dict[str, Any]:
    """
    Evaluate 3 fraud detection architectures on the future held-out test stream:
    1. Static Rule Baseline (Blocks only if amount > Rp 5.000.000 & untrusted device)
    2. Unsupervised Isolation Forest Only
    3. Hybrid Supervised LightGBM + Unsupervised Isolation Forest + Guardrails (NusaPay Shield)

    Computes industry-standard imbalanced fraud metrics:
    - PR-AUC (Average Precision) & ROC-AUC
    - Recall at 1% False Positive Rate (FPR)
    - Detection Rate per Fraud Pattern (Account Takeover, Velocity Card Testing, Syndicate Fraud Ring)
    - Financial Impact in IDR (Total Fraud Loss Prevented vs. False Positive Friction Cost)
    """
    t0 = time.perf_counter()
    detector = HybridFraudDetector()
    detector.fit(users_df, train_tx_df)
    train_time_sec = round(time.perf_counter() - t0, 3)

    # Extract causal streaming features on test_tx_df continuing from the trained state
    t_feat_start = time.perf_counter()
    X_test = detector.feature_engine_.transform_stream_dataframe(test_tx_df)
    y_test = test_tx_df["is_fraud"].to_numpy(dtype=np.int32)
    amounts_test = test_tx_df["amount_idr"].to_numpy(dtype=np.float64)
    patterns_test = test_tx_df["fraud_pattern"].to_numpy()

    hyb_probs, sup_probs, anom_scores = detector.predict_hybrid_probabilities(X_test)
    total_eval_ms = (time.perf_counter() - t_feat_start) * 1000.0
    avg_latency_per_tx_ms = round(total_eval_ms / max(1, len(test_tx_df)), 4)

    # Baseline 1: Simple Heuristic Rule score
    rule_scores = (
        0.45 * X_test["device_is_untrusted_for_user"].to_numpy()
        + 0.35 * np.clip(X_test["tx_amount_idr"].to_numpy() / 10_000_000.0, 0.0, 1.0)
        + 0.20 * X_test["temporal_is_odd_hour_00_05"].to_numpy()
    )

    models_scores = {
        "1_heuristic_rules_baseline": rule_scores,
        "2_unsupervised_isolation_forest": anom_scores,
        "3_hybrid_lgbm_isoforest_shield": hyb_probs,
    }

    comparison: dict[str, dict[str, Any]] = {}
    for m_name, scores in models_scores.items():
        pr_auc = float(average_precision_score(y_test, scores))
        roc_auc = float(roc_auc_score(y_test, scores))

        # Compute Recall at <= 1.0% False Positive Rate (FPR)
        fpr, tpr, roc_th = roc_curve(y_test, scores)
        valid_fpr_idxs = np.where(fpr <= 0.01)[0]
        recall_at_1pct_fpr = float(tpr[valid_fpr_idxs[-1]]) if len(valid_fpr_idxs) > 0 else 0.0

        # Evaluate at best F1 / Cost threshold
        prec, rec, pr_th = precision_recall_curve(y_test, scores)
        f1_vals = (2 * prec * rec) / np.maximum(1e-8, prec + rec)
        best_idx = int(np.argmax(f1_vals))
        chosen_th = float(pr_th[min(best_idx, len(pr_th) - 1)]) if len(pr_th) > 0 else 0.5

        preds = (scores >= chosen_th).astype(np.int32)
        tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()

        precision_val = float(tp / max(1, tp + fp))
        recall_val = float(tp / max(1, tp + fn))
        fpr_val = float(fp / max(1, fp + tn))

        # Financial impact calculation (in IDR)
        total_fraud_volume_idr = float(amounts_test[y_test == 1].sum())
        fraud_prevented_idr = float(amounts_test[(y_test == 1) & (preds == 1)].sum())
        missed_fraud_loss_idr = float(amounts_test[(y_test == 1) & (preds == 0)].sum())
        # False positive friction cost: Rp 25,000 per falsely flagged legitimate transaction
        fp_friction_cost_idr = float(fp * 25000.0)
        net_savings_idr = fraud_prevented_idr - fp_friction_cost_idr

        comparison[m_name] = {
            "pr_auc": round(pr_auc, 4),
            "roc_auc": round(roc_auc, 4),
            "recall_at_1pct_fpr": round(recall_at_1pct_fpr, 4),
            "precision": round(precision_val, 4),
            "recall": round(recall_val, 4),
            "false_positive_rate_pct": round(fpr_val * 100.0, 2),
            "total_fraud_exposure_idr": int(round(total_fraud_volume_idr)),
            "fraud_prevented_idr": int(round(fraud_prevented_idr)),
            "missed_fraud_loss_idr": int(round(missed_fraud_loss_idr)),
            "net_financial_savings_idr": int(round(net_savings_idr)),
        }

    # Recall breakdown by specific fraud attack pattern for our Hybrid Shield model
    shield_preds = (hyb_probs >= 0.38).astype(np.int32)
    pattern_recall: dict[str, float] = {}
    for pat in ["account_takeover_ato", "velocity_card_testing", "syndicate_fraud_ring"]:
        pat_mask = patterns_test == pat
        if np.any(pat_mask):
            pattern_recall[pat] = round(float(shield_preds[pat_mask].mean()), 4)

    # Reset feature engine state back to the end of train_tx_df before saving for live API serving
    detector.feature_engine_ = StreamingFraudFeatureEngine().initialize_users(users_df)
    detector.feature_engine_.transform_stream_dataframe(train_tx_df)
    detector.save()

    report = {
        "n_test_transactions": len(test_tx_df),
        "n_test_fraud_transactions": int(y_test.sum()),
        "training_time_sec": train_time_sec,
        "avg_inference_latency_per_tx_ms": avg_latency_per_tx_ms,
        "model_comparison": comparison,
        "detection_rate_by_attack_pattern": pattern_recall,
        "top_12_feature_importances": dict(list(detector.feature_importances_.items())[:12]),
    }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(ARTIFACTS_DIR / "stage3_fraud_metrics.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


if __name__ == "__main__":
    from src.models.detector import evaluate_fraud_detection_pipeline as _run_fraud_eval

    users_df = pd.read_csv(DATA_DIR / "users.csv")
    train_df = pd.read_csv(DATA_DIR / "train_transactions.csv")
    test_df = pd.read_csv(DATA_DIR / "test_transactions.csv")

    report = _run_fraud_eval(users_df, train_df, test_df)
    print("=== Project 03 Stage 2 & 3: Fraud Detection Evaluation Completed ===")
    print(json.dumps(report, indent=2))

