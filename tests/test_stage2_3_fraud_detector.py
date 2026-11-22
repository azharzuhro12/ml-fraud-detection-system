"""Unit tests for Project 03 Stage 2 & 3: Streaming Feature Engine & Hybrid Fraud Detector."""

from src.config import CITY_COORDINATES
from src.data.generator import (
    generate_transaction_stream,
    generate_users_table,
    temporal_train_test_split,
)
from src.features.pipeline import FRAUD_FEATURE_COLUMNS
from src.models.detector import HybridFraudDetector


def test_streaming_features_and_hybrid_detector_decisions() -> None:
    users_df = generate_users_table(n_users=80, seed=25)
    tx_df = generate_transaction_stream(
        users_df=users_df,
        n_normal_sessions=1600,
        n_fraud_incidents=55,
        seed=25,
    )
    train_df, _, cutoff_ts = temporal_train_test_split(tx_df, train_ratio=0.80)

    detector = HybridFraudDetector(n_estimators=50)
    detector.fit(users_df, train_df)

    assert len(detector.feature_importances_) == len(FRAUD_FEATURE_COLUMNS)

    u_row = users_df.iloc[0]
    u_id = str(u_row["user_id"])

    # 1. Normal daytime QRIS coffee purchase on trusted device in home city -> APPROVE
    benign_tx = {
        "user_id": u_id,
        "timestamp": cutoff_ts + 3600,
        "amount_idr": 45000,
        "merchant_category": "food_beverage",
        "device_id": str(u_row["trusted_device_id"]),
        "ip_subnet": str(u_row["trusted_ip_subnet"]),
        "tx_lat": float(u_row["home_lat"]),
        "tx_lon": float(u_row["home_lon"]),
        "payment_method": "qris_instant",
    }
    res_benign = detector.assess_single_transaction(benign_tx, update_state=True)
    assert res_benign.decision == "APPROVE"
    assert res_benign.risk_score_0_100 < 38

    # 2. 5 minutes later, Account Takeover attempt from Singapore VPN on unknown device for Rp 15,000,000 -> BLOCK
    sg_info = CITY_COORDINATES["singapore_vpn"]
    ato_tx = {
        "user_id": u_id,
        "timestamp": cutoff_ts + 3600 + 300,
        "amount_idr": 15000000,
        "merchant_category": "p2p_cash_transfer",
        "device_id": "dev_hacker_emulator_999",
        "ip_subnet": "ip_foreign_vpn_999",
        "tx_lat": float(sg_info["lat"]),
        "tx_lon": float(sg_info["lon"]),
        "payment_method": "p2p_transfer",
    }
    res_ato = detector.assess_single_transaction(ato_tx, update_state=False)
    assert res_ato.decision == "BLOCK"
    assert res_ato.risk_score_0_100 >= 75
    assert len(res_ato.risk_factors) >= 2

