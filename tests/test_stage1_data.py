"""Unit tests for Project 03 Stage 1: Payment Transaction & Multi-Pattern Fraud Generator."""

from src.data.generator import (
    compute_haversine_km,
    generate_transaction_stream,
    generate_users_table,
    temporal_train_test_split,
)


def test_users_and_imbalanced_fraud_transaction_generation() -> None:
    users_df = generate_users_table(n_users=100, seed=10)
    tx_df = generate_transaction_stream(
        users_df=users_df,
        n_normal_sessions=2000,
        n_fraud_incidents=60,
        seed=10,
    )

    assert len(users_df) == 100
    assert len(tx_df) > 2050
    fraud_rate = tx_df["is_fraud"].mean()
    # Realistic class imbalance between 2% and 10%
    assert 0.02 <= fraud_rate <= 0.10

    patterns = set(tx_df["fraud_pattern"].unique())
    assert patterns == {
        "legitimate",
        "account_takeover_ato",
        "velocity_card_testing",
        "syndicate_fraud_ring",
    }

    train_df, test_df, cutoff_ts = temporal_train_test_split(tx_df, train_ratio=0.80)
    assert train_df["timestamp"].max() < cutoff_ts
    assert test_df["timestamp"].min() >= cutoff_ts
    # Check distance Jakarta to Surabaya is > 600 km
    dist_jkt_sby = compute_haversine_km(-6.2088, 106.8456, -7.2575, 112.7521)
    assert float(dist_jkt_sby) > 600.0

