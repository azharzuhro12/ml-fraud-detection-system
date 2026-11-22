"""
Synthetic Payment Transaction & Multi-Pattern Fraud Dataset Generator.

Simulates realistic digital wallet transactions with ~4.5% fraud rate across 3 real-world attack vectors:
1. Account Takeover (ATO): Attacker logs in from a new device in a distant city/country (Impossible Travel),
   often at odd hours (00:00 - 04:30), attempting to drain wallet balance via electronics/crypto/P2P transfer.
2. Card / Wallet Velocity Testing: Bot script executes 4 to 7 rapid transactions within a few minutes.
3. Fraud Ring (Shared Device/IP Syndicate): A small pool of compromised emulator devices and proxy IPs
   shared across dozens of different user accounts (inspired by PayPal #233 & Grab #163).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import (
    CITY_COORDINATES,
    DATA_DIR,
    MERCHANT_CATEGORIES,
    WALLET_USER_PERSONAS,
)


def compute_haversine_km(
    lat1: float | np.ndarray,
    lon1: float | np.ndarray,
    lat2: float | np.ndarray,
    lon2: float | np.ndarray,
) -> float | np.ndarray:
    """Compute Haversine distance in kilometers between two coordinates."""
    r_earth = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    return 2.0 * r_earth * np.arcsin(np.sqrt(a))


def generate_users_table(n_users: int = 500, seed: int = 42) -> pd.DataFrame:
    """Generate wallet user profiles with primary trusted device, home city coordinates, and spending stats."""
    rng = np.random.default_rng(seed)
    persona_keys = list(WALLET_USER_PERSONAS.keys())

    users: list[dict[str, Any]] = []
    for idx in range(n_users):
        p_key = persona_keys[idx % len(persona_keys)]
        p_cfg = WALLET_USER_PERSONAS[p_key]
        city_key = str(p_cfg["home_city"])
        city_info = CITY_COORDINATES[city_key]

        home_lat = float(city_info["lat"]) + float(rng.normal(0, 0.025))
        home_lon = float(city_info["lon"]) + float(rng.normal(0, 0.025))

        mean_spend = int(max(25000, rng.normal(p_cfg["typical_spend_mean_idr"], p_cfg["typical_spend_mean_idr"] * 0.15)))
        std_spend = int(max(15000, rng.normal(p_cfg["typical_spend_std_idr"], p_cfg["typical_spend_std_idr"] * 0.15)))
        acct_age = int(max(14, rng.normal(p_cfg["account_age_days"], 90)))

        # First 4 users keep exact persona display names for clear UI selection
        display_name = (
            p_cfg["display_name"]
            if idx < 4
            else f"User #{idx:04d} ({city_info['name']})"
        )

        users.append(
            {
                "user_id": f"user_{idx:04d}",
                "persona": p_key,
                "display_name": display_name,
                "home_city": city_key,
                "home_city_label": str(city_info["name"]),
                "home_lat": round(home_lat, 5),
                "home_lon": round(home_lon, 5),
                "trusted_device_id": f"dev_trusted_{idx:04d}",
                "trusted_ip_subnet": f"ip_home_{idx // 2:04d}",
                "typical_spend_mean_idr": mean_spend,
                "typical_spend_std_idr": std_spend,
                "wallet_balance_idr": int(p_cfg["wallet_balance_idr"]),
                "account_age_days": acct_age,
                "primary_cat_1": p_cfg["primary_categories"][0],
                "primary_cat_2": p_cfg["primary_categories"][1],
            }
        )

    return pd.DataFrame(users)


def generate_transaction_stream(
    users_df: pd.DataFrame,
    n_normal_sessions: int = 22000,
    n_fraud_incidents: int = 680,
    simulation_days: int = 30,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a chronological stream of normal and fraudulent payment transactions.
    """
    rng = np.random.default_rng(seed + 100)
    all_categories = list(MERCHANT_CATEGORIES.keys())
    all_cities = list(CITY_COORDINATES.keys())

    # Small pool of syndicate devices and proxy IPs shared by fraudsters (Fraud Ring Graph)
    syndicate_devices = [f"dev_syndicate_ring_{i:02d}" for i in range(18)]
    syndicate_ips = [f"ip_proxy_ring_{i:02d}" for i in range(15)]

    base_ts = 1_710_000_000  # Base epoch timestamp
    total_sec = simulation_days * 86400

    records: list[dict[str, Any]] = []

    # 1. Generate Normal Legitimate Transactions
    for s_idx in range(n_normal_sessions):
        u_row = users_df.iloc[int(rng.integers(0, len(users_df)))]
        u_id = str(u_row["user_id"])

        # Legitimate transactions mostly happen during daytime/evening (07:00 - 22:00 local time)
        day_offset = int(rng.integers(0, simulation_days))
        hour_of_day = int(rng.choice(
            range(24),
            p=[
                0.005, 0.003, 0.002, 0.002, 0.003, 0.010,
                0.035, 0.065, 0.075, 0.070, 0.065, 0.085,
                0.095, 0.075, 0.060, 0.055, 0.060, 0.070,
                0.075, 0.045, 0.025, 0.012, 0.005, 0.003,
            ],
        ))
        ts = base_ts + day_offset * 86400 + hour_of_day * 3600 + int(rng.integers(0, 3599))

        # 88% in user's primary categories, 12% in other categories
        if rng.random() < 0.88:
            cat = str(rng.choice([u_row["primary_cat_1"], u_row["primary_cat_2"]]))
        else:
            cat = str(rng.choice(all_categories))

        # Spend amount normally distributed around user's typical spend, with 5% legitimate high-ticket purchases
        if rng.random() < 0.05:
            raw_amt = float(u_row["typical_spend_mean_idr"]) * rng.uniform(3.5, 9.5)
        else:
            raw_amt = rng.lognormal(
                mean=np.log(max(20000.0, float(u_row["typical_spend_mean_idr"]))),
                sigma=0.48,
            )
        amt_idr = int(round(np.clip(raw_amt, 12000.0, float(u_row["typical_spend_mean_idr"]) * 10.0) / 500.0) * 500)

        # 88% on user's primary trusted device, 8% on legitimate new/upgraded device, 4% on public Wi-Fi IP
        r_dev = rng.random()
        if r_dev < 0.88:
            dev_id = str(u_row["trusted_device_id"])
            ip_id = str(u_row["trusted_ip_subnet"])
        elif r_dev < 0.96:
            dev_id = f"dev_legit_new_{s_idx:05d}"
            ip_id = f"ip_legit_roam_{s_idx // 4:05d}"
        else:
            dev_id = str(u_row["trusted_device_id"])
            ip_id = f"ip_public_wifi_{s_idx % 40:03d}"

        # 94% near user's home city, 6% legitimate domestic travel to another city
        if rng.random() < 0.94:
            tx_lat = float(u_row["home_lat"]) + float(rng.normal(0, 0.03))
            tx_lon = float(u_row["home_lon"]) + float(rng.normal(0, 0.03))
            city_lbl = str(u_row["home_city_label"])
        else:
            travel_city = str(rng.choice(all_cities[:5]))
            tc_info = CITY_COORDINATES[travel_city]
            tx_lat = float(tc_info["lat"]) + float(rng.normal(0, 0.03))
            tx_lon = float(tc_info["lon"]) + float(rng.normal(0, 0.03))
            city_lbl = str(tc_info["name"])

        records.append(
            {
                "user_id": u_id,
                "timestamp": ts,
                "amount_idr": amt_idr,
                "merchant_category": cat,
                "device_id": dev_id,
                "ip_subnet": ip_id,
                "tx_lat": round(tx_lat, 5),
                "tx_lon": round(tx_lon, 5),
                "city_label": city_lbl,
                "payment_method": str(rng.choice(["wallet_balance", "qris_instant", "linked_card"], p=[0.50, 0.35, 0.15])),
                "is_fraud": 0,
                "fraud_pattern": "legitimate",
            }
        )

    # 2. Inject 3 Realistic Fraud Attack Patterns
    attack_types = ["account_takeover_ato", "velocity_card_testing", "syndicate_fraud_ring"]
    for f_idx in range(n_fraud_incidents):
        u_row = users_df.iloc[int(rng.integers(0, len(users_df)))]
        u_id = str(u_row["user_id"])
        attack = attack_types[f_idx % len(attack_types)]

        day_offset = int(rng.integers(0, simulation_days))

        if attack == "account_takeover_ato":
            # Happens mostly late night / early morning (00:00 - 05:00), from a distant city, high amount
            hour = int(rng.choice([0, 1, 2, 3, 4, 23]))
            ts = base_ts + day_offset * 86400 + hour * 3600 + int(rng.integers(60, 1800))

            # Pick a distant city different from user's home city
            distant_cities = [c for c in all_cities if c != u_row["home_city"]]
            chosen_city_key = str(rng.choice(distant_cities))
            c_info = CITY_COORDINATES[chosen_city_key]

            cat = str(rng.choice(["p2p_cash_transfer", "digital_vouchers_crypto", "electronics_gadgets"]))
            amt_idr = int(round(rng.uniform(4500000, 19500000) / 1000.0) * 1000)
            dev_id = f"dev_unknown_ato_{f_idx:04d}"
            ip_id = f"ip_foreign_ato_{f_idx // 3:04d}"

            records.append(
                {
                    "user_id": u_id,
                    "timestamp": ts,
                    "amount_idr": amt_idr,
                    "merchant_category": cat,
                    "device_id": dev_id,
                    "ip_subnet": ip_id,
                    "tx_lat": round(float(c_info["lat"]) + float(rng.normal(0, 0.02)), 5),
                    "tx_lon": round(float(c_info["lon"]) + float(rng.normal(0, 0.02)), 5),
                    "city_label": str(c_info["name"]),
                    "payment_method": "p2p_transfer" if cat == "p2p_cash_transfer" else "wallet_balance",
                    "is_fraud": 1,
                    "fraud_pattern": "account_takeover_ato",
                }
            )

        elif attack == "velocity_card_testing":
            # Rapid burst of 3 to 4 transactions within 3 minutes; 65% on stolen session token (same device_id!)
            hour = int(rng.integers(0, 24))
            burst_start_ts = base_ts + day_offset * 86400 + hour * 3600 + int(rng.integers(0, 3000))
            if rng.random() < 0.65:
                burst_dev = str(u_row["trusted_device_id"])
                burst_ip = f"ip_bot_roam_{f_idx:04d}"
            else:
                burst_dev = str(rng.choice(syndicate_devices))
                burst_ip = str(rng.choice(syndicate_ips))
            c_info = CITY_COORDINATES["singapore_vpn"] if rng.random() < 0.5 else CITY_COORDINATES[str(u_row["home_city"])]

            for b_step in range(3):
                # Starts with small probing amounts then escalates to a large cash-out
                if b_step < 2:
                    amt_idr = int(rng.choice([25000, 50000, 100000, 250000]))
                else:
                    amt_idr = int(round(rng.uniform(2500000, 9500000) / 1000.0) * 1000)

                records.append(
                    {
                        "user_id": u_id,
                        "timestamp": burst_start_ts + b_step * int(rng.integers(20, 55)),
                        "amount_idr": amt_idr,
                        "merchant_category": "digital_vouchers_crypto",
                        "device_id": burst_dev,
                        "ip_subnet": burst_ip,
                        "tx_lat": round(float(c_info["lat"]) + float(rng.normal(0, 0.01)), 5),
                        "tx_lon": round(float(c_info["lon"]) + float(rng.normal(0, 0.01)), 5),
                        "city_label": str(c_info["name"]),
                        "payment_method": "linked_card",
                        "is_fraud": 1,
                        "fraud_pattern": "velocity_card_testing",
                    }
                )

        else:  # syndicate_fraud_ring
            # Uses a device/IP from the shared syndicate ring across multiple accounts
            hour = int(rng.choice([1, 2, 3, 14, 15, 22, 23]))
            ts = base_ts + day_offset * 86400 + hour * 3600 + int(rng.integers(0, 3500))
            ring_dev = str(rng.choice(syndicate_devices))
            ring_ip = str(rng.choice(syndicate_ips))
            c_info = CITY_COORDINATES["medan"]

            records.append(
                {
                    "user_id": u_id,
                    "timestamp": ts,
                    "amount_idr": int(round(rng.uniform(1800000, 12500000) / 1000.0) * 1000),
                    "merchant_category": str(rng.choice(["electronics_gadgets", "digital_vouchers_crypto", "p2p_cash_transfer"])),
                    "device_id": ring_dev,
                    "ip_subnet": ring_ip,
                    "tx_lat": round(float(c_info["lat"]) + float(rng.normal(0, 0.02)), 5),
                    "tx_lon": round(float(c_info["lon"]) + float(rng.normal(0, 0.02)), 5),
                    "city_label": str(c_info["name"]),
                    "payment_method": "wallet_balance",
                    "is_fraud": 1,
                    "fraud_pattern": "syndicate_fraud_ring",
                }
            )

    df = pd.DataFrame(records).sort_values("timestamp").reset_index(drop=True)
    df.insert(0, "tx_id", [f"tx_{i:06d}" for i in range(len(df))])
    return df


def temporal_train_test_split(
    tx_df: pd.DataFrame,
    train_ratio: float = 0.80,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Split transaction stream strictly by timestamp (80% Train / 20% Test)."""
    sorted_df = tx_df.sort_values("timestamp").reset_index(drop=True)
    split_idx = int(len(sorted_df) * train_ratio)
    cutoff_ts = int(sorted_df.iloc[split_idx]["timestamp"])

    train_df = sorted_df[sorted_df["timestamp"] < cutoff_ts].copy().reset_index(drop=True)
    test_df = sorted_df[sorted_df["timestamp"] >= cutoff_ts].copy().reset_index(drop=True)
    return train_df, test_df, cutoff_ts


def build_and_save_fraud_dataset(
    output_dir: Path = DATA_DIR,
    n_users: int = 500,
    n_normal_sessions: int = 22000,
    n_fraud_incidents: int = 680,
    seed: int = 42,
) -> dict[str, Any]:
    """Run the full data generation and temporal split pipeline for Project 03."""
    output_dir.mkdir(parents=True, exist_ok=True)

    users_df = generate_users_table(n_users=n_users, seed=seed)
    tx_df = generate_transaction_stream(
        users_df=users_df,
        n_normal_sessions=n_normal_sessions,
        n_fraud_incidents=n_fraud_incidents,
        seed=seed,
    )
    train_df, test_df, cutoff_ts = temporal_train_test_split(tx_df, train_ratio=0.80)

    users_df.to_csv(output_dir / "users.csv", index=False)
    tx_df.to_csv(output_dir / "transactions.csv", index=False)
    train_df.to_csv(output_dir / "train_transactions.csv", index=False)
    test_df.to_csv(output_dir / "test_transactions.csv", index=False)

    fraud_rate = float(tx_df["is_fraud"].mean())
    summary = {
        "n_users_total": len(users_df),
        "n_transactions_total": len(tx_df),
        "n_train_transactions": len(train_df),
        "n_test_transactions": len(test_df),
        "overall_fraud_rate_pct": round(fraud_rate * 100.0, 2),
        "total_fraud_transactions": int(tx_df["is_fraud"].sum()),
        "total_legitimate_transactions": int((tx_df["is_fraud"] == 0).sum()),
        "temporal_cutoff_timestamp": cutoff_ts,
        "fraud_pattern_breakdown": tx_df["fraud_pattern"].value_counts().to_dict(),
        "merchant_category_breakdown": tx_df["merchant_category"].value_counts().to_dict(),
    }

    with open(output_dir / "dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    stats = build_and_save_fraud_dataset()
    print("=== Project 03 Stage 1: Fraud Dataset Pipeline Completed ===")
    print(json.dumps(stats, indent=2))
