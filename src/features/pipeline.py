"""
Layer 1: Streaming Velocity, Geospatial Impossible Travel, & Device/IP Graph Feature Engine.

Inspired by:
- PayPal (#193 & #233): Real-time sliding window velocity counters & Device/IP entity graph degrees
- Stripe (#1 & #273): Behavioral spend deviation & similarity clustering for fraud rings
- Lyft (#295) & Uber (#54): Impossible travel speed (km/h) & device fingerprint trust signals

Crucial Engineering Principle:
All features are computed strictly causally (using only transactions at t' < t) so there is
ZERO future information leakage during historical batch feature extraction or live streaming inference.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
import pandas as pd

from src.config import MERCHANT_CATEGORIES
from src.data.generator import compute_haversine_km


FRAUD_FEATURE_COLUMNS: list[str] = [
    # 1. Transaction & Behavioral Spend Deviation (5 features)
    "tx_amount_idr",
    "tx_amount_log",
    "amount_to_user_mean_ratio",
    "amount_user_zscore",
    "amount_to_wallet_balance_ratio",
    # 2. Sliding Window Velocity Counters (5 features)
    "velocity_tx_count_10m",
    "velocity_tx_count_1h",
    "velocity_amount_sum_1h_idr",
    "velocity_amount_to_mean_1h_ratio",
    "seconds_since_prev_user_tx",
    # 3. Geospatial & Impossible Travel Signals (3 features)
    "geo_dist_from_home_km",
    "geo_dist_from_prev_tx_km",
    "geo_travel_speed_kmh",
    # 4. Device & IP Entity Graph Signals (4 features)
    "device_is_untrusted_for_user",
    "ip_is_untrusted_for_user",
    "graph_device_unique_users_count",
    "graph_ip_unique_users_count",
    # 5. Merchant, Payment Method & Temporal Risk Signals (3 features)
    "merchant_category_base_risk",
    "is_high_risk_payment_method",
    "temporal_is_odd_hour_00_05",
]


class StreamingFraudFeatureEngine:
    """
    Maintains causal in-memory state (sliding time windows per user, device->users graph sets,
    ip->users graph sets, and user last location) to extract 20 real-time fraud features in < 1 ms.
    """

    def __init__(self) -> None:
        self.user_profiles_: dict[str, dict[str, Any]] = {}
        # device_id -> set of unique user_ids seen on this device so far
        self.device_users_graph_: dict[str, set[str]] = {}
        # ip_subnet -> set of unique user_ids seen on this IP subnet so far
        self.ip_users_graph_: dict[str, set[str]] = {}
        # user_id -> deque of recent (timestamp, amount_idr, lat, lon) within last 1 hour (3600s)
        self.user_recent_window_: dict[str, deque[tuple[int, float, float, float]]] = {}
        # user_id -> last transaction (timestamp, lat, lon)
        self.user_last_tx_: dict[str, tuple[int, float, float]] = {}

    def initialize_users(self, users_df: pd.DataFrame) -> StreamingFraudFeatureEngine:
        """Load static/historical baseline user profiles and register their initial trusted device/IP."""
        self.user_profiles_ = {}
        self.device_users_graph_ = {}
        self.ip_users_graph_ = {}
        self.user_recent_window_ = {}
        self.user_last_tx_ = {}

        for row in users_df.itertuples():
            u_id = str(row.user_id)
            t_dev = str(row.trusted_device_id)
            t_ip = str(row.trusted_ip_subnet)

            self.user_profiles_[u_id] = {
                "user_id": u_id,
                "persona": str(row.persona),
                "display_name": str(row.display_name),
                "home_city": str(row.home_city),
                "home_city_label": str(row.home_city_label),
                "home_lat": float(row.home_lat),
                "home_lon": float(row.home_lon),
                "trusted_device_id": t_dev,
                "trusted_ip_subnet": t_ip,
                "typical_spend_mean_idr": float(row.typical_spend_mean_idr),
                "typical_spend_std_idr": max(10000.0, float(row.typical_spend_std_idr)),
                "wallet_balance_idr": max(100000.0, float(row.wallet_balance_idr)),
                "account_age_days": int(row.account_age_days),
                "primary_cat_1": str(row.primary_cat_1),
                "primary_cat_2": str(row.primary_cat_2),
            }
            self.device_users_graph_.setdefault(t_dev, set()).add(u_id)
            self.ip_users_graph_.setdefault(t_ip, set()).add(u_id)
            self.user_recent_window_[u_id] = deque()

        return self

    def extract_features_for_event(
        self,
        tx_event: dict[str, Any],
        update_state: bool = True,
    ) -> dict[str, float]:
        """
        Compute the 20 causal risk features for a single incoming transaction event.
        If update_state=True, updates the sliding window and entity graph with this event.
        """
        u_id = str(tx_event["user_id"])
        u_prof = self.user_profiles_.get(u_id)
        if u_prof is None:
            u_prof = next(iter(self.user_profiles_.values()))

        ts = int(tx_event["timestamp"])
        amt = float(tx_event["amount_idr"])
        cat = str(tx_event["merchant_category"])
        dev_id = str(tx_event["device_id"])
        ip_sub = str(tx_event["ip_subnet"])
        tx_lat = float(tx_event["tx_lat"])
        tx_lon = float(tx_event["tx_lon"])
        pay_method = str(tx_event.get("payment_method", "wallet_balance"))

        # 1. Spend deviation features
        u_mean = u_prof["typical_spend_mean_idr"]
        u_std = u_prof["typical_spend_std_idr"]
        u_bal = u_prof["wallet_balance_idr"]

        amt_to_mean = amt / max(1000.0, u_mean)
        amt_zscore = (amt - u_mean) / u_std
        amt_to_bal = amt / u_bal

        # 2. Sliding window velocity features (strictly prior events within 10m = 600s and 1h = 3600s)
        win = self.user_recent_window_.setdefault(u_id, deque())
        # Evict events older than 1 hour relative to current timestamp
        while win and (ts - win[0][0]) > 3600:
            win.popleft()

        count_1h = len(win)
        sum_1h = sum(e[1] for e in win)
        count_10m = sum(1 for e in win if (ts - e[0]) <= 600)

        # Include current transaction in the 1h cumulative spend ratio
        amt_sum_1h_with_curr = sum_1h + amt
        vel_amt_ratio = amt_sum_1h_with_curr / max(1000.0, u_mean)

        # 3. Geospatial & Impossible Travel features
        dist_home_km = float(
            compute_haversine_km(u_prof["home_lat"], u_prof["home_lon"], tx_lat, tx_lon)
        )

        last_tx = self.user_last_tx_.get(u_id)
        if last_tx is not None:
            prev_ts, prev_lat, prev_lon = last_tx
            sec_since_prev = float(max(1, ts - prev_ts))
            dist_prev_km = float(compute_haversine_km(prev_lat, prev_lon, tx_lat, tx_lon))
            # Travel speed in km/h (floor elapsed time at 3 minutes = 0.05h to avoid micro-jitter division)
            hours_elapsed = max(sec_since_prev / 3600.0, 0.05)
            # Only flag travel speed when displacement is meaningful (> 15 km)
            travel_speed_kmh = (dist_prev_km / hours_elapsed) if dist_prev_km > 15.0 else 0.0
        else:
            sec_since_prev = 86400.0
            dist_prev_km = dist_home_km
            travel_speed_kmh = (dist_home_km / 0.25) if dist_home_km > 50.0 else 0.0

        # 4. Device & IP Entity Graph features
        is_untrusted_dev = 0.0 if (dev_id == u_prof["trusted_device_id"] or dev_id == f"dev_secondary_{u_id}") else 1.0
        is_untrusted_ip = 0.0 if ip_sub == u_prof["trusted_ip_subnet"] else 1.0

        dev_users_set = self.device_users_graph_.get(dev_id, set())
        ip_users_set = self.ip_users_graph_.get(ip_sub, set())

        # Count how many unique users (including current user) are linked to this device/IP
        dev_unique_users = float(len(dev_users_set | {u_id}))
        ip_unique_users = float(len(ip_users_set | {u_id}))

        # 5. Merchant, Payment Method & Temporal Odd-Hour risk
        m_cfg = MERCHANT_CATEGORIES.get(cat, {"base_risk": 0.25})
        m_base_risk = float(m_cfg["base_risk"])
        high_risk_method = 1.0 if pay_method in ("p2p_transfer", "linked_card") else 0.0

        hour_of_day = (ts % 86400) // 3600
        is_odd_hour = 1.0 if (0 <= hour_of_day <= 5) else 0.0

        if update_state:
            win.append((ts, amt, tx_lat, tx_lon))
            self.user_last_tx_[u_id] = (ts, tx_lat, tx_lon)
            self.device_users_graph_.setdefault(dev_id, set()).add(u_id)
            self.ip_users_graph_.setdefault(ip_sub, set()).add(u_id)

        return {
            "tx_amount_idr": amt,
            "tx_amount_log": float(np.log1p(amt)),
            "amount_to_user_mean_ratio": round(amt_to_mean, 4),
            "amount_user_zscore": round(amt_zscore, 4),
            "amount_to_wallet_balance_ratio": round(amt_to_bal, 4),
            "velocity_tx_count_10m": float(count_10m),
            "velocity_tx_count_1h": float(count_1h),
            "velocity_amount_sum_1h_idr": float(sum_1h),
            "velocity_amount_to_mean_1h_ratio": round(vel_amt_ratio, 4),
            "seconds_since_prev_user_tx": float(min(sec_since_prev, 86400.0)),
            "geo_dist_from_home_km": round(dist_home_km, 3),
            "geo_dist_from_prev_tx_km": round(dist_prev_km, 3),
            "geo_travel_speed_kmh": round(min(travel_speed_kmh, 15000.0), 2),
            "device_is_untrusted_for_user": is_untrusted_dev,
            "ip_is_untrusted_for_user": is_untrusted_ip,
            "graph_device_unique_users_count": dev_unique_users,
            "graph_ip_unique_users_count": ip_unique_users,
            "merchant_category_base_risk": m_base_risk,
            "is_high_risk_payment_method": high_risk_method,
            "temporal_is_odd_hour_00_05": is_odd_hour,
        }

    def transform_stream_dataframe(self, tx_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process a chronological DataFrame of transactions sequentially, returning the
        20-column feature matrix aligned with tx_df.
        """
        feature_rows: list[dict[str, float]] = []
        for row in tx_df.itertuples():
            evt = {
                "user_id": row.user_id,
                "timestamp": row.timestamp,
                "amount_idr": row.amount_idr,
                "merchant_category": row.merchant_category,
                "device_id": row.device_id,
                "ip_subnet": row.ip_subnet,
                "tx_lat": row.tx_lat,
                "tx_lon": row.tx_lon,
                "payment_method": row.payment_method,
            }
            feature_rows.append(self.extract_features_for_event(evt, update_state=True))

        return pd.DataFrame(feature_rows, columns=FRAUD_FEATURE_COLUMNS)

