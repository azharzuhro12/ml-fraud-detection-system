"""
FastAPI Microservice for NusaPay & Shield: Real-Time Digital Wallet & Payment Fraud Engine.

Endpoints:
- GET  /                    : Authentic Fintech Digital Wallet & Fraud Shield Web App
- GET  /health              : Service readiness check
- GET  /users               : 4 representative wallet user accounts with live balance & security stats
- POST /assess              : Evaluate an incoming transaction in real time (< 5 ms)
- POST /simulate/{scenario} : 1-Click realistic transaction & attack scenario simulator
- POST /verify-otp/{tx_id}  : Complete a Step-Up OTP challenge to approve a medium-risk legitimate payment
- POST /reset/{user_id}     : Reset user's live wallet balance & sliding window history
- GET  /metrics             : Offline benchmark evaluation metrics (PR-AUC, Recall@1%FPR, Net Savings IDR)
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, AsyncIterator, Literal

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from src.config import ARTIFACTS_DIR, CITY_COORDINATES, DATA_DIR, MERCHANT_CATEGORIES
from src.data.generator import build_and_save_fraud_dataset
from src.models.detector import HybridFraudDetector, evaluate_fraud_detection_pipeline

STATIC_DIR = Path(__file__).resolve().parent / "static"


class PaymentAssessmentRequest(BaseModel):
    """Payload for evaluating a custom payment transaction."""

    user_id: str = Field(default="user_0000")
    amount_idr: int = Field(..., ge=1000, le=100_000_000)
    merchant_category: str = Field(default="food_beverage")
    location_city: str = Field(default="jakarta")
    device_mode: Literal["trusted_owner_device", "new_unknown_device", "syndicate_ring_device"] = Field(
        default="trusted_owner_device"
    )
    time_mode: Literal["normal_daytime", "odd_hour_02am"] = Field(default="normal_daytime")
    payment_method: str = Field(default="qris_instant")
    merchant_name: str = Field(default="Merchant Pilihan")


class FraudServiceState:
    """Holds the trained HybridFraudDetector, user wallet balances, and live transaction feed."""

    def __init__(self) -> None:
        self.detector: HybridFraudDetector | None = None
        self.users_df: pd.DataFrame | None = None
        self.stage3_metrics: dict[str, Any] = {}
        self.sim_clock: int = 1_715_000_000
        self.tx_counter: int = 1
        self.live_balances: dict[str, int] = {}
        self.saved_fraud_totals: dict[str, int] = {}
        self.live_history: dict[str, list[dict[str, Any]]] = {}

    def initialize(self) -> None:
        if not (DATA_DIR / "train_transactions.csv").exists():
            build_and_save_fraud_dataset()

        self.users_df = pd.read_csv(DATA_DIR / "users.csv")
        train_df = pd.read_csv(DATA_DIR / "train_transactions.csv")
        test_df = pd.read_csv(DATA_DIR / "test_transactions.csv")

        detector_path = ARTIFACTS_DIR / "hybrid_fraud_detector.pkl"
        metrics_path = ARTIFACTS_DIR / "stage3_fraud_metrics.json"

        if not detector_path.exists() or not metrics_path.exists():
            evaluate_fraud_detection_pipeline(self.users_df, train_df, test_df)

        self.detector = HybridFraudDetector.load(detector_path)
        with open(metrics_path, encoding="utf-8") as f:
            self.stage3_metrics = json.load(f)

        # Ensure syndicate devices have >= 5 linked accounts in the entity graph for live demonstrations
        for ring_idx in range(5):
            r_dev = f"dev_syndicate_ring_{ring_idx:02d}"
            r_ip = f"ip_proxy_ring_{ring_idx:02d}"
            self.detector.feature_engine_.device_users_graph_[r_dev] = {
                f"user_{u:04d}" for u in range(400, 407)
            }
            self.detector.feature_engine_.ip_users_graph_[r_ip] = {
                f"user_{u:04d}" for u in range(400, 409)
            }

        self.live_balances = {}
        self.saved_fraud_totals = {}
        self.live_history = {}
        for row in self.users_df.head(12).itertuples():
            u_id = str(row.user_id)
            self.live_balances[u_id] = int(row.wallet_balance_idr)
            self.saved_fraud_totals[u_id] = 0
            self.live_history[u_id] = []


state = FraudServiceState()


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    state.initialize()
    yield


app = FastAPI(
    title="NusaPay & Shield: Real-Time Fraud Detection API",
    description="4-Layer Payment Fraud Risk Engine (Streaming Velocity & Entity Graph -> Guardrails -> Hybrid LightGBM + Isolation Forest -> 3-Tier Decision).",
    version="1.0.0",
    lifespan=lifespan,
)


def _ensure_ready() -> None:
    if state.detector is None or state.users_df is None:
        state.initialize()


@app.get("/health")
def health_check() -> dict[str, Any]:
    _ensure_ready()
    assert state.users_df is not None
    return {
        "status": "ready",
        "total_users": len(state.users_df),
    }


@app.get("/users")
def list_wallet_users() -> dict[str, Any]:
    """Return the 4 featured digital wallet user personas with live wallet balances."""
    _ensure_ready()
    assert state.detector is not None and state.users_df is not None

    out: list[dict[str, Any]] = []
    for row in state.users_df.head(4).itertuples():
        u_id = str(row.user_id)
        out.append(
            {
                "user_id": u_id,
                "persona": str(row.persona),
                "display_name": str(row.display_name),
                "home_city": str(row.home_city),
                "home_city_label": str(row.home_city_label),
                "typical_spend_mean_idr": int(row.typical_spend_mean_idr),
                "wallet_balance_idr": state.live_balances.get(u_id, int(row.wallet_balance_idr)),
                "saved_fraud_total_idr": state.saved_fraud_totals.get(u_id, 0),
                "trusted_device_id": str(row.trusted_device_id),
                "account_age_days": int(row.account_age_days),
                "recent_transactions": state.live_history.get(u_id, [])[:8],
            }
        )
    return {"users": out}


def _execute_and_record_tx(
    user_id: str,
    amount_idr: int,
    merchant_category: str,
    merchant_name: str,
    city_key: str,
    device_id: str,
    ip_subnet: str,
    timestamp: int,
    payment_method: str,
    scenario_title: str,
) -> dict[str, Any]:
    assert state.detector is not None
    u_prof = state.detector.feature_engine_.user_profiles_.get(user_id)
    if u_prof is None:
        raise HTTPException(status_code=404, detail=f"User '{user_id}' not found.")

    c_info = CITY_COORDINATES.get(city_key, CITY_COORDINATES[u_prof["home_city"]])

    tx_event = {
        "user_id": user_id,
        "timestamp": timestamp,
        "amount_idr": amount_idr,
        "merchant_category": merchant_category,
        "device_id": device_id,
        "ip_subnet": ip_subnet,
        "tx_lat": float(c_info["lat"]),
        "tx_lon": float(c_info["lon"]),
        "payment_method": payment_method,
    }

    # Only update user's trusted location history if transaction is NOT immediately blocked
    assessment = state.detector.assess_single_transaction(tx_event, update_state=False)
    if assessment.decision != "BLOCK":
        state.detector.feature_engine_.extract_features_for_event(tx_event, update_state=True)

    tx_id = f"NP-{state.tx_counter:05d}"
    state.tx_counter += 1

    curr_bal = state.live_balances.get(user_id, int(u_prof["wallet_balance_idr"]))
    if assessment.decision == "APPROVE":
        curr_bal = max(0, curr_bal - amount_idr)
        state.live_balances[user_id] = curr_bal
    elif assessment.decision == "BLOCK":
        state.saved_fraud_totals[user_id] = state.saved_fraud_totals.get(user_id, 0) + amount_idr

    cat_label = str(MERCHANT_CATEGORIES.get(merchant_category, {}).get("label", merchant_category))

    tx_record = {
        "tx_id": tx_id,
        "user_id": user_id,
        "scenario_title": scenario_title,
        "merchant_name": merchant_name,
        "merchant_category": merchant_category,
        "merchant_category_label": cat_label,
        "amount_idr": amount_idr,
        "city_label": str(c_info["name"]),
        "device_id": device_id,
        "payment_method": payment_method,
        "risk_score": assessment.risk_score_0_100,
        "decision": assessment.decision,
        "decision_label_id": assessment.decision_label_id,
        "decision_summary_id": assessment.decision_summary_id,
        "risk_factors": assessment.risk_factors,
        "safety_badges": assessment.safety_badges,
        "triggered_guardrail_rule": assessment.triggered_guardrail_rule,
        "latency_ms": assessment.latency_ms,
        "wallet_balance_after_idr": curr_bal,
        "saved_fraud_total_idr": state.saved_fraud_totals.get(user_id, 0),
        "otp_verified": False,
        "assessment_detail": asdict(assessment),
    }

    hist = state.live_history.setdefault(user_id, [])
    hist.insert(0, tx_record)
    state.live_history[user_id] = hist[:15]
    return tx_record


@app.post("/assess")
def assess_custom_payment(payload: PaymentAssessmentRequest) -> dict[str, Any]:
    """Evaluate a custom payment submitted from the NusaPay transfer/QRIS form."""
    _ensure_ready()
    assert state.detector is not None

    u_prof = state.detector.feature_engine_.user_profiles_.get(payload.user_id)
    if u_prof is None:
        raise HTTPException(status_code=404, detail=f"User '{payload.user_id}' not found.")

    state.sim_clock += 180
    # Adjust timestamp hour to daytime (13:00) or odd hour (02:30)
    day_base = (state.sim_clock // 86400) * 86400
    ts = day_base + (2 * 3600 + 1800 if payload.time_mode == "odd_hour_02am" else 13 * 3600 + (state.sim_clock % 1800))

    if payload.device_mode == "trusted_owner_device":
        dev_id = u_prof["trusted_device_id"]
        ip_sub = u_prof["trusted_ip_subnet"]
    elif payload.device_mode == "new_unknown_device":
        dev_id = f"dev_unknown_browser_{state.tx_counter:03d}"
        ip_sub = f"ip_roaming_unknown_{state.tx_counter:03d}"
    else:
        dev_id = "dev_syndicate_ring_01"
        ip_sub = "ip_proxy_ring_01"

    return _execute_and_record_tx(
        user_id=payload.user_id,
        amount_idr=payload.amount_idr,
        merchant_category=payload.merchant_category,
        merchant_name=payload.merchant_name,
        city_key=payload.location_city,
        device_id=dev_id,
        ip_subnet=ip_sub,
        timestamp=ts,
        payment_method=payload.payment_method,
        scenario_title="Transaksi Kustom Pengguna",
    )


@app.post("/simulate/{scenario_name}")
def simulate_preset_scenario(
    scenario_name: Literal["normal_qris", "travel_splurge_otp", "account_takeover_attack", "syndicate_velocity_bot"],
    user_id: str = "user_0000",
) -> dict[str, Any]:
    """
    1-Click realistic scenario simulator so anyone can test all 3 decision tiers immediately:
    1. normal_qris             -> Score ~4  (APPROVE INSTANTLY)
    2. travel_splurge_otp      -> Score ~52 (CHALLENGE_OTP: Legitimate user buying a high-ticket item out of town)
    3. account_takeover_attack -> Score ~98 (BLOCK: Attacker in Singapore at 02:30 AM trying to drain Rp 14.5M)
    4. syndicate_velocity_bot  -> Score ~96 (BLOCK: Rapid 4-tx card-testing burst from a syndicate ring device)
    """
    _ensure_ready()
    assert state.detector is not None

    u_prof = state.detector.feature_engine_.user_profiles_.get(user_id)
    if u_prof is None:
        raise HTTPException(status_code=404, detail=f"User '{user_id}' not found.")

    state.sim_clock += 300
    day_base = (state.sim_clock // 86400) * 86400
    daytime_ts = day_base + 14 * 3600 + (state.sim_clock % 1200)
    home_city = u_prof["home_city"]
    t_dev = u_prof["trusted_device_id"]
    t_ip = u_prof["trusted_ip_subnet"]
    u_mean = int(u_prof["typical_spend_mean_idr"])

    if scenario_name == "normal_qris":
        return _execute_and_record_tx(
            user_id=user_id,
            amount_idr=max(35000, int(round(u_mean * 0.45 / 1000.0) * 1000)),
            merchant_category="food_beverage",
            merchant_name="Kopi Nusantara & Bakery (QRIS)",
            city_key=home_city,
            device_id=t_dev,
            ip_subnet=t_ip,
            timestamp=daytime_ts,
            payment_method="qris_instant",
            scenario_title="Pembayaran QRIS Harian (Normal)",
        )

    if scenario_name == "travel_splurge_otp":
        # User is using their OWN trusted phone, during daytime, but buying an item 4.2x their normal average
        return _execute_and_record_tx(
            user_id=user_id,
            amount_idr=max(550000, int(round(u_mean * 4.2 / 5000.0) * 5000)),
            merchant_category="fashion_marketplace",
            merchant_name="Grand Mall Official Store",
            city_key=home_city,
            device_id=t_dev,
            ip_subnet=t_ip,
            timestamp=daytime_ts + 120,
            payment_method="wallet_balance",
            scenario_title="Belanja Nominal Lebih Besar dari Biasanya (Perlu OTP)",
        )

    if scenario_name == "account_takeover_attack":
        # First ensure a recent home-city presence 6 minutes ago so Impossible Travel triggers clearly
        c_home = CITY_COORDINATES[home_city]
        state.detector.feature_engine_.user_last_tx_[user_id] = (
            daytime_ts,
            float(c_home["lat"]),
            float(c_home["lon"]),
        )
        return _execute_and_record_tx(
            user_id=user_id,
            amount_idr=min(16500000, max(6500000, int(u_prof["wallet_balance_idr"] * 0.82))),
            merchant_category="p2p_cash_transfer",
            merchant_name="Transfer P2P ke Rekening Penampung Asing",
            city_key="singapore_vpn",
            device_id="dev_hacker_emulator_x99",
            ip_subnet="ip_foreign_vpn_sg99",
            timestamp=daytime_ts + 360,  # Only 6 minutes after being in Indonesia!
            payment_method="p2p_transfer",
            scenario_title="Serangan Pembajakan Akun (Account Takeover)",
        )

    # scenario_name == "syndicate_velocity_bot"
    # Pre-seed 3 rapid probing transactions within the last 90 seconds in the user's sliding window
    c_home = CITY_COORDINATES[home_city]
    win = state.detector.feature_engine_.user_recent_window_.setdefault(user_id, state.detector.feature_engine_.user_recent_window_[user_id])
    for step in range(3):
        win.append((daytime_ts + step * 20, 25000.0, float(c_home["lat"]), float(c_home["lon"])))

    return _execute_and_record_tx(
        user_id=user_id,
        amount_idr=4800000,
        merchant_category="digital_vouchers_crypto",
        merchant_name="Global Crypto & Game Voucher Instant",
        city_key="medan",
        device_id="dev_syndicate_ring_01",
        ip_subnet="ip_proxy_ring_01",
        timestamp=daytime_ts + 80,
        payment_method="linked_card",
        scenario_title="Serangan Bot Sindikat (Card Testing & Fraud Ring)",
    )


@app.post("/verify-otp/{tx_id}")
def verify_step_up_otp(tx_id: str, user_id: str = "user_0000") -> dict[str, Any]:
    """Allow the user to complete a Step-Up OTP challenge on a CHALLENGE_OTP transaction."""
    _ensure_ready()
    hist = state.live_history.get(user_id, [])
    for rec in hist:
        if rec["tx_id"] == tx_id:
            if rec["decision"] == "CHALLENGE_OTP" and not rec["otp_verified"]:
                rec["otp_verified"] = True
                rec["decision"] = "APPROVE"
                rec["decision_label_id"] = "DISETUJUI SETELAH VERIFIKASI OTP"
                rec["decision_summary_id"] = (
                    "Identitas pemilik akun berhasil dikonfirmasi melalui kode OTP. Pembayaran telah diselesaikan."
                )
                curr_bal = state.live_balances.get(user_id, 0)
                curr_bal = max(0, curr_bal - int(rec["amount_idr"]))
                state.live_balances[user_id] = curr_bal
                rec["wallet_balance_after_idr"] = curr_bal
            return rec
    raise HTTPException(status_code=404, detail=f"Transaction '{tx_id}' not found.")


@app.post("/reset/{user_id}")
def reset_user_wallet_state(user_id: str) -> dict[str, Any]:
    """Reset a user's live wallet balance, saved fraud counter, and sliding window history."""
    _ensure_ready()
    assert state.detector is not None and state.users_df is not None

    u_prof = state.detector.feature_engine_.user_profiles_.get(user_id)
    if u_prof is None:
        raise HTTPException(status_code=404, detail=f"User '{user_id}' not found.")

    state.live_balances[user_id] = int(u_prof["wallet_balance_idr"])
    state.saved_fraud_totals[user_id] = 0
    state.live_history[user_id] = []
    state.detector.feature_engine_.user_recent_window_[user_id].clear()
    state.detector.feature_engine_.user_last_tx_.pop(user_id, None)

    return {
        "status": "reset",
        "user_id": user_id,
        "wallet_balance_idr": state.live_balances[user_id],
        "saved_fraud_total_idr": 0,
    }


@app.get("/metrics")
def get_fraud_benchmark_metrics() -> dict[str, Any]:
    """Return offline evaluation metrics (PR-AUC, ROC-AUC, Recall@1%FPR, Net Financial Savings IDR)."""
    _ensure_ready()
    return state.stage3_metrics


@app.get("/", response_class=HTMLResponse)
def serve_nusapay_ui() -> HTMLResponse:
    """Serve the NusaPay & Shield digital wallet web application."""
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=500, detail="NusaPay index.html not found.")
    return HTMLResponse(content=index_path.read_text(encoding="utf-8"))

