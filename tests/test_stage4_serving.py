"""Integration tests for Project 03 Stage 4: NusaPay & Shield Fraud Detection Microservice & UI."""

from fastapi.testclient import TestClient

from src.serving.app import app

client = TestClient(app)


def test_health_users_and_metrics_endpoints() -> None:
    res_health = client.get("/health")
    assert res_health.status_code == 200
    assert res_health.json()["total_users"] == 500

    res_users = client.get("/users")
    assert res_users.status_code == 200
    assert len(res_users.json()["users"]) == 4

    res_metrics = client.get("/metrics")
    assert res_metrics.status_code == 200
    assert "model_comparison" in res_metrics.json()


def test_four_preset_scenarios_and_step_up_otp_flow() -> None:
    client.post("/reset/user_0000")

    # 1. Normal QRIS payment -> APPROVE
    res_norm = client.post("/simulate/normal_qris?user_id=user_0000")
    assert res_norm.status_code == 200
    assert res_norm.json()["decision"] == "APPROVE"

    # 2. Travel / High-ticket splurge on trusted phone -> CHALLENGE_OTP -> Verify OTP -> APPROVE
    res_otp = client.post("/simulate/travel_splurge_otp?user_id=user_0000")
    assert res_otp.status_code == 200
    otp_tx = res_otp.json()
    assert otp_tx["decision"] == "CHALLENGE_OTP"

    res_verified = client.post(f"/verify-otp/{otp_tx['tx_id']}?user_id=user_0000")
    assert res_verified.status_code == 200
    assert res_verified.json()["decision"] == "APPROVE"
    assert res_verified.json()["otp_verified"] is True

    # 3. Account Takeover (Impossible Travel from Singapore) -> BLOCK & increments saved_fraud_total_idr
    res_ato = client.post("/simulate/account_takeover_attack?user_id=user_0000")
    assert res_ato.status_code == 200
    ato_data = res_ato.json()
    assert ato_data["decision"] == "BLOCK"
    assert ato_data["risk_score"] >= 75
    assert ato_data["saved_fraud_total_idr"] >= ato_data["amount_idr"]

    # 4. Syndicate Velocity Card Testing Bot -> BLOCK
    res_bot = client.post("/simulate/syndicate_velocity_bot?user_id=user_0000")
    assert res_bot.status_code == 200
    assert res_bot.json()["decision"] == "BLOCK"

    res_ui = client.get("/")
    assert res_ui.status_code == 200
    assert "NusaPay" in res_ui.text

