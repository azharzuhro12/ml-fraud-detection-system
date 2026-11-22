"""Core configuration, merchant risk tiers, user profiles, and fraud attack scenarios for NusaPay Shield."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

# 3-Tier Risk Decision Thresholds (Risk Score 0 to 100)
RISK_TIER_THRESHOLDS: dict[str, int] = {
    "challenge_threshold": 38,  # 38..74 -> Step-Up OTP / Biometric Challenge
    "block_threshold": 75,      # 75..100 -> Automatic Block & Alert
}

# Indonesian City Coordinates for Geospatial & Impossible Travel Detection
CITY_COORDINATES: dict[str, dict[str, float | str]] = {
    "jakarta": {"name": "Jakarta", "lat": -6.2088, "lon": 106.8456},
    "bandung": {"name": "Bandung", "lat": -6.9175, "lon": 107.6191},
    "surabaya": {"name": "Surabaya", "lat": -7.2575, "lon": 112.7521},
    "medan": {"name": "Medan", "lat": 3.5952, "lon": 98.6722},
    "denpasar": {"name": "Denpasar (Bali)", "lat": -8.6705, "lon": 115.2126},
    "singapore_vpn": {"name": "Singapore (Foreign IP)", "lat": 1.3521, "lon": 103.8198},
}

# Merchant Categories and their Baseline Fraud Risk Multipliers
MERCHANT_CATEGORIES: dict[str, dict[str, float | str]] = {
    "food_beverage": {
        "label": "Makanan & Kedai Kopi (QRIS)",
        "base_risk": 0.08,
        "typical_amount_range": (15000, 180000),
    },
    "groceries_retail": {
        "label": "Supermarket & Kebutuhan Harian",
        "base_risk": 0.10,
        "typical_amount_range": (45000, 650000),
    },
    "fashion_marketplace": {
        "label": "Belanja Fashion & E-Commerce",
        "base_risk": 0.22,
        "typical_amount_range": (85000, 1800000),
    },
    "electronics_gadgets": {
        "label": "Elektronik, HP & Laptop",
        "base_risk": 0.55,
        "typical_amount_range": (500000, 18500000),
    },
    "digital_vouchers_crypto": {
        "label": "Voucher Game, Pulsa & Aset Digital",
        "base_risk": 0.82,
        "typical_amount_range": (50000, 12000000),
    },
    "p2p_cash_transfer": {
        "label": "Transfer Saldo Antar Akun / Tarik Tunai",
        "base_risk": 0.68,
        "typical_amount_range": (100000, 20000000),
    },
}

# Representative User Personas for Interactive Wallet Testing
WALLET_USER_PERSONAS: dict[str, dict] = {
    "karyawan_jakarta": {
        "display_name": "Budi Santoso (Karyawan Swasta - Jakarta)",
        "home_city": "jakarta",
        "typical_spend_mean_idr": 125000,
        "typical_spend_std_idr": 85000,
        "wallet_balance_idr": 18500000,
        "account_age_days": 640,
        "primary_categories": ["food_beverage", "groceries_retail", "fashion_marketplace"],
    },
    "mahasiswa_bandung": {
        "display_name": "Putri Lestari (Mahasiswi - Bandung)",
        "home_city": "bandung",
        "typical_spend_mean_idr": 55000,
        "typical_spend_std_idr": 35000,
        "wallet_balance_idr": 4200000,
        "account_age_days": 280,
        "primary_categories": ["food_beverage", "digital_vouchers_crypto", "fashion_marketplace"],
    },
    "pengusaha_surabaya": {
        "display_name": "Hendra Wijaya (Pengusaha - Surabaya)",
        "home_city": "surabaya",
        "typical_spend_mean_idr": 850000,
        "typical_spend_std_idr": 550000,
        "wallet_balance_idr": 65000000,
        "account_age_days": 1120,
        "primary_categories": ["electronics_gadgets", "p2p_cash_transfer", "food_beverage"],
    },
    "freelancer_bali": {
        "display_name": "Kevin Pratama (Desainer Freelance - Bali)",
        "home_city": "denpasar",
        "typical_spend_mean_idr": 210000,
        "typical_spend_std_idr": 140000,
        "wallet_balance_idr": 24000000,
        "account_age_days": 490,
        "primary_categories": ["food_beverage", "electronics_gadgets", "groceries_retail"],
    },
}

