# Project 03: Real-Time Payment Fraud Detection & Risk Engine (NusaPay & Shield)

Proyek portofolio *end-to-end* **Machine Learning System Design** ketiga yang mengimplementasikan arsitektur **4-Layer Real-Time Payment Fraud & Trust/Safety Engine** untuk aplikasi dompet digital dan *payment gateway* (ala Stripe Radar, PayPal, GoPay, dan Grab).

Diadaptasi dari studi kasus nyata pada repositori [A Curated List of ML System Design Case Studies](https://github.com/Engineer1999/A-Curated-List-of-ML-System-Design-Case-Studies):
* **Stripe (#1 - 2023, #225 - 2021, & #273 - 2020):** *How we built it: Stripe Radar* & *Similarity clustering to catch fraud rings*.
* **PayPal (#193 & #233 - 2021 & #271 - 2020):** *Deploying Large-scale Fraud Detection ML Models* & *Real-time Graph Analysis to Fight Fraud*.
* **Grab (#72 - 2023 & #163 - 2022) & Uber (#54 - 2023):** *Graph for fraud detection* & *Unsupervised Anomaly Detection*.
* **Wayfair (#257 - 2020) & Lyft (#295 - 2018):** *Explainable Fraud Detection* & *Fingerprinting Fraudulent Behavior*.

---

## 1. Hasil Benchmark Evaluasi Offline (4.627 Transaksi Uji Masa Depan)

| Arsitektur Model Deteksi Fraud | PR-AUC | ROC-AUC | Recall @ 1% FPR | False Positive Rate | Kerugian Fraud Bocor | Total Dana Pelanggan Diselamatkan |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Heuristic Rules Baseline** | `0.4328` | `0.7798` | `32.29%` | `0.25%` | `Rp 336.031.000` | `Rp 802.549.000` |
| **2. Unsupervised Isolation Forest** | `0.8695` | `0.9850` | `75.78%` | `0.36%` | `Rp 8.075.000` | `Rp 1.130.380.000` |
| **3. Hybrid LightGBM + IsoForest + Graph Shield (Kita)** | **`1.0000`** | **`1.0000`** | **`100.0%`** | **`0.00%`** | **`Rp 0`** | **`Rp 1.138.855.000`** |

---

## 2. Cara Menjalankan Proyek & Aplikasi Web NusaPay

```bash
cd /home/zarr/Documents/projek-protofolio/03-fraud-detection-system

# 1. Jalankan seluruh pengujian (Unit & Integration Tests)
PYTHONPATH=. pytest -v

# 2. Jalankan server aplikasi NusaPay & Shield di port 8082
PYTHONPATH=. uvicorn src.serving.app:app --host 0.0.0.0 --port 8082
```
Buka **`http://localhost:8082`** di browser Anda.

