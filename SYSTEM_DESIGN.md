# Production ML System Design: Real-Time Payment Fraud Detection & Risk Engine (NusaPay Shield)

Dokumen ini membedah rancangan arsitektur **Real-Time Fraud Detection & Trust/Safety Risk Engine** untuk dompet digital dan *payment gateway* (seperti GoPay, OVO, Stripe Radar, dan PayPal), yang diadaptasi langsung dari studi kasus nyata pada repositori *ML System Design Case Studies*:

- **Stripe (#1 - 2023, #225 - 2021, & #273 - 2020):** *How we built it: Stripe Radar* & *Similarity clustering to catch fraud rings* (Penilaian risiko transaksi $< 50\text{ ms}$ dengan kontrol *False Positive Rate* ketat).
- **PayPal (#193 & #233 - 2021 & #271 - 2020):** *Deploying Large-scale Fraud Detection ML Models* & *Real-time Graph Analysis to Fight Fraud* (Menggabungkan *streaming velocity counters* dengan sinyal graf perangkat/IP).
- **Grab (#72 - 2023 & #163 - 2022) & Uber (#54 - 2023 & #112 - 2022):** *Graph for fraud detection* & *Unsupervised Anomaly Detection* (Menggabungkan model *Supervised GBDT* dengan *Unsupervised Anomaly Detection* untuk mendeteksi modus serangan baru).
- **Wayfair (#257 - 2020) & Lyft (#295 & #297 - 2018):** *Explainable Fraud Detection* & *Fingerprinting Fraudulent Behavior* (Menghasilkan kode alasan risiko yang dapat dijelaskan ke pengguna dan analis).

---

## 1. Karakteristik Unik Sistem Fraud Detection dibanding Recommender / Search

| Dimensi Teknis | Recommender / Search (Proyek 01 & 02) | Real-Time Fraud Detection (Proyek 03) |
| :--- | :--- | :--- |
| **Distribusi Kelas** | Seimbang / Multi-level (*click, cart, buy*) | **Extreme Class Imbalance** (Transaksi penipuan umumnya hanya $1\% - 5\%$ dari total volume) |
| **Asimetri Biaya Kesalahan** | Salah rekomendasi hanya menurunkan CTR sedikit | **False Negative (FN):** Uang pengguna/platform hilang $100\%$ dari nominal transaksi.<br>**False Positive (FP):** Pelanggan asli diblokir $\rightarrow$ frustrasi & pindah ke kompetitor (*customer churn*). |
| **Musuh Aktif (Adversarial)** | Pengguna bekerja sama dengan sistem | **Pelaku Fraud Aktif Mengakali Sistem** (Menggunakan perangkat baru, membajak akun tengah malam, menguji kartu dengan nominal kecil beruntun). |
| **Metrik Evaluasi Utama** | `NDCG@K`, `Recall@K`, `MRR@K` | **`PR-AUC` (Precision-Recall AUC)**, **`Recall @ 1% FPR`**, dan **`Net Financial Savings (Rp)`**. *(Akurasi biasa `96%` tidak berguna jika model hanya menebak semua transaksi "Aman"!)*. |

---

## 2. Arsitektur 4-Lapisan Real-Time Risk Engine

```
[ Incoming Payment Event: (user_id, amount_idr, merchant_category, device_id, ip_hash, lat, lon, timestamp) ]
                                                    │
                                                    ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│ LAYER 1: STREAMING VELOCITY, GEO-IMPOSSIBLE TRAVEL & ENTITY GRAPH ENGINE (< 5 ms)       │
│  1. Sliding Window Velocity Counters:                                                   │
│     - tx_count_10m, tx_count_1h, tx_amount_sum_1h (mendeteksi serangan beruntun)        │
│  2. Geospatial Impossible Travel:                                                       │
│     - Jarak Haversine (km) & kecepatan perpindahan (km/jam) dari transaksi sebelumnya   │
│       (Misal: 10 menit lalu transaksi di Jakarta, sekarang di Surabaya -> > 4.000 km/j!)│
│  3. Device & IP Entity Graph Signals:                                                   │
│     - device_unique_users_count & ip_unique_users_count (1 HP dipakai 5 akun berbeda!)  │
│     - is_new_device_for_user (Perangkat asing yang belum pernah dipakai pemilik akun)   │
│  4. User Behavioral Deviation:                                                          │
│     - amount_to_user_avg_ratio & amount_zscore (Nominal melonjak 15x dari kebiasaan)    │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                                    │ (22 Real-Time Risk Features)
                                                    ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│ LAYER 2: DETERMINISTIC GUARDRAIL RULES (< 1 ms)                                         │
│  - Hard Rule 1: Perangkat berada di daftar hitam sindikat (Known Fraud Ring Device)     │
│  - Hard Rule 2: Impossible Travel (> 500 km/jam) + Perangkat Asing + Nominal Tinggi     │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                                    │
                                                    ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│ LAYER 3: HYBRID SUPERVISED (LightGBM) + UNSUPERVISED (Isolation Forest) (~8 ms)         │
│  - Supervised LightGBM Classifier: Memprediksi P(Fraud | x) dari pola historis          │
│  - Unsupervised Isolation Forest: Memberikan skor anomali untuk pola ekstrem baru       │
│  - Calibrated Hybrid Risk Score: Skala 0 - 100 + Top Risk Reason Codes                  │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                                    │
                                                    ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│ LAYER 4: 3-TIER COST-SENSITIVE DECISION ENGINE (< 1 ms)                                 │
│  - Score 0 - 34  --> [APPROVE]  Transaksi disetujui instan tanpa hambatan               │
│  - Score 35 - 74 --> [CHALLENGE / STEP-UP OTP] Minta verifikasi PIN/Biometrik tambahan  │
│  - Score 75 - 100--> [BLOCK]    Transaksi ditolak otomatis & akun diamankan             │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Rumus Teknis & Optimasi Ambang Keputusan (Cost-Sensitive Thresholding)

### 3.1 Kecepatan Perpindahan Geospasial (*Impossible Travel Velocity*)
Jika transaksi sebelumnya dari pengguna $u$ terjadi pada koordinat $(\phi_{t-1}, \lambda_{t-1})$ di waktu $t-1$, dan transaksi saat ini terjadi pada $(\phi_t, \lambda_t)$ di waktu $t$, maka kecepatan perpindahan fisik pengguna adalah:

$$V_{\text{geo}}(u, t) = \frac{d_{\text{Haversine}}\left((\phi_{t-1}, \lambda_{t-1}), (\phi_t, \lambda_t)\right)}{\max\left(\frac{t - t_{t-1}}{3600}, 0.0167\right)} \quad (\text{dalam km/jam})$$

Nilai $V_{\text{geo}} > 350\text{ km/jam}$ (apalagi disertai pergantian `device_id`) merupakan sinyal kuat terjadinya **Account Takeover (ATO)** atau pencurian kredensial jarak jauh.

### 3.2 Matriks Biaya Finansial (*Financial Cost Matrix*)
Mengacu pada studi kasus **PayPal (#271)** dan **Stripe (#1)**, memilih ambang batas (*threshold*) keputusan tidak boleh asal menggunakan $p = 0.50$, melainkan meminimalkan total kerugian finansial:
- **Biaya False Negative ($\text{Cost}_{\text{FN}}$):** Kehilangan $100\%$ nominal transaksi $A_i$ akibat *chargeback* / penggantian dana penipuan.
- **Biaya False Positive pada Blokir ($\text{Cost}_{\text{FP, Block}}$):** Kehilangan margin transaksi ($5\% \cdot A_i$) + biaya penalti *churn* pelanggan sebesar $\text{Rp } 75.000$.
- **Biaya Step-Up Challenge OTP ($\text{Cost}_{\text{Challenge}}$):** Gesekan UX ringan (~$\text{Rp } 3.500$), namun mampu menggagalkan $92\%$ pelaku *fraud* tanpa memblokir pelanggan asli!

