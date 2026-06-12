# PM₂.₅ & PM₁₀ Estimation using Satellite + ML



A scalable, physics-informed ML pipeline that estimates daily **PM₂.₅ and PM₁₀** concentrations across India by fusing INSAT satellite imagery, ERA5 meteorological reanalysis, and CPCB ground station data — delivering near-real-time air quality estimates even in sensor-sparse regions.

---

## Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Usage](#-usage)
- [Results](#-results)
- [Tech Stack](#-tech-stack)
- [Data Sources](#-data-sources)
- [Team](#-team)

---

## 🌍 Overview

India has **limited ground-based air quality sensors** and most existing PM models rely on raw, uncorrected satellite AOD — ignoring the actual physics of atmospheric dispersion. GeoMinds fixes this by:

- Bias-correcting INSAT AOD using AERONET ground truth
- Fusing 12+ ERA5 atmospheric variables (boundary layer height, wind, energy fluxes)
- Running a 3-model ensemble (Random Forest + XGBoost + Neural Network) stacked via GAM
- Producing sub-hourly predictions at locations with **no ground stations**

**Validation accuracy:** R² = **0.91** for PM₂.₅ and **0.849** for PM₁₀ against CPCB observations.

---

## 🗃️ Dataset

**File:** `finaldataset.parquet`  
**Size:** 395,099 rows × 23 columns  
**Period:** January 2021 – November 2023  
**Coverage:** Pan-India (Lat 8.5°–34.1°N, Lon 70.9°–94.6°E)

### Columns

| Column | Description | Unit |
|--------|-------------|------|
| `Date` | Observation date | YYYY-MM-DD |
| `latitude` | Location latitude | degrees N |
| `longitude` | Location longitude | degrees E |
| `pm25` | Ground PM₂.₅ concentration *(target)* | μg/m³ |
| `pm10` | Ground PM₁₀ concentration *(target)* | μg/m³ |
| `img_mir_radiance` | INSAT MIR channel radiance | W/m²/sr/μm |
| `img_mir_temp` | INSAT MIR brightness temperature | K |
| `img_swir_radiance` | INSAT SWIR channel radiance | W/m²/sr/μm |
| `img_tir1_radiance` | INSAT TIR-1 channel radiance | W/m²/sr/μm |
| `img_tir1_temp` | INSAT TIR-1 brightness temperature | K |
| `img_tir2_radiance` | INSAT TIR-2 channel radiance | W/m²/sr/μm |
| `img_tir2_temp` | INSAT TIR-2 brightness temperature | K |
| `img_vis_radiance` | INSAT VIS channel radiance | W/m²/sr/μm |
| `img_vis_albedo` | INSAT VIS surface albedo | dimensionless |
| `img_wv_radiance` | INSAT Water Vapor channel radiance | W/m²/sr/μm |
| `img_wv_temp` | INSAT Water Vapor brightness temperature | K |
| `PBLH` | Planetary Boundary Layer Height (ERA5) | m |
| `avg_population` | Population density around station | persons/km² |
| `sat_azimuth` | Satellite azimuth angle | degrees |
| `sat_elevation` | Satellite elevation angle | degrees |
| `sun_azimuth` | Solar azimuth angle | degrees |
| `sun_elevation` | Solar elevation angle | degrees |
| `lulc` | Land Use / Land Cover class | encoded float |

### Target Statistics

| | PM₂.₅ (μg/m³) | PM₁₀ (μg/m³) |
|--|--------------|--------------|
| **Mean** | 47.81 | 107.09 |
| **Std** | 45.86 | 89.97 |
| **Min** | 0.02 | 0.04 |
| **Median** | 33.25 | 81.50 |
| **Max** | 663.41 | 988.98 |
| **Nulls** | 656 (0.17%) | 517 (0.13%) |

---

## 🏗️ Architecture

### Stage 1 — Data Processing

```
INSAT-3D AOD ──────────────┐
                            ├──► Spatiotemporal Interpolation ──► AOD + Weather
AERONET AOD (bias ref) ────┘

ERA5 Reanalysis ──► Pearson Correlation Filtering ──► 14 Selected Features

CPCB PM Data ──► Ball Tree NNS (haversine < 10 km) ──► Spatial Match
                                                              │
                                                     Drop Null Rows
                                                              │
                                                   ✅ Processed Dataset
```

### Stage 2 — Two-Step Ensemble

```
Step 1:
  Processed Dataset ──► [Random Forest]  ─┐
                    ──► [XGBoost]         ├──► GAM ──► PM Estimate (Step 1)
                    ──► [Neural Network]  ┘

Step 2:
  Step-1 Estimate + Spatial Lag Features + Predictor Variables
                    ──► [Random Forest]  ─┐
                    ──► [XGBoost]         ├──► GAM ──► ✅ Final PM Estimate
                    ──► [Neural Network]  ┘
```

### Stage 3 — Trend Analysis

```
Final Predictions ──► Theil-Sen Estimator  ─┐
                  ──► Mann-Kendall Test      ├──► Trend Maps + Stats + Regional Output
                  ──► Polynomial Fit (°2)   ┘
```

---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/geominds-air-quality.git
cd geominds-air-quality
pip install -r requirements.txt
```

**requirements.txt**

```
numpy
pandas
pyarrow
scikit-learn
xgboost
tensorflow
pygam
joblib
duckdb
xarray
h5py
rasterio
geopandas
scipy
matplotlib
folium
shapely
pymannkendall
```

---

## 🚀 Usage

### Load the Dataset

```python
import pandas as pd

df = pd.read_parquet("data/finaldataset.parquet")
print(df.shape)   # (395099, 23)
print(df.head())
```

### Train the Ensemble

```python
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from pygam import LinearGAM
from sklearn.model_selection import train_test_split

FEATURES = [
    'img_mir_radiance', 'img_mir_temp', 'img_swir_radiance',
    'img_tir1_radiance', 'img_tir1_temp', 'img_tir2_radiance',
    'img_tir2_temp', 'img_vis_radiance', 'img_vis_albedo',
    'img_wv_radiance', 'img_wv_temp', 'PBLH',
    'sat_elevation', 'lulc'
]
TARGET = 'pm25'   # or 'pm10'

df_clean = df[FEATURES + [TARGET]].dropna()
X_train, X_test, y_train, y_test = train_test_split(
    df_clean[FEATURES], df_clean[TARGET], test_size=0.3, random_state=42
)

rf  = RandomForestRegressor(n_estimators=200, random_state=42).fit(X_train, y_train)
xgb = XGBRegressor(n_estimators=200, random_state=42).fit(X_train, y_train)

# Stack predictions into GAM
import numpy as np
stack_train = np.column_stack([rf.predict(X_train), xgb.predict(X_train)])
gam = LinearGAM().fit(stack_train, y_train)
```

### Run Inference

```python
import joblib

model = joblib.load("models/gam_ensemble.joblib")
predictions = model.predict(X_test)
```

### Trend Analysis

```python
import pymannkendall as mk
from sklearn.linear_model import TheilSenRegressor
import numpy as np

# Mann-Kendall test
result = mk.original_test(daily_pm25_series)
print(f"Trend: {result.trend}, p-value: {result.p:.4f}")

# Theil-Sen slope
ts = TheilSenRegressor()
X_time = np.arange(len(daily_pm25_series)).reshape(-1, 1)
ts.fit(X_time, daily_pm25_series)
print(f"Slope: {ts.coef_[0]:.4f} μg/m³/day")
```

---

## 📊 Results

### Model Performance (Test set, N=14,446)

| Metric | PM₂.₅ | PM₁₀ |
|--------|--------|------|
| **R²** | **0.91** | **0.849** |
| **RMSE** | 13.32 μg/m³ | 35.27 μg/m³ |
| **Slope (origin)** | 0.97 | 0.95 |

### Seasonal Trend (Jan–Oct 2024)

| | PM₂.₅ | PM₁₀ |
|--|-------|------|
| Mann-Kendall | No trend (p=0.283) | No trend (p=0.858) |
| Theil-Sen Slope | −0.098 μg/m³/day | −0.118 μg/m³/day |
| Peak Month | October (76.5 μg/m³) | May (204.2 μg/m³) |
| Lowest Month | August (29.3 μg/m³) | August (60.7 μg/m³) |

Both pollutants follow a **seasonal U-curve** — high in winter (temperature inversions) and post-monsoon, low during monsoon months. The May PM₁₀ spike reflects North Indian dust storm activity.

---

## 🛠️ Tech Stack

| Category | Libraries |
|----------|-----------|
| Core | `numpy`, `pandas`, `pyarrow` |
| ML Models | `scikit-learn`, `xgboost`, `tensorflow`, `pygam` |
| Tuning | `RandomizedSearchCV`, 5-fold CV |
| Geospatial | `xarray`, `rasterio`, `geopandas`, `scipy` |
| Visualization | `matplotlib`, `folium`, `shapely` |
| Trend Analysis | `pymannkendall`, `TheilSenRegressor` |
| Storage | `joblib`, `duckdb` |

---

## 📡 Data Sources

| Dataset | Provider | Link |
|---------|----------|------|
| INSAT-3D/3DR AOD (3RIMG L2G) | ISRO / MOSDAC | [mosdac.gov.in](https://mosdac.gov.in) |
| AERONET V3 Level 2.0 AOD | NASA GSFC | [aeronet.gsfc.nasa.gov](https://aeronet.gsfc.nasa.gov) |
| ERA5 Hourly Reanalysis | ECMWF / Copernicus | [cds.climate.copernicus.eu](https://cds.climate.copernicus.eu) |
| PM₂.₅ / PM₁₀ Ground Data | CPCB India | [cpcb.nic.in](https://cpcb.nic.in) |

---



---

## 📄 License

MIT License. Data sources retain their respective licenses.

---

*Submitted to Bharatiya Antariksh Hackathon 2025 — PS3: Monitoring Air Pollution from Space*
