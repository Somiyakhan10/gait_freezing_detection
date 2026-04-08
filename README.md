




# GaitLock – Real-Time Freezing of Gait Detection for Parkinson's Patients

**IMU-Based Machine Learning System for Freezing of Gait (FOG) Detection** · By Somiya Khan

> ⚠️ **Note:** This tool is designed for research and educational purposes only. It is not intended for clinical diagnosis or regulatory decision-making.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Model Performance](#model-performance)
- [Dataset](#dataset)
- [Feature Engineering](#feature-engineering)
- [How It Works](#how-it-works)
- [Technology Stack](#technology-stack)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Author](#author)
- [License](#license)

---

## 🔬 Overview

**GaitLock** is a machine learning system that detects **Freezing of Gait (FOG)** — a debilitating symptom in advanced Parkinson's disease — using wearable IMU sensor data. The system processes 9-axis acceleration signals from the Daphnet Freezing of Gait Dataset and achieves **98.9% AUC** in distinguishing normal gait from freezing episodes.

### Key Features

| Feature | Description |
|:--------|:------------|
| **Real-Time Detection** | 2.5-second sliding windows for near real-time FOG detection |
| **Multi-Sensor Fusion** | Combines ankle, thigh, and trunk accelerometer data |
| **117 Engineered Features** | Time-domain + frequency-domain features per sensor axis |
| **High Sensitivity** | 84.1% sensitivity with 97.7% specificity |
| **Clinical Relevance** | Enables fall prevention alerts for Parkinson's patients |

---

## 📊 Model Performance

| Metric | Value |
|:-------|:------|
| **AUC (ROC)** | **0.989** |
| **Sensitivity (Recall)** | 84.1% |
| **Specificity** | 97.7% |
| **Precision (PPV)** | 67.6% |
| **Accuracy** | 97.0% |
| **Cross-Validation AUC** | 0.993 ± 0.001 |

### Confusion Matrix

|  | Predicted Normal | Predicted Freeze |
|:--|:----------------:|:----------------:|
| **Actual Normal** | 1,394 (97.7%) | 33 (2.3%) |
| **Actual Freeze** | 13 (15.9%) | 69 (84.1%) |

---

## 📊 Dataset

| Property | Details |
|:---------|:--------|
| **Source** | Daphnet Freezing of Gait Dataset (UCI Machine Learning Repository) |
| **Subjects** | 10 Parkinson's patients (3 with freezing episodes) |
| **Sensors** | 3 accelerometers (ankle, thigh, trunk) — 9-axis total |
| **Sampling Rate** | 64 Hz |
| **Total Samples** | 241,632 |
| **Annotation Classes** | 0 = Rest, 1 = Normal walking, 2 = Freezing of Gait (FOG) |

### Data Distribution (After Windowing)

| Class | Windows | Percentage |
|:------|:-------:|:----------:|
| **Normal Gait** | 7,132 | 94.6% |
| **Freezing of Gait** | 410 | 5.4% |
| **Total** | 7,542 | 100% |

---

## 🧠 Feature Engineering

### Sliding Window Parameters

| Parameter | Value |
|:----------|:------|
| Window Size | 2.5 seconds (160 samples) |
| Step Size | 0.5 seconds (32 samples) |
| Overlap | 80% |

### Feature Categories (117 total features)

| Category | Features per Signal | Description |
|:---------|:-------------------|:------------|
| **Time-Domain** | Mean, Std, RMS, Max, Min, Range, Skew, Kurtosis, ZCR | Statistical properties of acceleration signals |
| **Frequency-Domain** | Dominant frequency, Spectral centroid, Total power, Freeze-band power (3-8 Hz) | FFT-based spectral analysis |

### Top 5 Most Important Features

| Rank | Feature | Importance |
|:----:|:--------|:----------:|
| 1 | trunk_z_zcr (Zero-crossing rate) | 0.0621 |
| 2 | thigh_x_zcr | 0.0473 |
| 3 | trunk_x_mean | 0.0462 |
| 4 | trunk_z_mean | 0.0437 |
| 5 | thigh_x_freeze_band_power | 0.0430 |

---

## ⚙️ How It Works

```
IMU Sensors (Ankle/Thigh/Trunk)
         │
         ▼
  Sliding Window (2.5 sec, 64 Hz)
         │
         ▼
  Feature Extraction (117 features)
         │
         ▼
     Standardization (Z-score)
         │
         ▼
   Random Forest Classifier (200 trees)
         │
         ▼
   FOG Prediction (0=Normal, 1=Freeze)
```

### Model Architecture

| Parameter | Value |
|:----------|:------|
| **Algorithm** | Random Forest Classifier |
| **Number of Trees** | 200 |
| **Max Depth** | 15 |
| **Min Samples Split** | 5 |
| **Class Weight** | Balanced |
| **Cross-Validation** | 5-fold Stratified |

---

## 💻 Technology Stack

| Category | Technologies |
|:---------|:-------------|
| **Language** | Python 3.x |
| **ML Framework** | scikit-learn |
| **Signal Processing** | NumPy, SciPy |
| **Data Processing** | Pandas |
| **Visualization** | Matplotlib, Seaborn |
| **Model Persistence** | joblib |
| **Environment** | Google Colab / Jupyter Notebook |

---

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/gaitlock-fog-detection.git
cd gaitlock-fog-detection
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download Dataset

Download the **Daphnet Freezing of Gait Dataset** from:
[https://archive.ics.uci.edu/static/public/245/daphnet+freezing+of+gait.zip](https://archive.ics.uci.edu/static/public/245/daphnet+freezing+of+gait.zip)

### 4. Run the Notebook

```bash
jupyter notebook "GaitLock_Real_Time_Freezing_Detection_for_Parkinson's_Patients.ipynb"
```

---

## 📁 Project Structure

```
gaitlock-fog-detection/
│
├── GaitLock_Real_Time_Freezing_Detection.ipynb   # Main analysis notebook
│
├── fog_detection_model.pkl                       # Trained Random Forest
├── scaler.pkl                                    # StandardScaler
├── fog_detection_report.txt                      # Performance summary
│
├── confusion_matrix.png                          # Confusion matrix
├── roc_curve.png                                 # ROC curve
├── feature_importance.png                        # Top features
├── data_exploration.png                          # Signal visualization
│
├── requirements.txt                              # Python dependencies
└── README.md                                     # This file
```

---

## 👩‍🔬 Author

**Somiya Khan**

- **Project:** GaitLock — Real-Time Freezing of Gait Detection for Parkinson's Patients
- **Field:** Biomedical Engineering / Rehabilitation Engineering
- **Focus:** Wearable sensors, IMU-based gait analysis, machine learning for movement disorders

---

## 📄 License

This project is licensed under the **MIT License**.

```
MIT License

Copyright (c) 2025 Somiya Khan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...
```

---

## 🙏 Acknowledgments

- **Daphnet Freezing of Gait Dataset** — UCI Machine Learning Repository
- **scikit-learn** — Open-source ML framework
- **Parkinson's Research Community** — For ongoing efforts in movement disorder characterization

---

**Built with ❤️ for Parkinson's research and rehabilitation engineering**
```

---

