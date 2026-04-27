# GaitLock — Real-Time Freezing of Gait Detection for Parkinson's Patients

**IMU-Based Machine Learning System for Freezing of Gait (FOG) Detection** 
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2%2B-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

⚠️ **Medical Disclaimer:** This tool is designed for research and educational purposes only. It is not intended for clinical diagnosis or regulatory decision-making. Always consult a qualified healthcare professional.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Model Performance](#-model-performance)
- [Dataset](#-dataset)
- [Feature Engineering](#-feature-engineering)
- [How It Works](#-how-it-works)
- [Technology Stack](#-technology-stack)
- [Installation](#-installation)
- [Project Structure](#-project-structure)
- [Results](#-results)
- [Future Work](#-future-work)
- [Author](#-author)
- [License](#-license)

---

## 🔬 Overview

**GaitLock** is a machine learning system that detects **Freezing of Gait (FOG)** — a debilitating symptom in advanced Parkinson's disease — using wearable IMU sensor data. The system processes 9-axis acceleration signals from the Daphnet Freezing of Gait Dataset and achieves **98.9% AUC** in distinguishing normal gait from freezing episodes.

Freezing of Gait affects **50-80% of advanced Parkinson's patients**, causing sudden falls, injuries, and loss of independence. This system enables real-time detection for fall prevention and caregiver alerts.
## 🏥 Clinical Relevance

### What is Freezing of Gait (FOG)?

Freezing of Gait is a sudden, temporary inability to move the feet forward while walking. It is one of the most debilitating symptoms of advanced Parkinson's disease.

| Statistic | Value |
| :--- | :--- |
| **Prevalence in advanced Parkinson's** | 50-80% |
| **Falls caused by FOG** | 60-80% |
| **Injury rate during falls** | 30-50% |

### Why Real-Time Detection Matters

| Current Problem | Solution |
| :--- | :--- |
| FOG is unpredictable | Continuous monitoring |
| Patients can't self-report during episodes | Automatic detection |
| Delayed response increases fall risk | Real-time alerts |
| No objective FOG data for clinicians | Quantified metrics |

### Clinical Applications

- **Fall Prevention** — Alert caregivers before patient falls
- **Rehabilitation Feedback** — Provide real-time biofeedback during physical therapy
- **Medication Monitoring** — Track FOG frequency to optimize medication
- **Long-term Tracking** — Monitor disease progression objectively
---

## ✨ Key Features

| Feature | Description |
| :--- | :--- |
| **Real-Time Detection** | 2.5-second sliding windows for near real-time FOG detection |
| **Multi-Sensor Fusion** | Combines ankle, thigh, and trunk accelerometer data |
| **117 Engineered Features** | Time-domain + frequency-domain features per sensor axis |
| **High Sensitivity** | 84.1% sensitivity with 97.7% specificity |
| **Clinical Relevance** | Enables fall prevention alerts for Parkinson's patients |

---

## 📊 Model Performance

| Metric | Value |
| :--- | :--- |
| **AUC (ROC)** | 0.989 |
| **Sensitivity (Recall)** | 84.1% |
| **Specificity** | 97.7% |
| **Precision (PPV)** | 67.6% |
| **Accuracy** | 97.0% |
| **Cross-Validation AUC** | 0.993 ± 0.001 |

### Confusion Matrix

| | Predicted Normal | Predicted Freeze |
| :--- | :--- | :--- |
| **Actual Normal** | 1,394 (97.7%) | 33 (2.3%) |
| **Actual Freeze** | 13 (15.9%) | 69 (84.1%) |

### ROC Curve

<img width="1200" height="900" alt="roc_curve" src="https://github.com/user-attachments/assets/1e6ce852-fe4d-4fec-9e96-769f80f5d3bf" />


### Confusion Matrix

<img width="1200" height="900" alt="confusion_matrix" src="https://github.com/user-attachments/assets/a4bc810f-c549-4f73-bc89-47b132a98676" />



### Feature Importance

<img width="1800" height="900" alt="feature_importance" src="https://github.com/user-attachments/assets/bfb5d280-8e58-4428-b714-35098f75536b" />

### Model Results
<img width="2682" height="1481" alt="model_results (1)" src="https://github.com/user-attachments/assets/e5dc7e61-421b-4975-851a-c083232b08e0" />



### Top 10 Feature Importances

| Rank | Feature | Importance |
| :--- | :--- | :--- |
| 1 | trunk_z_zcr (Zero-crossing rate) | 0.0621 |
| 2 | thigh_x_zcr | 0.0473 |
| 3 | trunk_x_mean | 0.0462 |
| 4 | trunk_z_mean | 0.0437 |
| 5 | thigh_x_freeze_band_power | 0.0430 |
| 6 | thigh_x_std | 0.0381 |
| 7 | trunk_y_freeze_band_power | 0.0268 |
| 8 | ankle_y_max | 0.0260 |
| 9 | ankle_y_mean | 0.0257 |
| 10 | thigh_x_min | 0.0239 |

---

## 📊 Dataset

| Property | Details |
| :--- | :--- |
| **Name** | Daphnet Freezing of Gait Dataset |
| **Source** | [UCI Machine Learning Repository](https://archive.ics.uci.edu/static/public/245/daphnet+freezing+of+gait.zip) |
| **Subjects** | 10 Parkinson's patients (3 with freezing episodes) |
| **Sensors** | 3 accelerometers (ankle, thigh, trunk) — 9-axis total |
| **Sampling Rate** | 64 Hz |
| **Total Samples** | 241,632 |
| **Annotation Classes** | 0 = Rest, 1 = Walking, 2 = Freezing of Gait |

### Data Distribution (After Windowing)

| Class | Windows | Percentage |
| :--- | :--- | :--- |
| Normal Gait | 7,132 | 94.6% |
| Freezing of Gait | 410 | 5.4% |
| **Total** | **7,542** | **100%** |

### Signal Visualization

<img width="2233" height="1780" alt="data_visualization" src="https://github.com/user-attachments/assets/55c83fc0-dca0-45ea-9708-1bfe0f91782a" />



---

## 🧠 Feature Engineering

### Sliding Window Parameters

| Parameter | Value |
| :--- | :--- |
| Window Size | 2.5 seconds (160 samples) |
| Step Size | 0.5 seconds (32 samples) |
| Overlap | 80% |

### Feature Categories (117 total features)

| Category | Features | Description |
| :--- | :--- | :--- |
| **Time-Domain** | Mean, Std, RMS, Max, Min, Range, Skew, Kurtosis, ZCR | Statistical properties of acceleration signals |
| **Frequency-Domain** | Dominant frequency, Spectral centroid, Total power, Freeze-band power (3-8 Hz) | FFT-based spectral analysis |

### Signal Processing Techniques

- **FFT (Fast Fourier Transform)** — Extracts frequency domain features
- **Zero-Crossing Rate** — Measures signal vibration frequency
- **Bandpass Filtering** — Isolates freeze-band power (3-8 Hz)
- **RMS (Root Mean Square)** — Measures signal energy
- **Spectral Centroid** — Identifies dominant frequency band

---


