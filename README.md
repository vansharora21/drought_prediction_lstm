🌾 LSTM-Based Drought Prediction
Proof-of-Concept Time-Series Forecasting Model

Undergraduate Research Project | 3rd Year BTech – Computer Science

📌 Project Summary

This project presents a proof-of-concept LSTM (Long Short-Term Memory) neural network for next-week drought prediction using 8 weeks of historical climate data from a single geographic region.

The objective is academic demonstration, not production deployment — focusing on clarity, interpretability, and correctness rather than scale or complexity.

🎯 Objectives

Demonstrate time-series forecasting using LSTM networks

Understand temporal dependency modeling

Apply sequence-based learning on climate data

Build an academically sound prototype suitable for:

Undergraduate research

Mini-project / capstone submission

Conference demo or poster presentation

🧠 Key Highlights

✅ Simple and interpretable architecture

✅ Single LSTM layer + Dense output

✅ Only 3 climate features

✅ Time-aware train / validation / test split

✅ Well-documented Jupyter Notebook

✅ Lightweight & CPU-friendly

📊 What Does the Model Do?
🔄 Workflow Overview

Load Climate Data (US Drought Monitor based)

Preprocess Data

Feature selection

Min–Max normalization

Create Sliding Windows

8-week historical input

Time-Aware Split

70% Train | 15% Validation | 15% Test

Build LSTM Model

Train for 50 Epochs

Evaluate (MAE & RMSE)

Visualize Predictions

📥 Input → 📤 Output
INPUT:
8 Weeks × 3 Features
- Precipitation
- Temperature
- Humidity

↓ LSTM Model ↓

OUTPUT:
- Drought Index for Week 9 (0–1 normalized)

🧠 Model Architecture
Input Shape: (8, 3)

LSTM Layer
- Units: 32
- Activation: ReLU

Dense Output Layer
- Units: 1

Total Parameters: 4,641

📈 Training Configuration
Parameter	Value
Loss Function	MAE
Optimizer	Adam (lr = 0.001)
Epochs	50
Batch Size	16
Lookback Window	8 weeks
Data Split	70 / 15 / 15 (time-aware)
📊 Expected Results

Test Set Performance (Approx.)

MAE (normalized):   0.06 – 0.08
RMSE (normalized):  0.08 – 0.10

Generated Visualizations

Training vs Validation Loss

Actual vs Predicted Drought Index

Saved to:

results/lstm_results_visualization.png

📁 Project Structure
DroughtPredictionProject/
│
├── README.md
├── requirements.txt
│
├── notebooks/
│   ├── LSTM_Drought_Prediction.ipynb   ⭐ Main Notebook
│   └── DroughtPrediction.ipynb
│
├── scripts/
│   ├── prepare_data.py
│   └── train_model.py
│
├── data/
│   └── (Downloaded separately – see below)
│
├── models/
│   └── lstm_model.keras
│
└── results/
    └── lstm_results_visualization.png

📂 Dataset Information
🔗 Data Download (Google Drive)

Due to large file size (~1.1 GB), the dataset is not included in the repository.

📥 Download Dataset Here:
👉 Google Drive – USDM Climate Dataset

📌 After Download

Extract the file

Place it inside the data/ directory:

data/USDMData.csv


⚠️ The notebook automatically:

Limits rows if memory is low

Generates synthetic data if file is missing

🛠️ Installation & Setup
🔹 Requirements

Python 3.8+

TensorFlow / Keras

Pandas, NumPy

Scikit-learn

Matplotlib

🔹 Install Dependencies
pip install -r requirements.txt

▶️ How to Run
jupyter notebook notebooks/LSTM_Drought_Prediction.ipynb


Run all cells top to bottom.

⏱️ Execution Time:

Total: ~5–10 minutes

Training: ~2–3 minutes

GPU not required

📓 Notebook Breakdown
Cell	Description
1–4	Data loading & exploration
5–8	Preprocessing & sequence creation
9–12	Time-aware splitting & model building
13–14	Model training
15–16	Evaluation (MAE, RMSE)
17–18	Visualization
19–20	Conclusions & summary
