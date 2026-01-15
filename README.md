# 🌾 LSTM-Based Drought Prediction
### Proof-of-Concept Time-Series Forecasting Model

**Undergraduate Research Project | 3rd Year BTech – Computer Science**

---

## 📌 Project Overview

This project presents a **proof-of-concept LSTM (Long Short-Term Memory) neural network** designed to **predict next-week drought conditions** using **8 weeks of historical climate data** for a **single geographic region**.

The goal is **academic demonstration**, focusing on **clarity, interpretability, and correct methodology**, rather than production-scale deployment.

---

## 🎯 Objectives

- Demonstrate time-series forecasting using LSTM networks  
- Understand temporal dependency modeling  
- Apply sliding-window sequence learning  
- Build an academically suitable prototype for undergraduate research  

---

## 🧠 Key Features

- Single LSTM layer + Dense output  
- 3 climate features: Precipitation, Temperature, Humidity  
- Time-aware train / validation / test split  
- Simple, interpretable architecture  
- Fully documented Jupyter Notebook  

---

## 📊 Input and Output

**Input:**  
8 weeks × 3 climate features  

**Output:**  
Predicted drought index for week 9 (normalized 0–1)

---

## 🧠 Model Architecture

- LSTM Layer: 32 units (ReLU)
- Dense Layer: 1 unit
- Total Parameters: 4,641

---

## 📈 Training Configuration

| Parameter | Value |
|---------|------|
| Loss Function | MAE |
| Optimizer | Adam (lr = 0.001) |
| Epochs | 50 |
| Batch Size | 16 |
| Lookback Window | 8 weeks |
| Data Split | 70 / 15 / 15 (time-aware) |

---

## 📁 Project Structure

```
DroughtPredictionProject/
├── README.md
├── requirements.txt
├── notebooks/
│   └── LSTM_Drought_Prediction.ipynb
├── scripts/
├── data/
├── models/
└── results/
```

---

## 📂 Dataset Download

The dataset is large (~1.1 GB) and hosted externally.

📥 **Google Drive Link:**  
https://drive.google.com/file/d/1r3BvjF2v23ad4qlYyd0_eMyoKhXOPrOu/view

After downloading, place the file inside:

```
data/USDMData.csv
```

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
jupyter notebook notebooks/LSTM_Drought_Prediction.ipynb
```

---

## 🎓 Academic Suitability

This project is suitable for:
- 3rd Year BTech Mini Project
- Undergraduate Research Work
- AI / ML Coursework Submission

---

## 📚 References

- Understanding LSTM Networks – http://colah.github.io/posts/2015-08-Understanding-LSTMs/
- TensorFlow Time Series Tutorials
- US Drought Monitor

---

## 👨‍🎓 Author

**Vansh Arora**  
3rd Year BTech – Computer Science  

*Last Updated: January 15, 2026*
