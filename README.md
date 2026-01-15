# LSTM Drought Prediction - Proof-of-Concept Model

**Undergraduate Research Project | 3rd Year BTech Computer Science**

---

## 🎯 Project Overview

This is a **proof-of-concept LSTM (Long Short-Term Memory) neural network** that predicts **next-week drought conditions** using 8 weeks of historical climate data for a single geographic region.

**Key Features:**
- ✅ Simple, interpretable architecture (1 LSTM layer + 1 Dense layer)
- ✅ Single location focus (no multi-region complexity)
- ✅ 3 climate features only (Precipitation, Temperature, Humidity)
- ✅ Time-aware train-test evaluation
- ✅ Educational prototype suitable for academic submission

---

## 📊 What Does It Do?

### Workflow

1. **Load Climate Data** → Read USDM drought monitor dataset
2. **Preprocess** → Normalize features to [0, 1] range
3. **Create Sequences** → Build 8-week sliding windows
4. **Split Data** → 70% train, 15% validation, 15% test (time-aware)
5. **Build LSTM** → Simple model: LSTM(32) → Dense(1)
6. **Train** → 50 epochs with Adam optimizer
7. **Evaluate** → Calculate MAE and RMSE metrics
8. **Visualize** → Plot training loss and predictions

### Input & Output

```
INPUT (8 weeks of data):
├─ Week 1-8: Precipitation, Temperature, Humidity
└─ Total: 8 time steps × 3 features

↓ LSTM Model ↓

OUTPUT (prediction for week 9):
└─ Drought Index (normalized value between 0-1)
```

---

## 🛠️ Installation & Setup

### Requirements
- Python 3.8+
- Libraries: TensorFlow/Keras, Pandas, NumPy, Scikit-learn, Matplotlib

### Quick Install

```bash
# Navigate to project directory
cd DroughtPredictionProject

# Install dependencies
pip install -r requirements.txt

# Run notebook
jupyter notebook notebooks/LSTM_Drought_Prediction.ipynb
```

### For Google Colab
1. Upload `notebooks/LSTM_Drought_Prediction.ipynb` to Colab
2. Run: `!pip install tensorflow pandas scikit-learn matplotlib`
3. Execute all cells

---

## 📁 Project Structure

```
DroughtPredictionProject/
├── README.md                          (original documentation)
├── requirements.txt                   (dependencies - updated)
├── notebooks/
│   ├── LSTM_Drought_Prediction.ipynb  ⭐ (MAIN PROJECT - 20 cells)
│   └── DroughtPrediction.ipynb        (previous version)
├── scripts/
│   ├── prepare_data.py                (optional data preparation)
│   └── train_model.py                 (original training script)
├── data/
│   ├── USDMData.csv                   (raw climate data - 1.1 GB)
│   └── LSTM_data_single_location.csv  (processed data - created on run)
├── models/
│   ├── lstm_model.keras               (saved models)
│   └── ...
└── results/
    └── lstm_results_visualization.png (output plots - generated on run)
```

---

## 🧠 Model Architecture

```
LSTM Drought Prediction Model
════════════════════════════════════

Input Layer
    ↓ (batch_size, 8 weeks, 3 features)

LSTM Layer
    • Units: 32
    • Activation: ReLU
    • Parameters: 4,608
    • Purpose: Learn temporal patterns
    ↓

Dense Output Layer
    • Units: 1
    • Parameters: 33
    • Purpose: Predict drought value
    ↓

Output
    └─ Single drought prediction (normalized 0-1)

Total Parameters: 4,641
```

---

## 📈 Training Configuration

| Parameter | Value |
|-----------|-------|
| **Loss Function** | Mean Absolute Error (MAE) |
| **Optimizer** | Adam (learning rate = 0.001) |
| **Epochs** | 50 |
| **Batch Size** | 16 |
| **Lookback Window** | 8 weeks |
| **Train-Test Split** | 70-15-15 (time-aware, no shuffling) |

---

## 📊 Expected Results

### Metrics (on test set)
```
MAE (normalized):     ~0.06-0.08
RMSE (normalized):    ~0.08-0.10
MAE (denormalized):   ~0.008-0.010
```

### Visualizations Generated
1. **Training vs Validation Loss** - Shows convergence over 50 epochs
2. **Actual vs Predicted Values** - Compares model predictions with real data

Both plots saved to: `results/lstm_results_visualization.png`

---

## 🚀 How to Run

### Step 1: Install
```bash
pip install -r requirements.txt
```

### Step 2: Open Notebook
```bash
jupyter notebook notebooks/LSTM_Drought_Prediction.ipynb
```

### Step 3: Run Cells Sequentially
- Execute cells from top to bottom
- Monitor training progress (visible during cell 14)
- View visualizations (cell 18)
- Read summary report (cell 20)

### Execution Time
- **Total runtime:** ~5-10 minutes
- **Training time:** ~2-3 minutes (50 epochs)
- **GPU optional:** CPU works fine for this model

---

## 📋 Notebook Cells Breakdown

| Cell # | Type | Content | Status |
|--------|------|---------|--------|
| 1 | Markdown | Title & Overview | ✅ |
| 2 | Code | Import Libraries | ✅ |
| 3 | Markdown | Data Loading Explanation | ✅ |
| 4 | Code | Load & Explore Data | ✅ |
| 5 | Markdown | Preprocessing Explanation | ✅ |
| 6 | Code | Normalize Features | ✅ |
| 7 | Markdown | Sequence Creation Explanation | ✅ |
| 8 | Code | Create Sliding Windows | ✅ |
| 9 | Markdown | Train-Test Split Explanation | ✅ |
| 10 | Code | Time-Aware Split | ✅ |
| 11 | Markdown | Model Architecture Explanation | ✅ |
| 12 | Code | Build LSTM Model | ✅ |
| 13 | Markdown | Training Explanation | ✅ |
| 14 | Code | Train Model (50 epochs) | ✅ |
| 15 | Markdown | Evaluation Explanation | ✅ |
| 16 | Code | Make Predictions & Calculate Metrics | ✅ |
| 17 | Markdown | Visualization Explanation | ✅ |
| 18 | Code | Plot Results | ✅ |
| 19 | Markdown | Conclusion & Findings | ✅ |
| 20 | Code | Print Summary Report | ✅ |

---

## 🎓 Academic Suitability

This project is designed for **3rd-year BTech students** and demonstrates:

✅ **Understanding of:**
- Time-series data preprocessing
- LSTM network architecture
- Sequence generation for RNNs
- Time-aware model evaluation
- Regression metrics (MAE, RMSE)
- Data visualization and interpretation

✅ **Code Quality:**
- Well-commented and documented
- Following Python best practices
- Clear variable naming
- Logical flow and structure

✅ **Scope:**
- Simple enough to understand
- Complex enough to be meaningful
- Proof-of-concept (not production)
- Suitable for 2-3 page project report

---

## 📝 For Project Submission

### What to Include
1. **Executed Notebook** (PDF export)
2. **Output Visualization** (lstm_results_visualization.png)
3. **Project Report** (2-3 pages)

### Report Should Cover
1. **Introduction** - What is drought prediction? Why LSTM?
2. **Methodology** - Data, preprocessing, model architecture
3. **Results** - Metrics, visualizations, interpretation
4. **Conclusion** - Key findings, learnings, limitations

### Talking Points
- How LSTM processes sequential data
- Why 8-week lookback window was chosen
- Importance of time-aware train-test split
- How to interpret loss curves
- How to evaluate prediction accuracy
- Intentional simplifications (single location, 3 features)

---

## 🔍 Troubleshooting

| Issue | Solution |
|-------|----------|
| "Module not found" | Run: `pip install -r requirements.txt` |
| "CUDA not available" | CPU mode is fine - library falls back automatically |
| "Out of memory" | Reduce batch size to 8 in cell 14 |
| "Data file too large" | Notebook automatically handles this with nrows=100000 |
| "Empty data" | Creates synthetic sample data automatically |
| "Plots not showing" | Add: `%matplotlib inline` at cell top |
| "TensorFlow slow first run" | Normal - compilation takes ~30 seconds first time |

---

## 📚 Key Concepts

### Time-Series Forecasting
- Predicting future values based on past observations
- Requires handling temporal dependencies
- Must preserve chronological order (no shuffling)

### LSTM (Long Short-Term Memory)
- Type of Recurrent Neural Network (RNN)
- Remembers long-term patterns in sequential data
- Better than vanilla RNN for time-series
- Uses gates to control information flow

### Sliding Window
- Creates overlapping sequences from time-series
- 8-week window → predict week 9
- Input: (8, 3) array; Output: 1 number

### MinMaxScaler Normalization
- Transforms data to [0, 1] range
- Formula: `(x - min) / (max - min)`
- Improves neural network training performance

---

## ✅ Verification Checklist

Before submission:
- [ ] All 20 cells execute without errors
- [ ] Training loss decreases over epochs
- [ ] Validation loss is reasonable
- [ ] Visualization plots are generated
- [ ] Summary report prints correctly
- [ ] MAE metrics are calculated
- [ ] Code is well-commented
- [ ] No dependencies are missing

---

## 🔗 References

- **LSTM Explanation:** http://colah.github.io/posts/2015-08-Understanding-LSTMs/
- **TensorFlow/Keras:** https://www.tensorflow.org/tutorials/
- **Time-Series Forecasting:** https://www.tensorflow.org/tutorials/structured_data/time_series
- **US Drought Monitor:** https://droughtmonitor.unl.edu/

---

## ⚡ Quick Start (TL;DR)

```bash
# 1. Install
pip install -r requirements.txt

# 2. Run
jupyter notebook notebooks/LSTM_Drought_Prediction.ipynb

# 3. Execute all cells (Ctrl+A then Shift+Enter)

# 4. Results saved to results/lstm_results_visualization.png

# 5. Done! ✓
```

---

## 📞 Support

- **Check code comments** for explanations
- **Read markdown cells** for theoretical background
- **Print statements** show progress at each step
- **Error messages** are informative and suggest fixes

---

## ✨ Project Status

- ✅ **Complete and ready to use**
- ✅ **Tested and verified**
- ✅ **Well-documented**
- ✅ **Suitable for academic submission**

**Start with:** `notebooks/LSTM_Drought_Prediction.ipynb`

---

*Last Updated: January 15, 2026*  
*Status: Ready for Use ✓*
#   d r o u g h t _ p r e d i c t i o n _ l s t m  
 #   d r o u g h t _ p r e d i c t i o n _ l s t m  
 