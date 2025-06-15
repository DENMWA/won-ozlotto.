# 🧠 Won-OzLotto: Hybrid Oz Lotto Predictor

A powerful **Oz Lotto prediction engine** using hybrid techniques that blend:
- Constructive prediction (Mode C)
- Statistical post-filter scoring with adjustable weights (Final Formula: α, β, γ)

Built in **Python** with **Streamlit**, this app allows you to simulate, score, and optimize your number sets using entropy, Mahalanobis distance, and historical frequency patterns.

---

## 🚀 Features

### 🎯 Predictive Modes
- **Mode C – Constructive Prediction**: Dynamically builds 7-number sets from [1–47] using:
  - Fourier frequency boosts
  - Entropy normalization
  - Co-draw probability filters
  - Randomized cryptographic weights

- **Final Formula Post-Scoring**: Uses adjustable α, β, γ to tune:
  - Frequency weighting
  - Hot zone & cold zone pressure
  - Entropy bonus and Mahalanobis distance for set diversity

### 📊 Visual Tools
- Live bar chart of simulated historical frequencies
- Slider-adjustable formula weights (α, β, γ)
- Entropy, sequential gap penalty, and co-draw logic

### 📥 Output Options
- Displays top 10 highest-scoring prediction sets
- Allows CSV export of prediction sets

---

## 📦 Installation

1. Clone the repo:

```bash
git clone https://github.com/YOUR_USERNAME/won-ozlotto.git
cd won-ozlotto
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run Streamlit app:

```bash
streamlit run streamlit_app.py
```

---

## 📈 Future Enhancements

- Integration of real Oz Lotto historical data
- Supplementary number modeling
- Result matching against actual past draws
- Multi-mode comparative heatmaps

---

## 💡 Disclaimer

This tool is for educational and entertainment purposes only. There is no guarantee of lottery success. Please play responsibly.

---
