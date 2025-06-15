# oz_lotto_hybrid_predictor.py
import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import entropy
from scipy.spatial import distance

# ----- Constants and Configs ----- #
NUMBERS_RANGE = list(range(1, 48))  # Oz Lotto: 1 to 47
NUM_MAIN = 7
NUM_SETS = 100

# ----- User Input Weights ----- #
st.sidebar.header("Weight Adjustment (Final Formula)")
alpha = st.sidebar.slider("Alpha – Frequency Weight", 0.0, 2.0, 1.0, 0.1)
beta = st.sidebar.slider("Beta – Hot Zone Weight", 0.0, 2.0, 1.0, 0.1)
gamma = st.sidebar.slider("Gamma – Cold Zone Weight", 0.0, 2.0, 1.0, 0.1)

st.title("🧠 Oz Lotto Hybrid Predictor")
st.markdown("---")

# ----- Simulated Historical Data (Placeholder) ----- #
st.subheader("Simulated Historical Frequencies")
np.random.seed(42)
historical_freq = pd.Series(np.random.randint(5, 50, size=47), index=NUMBERS_RANGE)
st.bar_chart(historical_freq)

# ----- Formula Components ----- #
def fourier_score(freqs):
    ft = np.fft.fft(freqs)
    return np.abs(ft[:len(freqs)//2])  # Only real component

def hot_zone_score(freqs):
    zone_thresh = np.percentile(freqs, 75)
    return (freqs >= zone_thresh).astype(int)

def cold_zone_score(freqs):
    zone_thresh = np.percentile(freqs, 25)
    return (freqs <= zone_thresh).astype(int)

def sequential_penalty(numbers):
    numbers = sorted(numbers)
    penalties = sum(1 for i in range(len(numbers) - 1) if numbers[i+1] - numbers[i] == 1)
    return penalties

def entropy_score(numbers):
    probs = np.array([historical_freq[n]/historical_freq.sum() for n in numbers])
    return entropy(probs, base=2)

def mahalanobis_distance(numbers, historical_matrix):
    if len(historical_matrix) < 2:
        return 0  # Skip if not enough history
    mu = np.mean(historical_matrix, axis=0)
    cov = np.cov(historical_matrix, rowvar=False)
    try:
        return distance.mahalanobis(numbers, mu, np.linalg.inv(cov))
    except:
        return 0

# ----- Prediction Generators ----- #
def generate_mode_c_predictions():
    predictions = []
    for _ in range(NUM_SETS):
        scores = historical_freq.copy()
        scores += np.random.randn(47) * 0.5  # Inject randomness ξ
        scores += hot_zone_score(historical_freq) * 1.5
        scores -= cold_zone_score(historical_freq) * 0.5
        probs = scores / scores.sum()
        picks = np.random.choice(NUMBERS_RANGE, size=NUM_MAIN, replace=False, p=probs)
        predictions.append(sorted(picks))
    return predictions

def evaluate_final_formula(predictions, historical_matrix):
    scored = []
    for entry in predictions:
        F = np.mean([historical_freq[n]/historical_freq.sum() for n in entry])
        H = np.mean([hot_zone_score(historical_freq)[n-1] for n in entry])
        C = np.mean([cold_zone_score(historical_freq)[n-1] for n in entry])
        S = sequential_penalty(entry)
        E = entropy_score(entry)
        M = mahalanobis_distance(entry, historical_matrix)
        score = (alpha*F + beta*H + gamma*C) - S + E + M
        scored.append((entry, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored

# ----- Simulate and Display ----- #
st.subheader("🧪 Prediction Simulation")
mode_c_preds = generate_mode_c_predictions()
historical_matrix = [sorted(np.random.choice(NUMBERS_RANGE, NUM_MAIN, replace=False)) for _ in range(50)]
evaluated = evaluate_final_formula(mode_c_preds, historical_matrix)

# Display top 10
st.markdown("### 🔝 Top 10 Predicted Sets")
top_df = pd.DataFrame([x[0] for x in evaluated[:10]], columns=[f"N{i+1}" for i in range(NUM_MAIN)])
top_df["Score"] = [round(x[1], 3) for x in evaluated[:10]]
st.dataframe(top_df)

# Optional download
csv = top_df.to_csv(index=False).encode('utf-8')
st.download_button("⬇ Download Top Predictions", csv, "oz_lotto_predictions.csv", "text/csv")
