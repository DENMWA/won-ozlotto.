import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import entropy
from scipy.spatial import distance

NUMBERS_RANGE = list(range(1, 48))  # Oz Lotto: 1 to 47
NUM_MAIN = 7
NUM_SETS = 100

# Sidebar weights
st.sidebar.header("Weight Adjustment (Final Formula)")
alpha = st.sidebar.slider("Alpha – Frequency Weight", 0.0, 2.0, 1.0, 0.1)
beta = st.sidebar.slider("Beta – Hot Zone Weight", 0.0, 2.0, 1.0, 0.1)
gamma = st.sidebar.slider("Gamma – Cold Zone Weight", 0.0, 2.0, 1.0, 0.1)

st.title("🧠 Oz Lotto Hybrid Predictor")
st.markdown("---")

# Upload historical data
st.subheader("📁 Upload Historical Draw Data (CSV)")
uploaded_file = st.file_uploader("Upload CSV with 7 columns for main numbers", type="csv")

if uploaded_file:
    historical_df = pd.read_csv(uploaded_file)
    if historical_df.shape[1] < NUM_MAIN:
        st.error("Please upload a file with at least 7 columns for main numbers.")
        st.stop()
    historical_numbers = historical_df.iloc[:, :NUM_MAIN]
else:
    # Fallback to simulated historical data
    np.random.seed(42)
    historical_numbers = pd.DataFrame(
        [np.random.choice(NUMBERS_RANGE, NUM_MAIN, replace=False) for _ in range(100)],
        columns=[f"N{i+1}" for i in range(NUM_MAIN)]
    )

# Frequency table from historical data
flat_numbers = historical_numbers.values.flatten()
historical_freq = pd.Series(0, index=NUMBERS_RANGE)
for num in flat_numbers:
    try:
        clean_num = int(float(num))  # handles strings and floats
        if clean_num in historical_freq.index:
            historical_freq[clean_num] += 1
    except:
        continue

st.subheader("📊 Historical Frequency Chart")
st.bar_chart(historical_freq)

# Components
def hot_zone_score(freqs):
    zone_thresh = np.percentile(freqs, 75)
    return (freqs >= zone_thresh).astype(int)

def cold_zone_score(freqs):
    zone_thresh = np.percentile(freqs, 25)
    return (freqs <= zone_thresh).astype(int)

def sequential_penalty(numbers):
    numbers = sorted(numbers)
    return sum(1 for i in range(len(numbers) - 1) if numbers[i+1] - numbers[i] == 1)

def entropy_score(numbers):
    probs = np.array([historical_freq[n]/historical_freq.sum() for n in numbers])
    return entropy(probs, base=2)

def mahalanobis_distance(numbers, historical_matrix):
    if len(historical_matrix) < 2:
        return 0
    mu = np.mean(historical_matrix, axis=0)
    cov = np.cov(historical_matrix, rowvar=False)
    try:
        return distance.mahalanobis(numbers, mu, np.linalg.inv(cov))
    except:
        return 0

def generate_mode_c_predictions():
    predictions = []
    for _ in range(NUM_SETS):
        scores = historical_freq.copy()
        scores += np.random.randn(47) * 0.5
        scores += hot_zone_score(historical_freq) * 1.5
        scores -= cold_zone_score(historical_freq) * 0.5

        scores = np.maximum(scores.to_numpy(), 0)  # Ensure NumPy array for math

        if scores.sum() == 0:
            probs = np.ones_like(scores) / len(scores)
        else:
            probs = scores / scores.sum()

        picks = np.random.choice(NUMBERS_RANGE, size=NUM_MAIN, replace=False, p=probs)
        predictions.append(sorted(picks))
    return predictions

def evaluate_final_formula(predictions, historical_matrix):
    hz_score = hot_zone_score(historical_freq)
    cz_score = cold_zone_score(historical_freq)
    scored = []
    for entry in predictions:
        F = np.mean([historical_freq[n]/historical_freq.sum() for n in entry])
        H = np.mean([hz_score[n] for n in entry])
        C = np.mean([cz_score[n] for n in entry])
        S = sequential_penalty(entry)
        E = entropy_score(entry)
        M = mahalanobis_distance(entry, historical_matrix)
        score = (alpha*F + beta*H + gamma*C) - S + E + M
        scored.append((entry, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored

# Run predictions
st.subheader("🔮 Predictive Simulation")
mode_c_preds = generate_mode_c_predictions()
historical_matrix = historical_numbers.values.tolist()
evaluated = evaluate_final_formula(mode_c_preds, historical_matrix)

top_df = pd.DataFrame([x[0] for x in evaluated[:10]], columns=[f"N{i+1}" for i in range(NUM_MAIN)])
top_df["Score"] = [round(x[1], 3) for x in evaluated[:10]]
st.markdown("### 🏆 Top 10 Predicted Sets")
st.dataframe(top_df)

csv = top_df.to_csv(index=False).encode('utf-8')
st.download_button("⬇ Download Top Predictions", csv, "oz_lotto_predictions.csv", "text/csv")
