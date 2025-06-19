
import pandas as pd
import numpy as np
import random

def generate_mode_c_predictions(csv_path):
    df = pd.read_csv(csv_path)

    frequency = [0] * 35
    gap_score = [0] * 35

    # Parse and validate number frequencies
    for col in df.columns:
        if col.startswith('N'):
            for n in df[col].dropna():
                if isinstance(n, (int, float)):
                    n = int(n)
                    if 1 <= n <= 35:
                        frequency[n - 1] += 1

    frequency = np.array(frequency)
    entropy = np.std(frequency)

    predictions = []
    for _ in range(100):
        selected = []
        weights = frequency + np.random.rand(35)

        while len(selected) < 7:
            choice = int(np.argmax(weights)) + 1
            if choice not in selected:
                selected.append(choice)
                weights[choice - 1] = -np.inf  # mark as used

        selected.sort()
        powerball = random.randint(1, 20)
        predictions.append(selected + [powerball])

    cols = [f"N{i+1}" for i in range(7)] + ["Powerball"]
    return pd.DataFrame(predictions, columns=cols)
