# mindweave_env\server\evaluation\plot_results.py

import matplotlib.pyplot as plt
import json
import os
import numpy as np

# 1. SETUP PATHS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Point to the results folder
RESULTS_DIR = os.path.join(BASE_DIR, "results")

def load_data(filename):
    # Try results folder first, then local
    paths = [os.path.join(RESULTS_DIR, filename), os.path.join(BASE_DIR, filename)]
    for path in paths:
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    raise FileNotFoundError(f". Could not find {filename}")

try:
    # 2. LOAD SCORES
    baseline = np.array(load_data("baseline.json")) # Make sure names match
    mindweave = np.array(load_data("mindweave_scores.json"))

    # 3. PLOTTING
    plt.figure(figsize=(10, 6))

    # Plot raw points
    plt.scatter(range(len(baseline)), baseline, color='gray', alpha=0.3)
    plt.scatter(range(len(mindweave)), mindweave, color='blue', alpha=0.3)

    # Plot trend lines
    plt.plot(baseline, label="Baseline (LLM)", linestyle="--", color='gray', marker='o')
    plt.plot(mindweave, label="MindWeave (RL + Agents)", color='blue', linewidth=2, marker='s')

    # 4. AVERAGES (The "Money" Lines for the demo)
    plt.axhline(np.mean(baseline), color='gray', linestyle=":", label=f"Baseline Avg: {np.mean(baseline):.2f}")
    plt.axhline(np.mean(mindweave), color='blue', linestyle=":", label=f"MindWeave Avg: {np.mean(mindweave):.2f}")

    plt.xlabel("Test Case / Episode")
    plt.ylabel("Therapeutic Score (0-1)")
    plt.title("MindWeave Performance vs. Standard LLM")
    plt.ylim(0, 1.1)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.5)

    # 5. SAVE THE PLOT
    plot_path = os.path.join(RESULTS_DIR, "performance_chart.png")
    plt.savefig(plot_path)
    print(f".Chart saved to: {plot_path}")
    
    plt.show()

    # SUMMARY
    print("\n. FINAL PERFORMANCE SUMMARY:")
    print(f"Baseline Avg:  {np.mean(baseline):.3f}")
    print(f"MindWeave Avg: {np.mean(mindweave):.3f}")
    print(f"Improvement:   {((np.mean(mindweave) - np.mean(baseline)) / np.mean(baseline) * 100):.1f}%")

except Exception as e:
    print(e)