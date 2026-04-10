#server\evaluation\compare_outputs.py

import json
import os

# 1. Setup paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# 2. File Names
BASELINE_FILE = "baseline_full.json"
MINDWEAVE_FILE = "mindweave_report_full.json"
OUTPUT_FILE = "comparison_summary.txt" # The new file we will create

def load_data():
    # Look in the results folder specifically
    b_path = os.path.join(RESULTS_DIR, BASELINE_FILE)
    m_path = os.path.join(RESULTS_DIR, MINDWEAVE_FILE)
    
    if not os.path.exists(b_path) or not os.path.exists(m_path):
        raise FileNotFoundError(f". Missing files in {RESULTS_DIR}. Ensure they are moved there first!")
        
    with open(b_path) as f: b = json.load(f)
    with open(m_path) as f: m = json.load(f)
    return b, m

try:
    baseline, mindweave = load_data()
    
    # We will collect the text in a list to save it later
    report_lines = ["=== MINDWEAVE VS BASELINE EVALUATION REPORT ===\n"]

    for i, (b, m) in enumerate(zip(baseline, mindweave), 1):
        case_text = (
            f"\n" + "="*70 +
            f"\n. Case {i}" +
            f"\n. User: {b['input']}" +
            f"\n\n. Baseline Response: {b['response']}" +
            f"\n. Baseline Score: {b['score']:.2f}" +
            f"\n\n. MindWeave Response: {m['response']}" +
            f"\n. MindWeave Agent: {m['agent']}" +
            f"\n. MindWeave Score: {m['score']:.2f}\n"
        )
        print(case_text)
        report_lines.append(case_text)

    # 3. SAVE THE OUTPUT TO THE RESULTS FOLDER
    save_path = os.path.join(RESULTS_DIR, OUTPUT_FILE)
    with open(save_path, "w", encoding="utf-8") as f:
        f.writelines(report_lines)
        
    print(f"\n.SUCCESS: Comparison report saved to: {save_path}")

except Exception as e:
    print(e)