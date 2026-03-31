import os
import numpy as np
from pathlib import Path

def calculate_fnr_fpr_from_val_stats(tp, fp, fn, total_images=None):
    total_positives = tp + fn  
    total_predictions = tp + fp  
    
    precision = tp / total_predictions if total_predictions > 0 else 0
    recall = tp / total_positives if total_positives > 0 else 0
    fnr = fn / total_positives if total_positives > 0 else 0

    fppi = fp / total_images if total_images else None
    
    return {
        'Precision': precision,
        'Recall': recall,
        'FNR': fnr,
        'FPPI': fppi,
        'TP': tp,
        'FP': fp,
        'FN': fn,
        'Total_Positives': total_positives,
        'Total_Predictions': total_predictions
    }

if __name__ == "__main__":
    experiments = [        {"name": "Clean", "P": 0.0, "R": 0.0, "images": 0, "instances": 0},
        {"name": "AAP", "P": 0.0, "R": 0.0, "images": 0, "instances": 0},
        {"name": "AdvPatch", "P": 0.0, "R": 0.0, "images": 0, "instances": 0}, 
        {"name": "DPatch", "P": 0.0, "R": 0.0, "images": 0, "instances": 0},
        {"name": "RobustDPatch", "P": 0.0, "R": 0.0, "images": 0, "instances": 0}
    ]
    
    print(f"{'Method':12} | {'P':<6} {'R':<6} | {'FNR':<6} {'FPPI':<6} | {'TP':<4} {'FP':<4} {'FN':<4}")
    print("-" * 70)
    
    for exp in experiments:
        tp = exp["R"] * exp["instances"]
        fp = (tp / exp["P"]) - tp if exp["P"] > 0 else 0
        fn = exp["instances"] - tp
        
        result = calculate_fnr_fpr_from_val_stats(tp, fp, fn, exp["images"])
        
        print(f"{exp['name']:12} | {exp['P']:<6.3f} {exp['R']:<6.3f} | "
              f"{result['FNR']:<6.3f} {result['FPPI']:<6.3f} | "
              f"{result['TP']:<4.0f} {result['FP']:<4.0f} {result['FN']:<4.0f}")