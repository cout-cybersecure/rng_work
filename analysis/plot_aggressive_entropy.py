import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    
    df = pd.read_csv(args.csv)
    df = df.dropna(subset=['window_index', 'observed_min_entropy', 'predicted_min_entropy'])
    if df.empty:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        plt.figure(figsize=(10, 6))
        plt.title('Min-Entropy Series (empty)')
        plt.savefig(args.out, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        return
    
    df = df.sort_values('window_index')
    x = df['window_index'].to_numpy()
    observed = df['observed_min_entropy'].to_numpy()
    predicted = df['predicted_min_entropy'].to_numpy()
    
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, observed, label='Observed Min-Entropy', linewidth=2, color='green')
    plt.plot(x, predicted, label='Predicted Min-Entropy', linewidth=2, color='orange')
    plt.xlabel('Window Index (256 steps)')
    plt.ylabel('Min-Entropy (bits)')
    plt.title('Observed vs Predicted Rolling Min-Entropy (window=256)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim(0, max(np.max(observed), np.max(predicted)) * 1.1 if len(x) > 0 else 1)
    plt.xlim(x.min(), x.max())
    plt.savefig(args.out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

if __name__ == "__main__":
    main()

