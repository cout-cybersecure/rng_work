import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def load_dispatcher(path):
    df = pd.read_csv(path, low_memory=False)
    df = df[df['recv_ns'] != 'recv_ns']
    df['recv_ns'] = pd.to_numeric(df['recv_ns'], errors='coerce')
    df['jitter_ns'] = pd.to_numeric(df['jitter_ns'], errors='coerce')
    df['offset'] = pd.to_numeric(df['offset'], errors='coerce')
    df['path_id'] = pd.to_numeric(df['path_id'], errors='coerce')
    df = df.dropna()
    df = df.sort_values('recv_ns').reset_index(drop=True)
    return df

def build_rows(df):
    recv = df['recv_ns'].to_numpy(dtype=np.int64)
    jitter = df['jitter_ns'].to_numpy(dtype=np.int64)
    offset = df['offset'].to_numpy(dtype=np.int64)
    path = df['path_id'].astype(int).to_numpy()
    n = len(recv)
    rows = []
    if n == 0:
        return rows
    dt = np.zeros(n, dtype=np.int64)
    if n > 1:
        dt[1:] = recv[1:] - recv[:-1]
        median_dt = int(np.median(dt[1:]))
    else:
        median_dt = 0
    dt[0] = median_dt
    for i in range(n):
        row = {
            'index': i,
            'path_id': int(path[i]),
            'recv_ns': int(recv[i]),
            'jitter_ns': int(jitter[i]),
            'offset': int(offset[i]),
            'dt_ns': int(dt[i]),
            'offset_mod_256': int(offset[i] % 256),
            'offset_mod_1024': int(offset[i] % 1024),
            'prev_path_1': int(path[i - 1]) if i > 0 else -1,
            'prev_path_2': int(path[i - 2]) if i > 1 else -1,
            'prev_path_3': int(path[i - 3]) if i > 2 else -1,
            'jitter_bucket': int(min(jitter[i] // 2000, 63)),
            'dt_bucket': int(min(dt[i] // 50000, 63)),
            'off256_bucket': int((offset[i] % 256) // 8),
            'off1024_bucket': int((offset[i] % 1024) // 32)
        }
        rows.append(row)
    return rows

def evaluate_rows(rows, paths):
    T_freq = defaultdict(lambda: defaultdict(int))
    T_joint = defaultdict(lambda: defaultdict(int))
    T_time = defaultdict(lambda: defaultdict(int))
    correct_top1 = 0
    for row in rows:
        prev_path = row['prev_path_1']
        if prev_path >= 0:
            T_freq[prev_path][row['path_id']] += 1
            T_joint[(prev_path, row['jitter_bucket'], row['off256_bucket'])][row['path_id']] += 1
            T_time[(prev_path, row['dt_bucket'])][row['path_id']] += 1
        counts = defaultdict(int)
        joint_key = (prev_path, row['jitter_bucket'], row['off256_bucket'])
        time_key = (prev_path, row['dt_bucket'])
        if joint_key in T_joint and len(T_joint[joint_key]) > 0:
            for p, c in T_joint[joint_key].items():
                counts[p] += c
        elif time_key in T_time and len(T_time[time_key]) > 0:
            for p, c in T_time[time_key].items():
                counts[p] += c
        elif prev_path in T_freq and len(T_freq[prev_path]) > 0:
            for p, c in T_freq[prev_path].items():
                counts[p] += c
        else:
            for p in range(paths):
                counts[p] = 1
        if len(counts) == 0:
            predicted_path = 0
        else:
            predicted_path = max(counts.items(), key=lambda x: x[1])[0]
        if row['path_id'] == predicted_path:
            correct_top1 += 1
    acc = correct_top1 / len(rows) if len(rows) > 0 else 0.0
    observed = []
    predicted = []
    windows = []
    window_idx = 0
    for window_start in range(0, len(rows) - 255, 256):
        window_end = min(window_start + 256, len(rows))
        window_rows = rows[window_start:window_end]
        path_counts = defaultdict(int)
        for r in window_rows:
            path_counts[r['path_id']] += 1
        if len(path_counts) > 0 and len(window_rows) > 0:
            pmax = max(path_counts.values()) / len(window_rows)
            H_obs = -np.log2(pmax) if pmax > 0 else 0.0
        else:
            H_obs = 0.0
        observed.append(H_obs)
        T_freq_w = defaultdict(lambda: defaultdict(int))
        T_joint_w = defaultdict(lambda: defaultdict(int))
        T_time_w = defaultdict(lambda: defaultdict(int))
        for j in range(window_start):
            r = rows[j]
            prev_path = r['prev_path_1']
            if prev_path >= 0:
                T_freq_w[prev_path][r['path_id']] += 1
                T_joint_w[(prev_path, r['jitter_bucket'], r['off256_bucket'])][r['path_id']] += 1
                T_time_w[(prev_path, r['dt_bucket'])][r['path_id']] += 1
        pred_counts = defaultdict(float)
        for j in range(window_start, window_end):
            if j == 0:
                continue
            r = rows[j]
            prev_path = r['prev_path_1']
            counts = defaultdict(int)
            joint_key = (prev_path, r['jitter_bucket'], r['off256_bucket'])
            time_key = (prev_path, r['dt_bucket'])
            if joint_key in T_joint_w and len(T_joint_w[joint_key]) > 0:
                for p, c in T_joint_w[joint_key].items():
                    counts[p] += c
            elif time_key in T_time_w and len(T_time_w[time_key]) > 0:
                for p, c in T_time_w[time_key].items():
                    counts[p] += c
            elif prev_path in T_freq_w and len(T_freq_w[prev_path]) > 0:
                for p, c in T_freq_w[prev_path].items():
                    counts[p] += c
            else:
                for p in range(paths):
                    counts[p] = 1
            if len(counts) > 0:
                total = sum(counts.values())
                for p in range(paths):
                    prob = counts.get(p, 0) / total
                    pred_counts[p] += prob
        if len(pred_counts) > 0 and len(window_rows) > 0:
            pmax = max(pred_counts.values()) / len(window_rows)
            H_pred = -np.log2(pmax) if pmax > 0 else 0.0
        else:
            H_pred = 0.0
        predicted.append(H_pred)
        windows.append(window_idx)
        window_idx += 1
    return {
        'acc': acc,
        'attacker_success': acc,
        'observed': observed,
        'predicted': predicted,
        'windows': windows
    }

def cross_match(df_a, df_b):
    times_a = df_a['recv_ns'].to_numpy(dtype=np.int64)
    times_b = df_b['recv_ns'].to_numpy(dtype=np.int64)
    path_a = df_a['path_id'].astype(int).to_numpy()
    path_b = df_b['path_id'].astype(int).to_numpy()
    shifts = list(range(-5000000, 5000001, 50000))
    tol = 100000
    best_rate = 0.0
    best_shift = shifts[0]
    match_rates = []
    for shift in shifts:
        shifted = times_b + shift
        idx = np.searchsorted(shifted, times_a)
        matches = 0
        overlap = 0
        m = len(shifted)
        for i in range(len(times_a)):
            pos = idx[i]
            best_j = -1
            best_diff = tol + 1
            if pos < m:
                diff = abs(int(shifted[pos] - times_a[i]))
                if diff <= tol and diff < best_diff:
                    best_j = pos
                    best_diff = diff
            if pos > 0:
                diff = abs(int(shifted[pos - 1] - times_a[i]))
                if diff <= tol and diff < best_diff:
                    best_j = pos - 1
                    best_diff = diff
            if best_j != -1:
                overlap += 1
                if path_a[i] == path_b[best_j]:
                    matches += 1
        rate = matches / overlap if overlap > 0 else 0.0
        match_rates.append(rate)
        if rate > best_rate:
            best_rate = rate
            best_shift = shift
    return shifts, match_rates, best_shift, best_rate

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--good', required=True)
    parser.add_argument('--bad', required=True)
    parser.add_argument('--bad2', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    good_df = load_dispatcher(args.good)
    bad_df = load_dispatcher(args.bad)
    bad2_df = load_dispatcher(args.bad2)
    good_rows = build_rows(good_df)
    bad_rows = build_rows(bad_df)
    good_metrics = evaluate_rows(good_rows, 3)
    bad_metrics = evaluate_rows(bad_rows, 3)
    shifts, match_rates, best_shift, best_rate = cross_match(bad_df, bad2_df)
    print(f"good_acc={good_metrics['acc']:.6f}")
    print(f"good_attacker_success={good_metrics['attacker_success']:.6f}")
    print(f"bad_acc={bad_metrics['acc']:.6f}")
    print(f"bad_attacker_success={bad_metrics['attacker_success']:.6f}")
    print(f"cross_match_rate={best_rate:.6f}")
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    if good_metrics['windows']:
        axes[0].plot(good_metrics['windows'], good_metrics['observed'], label='Observed', linewidth=2, color='green')
        axes[0].plot(good_metrics['windows'], good_metrics['predicted'], label='Predicted', linewidth=2, color='orange')
    axes[0].set_title('Independent DRBG Min-Entropy (window=256)')
    axes[0].set_ylabel('Min-Entropy (bits)')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    if bad_metrics['windows']:
        axes[1].plot(bad_metrics['windows'], bad_metrics['observed'], label='Observed', linewidth=2, color='green')
        axes[1].plot(bad_metrics['windows'], bad_metrics['predicted'], label='Predicted', linewidth=2, color='orange')
    axes[1].set_title('Same-Seed DRBG Min-Entropy (window=256)')
    axes[1].set_ylabel('Min-Entropy (bits)')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    axes[2].plot(shifts, match_rates, linewidth=2, color='blue')
    axes[2].set_title('Cross-Node Match Rate vs Time Shift')
    axes[2].set_xlabel('Shift (ns)')
    axes[2].set_ylabel('Match Rate')
    axes[2].grid(True, alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches='tight', facecolor='white')
    secondary = os.path.join('analysis', 'quick_demo.png')
    if os.path.abspath(args.out) != os.path.abspath(secondary):
        os.makedirs(os.path.dirname(secondary), exist_ok=True)
        fig.savefig(secondary, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)

if __name__ == "__main__":
    main()

