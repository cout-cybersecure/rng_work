import argparse
import json
import os
import sys
import math
import pandas as pd
import numpy as np
from collections import defaultdict

def read_dispatcher_csv(path):
    df = pd.read_csv(path, low_memory=False)
    df = df[df['recv_ns'] != 'recv_ns']
    df['recv_ns'] = pd.to_numeric(df['recv_ns'], errors='coerce')
    df['jitter_ns'] = pd.to_numeric(df['jitter_ns'], errors='coerce')
    df['offset'] = pd.to_numeric(df['offset'], errors='coerce')
    df['path_id'] = pd.to_numeric(df['path_id'], errors='coerce')
    df = df.dropna()
    df = df.sort_values('recv_ns').reset_index(drop=True)
    return df

def single_mode(dispatcher_path, paths, out_path):
    df = read_dispatcher_csv(dispatcher_path)
    
    rows = []
    median_dt = df['recv_ns'].diff().median()
    if pd.isna(median_dt):
        median_dt = 0
    
    for i in range(len(df)):
        row = {
            'index': i,
            'path_id': int(df.iloc[i]['path_id']),
            'jitter_ns': int(df.iloc[i]['jitter_ns']),
            'recv_ns': int(df.iloc[i]['recv_ns']),
            'offset': int(df.iloc[i]['offset'])
        }
        
        if i == 0:
            row['dt_ns'] = int(median_dt)
        else:
            row['dt_ns'] = int(df.iloc[i]['recv_ns'] - df.iloc[i-1]['recv_ns'])
        
        row['offset_mod_256'] = row['offset'] % 256
        row['offset_mod_1024'] = row['offset'] % 1024
        
        row['prev_path_1'] = int(df.iloc[i-1]['path_id']) if i > 0 else -1
        row['prev_path_2'] = int(df.iloc[i-2]['path_id']) if i > 1 else -1
        row['prev_path_3'] = int(df.iloc[i-3]['path_id']) if i > 2 else -1
        
        rows.append(row)
    
    T_freq = defaultdict(lambda: defaultdict(int))
    T_joint = defaultdict(lambda: defaultdict(int))
    T_time = defaultdict(lambda: defaultdict(int))
    
    predictions = []
    correct_top1 = 0
    correct_top2 = 0
    
    for i, row in enumerate(rows):
        if i == 0:
            predictions.append({
                'predicted_path': 0,
                'top2_set': [0, 1] if paths > 1 else [0],
                'attacker_success_top1': 0
            })
            continue
        
        jitter_bucket = min(row['jitter_ns'] // 2000, 63)
        dt_bucket = min(row['dt_ns'] // 50000, 63)
        off256_bucket = row['offset_mod_256'] // 8
        off1024_bucket = row['offset_mod_1024'] // 32
        
        prev_path = row['prev_path_1']
        
        if prev_path >= 0:
            T_freq[prev_path][row['path_id']] += 1
            T_joint[(prev_path, jitter_bucket, off256_bucket)][row['path_id']] += 1
            T_time[(prev_path, dt_bucket)][row['path_id']] += 1
        
        counts = defaultdict(int)
        
        joint_key = (prev_path, jitter_bucket, off256_bucket)
        time_key = (prev_path, dt_bucket)
        
        if joint_key in T_joint and len(T_joint[joint_key]) > 0:
            for path, count in T_joint[joint_key].items():
                counts[path] += count
        elif time_key in T_time and len(T_time[time_key]) > 0:
            for path, count in T_time[time_key].items():
                counts[path] += count
        elif prev_path in T_freq and len(T_freq[prev_path]) > 0:
            for path, count in T_freq[prev_path].items():
                counts[path] += count
        else:
            for p in range(paths):
                counts[p] = 1
        
        if len(counts) == 0:
            predicted_path = 0
            top2_set = sorted(range(paths))[:2] if paths > 1 else [0]
        else:
            sorted_paths = sorted(counts.items(), key=lambda x: x[1], reverse=True)
            predicted_path = sorted_paths[0][0]
            top2_set = [p for p, _ in sorted_paths[:2]]
            if len(top2_set) < 2 and paths > 1:
                for p in range(paths):
                    if p not in top2_set:
                        top2_set.append(p)
                        if len(top2_set) >= 2:
                            break
            top2_set = top2_set[:2] if paths > 1 else top2_set[:1]
        
        attacker_success_top1 = 1 if row['path_id'] == predicted_path else 0
        attacker_success_top2 = 1 if row['path_id'] in top2_set else 0
        
        correct_top1 += attacker_success_top1
        correct_top2 += attacker_success_top2
        
        predictions.append({
            'predicted_path': int(predicted_path),
            'top2_set': [int(p) for p in top2_set],
            'attacker_success_top1': attacker_success_top1
        })
    
    observed_entropies = []
    predicted_entropies = []
    entropy_series = []
    
    window_idx = 0
    for window_start in range(0, len(rows) - 255, 256):
        window_end = min(window_start + 256, len(rows))
        window_rows = rows[window_start:window_end]
        
        path_counts = defaultdict(int)
        for r in window_rows:
            path_counts[r['path_id']] += 1
        
        if len(path_counts) > 0 and len(window_rows) > 0:
            pmax = max(path_counts.values()) / len(window_rows)
            H_obs = -math.log2(pmax) if pmax > 0 else 0.0
        else:
            H_obs = 0.0
        observed_entropies.append(H_obs)
        
        T_freq_window = defaultdict(lambda: defaultdict(int))
        T_joint_window = defaultdict(lambda: defaultdict(int))
        T_time_window = defaultdict(lambda: defaultdict(int))
        
        for i in range(window_start):
            if i == 0:
                continue
            row = rows[i]
            prev_path = row['prev_path_1']
            if prev_path >= 0:
                jitter_bucket = min(row['jitter_ns'] // 2000, 63)
                dt_bucket = min(row['dt_ns'] // 50000, 63)
                off256_bucket = row['offset_mod_256'] // 8
                T_freq_window[prev_path][row['path_id']] += 1
                T_joint_window[(prev_path, jitter_bucket, off256_bucket)][row['path_id']] += 1
                T_time_window[(prev_path, dt_bucket)][row['path_id']] += 1
        
        pred_counts = defaultdict(int)
        for i in range(window_start, window_end):
            if i == 0:
                continue
            row = rows[i]
            prev_path = row['prev_path_1']
            jitter_bucket = min(row['jitter_ns'] // 2000, 63)
            dt_bucket = min(row['dt_ns'] // 50000, 63)
            off256_bucket = row['offset_mod_256'] // 8
            
            counts = defaultdict(int)
            joint_key = (prev_path, jitter_bucket, off256_bucket)
            time_key = (prev_path, dt_bucket)
            
            if joint_key in T_joint_window and len(T_joint_window[joint_key]) > 0:
                for path, count in T_joint_window[joint_key].items():
                    counts[path] += count
            elif time_key in T_time_window and len(T_time_window[time_key]) > 0:
                for path, count in T_time_window[time_key].items():
                    counts[path] += count
            elif prev_path in T_freq_window and len(T_freq_window[prev_path]) > 0:
                for path, count in T_freq_window[prev_path].items():
                    counts[path] += count
            else:
                for p in range(paths):
                    counts[p] = 1
            
            if len(counts) > 0:
                total = sum(counts.values())
                for path in range(paths):
                    prob = counts.get(path, 0) / total
                    pred_counts[path] += prob
        
        if len(pred_counts) > 0 and len(window_rows) > 0:
            pmax = max(pred_counts.values()) / len(window_rows)
            H_pred = -math.log2(pmax) if pmax > 0 else 0.0
        else:
            H_pred = 0.0
        predicted_entropies.append(H_pred)
        entropy_series.append({
            'window_index': window_idx,
            'observed_min_entropy': H_obs,
            'predicted_min_entropy': H_pred
        })
        window_idx += 1
    
    acc_backoff = correct_top1 / len(rows) if len(rows) > 0 else 0.0
    top2_backoff = correct_top2 / len(rows) if len(rows) > 0 else 0.0
    avg_observed = np.mean(observed_entropies) if observed_entropies else 0.0
    avg_predicted = np.mean(predicted_entropies) if predicted_entropies else 0.0
    
    result = {
        "samples": len(rows),
        "paths": paths,
        "acc_backoff": float(acc_backoff),
        "top2_backoff": float(top2_backoff),
        "attacker_success_top1": float(acc_backoff),
        "avg_observed_min_entropy_256": float(avg_observed),
        "avg_predicted_min_entropy_256": float(avg_predicted)
    }
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    entropy_csv_path = os.path.splitext(out_path)[0] + "_entropy.csv"
    if entropy_series:
        entropy_df = pd.DataFrame(entropy_series)
        entropy_df.to_csv(entropy_csv_path, index=False)
    else:
        pd.DataFrame(columns=['window_index', 'observed_min_entropy', 'predicted_min_entropy']).to_csv(entropy_csv_path, index=False)

def pair_mode(dispatcher_a_path, dispatcher_b_path, paths, out_path):
    df_a = read_dispatcher_csv(dispatcher_a_path)
    df_b = read_dispatcher_csv(dispatcher_b_path)
    
    best_shift = 0
    best_match_rate = 0.0
    
    for shift_ns in range(-5000000, 5000001, 50000):
        df_b_shifted = df_b.copy()
        df_b_shifted['recv_ns'] = df_b_shifted['recv_ns'] + shift_ns
        
        merged = pd.merge_asof(df_a.sort_values('recv_ns'), df_b_shifted.sort_values('recv_ns'), 
                               on='recv_ns', direction='nearest', suffixes=('_a', '_b'), tolerance=100000)
        merged = merged.dropna(subset=['path_id_a', 'path_id_b'])
        
        if len(merged) == 0:
            continue
        
        matches = (merged['path_id_a'] == merged['path_id_b']).sum()
        match_rate = matches / len(merged)
        
        if match_rate > best_match_rate:
            best_match_rate = match_rate
            best_shift = shift_ns
    
    df_b_aligned = df_b.copy()
    df_b_aligned['recv_ns'] = df_b_aligned['recv_ns'] + best_shift
    
    merged = pd.merge_asof(df_a.sort_values('recv_ns'), df_b_aligned.sort_values('recv_ns'),
                           on='recv_ns', direction='nearest', suffixes=('_a', '_b'), tolerance=100000)
    merged = merged.dropna(subset=['path_id_a', 'path_id_b'])
    
    aligned_sequence = []
    for _, row in merged.iterrows():
        a_path = int(row['path_id_a'])
        a_jitter = int(row['jitter_ns_a'])
        a_offset = int(row['offset_a'])
        b_path = int(row['path_id_b'])
        
        a_jitter_bucket = min(a_jitter // 2000, 63)
        a_off256_bucket = (a_offset % 256) // 8
        
        aligned_sequence.append({
            'A_path': a_path,
            'A_jitter_bucket': a_jitter_bucket,
            'A_off256_bucket': a_off256_bucket,
            'B_path': b_path
        })
    
    X1 = defaultdict(lambda: defaultdict(int))
    X2 = defaultdict(lambda: defaultdict(int))
    X3 = defaultdict(lambda: defaultdict(int))
    
    correct_top1 = 0
    
    for i, step in enumerate(aligned_sequence):
        if i > 0:
            X1[step['A_path']][step['B_path']] += 1
            X2[(step['A_path'], step['A_jitter_bucket'])][step['B_path']] += 1
            X3[(step['A_path'], step['A_jitter_bucket'], step['A_off256_bucket'])][step['B_path']] += 1
        
        if i == 0:
            continue
        
        counts = defaultdict(int)
        key3 = (step['A_path'], step['A_jitter_bucket'], step['A_off256_bucket'])
        key2 = (step['A_path'], step['A_jitter_bucket'])
        key1 = step['A_path']
        
        if key3 in X3 and len(X3[key3]) > 0:
            for path, count in X3[key3].items():
                counts[path] += count
        elif key2 in X2 and len(X2[key2]) > 0:
            for path, count in X2[key2].items():
                counts[path] += count
        elif key1 in X1 and len(X1[key1]) > 0:
            for path, count in X1[key1].items():
                counts[path] += count
        else:
            for p in range(paths):
                counts[p] = 1
        
        if len(counts) > 0:
            predicted_path = max(counts.items(), key=lambda x: x[1])[0]
        else:
            predicted_path = 0
        
        if predicted_path == step['B_path']:
            correct_top1 += 1
    
    cross_match_rate = best_match_rate
    cross_attacker_success_top1 = correct_top1 / (len(aligned_sequence) - 1) if len(aligned_sequence) > 1 else 0.0
    
    window_entropies = []
    for window_start in range(0, len(aligned_sequence) - 255, 256):
        window_end = min(window_start + 256, len(aligned_sequence))
        window_steps = aligned_sequence[window_start:window_end]
        
        path_counts = defaultdict(int)
        for step in window_steps:
            path_counts[step['B_path']] += 1
        
        if len(path_counts) > 0:
            pmax = max(path_counts.values()) / len(window_steps)
            H = -math.log2(pmax) if pmax > 0 else 0.0
            window_entropies.append(H)
    
    cross_window_min_entropy_256 = np.mean(window_entropies) if window_entropies else 0.0
    
    result = {
        "cross_match_rate": float(cross_match_rate),
        "cross_attacker_success_top1": float(cross_attacker_success_top1),
        "cross_window_min_entropy_256": float(cross_window_min_entropy_256),
        "best_shift_ns": int(best_shift),
        "samples_overlap": len(aligned_sequence)
    }
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode', required=True)
    
    single_parser = subparsers.add_parser('single')
    single_parser.add_argument('--dispatcher', required=True)
    single_parser.add_argument('--paths', type=int, required=True)
    single_parser.add_argument('--out', required=True)
    
    pair_parser = subparsers.add_parser('pair')
    pair_parser.add_argument('--dispatcherA', required=True)
    pair_parser.add_argument('--dispatcherB', required=True)
    pair_parser.add_argument('--paths', type=int, required=True)
    pair_parser.add_argument('--out', required=True)
    
    args = parser.parse_args()
    
    if args.mode == 'single':
        single_mode(args.dispatcher, args.paths, args.out)
    elif args.mode == 'pair':
        pair_mode(args.dispatcherA, args.dispatcherB, args.paths, args.out)

if __name__ == "__main__":
    main()

