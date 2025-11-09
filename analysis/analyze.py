import argparse
import os
import sys
import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_latency_cdf(generator_df, outdir):
    rt_ns = generator_df['rt_ns'].sort_values().values
    n = len(rt_ns)
    if n == 0:
        return
    cdf_y = np.linspace(0, 1, n, endpoint=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(rt_ns, cdf_y, linewidth=2, color='blue')
    ax.set_xlabel('rt_ns (ns)')
    ax.set_ylabel('Empirical CDF')
    ax.set_title('Latency CDF')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    ax.set_xlim(rt_ns.min(), rt_ns.max())
    fig.patch.set_facecolor('white')
    fig.savefig(os.path.join(outdir, 'latency_cdf.png'), dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)

def plot_path_freq(dispatcher_df, outdir):
    path_counts = dispatcher_df['path_id'].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(path_counts.index, path_counts.values, color='steelblue', edgecolor='black', linewidth=1.5, width=0.8)
    ax.set_xlabel('path_id')
    ax.set_ylabel('Frequency')
    ax.set_title('Path Frequency')
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_xticks(path_counts.index)
    ax.set_ylim(0, path_counts.values.max() * 1.1)
    fig.patch.set_facecolor('white')
    fig.savefig(os.path.join(outdir, 'path_freq.png'), dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)

def plot_min_entropy_rolling(dispatcher_df, outdir):
    if len(dispatcher_df) < 256:
        return
    path_ids = dispatcher_df['path_id'].values
    counts = {}
    entropies = []
    for i, path_id in enumerate(path_ids):
        counts[path_id] = counts.get(path_id, 0) + 1
        if i >= 255:
            if i >= 256:
                old_path_id = path_ids[i - 256]
                counts[old_path_id] -= 1
                if counts[old_path_id] == 0:
                    del counts[old_path_id]
            window_size = 256
            max_count = max(counts.values()) if counts else 0
            if max_count > 0:
                entropy = -math.log2(max_count / window_size)
            else:
                entropy = 0.0
            entropies.append(entropy)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(255, len(path_ids)), entropies, linewidth=2, color='green')
    ax.set_xlabel('Index')
    ax.set_ylabel('Min-Entropy (bits)')
    ax.set_title('Rolling Min-Entropy (window=256)')
    ax.grid(True, alpha=0.3)
    if entropies:
        ax.set_ylim(0, max(entropies) * 1.1)
    ax.set_xlim(255, len(path_ids) - 1)
    fig.patch.set_facecolor('white')
    fig.savefig(os.path.join(outdir, 'min_entropy_rolling.png'), dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)

def plot_jitter_hist(dispatcher_df, outdir):
    jitter_ns = dispatcher_df['jitter_ns'].values
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(jitter_ns, bins=50, edgecolor='black', linewidth=1.0, color='steelblue', alpha=0.7)
    ax.set_xlabel('jitter_ns (ns)')
    ax.set_ylabel('Frequency')
    ax.set_title('Jitter Histogram')
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_xlim(jitter_ns.min(), jitter_ns.max())
    ax.set_ylim(0, None)
    fig.patch.set_facecolor('white')
    fig.savefig(os.path.join(outdir, 'jitter_hist.png'), dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dispatcher', required=True, help='Dispatcher CSV file')
    parser.add_argument('--generator', required=True, help='Generator CSV file')
    parser.add_argument('--outdir', required=True, help='Output directory for plots')
    args = parser.parse_args()
    
    if not os.path.exists(args.dispatcher):
        print(f"Error: Dispatcher CSV file not found: {args.dispatcher}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(args.generator):
        print(f"Error: Generator CSV file not found: {args.generator}", file=sys.stderr)
        sys.exit(1)
    
    os.makedirs(args.outdir, exist_ok=True)
    
    try:
        dispatcher_df = pd.read_csv(args.dispatcher, usecols=['recv_ns', 'send_ns', 'jitter_ns', 'rng_source', 'offset', 'path_id'], low_memory=False)
        dispatcher_df = dispatcher_df[dispatcher_df['recv_ns'] != 'recv_ns']
        dispatcher_df['recv_ns'] = pd.to_numeric(dispatcher_df['recv_ns'], errors='coerce')
        dispatcher_df['send_ns'] = pd.to_numeric(dispatcher_df['send_ns'], errors='coerce')
        dispatcher_df['jitter_ns'] = pd.to_numeric(dispatcher_df['jitter_ns'], errors='coerce')
        dispatcher_df['rng_source'] = pd.to_numeric(dispatcher_df['rng_source'], errors='coerce')
        dispatcher_df['offset'] = pd.to_numeric(dispatcher_df['offset'], errors='coerce')
        dispatcher_df['path_id'] = pd.to_numeric(dispatcher_df['path_id'], errors='coerce')
        dispatcher_df = dispatcher_df.dropna()
        dispatcher_df = dispatcher_df.astype({'recv_ns': 'int64', 'send_ns': 'int64', 'jitter_ns': 'int64', 'rng_source': 'int32', 'offset': 'int32', 'path_id': 'int32'})
        
        generator_df = pd.read_csv(args.generator, usecols=['seq', 'send_ns', 'recv_ns', 'rt_ns', 'path_id'], low_memory=False)
        generator_df = generator_df[generator_df['seq'] != 'seq']
        generator_df['seq'] = pd.to_numeric(generator_df['seq'], errors='coerce')
        generator_df['send_ns'] = pd.to_numeric(generator_df['send_ns'], errors='coerce')
        generator_df['recv_ns'] = pd.to_numeric(generator_df['recv_ns'], errors='coerce')
        generator_df['rt_ns'] = pd.to_numeric(generator_df['rt_ns'], errors='coerce')
        generator_df['path_id'] = pd.to_numeric(generator_df['path_id'], errors='coerce')
        generator_df = generator_df.dropna()
        generator_df = generator_df.astype({'seq': 'int64', 'send_ns': 'int64', 'recv_ns': 'int64', 'rt_ns': 'int64', 'path_id': 'int32'})
    except KeyError as e:
        print(f"Error: Missing required column in CSV: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading CSV files: {e}", file=sys.stderr)
        sys.exit(1)
    
    plot_latency_cdf(generator_df, args.outdir)
    plot_path_freq(dispatcher_df, args.outdir)
    plot_min_entropy_rolling(dispatcher_df, args.outdir)
    plot_jitter_hist(dispatcher_df, args.outdir)

if __name__ == "__main__":
    main()

