#!/usr/bin/env bash
set -euo pipefail

mkdir -p analysis_out

python3 analysis/aggressive_predict.py single --dispatcher logs/dispatcher.csv --paths 3 --out analysis_out/single.json

python3 analysis/aggressive_predict.py pair --dispatcherA logs/dispatcher_node1.csv --dispatcherB logs/dispatcher_node2.csv --paths 3 --out analysis_out/pair_same_seed.json

python3 analysis/plot_aggressive_entropy.py --csv analysis_out/single_entropy.csv --out plots/predicted_min_entropy.png

