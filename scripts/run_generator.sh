#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs

./generator --dst 127.0.0.1:5000 \
    --requests 100000 \
    --interval_us 100 \
    --run 0 \
    --trial 0 \
    --log logs/generator.csv

