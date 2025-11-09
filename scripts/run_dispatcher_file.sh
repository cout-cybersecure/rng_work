#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <bitstream_path>"
    exit 1
fi

BITSTREAM_PATH="$1"
mkdir -p logs

./dispatcher --bind 127.0.0.1:5000 \
    --backends 127.0.0.1:6001,127.0.0.1:6002,127.0.0.1:6003 \
    --jitter_ns 100000 \
    --rng file \
    --bitstream "$BITSTREAM_PATH" \
    --run 0 \
    --trial 0 \
    --log logs/dispatcher.csv

