#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs

./dispatcher --bind 127.0.0.1:5000 \
    --backends 127.0.0.1:6001,127.0.0.1:6002,127.0.0.1:6003 \
    --jitter_ns 100000 \
    --rng drbg \
    --seed 0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef \
    --nonce 000000000000000000000000 \
    --run 0 \
    --trial 0 \
    --log logs/dispatcher.csv

