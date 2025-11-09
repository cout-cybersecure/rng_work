# Multipath-Routing Randomness Study

Complete repository for a multipath-routing randomness study targeting Linux and C++17.

## Build

```bash
mkdir -p cmake && cmake -S . -B cmake -DCMAKE_BUILD_TYPE=Release && cmake --build cmake -j && cp cmake/generator cmake/dispatcher cmake/backend cmake/predictor ./
```

## Run Single Node

### Terminal 1: Start Backends

```bash
bash scripts/run_backend.sh
```

### Terminal 2: Start Dispatcher (DRBG Mode)

```bash
bash scripts/run_dispatcher_drbg.sh
```

### Terminal 3: Start Generator

```bash
bash scripts/run_generator.sh
```

## Run Two Node Experiments

### Same Seed Two Nodes

```bash
bash scripts/run_same_seed_two_nodes.sh
```

After completion, analyze:

```bash
bash scripts/analyze_same_seed.sh
```

### Quantum RNG Two Nodes

```bash
bash scripts/run_qrng_two_nodes.sh
```

After completion, analyze:

```bash
bash scripts/analyze_qrng.sh
```

## Components

- **backend**: UDP server that echoes probes
- **dispatcher**: UDP server that selects backend paths using RNG and forwards probes
- **generator**: UDP client that sends probes and receives echoes
- **predictor**: Analysis tool for dispatcher CSV logs

## Setup (Optional)

For optimal performance, run once:

```bash
bash scripts/setup_linux.sh
```

This disables GRO, GSO, TSO, LRO on all NICs (except lo) and sets CPU governor to performance.

## Analysis

```bash
python analysis/analyze.py --dispatcher logs/dispatcher.csv --generator logs/generator.csv --outdir plots
```

