# 🚁 Gym-PyBullet Integrated Simulation

## Overview

This simulation implements the complete integrated training+testing approach for federated multi-task learning with **15 drones per cluster (45 total)**, optimized for fast execution with **no delays**.

## Features

✅ **15 Drones per Cluster** (45 total)  
✅ **Per-Cluster Testing** (Equal + Dirichlet splits)  
✅ **CH1 as Global Aggregator** (Hierarchical: CH0/CH2 → CH1 → broadcast)  
✅ **CH0 Compromise Scenarios** (Convergence + Transient)  
✅ **Model Checkpointing** (Every round saved to disk)  
✅ **All 3 Tasks** (Traffic, Duration, Bandwidth classification)  
✅ **NO Delays** (Fast simulation)  

## Quick Start

### Run Both Scenarios
```bash
python gym_integrated_simulation.py --scenario both
```

### Run Convergence Only (125 rounds)
```bash
python gym_integrated_simulation.py --scenario convergence
```

### Run Transient Only (30 rounds)
```bash
python gym_integrated_simulation.py --scenario transient
```

## Configuration

Edit `SimulationConfig` in the script to customize:

```python
@dataclass
class SimulationConfig:
    n_drones_per_cluster: int = 15      # Drones per cluster
    n_clusters: int = 3                  # Number of clusters
    total_drones: int = 45               # Total drones
    
    local_epochs: int = 1                # Local training epochs
    lr: float = 1e-3                     # Learning rate
    
    global_aggregator_cluster: int = 1   # CH1 as global aggregator
    client_frac: float = 1.0             # 100% participation
    
    convergence_rounds: int = 125        # Convergence scenario rounds
    convergence_compromise_round: int = 111  # Compromise round
    transient_rounds: int = 30           # Transient scenario rounds
    transient_compromise_round: int = 11     # Compromise round
```

## Output Structure

### Convergence Scenario
```
trained_models/gym_convergence_integrated/
├── model_round_1.pkl
├── model_round_2.pkl
├── ...
└── model_round_125.pkl
```

### Transient Scenario
```
trained_models/gym_transient_integrated/
├── model_round_1.pkl
├── model_round_2.pkl
├── ...
└── model_round_30.pkl
```

## Checkpoint Format

Each checkpoint contains:

```python
checkpoint = {
    'round': int,                          # Round number
    'global_params': list[np.ndarray],     # Global model parameters
    'cluster_params': {                    # Per-cluster parameters
        0: list[np.ndarray],
        1: list[np.ndarray],
        2: list[np.ndarray]
    },
    'cluster_weights': {0: int, 1: int, 2: int},
    'participating_clusters': [0, 1, 2],
    'recovery_phase': str,                 # 'normal', 'detection', 'continuity', 'complete'
    'ch_compromised': bool,
    'test_results': {
        'equal_split': {
            0: {'traffic_accuracy': float, 'duration_accuracy': float, 'bandwidth_accuracy': float},
            1: {...},
            2: {...}
        },
        'dirichlet_split': {
            0: {...},
            1: {...},
            2: {...}
        }
    },
    'avg_accuracy': float
}
```

## Load and Analyze Results

```python
import pickle
import numpy as np

# Load a checkpoint
with open('trained_models/gym_convergence_integrated/model_round_100.pkl', 'rb') as f:
    checkpoint = pickle.load(f)

# Access cluster 1's test results
print(f"Round: {checkpoint['round']}")
print(f"Recovery Phase: {checkpoint['recovery_phase']}")
print(f"Cluster 1 Equal Split:")
print(f"  Traffic: {checkpoint['test_results']['equal_split'][1]['traffic_accuracy']:.4f}")
print(f"  Duration: {checkpoint['test_results']['equal_split'][1]['duration_accuracy']:.4f}")
print(f"  Bandwidth: {checkpoint['test_results']['equal_split'][1]['bandwidth_accuracy']:.4f}")
```

## Scenarios

### Convergence (125 rounds)
- **Normal Training**: Rounds 1-110
- **CH0 Compromise**: Round 111
- **Detection & D&R-E**: Rounds 112-118 (7 rounds, CH0 offline)
- **Continuity**: Rounds 119-121 (30% → 70% → 100% re-entry)
- **Stabilization**: Rounds 122-125

### Transient (30 rounds)
- **Normal Training**: Rounds 1-10
- **CH0 Compromise**: Round 11
- **Detection & D&R-E**: Rounds 12-18 (7 rounds, CH0 offline)
- **Continuity**: Rounds 19-21 (30% → 70% → 100% re-entry)
- **Stabilization**: Rounds 22-30

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    HIERARCHICAL FEDERATION                   │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Cluster 0           Cluster 1           Cluster 2          │
│  (15 drones)        (15 drones)          (15 drones)        │
│       │                  │                   │              │
│       ↓                  ↓                   ↓              │
│      CH0                CH1                 CH2             │
│       │                  │                   │              │
│       └────────────→    CH1*   ←────────────┘              │
│                    (Global Aggregator)                       │
│                          │                                   │
│                    Broadcast Back                            │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Key Differences from Notebook Implementation

| Feature | Notebook | Gym Simulation |
|---------|----------|----------------|
| Clients | 600 (200/cluster) | 45 (15/cluster) |
| Training Data | Real CESNET data | Dummy data (placeholder) |
| Delays | Optional | None (fast) |
| Visualization | PyBullet GUI | Console output |
| Purpose | Full experiment | Quick testing |

## Performance

With 45 drones and no delays:
- **Convergence (125 rounds)**: ~2-5 minutes
- **Transient (30 rounds)**: ~30-90 seconds

## Next Steps

1. **Replace dummy data**: Load real training data partitions from the notebook
2. **Add PyBullet visualization**: Integrate with `run_visual_simulation.py`
3. **Add KPI tracking**: Integrate `ComprehensiveKPITracker` from notebook
4. **Visualize results**: Create plots for per-cluster accuracy curves

## Requirements

```bash
pip install tensorflow flwr numpy
```

## Notes

- The simulation uses dummy training data for speed. To use real data, load the partitions from `final_experiment.ipynb`
- All test data is real (from preprocessed CESNET dataset)
- Checkpoints are fully compatible with the notebook implementation
- NO delays means faster iteration for development and testing

## Troubleshooting

**Issue**: `FileNotFoundError: trained_models/preprocessed_test_data.pkl`  
**Solution**: Run the data preprocessing cells in `final_experiment.ipynb` first

**Issue**: Out of memory  
**Solution**: Reduce `n_drones_per_cluster` or `batch_size` in config

**Issue**: Slow execution  
**Solution**: Reduce `convergence_rounds` or use `--scenario transient` for quick tests

