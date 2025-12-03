# FMTL Visualization System - README

## Overview
Complete PyBullet-based 3D visualization system for Hierarchical Federated Multi-Task Learning with CUAV attack simulation.

## Features
- **600 UAVs** in 3 clusters with visual cluster heads
- **CUAV Attack** visualization with red attacker drone
- **4-Tier Communication** tracking (Members → CH → CH* → CH → Members)
- **D&R-E Protocol** with cluster isolation and gradual recovery
- **CH Re-Election** using LEACH-inspired formula (α·Energy + β·RSSI)
- **HUD Overlay** showing round, phase, accuracies, and cluster status
- **Frame Capture** for creating GIFs/videos of attack sequences

## Quick Start

```python
# Run standalone visualization
from fmtl_visualization import run_visualization_demo
run_visualization_demo('convergence_compromise')
```

```bash
# Run with communication tracking
python test_convergence_with_visualization.py
```

## Scenarios

### Convergence Compromise
- Attack at round 111 (after convergence at ~90)
- D&R-E: Rounds 112-118 (7 rounds)
- Continuity: Rounds 119-121 (30% → 70% → 100%)
- Demonstrates "training in vain" effect

### Transient Compromise  
- Attack at round 11 (early in training)
- D&R-E: Rounds 12-18 (7 rounds)
- Continuity: Rounds 19-21 (30% → 70% → 100%)
- Shows recovery and continued learning

## Architecture

```
Members (200/cluster) → CH (local agg) → CH* (global agg) → CH → Members
                       Tier 1          Tier 2            Tier 3   Tier 4
```

## Files
- `fmtl_visualization/` - Main visualization package
  - `scenario.py` - Scenario configuration
  - `scene.py` - PyBullet scene with 600 UAVs
  - `attack.py` - CUAV attacker logic
  - `ch_election.py` - CH re-election (LEACH formula)
  - `hud.py` - HUD overlay
  - `frame_capture.py` - Frame export (GIF/MP4)
  - `comparison.py` - Side-by-side scenarios
  - `main.py` - Integration with training

- `test_convergence_with_visualization.py` - Main simulation script
- `ch_simulation.py` - Communication simulator

## Requirements
```
pybullet
numpy
pillow
opencv-python (optional, for MP4 export)
```

## Usage with Training Loop

```python
from fmtl_visualization import FMTLVisualization, ScenarioConfig

# Initialize
config = ScenarioConfig(scenario_type='convergence_compromise')
viz = FMTLVisualization(config, use_gui=True)

# In your training loop
for round_num in range(1, 126):
    # ... your FL training code ...
    
    # Update visualization
    viz.update_round(
        round_num=round_num,
        accuracies={'traffic': 0.85, 'duration': 0.82, 'bandwidth': 0.88},
        communication_stats={'round_data': 297.15, 'total_data': 32.5}
    )

viz.close()
```

## Communication Tracking

The system tracks all 4 tiers of hierarchical communication with byte-level precision:

- **Normal Round**: ~297 MB (all 3 clusters)
- **D&R-E Round**: ~198 MB (2 clusters, 33% reduction)
- **125 Rounds Total**: ~35.82 GB

## Key Moments

- **Round 111**: CH0 compromised (CUAV reaches jamming distance)
- **Round 112**: Attack detected, D&R-E phase starts, Cluster 0 offline
- **Rounds 112-118**: D&R-E phase, members retain R110 model
- **Round 119**: New CH0 elected, 30% recovery
- **Round 120**: 70% participation
- **Round 121**: 100% participation
- **Rounds 122-125**: Stabilization

## Member Old Model Retention

Critical defense mechanism:
```
Round 110: Cluster_0_members use R110_global_agg (CLEAN)
Round 111: CH0 compromised, sends poisoned R111 model
Round 112-118: Members REVERT to R110_global_agg (old, clean)
Round 119+: Members receive new clean models from new CH0
```

## Export Attack Sequence

```python
viz = FMTLVisualization(config, capture_frames=True)
# ... run simulation ...
viz.export_attack_video("attack.gif")
```

## Comparison Mode

```python
from fmtl_visualization.comparison import ComparisonView

comparison = ComparisonView()
results = comparison.run_both(max_rounds=125)
```

## Author
Implementation based on "Scalable Federated Multi-Task Learning with Adaptive Aggregation for Intrusion Detection" research.
