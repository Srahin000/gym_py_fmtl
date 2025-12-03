# KPI Integration Summary for Scalable FMTL Experiment

## Overview

This document summarizes the comprehensive Key Performance Indicator (KPI) tracking system integrated into the Federated Multi-Task Learning (FMTL) experiment notebook. The system tracks metrics across **TIER 1** (Learning Performance, Model Architecture, Communication Efficiency) and **TIER 2** (Attack Impact, Cluster Health, CH Selection) categories.

## Problem Statement

The original notebook had a **save-and-test workflow**:
1. Training phase: Save model weights after each round
2. Testing phase: Load saved models and evaluate on test data

However, many KPIs required for the study were either:
- Not tracked at all
- Only partially computed
- Missing real-time measurements during training

## Solution Architecture

We implemented a **dual-phase KPI tracking system** that works seamlessly with the existing workflow:

### Phase 1: Real-Time Training Tracking
- **Component**: `ComprehensiveKPITracker` class
- **Integration**: KPI-enabled training strategies
- **Output**: KPI snapshots saved with each model checkpoint

### Phase 2: Post-Training Analysis
- **Component**: Retroactive KPI computation functions
- **Integration**: Enhanced testing functions
- **Output**: Complete KPI dictionary for visualization

## What Was Added

### 1. Core KPI Tracker Class (`ComprehensiveKPITracker`)

**Location**: Cell 45

**Purpose**: Real-time metric tracking during training

**Key Features**:
- Automatic round timing (`start_round()`, `end_round()`)
- Computational load monitoring (CPU, memory via `psutil`)
- Convergence detection (variance-based, 5-round window)
- Attack phase tracking (`record_attack_start()`, `record_attack_detected()`)
- Model divergence calculation (L2 norm between cluster and global weights)
- CH re-election tracking

**Key Methods**:
```python
kpi_tracker.start_experiment()      # Start experiment timer
kpi_tracker.start_round()           # Mark round start
kpi_tracker.end_round(round, accuracies, phase)  # Record round metrics
kpi_tracker.measure_inference_latency(test_samples)  # Measure inference time
kpi_tracker.measure_computational_load()  # CPU/memory snapshot
kpi_tracker.record_model_divergence(cluster_weights, global_weights)  # L2 norm
kpi_tracker.print_summary()         # Formatted output
```

### 2. KPI-Enabled Training Strategies

**Location**: Cell 43a

**Components**:
- `TrainingOnlyStrategyWithKPIs` (single cluster)
- `HierarchicalTrainingOnlyStrategyWithKPIs` (hierarchical)

**Key Changes from Original**:
1. Accepts `kpi_tracker` parameter in `__init__`
2. Calls `kpi_tracker.start_round()` at beginning of `aggregate_fit()`
3. Measures computational load during aggregation
4. Saves KPI snapshots with each model checkpoint
5. Tracks per-cluster participation and accuracies

**Saved Data Structure**:
```python
{
    'round': round_num,
    'weights': model_weights,
    'cluster_params': {...},  # For hierarchical
    'kpi_snapshot': {         # 🆕 NEW
        'round_duration': float,
        'cumulative_time': float,
        'cpu_percent': float,
        'memory_mb': float,
        'participating_clients_per_cluster': {...}
    }
}
```

### 3. Enhanced Testing Functions

**Location**: Cell 54

**Functions**:
- `enhanced_test_evaluation_with_kpis()`: Extracts KPIs from saved models during testing
- `aggregate_kpis_from_saved_models()`: Aggregates all training KPIs post-hoc

**Key Feature**: Extracts KPI snapshots from saved model files, enabling analysis without retraining.

### 4. Retroactive KPI Computation

**Location**: Cell 46

**Function**: `compute_kpis_from_test_results()`

**Purpose**: Computes KPIs from existing `test_results` dictionary (works with old training runs)

**What It Computes**:
- Global and per-task accuracies
- Convergence round (variance-based detection)
- Model parameter size and architecture overhead
- Inference latency (if test data provided)
- Communication costs (formula: `W = 2 × N × ω`)
- Attack impact metrics (degradation, recovery time)
- Per-cluster metrics
- CH selection metrics

### 5. Visualization Functions

**Location**: Cell 47

**Functions**:
- `visualize_kpis()`: 6-panel dashboard
  - Per-task recovery curves
  - Communication breakdown by phase
  - Gradual re-integration effect
  - Task-specific attack impact
  - Model divergence during isolation
  - KPI summary dashboard
- `plot_per_cluster_communication()`: Per-cluster communication analysis

## Complete KPI List

### TIER 1: Learning Performance
- ✅ Global accuracy (per round)
- ✅ Per-cluster accuracy (C0, C1, C2)
- ✅ Per-task accuracy (traffic, duration, bandwidth)
- ✅ Convergence round (variance < 0.01 threshold)
- 🆕 **Convergence time (wall-clock seconds)**
- 🆕 **Round duration (per round)**

### TIER 1: Model Architecture & Resources
- ✅ Model parameter size (bytes/KB)
- 🆕 **Model architecture storage overhead** (sys.getsizeof + pickle)
- 🆕 **Inference latency** (average over 100 samples, ms)
- 🆕 **Computational load per UAV** (CPU %, memory RSS MB)

### TIER 1: Communication Efficiency
- ✅ Communication cost per round (2 × N × ω)
- 🆕 **Communication overhead breakdown** (normal/attack/recovery phases)
- 🆕 **Extra cost due to attack** (attack + recovery - baseline equivalent)
- 🆕 **Per-cluster communication** (bytes per cluster per round)
- 🆕 **Bytes per federation round** (average)

### TIER 2: Attack Impact & Recovery
- ✅ Detection time (rounds between attack and detection)
- ✅ Recovery time breakdown (detection/isolation/reintegration rounds)
- 🆕 **Recovery time in real seconds** (wall-clock)
- 🆕 **Accuracy degradation during attack** (pre-attack - attack round)
- 🆕 **Time to restore pre-attack accuracy** (first round ≥ 99% pre-attack)
- 🆕 **Model divergence during isolation** (L2 norm, rounds 112-118)
- 🆕 **Task-specific attack impact** (percentage drop per task)
- 🆕 **Per-task recovery curves** (accuracy over rounds 111-125)

### TIER 2: Cluster Health & Participation
- ✅ Connectivity/participation rate per cluster
- ✅ Cluster 0 isolation impact (C1, C2 accuracy during isolation)
- 🆕 **Gradual re-integration effect** (accuracy at 30%, 70%, 100% participation)
- 🆕 **Per-phase participation correlation** (Pearson correlation with accuracy)

### TIER 2: CH Selection & Load
- ✅ CH load (members per CH, typically 200)
- ✅ CH duty cycle (energy estimate)
- 🆕 **CH selection frequency** (number of re-elections)
- 🆕 **CH re-election time** (seconds per re-election)
- 🆕 **New CH0 characteristics** (energy_residual, rssi_avg)
- 🆕 **Context-aware selection effectiveness** (α×E + β×RSSI score)

## Usage Patterns

### Pattern 1: Quick Retroactive Analysis (No Retraining)

Works with **existing trained models**:

```python
# Compute KPIs from existing test results
kpis = compute_kpis_from_test_results(
    test_results=test_results,  # Your existing test_results dict
    cfg=CFG,
    model=global_model_template,
    test_data=test_data
)

# Print summary
print_kpi_summary(kpis)

# Visualize
visualize_kpis(kpis)
```

**When to use**: Analyzing already-trained models, no need to retrain.

### Pattern 2: Full Real-Time Tracking (New Training)

For new training runs with complete KPI tracking:

```python
# Step 1: Create KPI tracker
kpi_tracker = ComprehensiveKPITracker(
    cfg=CFG,
    model=global_model_template,
    n_clusters=3,
    clients_per_cluster=200
)

# Step 2: Measure inference latency once
kpi_tracker.measure_inference_latency(X_traffic_test[:100])

# Step 3: Create KPI-enabled strategy
strategy = HierarchicalTrainingOnlyStrategyWithKPIs(
    save_dir='trained_models/hierarchical_with_kpis',
    kpi_tracker=kpi_tracker,  # 🔥 Pass tracker
    fraction_fit=CFG['client_frac'],
    fraction_evaluate=CFG['client_frac'],
    # ... other parameters
)

# Step 4: Train (KPIs tracked automatically)
history = fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=len(clients),
    config=fl.server.ServerConfig(num_rounds=125),
    strategy=strategy,
    client_resources={'num_cpus': 1.0, 'num_gpus': 0.0},
)

# Step 5: Get training KPIs
training_kpis = kpi_tracker.get_summary()
kpi_tracker.print_summary()

# Step 6: Test with KPI extraction
result = enhanced_test_evaluation_with_kpis(
    model_dir='trained_models/hierarchical_with_kpis',
    test_data_dict=test_data_equal,
    model_type='global',
    extract_kpis=True
)

# Step 7: Aggregate all KPIs
all_training_kpis = aggregate_kpis_from_saved_models(
    'trained_models/hierarchical_with_kpis'
)

# Step 8: Combine and visualize
final_kpis = compute_kpis_from_test_results(...)
# Merge training KPIs
final_kpis['round_durations'] = all_training_kpis['round_durations']
final_kpis['cumulative_time'] = all_training_kpis['cumulative_times']
# ... etc

visualize_kpis(final_kpis)
```

**When to use**: New training runs where you want complete real-time tracking.

## Integration Points

### Training Phase Integration

**Before**:
```python
strategy = HierarchicalTrainingOnlyStrategy(
    save_dir='trained_models/hierarchical_equal',
    # ... parameters
)
```

**After**:
```python
kpi_tracker = ComprehensiveKPITracker(CFG, model)
strategy = HierarchicalTrainingOnlyStrategyWithKPIs(
    save_dir='trained_models/hierarchical_with_kpis',
    kpi_tracker=kpi_tracker,  # 🔥 Only change needed
    # ... same parameters
)
```

### Testing Phase Integration

**Before**:
```python
# Load and test models
for round_num in range(1, 101):
    with open(f'model_round_{round_num}.pkl', 'rb') as f:
        saved = pickle.load(f)
    # Evaluate model...
```

**After**:
```python
# Enhanced testing extracts KPIs automatically
result = enhanced_test_evaluation_with_kpis(
    model_dir='trained_models/hierarchical_with_kpis',
    test_data_dict=test_data_equal,
    extract_kpis=True  # 🔥 Extracts KPI snapshots
)
test_results = result['test_results']  # Accuracies
kpi_data = result['kpi_data']  # Round duration, CPU, memory, divergence
```

## Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    TRAINING PHASE                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ComprehensiveKPITracker                                    │
│  ├─ start_round() → Records round start time                │
│  ├─ measure_computational_load() → CPU/memory snapshot      │
│  └─ end_round() → Records duration, accuracies, convergence│
│                                                              │
│  HierarchicalTrainingOnlyStrategyWithKPIs                    │
│  ├─ aggregate_fit() → Calls tracker methods                 │
│  └─ Saves model + KPI snapshot to .pkl file                 │
│                                                              │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    MODEL CHECKPOINTS                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  model_round_N.pkl                                          │
│  ├─ weights: Model parameters                               │
│  ├─ cluster_params: Per-cluster weights (hierarchical)     │
│  └─ kpi_snapshot: {                                         │
│       round_duration, cumulative_time,                      │
│       cpu_percent, memory_mb,                               │
│       participating_clients_per_cluster                     │
│     }                                                        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    TESTING PHASE                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  enhanced_test_evaluation_with_kpis()                       │
│  ├─ Loads each model_round_N.pkl                            │
│  ├─ Extracts kpi_snapshot                                   │
│  ├─ Evaluates model on test data                            │
│  └─ Returns: {test_results, kpi_data}                       │
│                                                              │
│  aggregate_kpis_from_saved_models()                         │
│  └─ Aggregates all KPI snapshots across all rounds          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    ANALYSIS PHASE                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  compute_kpis_from_test_results()                           │
│  ├─ Computes communication costs                            │
│  ├─ Calculates attack impact metrics                        │
│  ├─ Determines convergence round                            │
│  └─ Returns complete KPI dictionary                         │
│                                                              │
│  visualize_kpis()                                           │
│  └─ Creates 6-panel dashboard                              │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Key Design Decisions

### 1. Dual-Phase Approach
- **Why**: Supports both existing models (retroactive) and new training (real-time)
- **Benefit**: No need to retrain to get KPIs

### 2. KPI Snapshots in Model Files
- **Why**: KPIs saved with each checkpoint, enabling post-hoc analysis
- **Benefit**: Can extract training metrics anytime without retraining

### 3. Automatic vs Manual Tracking
- **Automatic**: Round duration, CPU, memory, convergence (via tracker)
- **Manual**: Attack phases, CH re-election (via explicit calls)
- **Why**: Some events require explicit marking (e.g., attack start)

### 4. Variance-Based Convergence Detection
- **Method**: Check if variance of last 5 rounds < 0.01
- **Why**: Simple, effective, matches study requirements
- **Location**: `ComprehensiveKPITracker.end_round()`

## File Structure

```
final_experiment.ipynb
├─ Cell 44: KPI Documentation (markdown)
├─ Cell 45: ComprehensiveKPITracker class
├─ Cell 43a: KPI-Enabled Training Strategies
├─ Cell 46: Retroactive KPI Computation
├─ Cell 47: KPI Visualization Functions
├─ Cell 54: Enhanced Testing Functions
├─ Cell 55-56: Usage Examples
└─ Cell 57: Integration Summary
```

## Dependencies

**New dependencies** (if not already installed):
- `psutil` - For CPU and memory monitoring
- `scipy.stats` - For Pearson correlation (already used)

**Existing dependencies** (already in notebook):
- `numpy`, `tensorflow`, `flwr`, `pickle`, `time`, `sys`

## Testing Checklist

To verify the integration works:

- [ ] Create KPI tracker instance
- [ ] Pass tracker to training strategy
- [ ] Run training (verify models save with KPI snapshots)
- [ ] Extract KPIs from saved models
- [ ] Compute retroactive KPIs from test results
- [ ] Generate visualizations
- [ ] Verify all KPIs in summary output

## Known Limitations

1. **Model Divergence**: Requires hierarchical models with `cluster_params` saved
2. **CH Re-election Time**: Currently estimated (5ms) - would need actual LEACH implementation timing
3. **Inference Latency**: Measured on test data, not training data
4. **Computational Load**: Single-process measurement (may not reflect distributed load accurately)

## Future Enhancements

1. **Distributed Load Tracking**: Track CPU/memory across all Ray workers
2. **Real CH Re-election**: Integrate with actual LEACH implementation for precise timing
3. **Communication Protocol Tracking**: Track actual bytes sent/received (if Flower supports it)
4. **Energy Consumption**: More accurate CH duty cycle calculation with real energy models

## Contact & Questions

For questions about the KPI integration:
- Review the usage examples in cells 55-56
- Check the `ComprehensiveKPITracker` docstrings
- See the integration summary in cell 57

---

**Last Updated**: Integration completed with full support for both training and testing phases.

