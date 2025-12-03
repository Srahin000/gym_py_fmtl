# KPI Integration - Quick Reference

## TL;DR

Added comprehensive KPI tracking to the FMTL experiment notebook. Works with existing save-and-test workflow. Two usage modes: **retroactive** (works now) or **real-time** (for new training).

## What Changed

### New Components Added

1. **`ComprehensiveKPITracker`** (Cell 45)
   - Tracks metrics during training
   - Methods: `start_round()`, `end_round()`, `measure_inference_latency()`, etc.

2. **KPI-Enabled Strategies** (Cell 43a)
   - `TrainingOnlyStrategyWithKPIs`
   - `HierarchicalTrainingOnlyStrategyWithKPIs`
   - Auto-save KPI snapshots with each model

3. **Enhanced Testing** (Cell 54)
   - `enhanced_test_evaluation_with_kpis()` - Extracts KPIs from saved models
   - `aggregate_kpis_from_saved_models()` - Aggregates training KPIs

4. **Retroactive Computation** (Cell 46)
   - `compute_kpis_from_test_results()` - Computes KPIs from existing test results

5. **Visualization** (Cell 47)
   - `visualize_kpis()` - 6-panel dashboard
   - `plot_per_cluster_communication()` - Per-cluster analysis

## Quick Start

### Option 1: Use with Existing Models (No Retraining)

```python
# Works with your current test_results!
kpis = compute_kpis_from_test_results(test_results, CFG, model, test_data)
print_kpi_summary(kpis)
visualize_kpis(kpis)
```

### Option 2: New Training with Full Tracking

```python
# Create tracker
kpi_tracker = ComprehensiveKPITracker(CFG, model)

# Use KPI-enabled strategy
strategy = HierarchicalTrainingOnlyStrategyWithKPIs(
    save_dir='trained_models/with_kpis',
    kpi_tracker=kpi_tracker  # Only change needed
)

# Train (KPIs saved automatically)
fl.simulation.start_simulation(..., strategy=strategy)

# Extract KPIs from saved models
kpis = aggregate_kpis_from_saved_models('trained_models/with_kpis')
```

## All KPIs Tracked

### Training Phase (Real-Time)
- Round duration
- Cumulative time
- CPU usage
- Memory usage
- Model divergence (hierarchical)
- Convergence detection

### Testing Phase (From Test Results)
- Global accuracy
- Per-cluster accuracy
- Per-task accuracy
- Convergence round

### Post-Processing (Computed)
- Communication costs
- Attack impact metrics
- Recovery time
- CH selection metrics
- Model architecture metrics

## Key Files

- **Main Summary**: `KPI_INTEGRATION_SUMMARY.md` (detailed documentation)
- **Notebook**: `final_experiment.ipynb` (cells 43a, 45-47, 54-57)

## Migration Guide

### Before
```python
strategy = HierarchicalTrainingOnlyStrategy(...)
```

### After
```python
kpi_tracker = ComprehensiveKPITracker(CFG, model)
strategy = HierarchicalTrainingOnlyStrategyWithKPIs(
    kpi_tracker=kpi_tracker,
    ...  # same other params
)
```

That's it! Everything else works the same.

