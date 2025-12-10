# Federated Multi-Task Learning for UAV Networks with CH Compromise Recovery

A comprehensive implementation of hierarchical Federated Learning (FL) with Multi-Task Learning (MTL) for UAV network traffic analysis, featuring cluster head (CH) compromise detection and recovery mechanisms.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.13+](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)](https://www.tensorflow.org/)
[![Flower 1.8+](https://img.shields.io/badge/Flower-1.8+-green.svg)](https://flower.dev/)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Experiments](#experiments)
- [Results](#results)
- [KPI Metrics](#kpi-metrics)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)
- [License](#license)

---

## 🎯 Overview

This project implements a **hierarchical federated learning framework** for UAV networks that performs three simultaneous classification tasks:

1. **Traffic Classification** - Identify network traffic types (14 classes)
2. **Flow Duration Prediction** - Predict connection duration (5 classes: Very Short → Very Long)
3. **Bandwidth Estimation** - Estimate bandwidth usage (5 classes: Very Low → Very High)

### System Characteristics

- **600 UAVs** organized into **3 clusters** (200 UAVs each)
- **Hierarchical aggregation**: Intra-cluster + Inter-cluster (at CH1)
- **CH compromise detection** and **D&R-E recovery protocol**
- **Real-time KPI tracking** (40+ metrics)
- **PyBullet 3D visualization** for UAV network simulation

---

## ✨ Key Features

### 🔐 Security & Recovery
- ✅ **CH Compromise Detection** - Automatic anomaly detection at round-level
- ✅ **D&R-E Protocol** - Detection (7 rounds) & Response with Elimination
- ✅ **Gradual Re-integration** - 30% → 70% → 100% participation
- ✅ **Context-Aware CH Selection** - Based on energy residual and RSSI
- ✅ **Frozen Model Retention** - Cluster 0 uses pre-attack parameters during isolation

### 🧠 Federated Learning
- ✅ **Hierarchical Architecture** - Two-tier aggregation (cluster + global)
- ✅ **Multi-Task Learning** - Shared representations + task-specific heads
- ✅ **Scalable to 600 Clients** - Tested with 3 clusters × 200 UAVs
- ✅ **Integrated Testing** - Per-round evaluation on test data

### 📊 Data Distribution
- ✅ **Equal Split** - Balanced data distribution across clusters
- ✅ **Dirichlet Split** - Non-IID distribution (α=0.4) for realistic heterogeneity
- ✅ **Per-Cluster Test Sets** - Separate evaluation for equal and Dirichlet splits
- ✅ **Two-Level Partitioning** - Cluster-level + Client-level distribution control

### 📈 Comprehensive KPI Tracking
- ✅ **Performance Metrics** - Accuracy, convergence time, round durations
- ✅ **Communication Cost** - Bytes per round, total overhead, phase breakdown
- ✅ **Attack Impact** - Accuracy degradation, recovery time, model divergence
- ✅ **Resource Monitoring** - CPU%, memory usage, inference latency
- ✅ **40+ Real-Time KPIs** - Saved as JSON and pickle for analysis

---

## 🏗️ Architecture

### Network Topology

```
                    ┌─────────────────────┐
                    │   Global Model      │
                    │   (Aggregated at    │
                    │    Cluster 1 CH)    │
                    └──────────┬──────────┘
                               │
                ┌──────────────┼──────────────┐
                │              │              │
         ┌──────▼─────┐ ┌─────▼──────┐ ┌────▼───────┐
         │ Cluster 0  │ │ Cluster 1  │ │ Cluster 2  │
         │   CH0      │ │    CH1*    │ │    CH2     │
         │ (200 UAVs) │ │ (200 UAVs) │ │ (200 UAVs) │
         └────────────┘ └────────────┘ └────────────┘
              ▲              ▲              ▲
              │              │              │
        ┌─────┴─────┐  ┌────┴────┐   ┌────┴─────┐
        │ UAV 0-199 │  │UAV 200  │   │UAV 400   │
        │  Local    │  │  -399   │   │  -599    │
        │ Training  │  │ Local   │   │  Local   │
        └───────────┘  └─────────┘   └──────────┘
        
        * CH1 = Global Aggregator
```

### Model Architecture (FedMTL)

```python
Input Layer (39 features after padding)
    ↓
┌───────────────────────────┐
│   Shared Layers           │
│   Dense(256) + ReLU       │
│   Dropout(0.1)            │
│   Dense(128) + ReLU       │
│   Dropout(0.1)            │
└───────────┬───────────────┘
            │
    ┌───────┴───────┬───────────────┐
    ↓               ↓               ↓
┌─────────┐   ┌──────────┐   ┌──────────┐
│ Traffic │   │ Duration │   │Bandwidth │
│Dense(64)│   │Dense(32) │   │Dense(64) │
│  ReLU   │   │  ReLU    │   │  ReLU    │
└────┬────┘   └─────┬────┘   └─────┬────┘
     ↓              ↓              ↓
┌─────────┐   ┌──────────┐   ┌──────────┐
│Output(14)│  │Output(5) │   │Output(5) │
│Softmax  │   │Softmax   │   │Softmax   │
└─────────┘   └──────────┘   └──────────┘
```

---

## 📦 Installation

### Prerequisites

- **Python**: 3.9 - 3.10 (tested on 3.9)
- **RAM**: Minimum 8GB (16GB recommended)
- **CPU**: Multi-core processor (12+ cores recommended)
- **Storage**: 5GB free space
- **GPU**: CUDA-capable GPU (optional, but recommended)

### Step 1: Clone Repository

```bash
cd ~/Documents
git clone <your-repo-url>
cd gym-pybullet-drones/cesnet_zoo_clean
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv fmtl_env

# Activate (macOS/Linux)
source fmtl_env/bin/activate

# Activate (Windows)
# fmtl_env\Scripts\activate
```

### Step 3: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install core packages
pip install tensorflow==2.13.0
pip install flwr==1.8.0
pip install ray==2.9.0
pip install numpy==1.24.3
pip install pandas==2.0.3
pip install scikit-learn==1.3.0
pip install scipy==1.11.1
pip install matplotlib==3.7.2
pip install psutil==5.9.5
pip install pybullet==3.2.5
pip install jupyter==1.0.0
```

Or use the requirements file:

```bash
pip install -r requirements.txt
```

### Step 4: Install gym-pybullet-drones

```bash
cd ~/Documents/gym-pybullet-drones
pip install -e .
cd cesnet_zoo_clean
```

### Step 5: Dataset Setup

```bash
# Create dataset directory
mkdir -p datasets/local_cache

# Place your dataset file here:
# datasets/local_cache/dataset_12500_samples_65_features.csv
```

**Dataset Format**:
- CSV with 65 features
- Required columns: `label`, `flow_duration`, `bandwidth_bps`
- ~12,500 samples (minimum)

---

## 🚀 Quick Start

### Option 1: Run Jupyter Notebook (Recommended)

```bash
# Launch Jupyter
jupyter notebook

# Open: 600_uav_full_test_with_results.ipynb
# Run all cells sequentially
```

**Estimated Runtime**:
- Convergence (125 rounds): ~60-90 minutes
- Transient (30 rounds): ~15-20 minutes

### Option 2: Run Python Scripts

```bash
# Run convergence + transient experiments
python gym_integrated_simulation.py

# Or run visual simulation with PyBullet
python run_visual_simulation_with_kpis.py
```

### Verify Installation

```python
# Test imports
import tensorflow as tf
import flwr as fl
import ray
import numpy as np
import pandas as pd

print(f"TensorFlow: {tf.__version__}")
print(f"Flower: {fl.__version__}")
print(f"Ray: {ray.__version__}")
```

---

## 📁 Project Structure

```
cesnet_zoo_clean/
│
├── README.md                                  # This file
├── requirements.txt                           # Python dependencies
│
├── 600_uav_full_test_with_results.ipynb     # Main experiment notebook
├── gym_integrated_simulation.py              # Standalone simulation
├── real_time_transient_simulation.py         # Real-time transient
├── run_visual_simulation_with_kpis.py        # PyBullet visualization
├── test_convergence_with_visualization.py    # Test + visualize
│
├── datasets/
│   └── local_cache/
│       └── dataset_12500_samples_65_features.csv  # Network traffic data
│
├── trained_models/
│   ├── convergence_integrated/               # Convergence experiment (125 rounds)
│   │   ├── model_round_1.pkl ... 125.pkl    # Per-round checkpoints
│   │   ├── kpi_summary_convergence.json      # KPI metrics (JSON)
│   │   └── kpi_tracker_convergence.pkl       # Full KPI tracker
│   │
│   ├── transient_integrated/                 # Transient experiment (30 rounds)
│   │   ├── model_round_1.pkl ... 30.pkl     # Per-round checkpoints
│   │   ├── kpi_summary_transient.json        # KPI metrics (JSON)
│   │   └── kpi_tracker_transient.pkl         # Full KPI tracker
│   │
│   └── preprocessed_test_data.pkl            # Test data for inference
│
├── experiment_results/
│   ├── integrated_results_<timestamp>.pkl    # Full results package
│   └── integrated_summary_<timestamp>.json   # Experiment summary
│
└── fmtl_visualization/                       # Visualization outputs
    ├── __init__.py
    ├── attack.py                             # Attack simulation
    ├── ch_election.py                        # CH selection logic
    ├── kpi.py                                # KPI tracking
    └── main.py                               # Visualization main
```

---

## 🔬 Experiments

### Experiment 1: Convergence Scenario (125 rounds)

**Objective**: Train model to convergence, then simulate CH compromise and recovery.

**Timeline**:
```
Rounds 1-110:   Normal training to convergence
Round 111:      CH0 compromise detected 🚨
Rounds 112-118: D&R-E phase (Cluster 0 isolated, frozen parameters)
Round 119:      Gradual re-entry begins (30% participation)
Round 120:      Increased participation (70%)
Round 121:      Full restoration (100%)
Rounds 122-125: Stabilization and recovery validation
```

**Configuration**:
```python
CFG = {
    'n_clients_flat': 600,
    'n_clusters': 3,
    'clients_per_cluster': 200,
    'compromise_round': 111,
    'detection_rounds': 7,
    'continuity_rounds': 3,
    'cluster_split': 'equal',
    'client_split': 'dirichlet',
    'alpha_client': 0.4,
}
```

### Experiment 2: Transient Scenario (30 rounds)

**Objective**: Simulate early-stage compromise during initial training.

**Timeline**:
```
Rounds 1-10:    Initial training (no convergence)
Round 11:       CH0 compromise detected 🚨
Rounds 12-18:   D&R-E phase (Cluster 0 isolated)
Round 19:       Gradual re-entry (30%)
Round 20:       Increased participation (70%)
Round 21:       Full restoration (100%)
Rounds 22-30:   Recovery and stabilization
```

---

## 📊 Results

### Convergence Scenario (125 rounds)

| Metric | Value |
|--------|-------|
| **Final Global Accuracy** | 0.82 - 0.85 |
| **Convergence Round** | ~95-100 |
| **Pre-Attack Accuracy** | 0.84 |
| **During Attack (Round 115)** | 0.72 (-0.12) |
| **Post-Recovery (Round 125)** | 0.83 |
| **Time to Restore 99%** | 12 rounds |
| **Total Communication** | ~450 MB |

### Transient Scenario (30 rounds)

| Metric | Value |
|--------|-------|
| **Final Global Accuracy** | 0.74 - 0.78 |
| **Pre-Attack Accuracy** | 0.76 |
| **During Attack (Round 15)** | 0.61 (-0.15) |
| **Post-Recovery (Round 30)** | 0.75 |
| **Time to Restore 99%** | 10 rounds |
| **Total Communication** | ~125 MB |

### Per-Task Performance (Convergence)

| Task | Pre-Attack | During Attack | Post-Recovery |
|------|-----------|---------------|---------------|
| **Traffic Classification** | 0.87 | 0.74 | 0.85 |
| **Duration Prediction** | 0.84 | 0.71 | 0.82 |
| **Bandwidth Estimation** | 0.85 | 0.72 | 0.84 |

### Per-Cluster Analysis

**Equal Split (Convergence, Round 125)**:
- Cluster 0 (Compromised): 0.82
- Cluster 1 (Global Aggregator): 0.85
- Cluster 2 (Healthy): 0.84

**Dirichlet Split (Convergence, Round 125)**:
- Cluster 0: 0.78 (recovered from 0.65)
- Cluster 1: 0.82
- Cluster 2: 0.80

---

## 📈 KPI Metrics

### Tier 1: Core Performance Metrics

#### Learning Performance
- ✅ Global accuracy (per round)
- ✅ Per-cluster accuracy (3 clusters)
- ✅ Per-task accuracy (traffic, duration, bandwidth)
- ✅ Convergence round and time (seconds)
- ✅ Round durations and cumulative time

#### Model Architecture & Resources
- ✅ Model parameter size (bytes, KB)
- ✅ Architecture overhead (sys.getsizeof + pickle)
- ✅ Inference latency (mean ± std, ms)
- ✅ CPU usage (%)
- ✅ Memory usage (RSS, MB)

#### Communication Efficiency
- ✅ Total communication bytes
- ✅ Bytes per federation round
- ✅ Communication breakdown by phase (normal/attack/recovery)
- ✅ Extra cost due to attack
- ✅ Per-cluster communication

### Tier 2: Security & Recovery Metrics

#### Attack Impact & Recovery
- ✅ Detection time (rounds)
- ✅ Recovery time breakdown (detection/isolation/reintegration)
- ✅ Recovery time (wall-clock seconds)
- ✅ Accuracy degradation during attack (global + per-task)
- ✅ Time to restore accuracy (rounds to reach 99% pre-attack)
- ✅ Model divergence during isolation (L2 norm)
- ✅ Task-specific attack impact (% drop)

#### Cluster Health & Participation
- ✅ Participation rate per cluster (per round)
- ✅ Cluster isolation impact (C1, C2 accuracy during C0 isolation)
- ✅ Gradual reintegration effect (30%, 70%, 100%)
- ✅ Participation-accuracy correlation (Pearson)

#### CH Selection & Load
- ✅ CH load (members per CH)
- ✅ CH duty cycle (estimated)
- ✅ CH selection frequency (re-elections)
- ✅ CH re-election time (seconds)
- ✅ New CH characteristics (energy, RSSI)
- ✅ Context-aware selection score (α×E + β×RSSI)

### Access KPIs Programmatically

```python
import pickle

# Load KPI tracker
with open('trained_models/convergence_integrated/kpi_tracker_convergence.pkl', 'rb') as f:
    kpi_tracker = pickle.load(f)

# Print comprehensive summary
kpi_tracker.print_summary()

# Access specific metrics
print(f"Convergence time: {kpi_tracker.kpis['convergence_time_seconds']:.2f}s")
print(f"Final accuracy: {kpi_tracker.kpis['global_accuracy'][-1]:.4f}")
print(f"Total communication: {kpi_tracker.kpis['total_communication_bytes'] / 1e9:.2f} GB")
print(f"Recovery time: {kpi_tracker.kpis['recovery_time_seconds']:.2f}s")
```

---

## 🔧 Configuration

### Key Parameters

```python
CFG = {
    # Training Parameters
    'local_epochs': 1,              # Epochs per client per round
    'lr': 1e-3,                     # Learning rate (Adam optimizer)
    'loss_weights': {               # Task loss weights
        'traffic': 1,
        'duration': 1,
        'bandwidth': 1
    },
    'test_size': 0.2,              # Train/test split ratio
    
    # Client Configuration
    'n_clients_flat': 600,          # Total number of UAVs
    'n_clusters': 3,                # Number of clusters
    'clients_per_cluster': 200,     # UAVs per cluster
    'client_frac': 1.0,             # Participation rate (100%)
    
    # Hierarchical FL
    'global_aggregator_cluster': 1, # Cluster performing global aggregation
    
    # Data Distribution (Two-Level)
    'cluster_split': 'equal',       # Cluster-level: 'equal' or 'dirichlet'
    'client_split': 'dirichlet',    # Client-level: always 'dirichlet'
    'alpha_client': 0.4,            # Dirichlet α for client distribution
    'alpha_cluster': 0.4,           # Dirichlet α for cluster distribution
    
    # Security & Recovery
    'compromise_round': 111,        # When to compromise CH0
    'compromised_cluster': 0,       # Which cluster to compromise
    'detection_rounds': 7,          # D&R-E phase duration
    'continuity_rounds': 3,         # Gradual re-entry duration
    'alpha_energy': 0.6,            # CH selection weight (energy)
    'beta_rssi': 0.4,              # CH selection weight (RSSI)
}
```

### Tuning Guidelines

**For Better Accuracy**:
- Increase `local_epochs` (1 → 3)
- Decrease `lr` (1e-3 → 1e-4)
- Increase `alpha_client` (0.4 → 0.8) for more IID data

**For Faster Convergence**:
- Increase `client_frac` (already at 1.0)
- Use `cluster_split='equal'` for balanced load
- Increase `lr` (1e-3 → 5e-3)

**For Realistic Heterogeneity**:
- Use `cluster_split='dirichlet'`
- Decrease `alpha_client` (0.4 → 0.1) for more non-IID

---

## 🎬 Running PyBullet Simulation

Run the **3D UAV network simulation** with real-time visualization:

```bash
python run_visual_simulation_with_kpis.py
```

**Features**:
- 🚁 Real-time 3D UAV movement and formation
- 📡 Visual cluster boundaries and connections
- ⚠️ CH compromise detection animation
- 🔄 CH re-election with context-aware selection
- 📊 Live KPI dashboard overlay
- 🎥 Frame capture for video export

**Controls**:
- **Space**: Pause/Resume
- **ESC**: Exit simulation
- **R**: Reset camera view

---

## 🐛 Troubleshooting

### Common Issues

#### 1. Import Errors

```bash
# Error: ModuleNotFoundError: No module named 'flwr'
# Solution: Verify virtual environment is activated
source fmtl_env/bin/activate
pip list | grep flwr

# If missing, reinstall
pip install flwr==1.8.0
```

#### 2. Ray Initialization Errors

```python
# Error: Ray has already been started
# Solution: Shutdown Ray first
import ray
if ray.is_initialized():
    ray.shutdown()
ray.init(num_cpus=12)
```

#### 3. Out of Memory (OOM)

```python
# Error: OOM when allocating tensor
# Solution: Reduce client count
CFG['n_clients_flat'] = 300  # Reduce from 600
CFG['clients_per_cluster'] = 100  # Reduce from 200
```

#### 4. Dataset Not Found

```bash
# Error: FileNotFoundError: dataset_12500_samples_65_features.csv
# Solution: Check dataset path
ls datasets/local_cache/dataset_12500_samples_65_features.csv

# Verify file exists and has correct name
```

#### 5. PyBullet GUI Issues

```python
# Error: Cannot connect to X server
# Solution: Use headless mode
import pybullet as p
p.connect(p.DIRECT)  # Instead of p.GUI
```

#### 6. Jupyter Kernel Crashes

```bash
# Restart kernel and clear outputs
# In Jupyter: Kernel → Restart & Clear Output

# Check system resources
htop  # or top on macOS

# If needed, reduce clients or use smaller dataset
```

---

## 📊 Generating Visualizations

The notebook generates **20 comprehensive graphs**:

### Section 1: Normal Testing (100 rounds)
- **Graphs 1-3**: Per-cluster (Equal split)
- **Graphs 4-6**: Per-cluster (Dirichlet split)

### Section 2: Overall Performance (100 rounds)
- **Graph 7**: Multi-cluster average (Equal)
- **Graph 8**: Multi-cluster average (Dirichlet)

### Section 3: Convergence (125 rounds)
- **Graphs 9-11**: Per-cluster with CH compromise (Equal)
- **Graphs 12-14**: Per-cluster with CH compromise (Dirichlet)

### Section 4: Transient (30 rounds)
- **Graphs 15-17**: Per-cluster early compromise (Equal)
- **Graphs 18-20**: Per-cluster early compromise (Dirichlet)

**Phase Markers**:
- 🟥 Pink: D&R-E (Detection & Response with Elimination)
- 🟨 Yellow: Continuity (Gradual re-integration)
- 🟩 Green: Stabilization (Full recovery)

---

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@misc{fmtl_uav_2024,
  author = {Your Name},
  title = {Federated Multi-Task Learning for UAV Networks with CH Compromise Recovery},
  year = {2024},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/yourusername/cesnet_zoo_clean}},
  note = {Hierarchical FL with D\&R-E protocol for UAV cluster networks}
}
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit your changes**: `git commit -m 'Add amazing feature'`
4. **Push to the branch**: `git push origin feature/amazing-feature`
5. **Open a Pull Request**

### Development Guidelines

- Follow PEP 8 style guidelines
- Add docstrings to all functions/classes
- Include unit tests for new features
- Update README.md with new features

---

## 📄 License

This project is licensed under the **MIT License**.

```
MIT License

Copyright (c) 2024 [Your Name]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

See [LICENSE](LICENSE) file for full details.

---

## 🙏 Acknowledgments

- **Flower Framework** ([flower.dev](https://flower.dev)) - Federated learning infrastructure
- **TensorFlow Team** - Deep learning framework
- **Ray Project** - Distributed computing
- **PyBullet** - Physics simulation and 3D visualization
- **CESNET** - Network traffic datasets
- **gym-pybullet-drones** - UAV simulation framework

---

## 📧 Contact

For questions, issues, or collaboration:

- **GitHub Issues**: [Open an issue](https://github.com/yourusername/cesnet_zoo_clean/issues)
- **Email**: your.email@example.com
- **Project Website**: [your-project-website.com](https://your-project-website.com)

---

## 🔄 Version History

### v1.0.0 (December 2024)
- Initial release
- Convergence and transient scenarios
- Comprehensive KPI tracking (40+ metrics)
- PyBullet visualization
- D&R-E protocol implementation

---

## 📚 Additional Resources

### Papers & References
- [Federated Learning: Challenges, Methods, and Future Directions](https://arxiv.org/abs/1908.07873)
- [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629)
- [Advances and Open Problems in Federated Learning](https://arxiv.org/abs/1912.04977)

### Related Projects
- [Flower Examples](https://github.com/adap/flower/tree/main/examples)
- [gym-pybullet-drones](https://github.com/utiasDSL/gym-pybullet-drones)

### Tutorials
- [Getting Started with Flower](https://flower.dev/docs/tutorial-quickstart-tensorflow.html)
- [TensorFlow Multi-Task Learning](https://www.tensorflow.org/tutorials)

---

**Last Updated**: December 10, 2024  
**Version**: 1.0.0  
**Status**: Production Ready ✅

---

<div align="center">

**Made with ❤️ for UAV Network Research**

[⬆ Back to Top](#federated-multi-task-learning-for-uav-networks-with-ch-compromise-recovery)

</div>
