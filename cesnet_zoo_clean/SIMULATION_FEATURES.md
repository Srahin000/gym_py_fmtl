# Real-Time Transient Training Simulation Features

## ✅ Implemented Features

### 1. **PyBullet 3D Visualization**
- Main GUI window showing all 45 UAVs (3 clusters × 15 drones)
- Cluster heads displayed as Crazyflie cf2x drones (scaled 3x)
- Regular drones displayed as colored spheres
- Drones orbit around their cluster heads with realistic physics
- Camera view labels for each cluster (Press 1/2/3 for zoomed views, 0 for overview)

### 2. **CH0 Compromise Animation** 
- **Red Attacker UAV** appears at round 11 (COMPROMISE_ROUND)
- Attacker approaches CH0 from distance with red attack lines
- **Pulsing red effect** on CH0 when compromised
- **Pause duration**: ~4 seconds for compromise animation
- CH0 turns dark red (compromised color) permanently

### 3. **Frozen Drone Movement During Detection**
- **All Cluster 0 drones STOP moving** during D&R-E phase (rounds 11-17)
- `is_frozen` flag prevents position updates
- Drones remain stationary to indicate isolation
- Movement resumes during recovery phases

### 4. **No Communication Lines During Isolation**
- **No green lines** from Cluster 0 drones to CH0 during detection phase
- **No yellow lines** from CH0 to CH1 (global aggregation skipped)
- Other clusters (1 & 2) continue showing normal communication lines
- Lines resume during continuity phase

### 5. **Gradual Recovery Visualization**
- **30% Participation (Round 18)**: 30% of Cluster 0 drones unfreeze and show orange lines
- **70% Participation (Round 19)**: 70% of drones unfreeze and show yellow lines  
- **100% Participation (Round 20+)**: All drones unfreeze and show green lines
- Visual indication through line colors and drone movement

### 6. **KPI Metrics Display**
Each round shows:
- **Selected clients**: e.g., "C0-D1, C0-D5, C1-D3..." (up to 10 shown)
- **Phase**: NORMAL, DETECTION, CONTINUITY, STABILIZATION
- **Cluster accuracies**: Traffic, Duration, Bandwidth for each cluster
- **Round duration**: Time taken for LOCAL + GLOBAL aggregation
- **Number of clients**: How many participated

### 7. **Parameter Passing Lines**
- **LOCAL round** (green): Drones → Cluster Heads
  - Green for full participation
  - Yellow for 70% participation
  - Orange for 30% participation
- **GLOBAL round** (yellow/blue): Cluster Heads → CH1
  - Yellow for CH0 (when participating)
  - Blue for CH2

### 8. **Comprehensive Summary Table**
Printed after all 30 rounds:
```
Round | Phase          | C0 Traffic | C1 Traffic | C2 Traffic | Duration | Clients
1     | NORMAL         | 0.6783     | 0.6267     | 0.5480     | 15.23    | 9
11    | DETECTION      | 0.5198     | 0.6783     | 0.6450     | 14.87    | 6
18    | CONTINUITY     | 0.5762     | 0.6892     | 0.6521     | 15.01    | 8
...
```

### 9. **Key Statistics**
- Average Cluster 0 accuracy before compromise
- Average Cluster 0 accuracy during D&R-E (using frozen params)
- Average Cluster 0 accuracy after recovery
- Total simulation time
- Average round time

### 10. **Phase Tracking**
- **NORMAL** (Rounds 1-10): All clusters train normally
- **DETECTION** (Rounds 11-17): CH0 compromised, Cluster 0 isolated, frozen params used
- **CONTINUITY** (Rounds 18-20): Gradual recovery (30% → 70% → 100%)
- **STABILIZATION** (Rounds 21-30): Full operation resumed

## 🎮 Camera Controls
- **Press 0**: Overview of all 3 clusters
- **Press 1**: Zoom to Cluster 0 (Compromised cluster)
- **Press 2**: Zoom to Cluster 1 (Global aggregator)
- **Press 3**: Zoom to Cluster 2 (Normal cluster)

## 📊 Performance Optimizations
- Single Ray instance for all 30 rounds (no reinitialization)
- `min_fit_clients=8` to match CPU cores (8 concurrent clients)
- Efficient visualization with 15-frame animations per phase
- ~15 seconds per round (includes training + visualization)

## 🔧 Technical Details
- **Flower 2.0+**: federated learning framework
- **TensorFlow/Keras 3.12**: model training
- **PyBullet**: 3D physics simulation
- **Ray 2.52.1**: distributed training backend
- **Dataset**: CESNET-TLS22, 12500 samples, 42 features, 5 classes

## 📝 Output Files
- Console output shows real-time progress
- Final summary table with all rounds
- Visual PyBullet GUI remains open after training

## ⚡ Quick Start
```bash
conda activate fl_sim
python real_time_transient_simulation.py
```

## 🎯 Expected Behavior
1. PyBullet GUI opens with 45 drones
2. Rounds 1-10: Normal training, all drones moving
3. Round 11: Red attacker compromises CH0 (animation pause)
4. Rounds 11-17: Cluster 0 drones frozen, no lines, frozen params used
5. Round 18: 30% of C0 drones resume (orange lines)
6. Round 19: 70% of C0 drones resume (yellow lines)
7. Round 20+: 100% participation, green lines
8. Final summary table printed
9. GUI stays open for exploration

## 🐛 Known Issues
- Camera keyboard controls require focus on PyBullet window
- Ray import warning (cosmetic, doesn't affect functionality)
