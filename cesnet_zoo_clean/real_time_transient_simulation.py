"""
REAL-TIME TRANSIENT CH COMPROMISE SIMULATION WITH PYBULLET
===========================================================

Features:
- REAL Flower federated learning training (30 rounds)
- CH0 compromise at round 11 (transient case)
- Visual parameter passing with lines
- Drones fly around CHs during training
- Real-time KPI display in terminal
- Uses actual CESNET data from your notebook
"""

import numpy as np
import pickle
import os
import sys
import time
from datetime import datetime
from collections import defaultdict
import pybullet as p
import pybullet_data
import cv2  # For displaying camera images
import threading

# Flower and TensorFlow imports
import flwr as fl
from flwr.common import Context
import tensorflow as tf
from tensorflow import keras
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

# Set seeds for reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Drone configuration
NUM_CLUSTERS = 3
UAVS_PER_CLUSTER = 5
TOTAL_UAVS = NUM_CLUSTERS * UAVS_PER_CLUSTER
TOTAL_ROUNDS = 30
COMPROMISE_ROUND = 11
DETECTION_ROUNDS = 7
CONTINUITY_ROUNDS = 3

# Cluster positions (closer together, 3x size)
CLUSTER_POSITIONS = [
    np.array([-20.0, 0.0, 30.0]),  # Cluster 0 (left)
    np.array([0.0, 0.0, 30.0]),     # Cluster 1 (center)
    np.array([20.0, 0.0, 30.0])     # Cluster 2 (right)
]

# Colors
CLUSTER_COLORS = {
    0: [0.2, 1.0, 0.2, 1.0],   # Green
    1: [0.2, 0.5, 1.0, 1.0],   # Blue
    2: [0.8, 0.2, 1.0, 1.0]    # Purple
}
COLOR_CH = [1.0, 0.84, 0.0, 1.0]  # Golden for original CHs
COLOR_NEW_CH = [1.0, 0.84, 0.0, 1.0]  # Golden for newly elected CHs
COLOR_COMPROMISED = [0.3, 0.3, 0.3, 0.7]
COLOR_PARAM_LINE = [1.0, 1.0, 0.0, 0.5]  # Yellow for parameter passing

# Get drone model path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'gym_pybullet_drones', 'assets')
DRONE_URDF = os.path.join(ASSETS_DIR, 'cf2x.urdf')

# ============================================================================
# LOAD PREPROCESSED DATA
# ============================================================================

print("="*100)
print("REAL-TIME TRANSIENT TRAINING WITH PYBULLET VISUALIZATION")
print("="*100)
print("\nLoading preprocessed CESNET data...")

with open('trained_models/preprocessed_test_data.pkl', 'rb') as f:
    preprocessed_data = pickle.load(f)

X_traffic_test = preprocessed_data['X_traffic']
X_duration_test = preprocessed_data['X_duration']
X_bandwidth_test = preprocessed_data['X_bandwidth']
y_traffic_test = preprocessed_data['y_traffic']
y_duration_test = preprocessed_data['y_duration']
y_bandwidth_test = preprocessed_data['y_bandwidth']

input_dim = preprocessed_data['input_dim']
n_classes = {
    'traffic': len(np.unique(y_traffic_test)),
    'duration': len(np.unique(y_duration_test)),
    'bandwidth': len(np.unique(y_bandwidth_test))
}

print(f"Data loaded:")
print(f"  Input dimension: {input_dim}")
print(f"  Traffic classes: {n_classes['traffic']}")
print(f"  Duration classes: {n_classes['duration']}")
print(f"  Bandwidth classes: {n_classes['bandwidth']}")

# Load training data
print("\nLoading training data...")
import pandas as pd

df = pd.read_csv('datasets/local_cache/dataset_12500_samples_65_features.csv')

# Drop non-numeric and high leakage features
cols_to_drop = [
    'src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol',  # Non-numeric
    'fin_flag_cnt', 'syn_flag_cnt', 'rst_flag_cnt', 'psh_flag_cnt',
    'ack_flag_cnt', 'urg_flag_cnt', 'cwe_flag_cnt', 'ece_flag_cnt',
    'fwd_header_length', 'bwd_header_length',
    'active_mean', 'active_std', 'active_max', 'active_min',
    'idle_mean', 'idle_std', 'idle_max', 'idle_min',
    'subflow_fwd_bytes'
]
df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

# Prepare training data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Extract features and labels - keep only numeric columns
X = df.drop(columns=['label']).select_dtypes(include=[np.number]).values

# Encode string labels to integers
le = LabelEncoder()
y_traffic = le.fit_transform(df['label'])  # Converts 'instagram' → 0, 'youtube' → 1, etc.

print(f"Label encoding: {len(le.classes_)} classes")
print(f"  Classes: {le.classes_[:5]}...")  # Show first 5

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y_traffic, test_size=0.2, random_state=SEED)

# Force types for TensorFlow compatibility
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train).astype(np.float32)
X_test = scaler.transform(X_test).astype(np.float32)

# Update input_dim to match training data
input_dim = X_train.shape[1]

print(f"Training data prepared: {len(X_train)} samples, {X_train.shape[1]} features")
print(f"Test data prepared: {len(X_test)} samples, {X_test.shape[1]} features")

# ============================================================================
# MULTI-TASK MODEL
# ============================================================================

class FedMTLModel(keras.Model):
    """
    Multi-Task Learning Model - MATCHES NOTEBOOK ARCHITECTURE
    
    Architecture:
    - Shared layers: 2 dense layers (256 → 128) with dropout
    - Task-specific layers: 1 dense layer per task (traffic: 64, duration: 32, bandwidth: 64)
    - Task heads: 3 classification heads (traffic, duration, bandwidth)
    """
    
    def __init__(self, in_dims, n_classes, dropout=0.1):
        super().__init__()
        self.in_dims = in_dims
        self.n_classes = n_classes
        self.tasks = ['traffic', 'duration', 'bandwidth']
        
        # Shared layers (learned across all tasks)
        self.shared_dense1 = keras.layers.Dense(256, activation='relu', name='shared_dense1')
        self.shared_drop1 = keras.layers.Dropout(dropout)
        self.shared_dense2 = keras.layers.Dense(128, activation='relu', name='shared_dense2')
        self.shared_drop2 = keras.layers.Dropout(dropout)
        
        # Task-specific layers (MISSING IN OLD VERSION!)
        self.task_dense = {
            'traffic':   keras.layers.Dense(64, activation='relu', name='task_traffic_dense'),
            'duration':  keras.layers.Dense(32, activation='relu', name='task_duration_dense'),
            'bandwidth': keras.layers.Dense(64, activation='relu', name='task_bandwidth_dense'),
        }
        
        # Task heads (output logits)
        self.task_heads = {
            'traffic':   keras.layers.Dense(n_classes['traffic'], name='traffic_output'),
            'duration':  keras.layers.Dense(n_classes['duration'], name='duration_output'),
            'bandwidth': keras.layers.Dense(n_classes['bandwidth'], name='bandwidth_output'),
        }
    
    def call(self, inputs, task='traffic', training=False):
        """Forward pass for a specific task"""
        # Shared layers
        x = self.shared_dense1(inputs)
        x = self.shared_drop1(x, training=training)
        x = self.shared_dense2(x)
        x = self.shared_drop2(x, training=training)
        
        # Task-specific branch
        x = self.task_dense[task](x)
        
        # Final classification head
        return self.task_heads[task](x)
    
    def predict_task(self, inputs, task='traffic', training=False):
        """Predict for a specific task"""
        return self.call(inputs, task=task, training=training)
    
    def build_all(self, max_dim):
        """Build all task heads with a dummy forward pass"""
        tf.random.set_seed(SEED)
        dummy = tf.zeros((1, max_dim))
        for task in self.tasks:
            _ = self.call(dummy, task=task, training=False)

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class DroneData:
    drone_id: int
    cluster_id: int
    ds: Dict[str, Tuple[np.ndarray, np.ndarray]]
    energy_residual: float = 0.85
    rssi_avg: float = 0.80

@dataclass
class UAVMetrics:
    client_id: int
    cluster_id: int
    energy_residual: float
    rssi_avg: float
    num_examples: int
    param_change: float = 0.01

# ============================================================================
# DATA PARTITIONING
# ============================================================================

def partition_data_hierarchical(X_train, y_train, n_clusters=3, clients_per_cluster=15):
    """Partition data hierarchically among clusters and clients"""
    
    n_samples = len(X_train)
    samples_per_cluster = n_samples // n_clusters
    
    clients = []
    
    for cluster_id in range(n_clusters):
        # Get cluster's data
        start_idx = cluster_id * samples_per_cluster
        end_idx = start_idx + samples_per_cluster if cluster_id < n_clusters - 1 else n_samples
        
        cluster_X = X_train[start_idx:end_idx]
        cluster_y = y_train[start_idx:end_idx]
        
        # Shuffle within cluster
        indices = np.random.permutation(len(cluster_X))
        cluster_X = cluster_X[indices]
        cluster_y = cluster_y[indices]
        
        # Divide among clients in cluster
        samples_per_client = len(cluster_X) // clients_per_cluster
        
        for local_id in range(clients_per_cluster):
            start = local_id * samples_per_client
            end = start + samples_per_client if local_id < clients_per_cluster - 1 else len(cluster_X)
            
            client_X = cluster_X[start:end]
            client_y = cluster_y[start:end]
            
            # Create client data (simplified - single task for speed)
            drone_id = cluster_id * clients_per_cluster + local_id
            client_ds = {
                'traffic': (client_X, client_y),
                'duration': (client_X, client_y),  # Reuse for simplicity
                'bandwidth': (client_X, client_y)
            }
            
            clients.append(DroneData(
                drone_id=drone_id,
                cluster_id=cluster_id,
                ds=client_ds,
                energy_residual=np.random.uniform(0.70, 0.95),  # Random initial energy
                rssi_avg=np.random.uniform(0.65, 0.90)  # Random initial RSSI
            ))
    
    print(f"Created {len(clients)} drone clients")
    print(f"  Samples per client: ~{samples_per_client}")
    
    return clients

# Create clients
drone_clients = partition_data_hierarchical(X_train, y_train)

# ============================================================================
# TEST DATA PARTITIONING
# ============================================================================

def create_per_cluster_test_data(n_clusters=3):
    """Create per-cluster test data (equal split) - use CSV test split"""
    cluster_test_data = {}
    
    n_samples = len(y_test)
    samples_per_cluster = n_samples // n_clusters
    
    for cluster_id in range(n_clusters):
        start = cluster_id * samples_per_cluster
        end = start + samples_per_cluster if cluster_id < n_clusters - 1 else n_samples
        
        # Use same data for all tasks (simplified)
        cluster_test_data[cluster_id] = {
            'traffic': (X_test[start:end], y_test[start:end]),
            'duration': (X_test[start:end], y_test[start:end]),  # Reuse
            'bandwidth': (X_test[start:end], y_test[start:end])  # Reuse
        }
    
    return cluster_test_data

test_data_per_cluster = create_per_cluster_test_data()

# ============================================================================
# FLOWER CLIENT
# ============================================================================

class DroneClient(fl.client.NumPyClient):
    """Flower client for drone"""
    
    def __init__(self, drone_data: DroneData, model: FedMTLModel, cfg: dict):
        self.drone_data = drone_data
        self.model = model
        self.cfg = cfg
        
    def get_parameters(self, config):
        return self.model.get_weights()
    
    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        
        # Train on traffic task only (for speed)
        X, y = self.drone_data.ds['traffic']
        
        # Setup optimizer and loss
        optimizer = keras.optimizers.Adam(self.cfg['lr'])
        loss_fn = keras.losses.SparseCategoricalCrossentropy()
        
        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        dataset = dataset.batch(self.cfg['batch_size'])
        
        # Custom training loop
        for epoch in range(self.cfg['local_epochs']):
            for X_batch, y_batch in dataset:
                with tf.GradientTape() as tape:
                    # Get model output and extract traffic logits
                    logits = self.model.predict_task(X_batch, task='traffic', training=True)
                    loss = loss_fn(y_batch, logits)
                
                # Compute and apply gradients
                gradients = tape.gradient(loss, self.model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        return self.model.get_weights(), len(X), {
            'cluster_id': self.drone_data.cluster_id,
            'client_id': self.drone_data.drone_id,
            'energy_residual': self.drone_data.energy_residual,
            'rssi_avg': self.drone_data.rssi_avg
        }
    
    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        
        X, y = self.drone_data.ds['traffic']
        X_tensor = tf.convert_to_tensor(X, dtype=tf.float32)
        logits = self.model.predict_task(X_tensor, task='traffic', training=False)
        preds = tf.argmax(logits, axis=1).numpy()
        acc = float(np.mean(preds == y))
        
        return 0.0, len(X), {'accuracy': acc}

# ============================================================================
# INTEGRATED HIERARCHICAL STRATEGY (TRANSIENT)
# ============================================================================

class IntegratedHierarchicalCHStrategy(fl.server.strategy.FedAvg):
    """Strategy with CH compromise and integrated testing"""
    
    def __init__(
        self,
        test_data,
        model_class,
        in_dims,
        n_classes,
        compromise_round,
        detection_rounds=7,
        continuity_rounds=3,
        uavs=None,
        p=None,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.test_data = test_data
        self.model_class = model_class
        self.in_dims = in_dims
        self.n_classes = n_classes
        self.compromise_round = compromise_round
        self.detection_rounds = detection_rounds
        self.continuity_rounds = continuity_rounds
        
        self.ch_compromised = False
        self.compromise_detected_round = None
        self.recovery_phase = None
        self._frozen_cluster_params = None
        
        self.cluster_test_accuracies = []
        
        # Visualization
        self.uavs = uavs
        self.p = p
        self.recovery_log = []
    
    def _ndarrays_weighted_average(self, param_list):
        if not param_list:
            return None
        total_weight = float(sum(w for _, w in param_list))
        if total_weight <= 0:
            total_weight = 1.0
        summed = [np.zeros_like(arr) for arr in param_list[0][0]]
        for arrays, w in param_list:
            for i, arr in enumerate(arrays):
                summed[i] = summed[i] + (arr * (w / total_weight))
        return summed
    
    def _get_participation_fraction(self, rounds_since_detection):
        if rounds_since_detection < self.detection_rounds:
            return 0.0
        elif rounds_since_detection == self.detection_rounds:
            return 0.3
        elif rounds_since_detection == self.detection_rounds + 1:
            return 0.7
        else:
            return 1.0
    
    def _show_compromise_animation(self):
        """Show red attacker UAV compromising CH0"""
        if self.uavs is None or self.p is None:
            return
        
        # Find CH0
        ch0 = next((u for u in self.uavs if u.is_ch and u.cluster_id == 0), None)
        if not ch0:
            return
        
        # Create red attacker UAV (Crazyflie drone) approaching from distance
        attacker_start = ch0.position + np.array([15, 15, 10])
        
        # Load attacker as Crazyflie drone if URDF exists
        if os.path.exists(DRONE_URDF):
            attacker_id = self.p.loadURDF(
                DRONE_URDF, 
                attacker_start,
                self.p.getQuaternionFromEuler([0, 0, 0]),
                useFixedBase=False,
                globalScaling=2.0
            )
            # Make it RED
            self.p.changeVisualShape(attacker_id, -1, rgbaColor=[1, 0, 0, 1])
            try:
                num_joints = self.p.getNumJoints(attacker_id)
                for j in range(num_joints):
                    self.p.changeVisualShape(attacker_id, j, rgbaColor=[1, 0, 0, 1])
            except:
                pass
        else:
            # Fallback to red sphere
            collision = self.p.createCollisionShape(self.p.GEOM_SPHERE, radius=1.2)
            visual = self.p.createVisualShape(self.p.GEOM_SPHERE, radius=1.2)
            attacker_id = self.p.createMultiBody(0.1, collision, visual, attacker_start)
            self.p.changeVisualShape(attacker_id, -1, rgbaColor=[1, 0, 0, 1])
        
        print(f"   Red Crazyflie attacker approaching CH0...")
        
        # Animate attacker approaching CH0
        for step in range(100):
            progress = step / 100.0
            current_pos = attacker_start * (1 - progress) + ch0.position * progress
            self.p.resetBasePositionAndOrientation(
                attacker_id,
                current_pos,
                self.p.getQuaternionFromEuler([0, 0, 0])
            )
            
            # Draw attack line
            if step % 10 == 0:
                self.p.addUserDebugLine(
                    current_pos, ch0.position,
                    lineColorRGB=[1, 0, 0], lineWidth=5, lifeTime=0.2
                )
            
            # Update other UAVs and camera views
            for uav in self.uavs:
                uav.update_position()
            
            if step % 10 == 0:
                update_camera_windows(self.uavs)
            
            self.p.stepSimulation()
            time.sleep(0.02)
        
        # Pause at CH0 with pulsing effect
        print(f"   CH0 COMPROMISED!")
        for pulse in range(20):
            # Pulse effect
            if pulse % 4 < 2:
                ch0.set_color([1, 0, 0, 1])  # Red
            else:
                ch0.set_color([0.5, 0, 0, 1])  # Dark red
            
            if pulse % 5 == 0:
                update_camera_windows(self.uavs)
            
            self.p.stepSimulation()
            time.sleep(0.1)
        
        # Set final compromised state
        ch0.set_color(COLOR_COMPROMISED)
        
        # Remove attacker
        self.p.removeBody(attacker_id)
        print(f"   CH0 marked as compromised\n")
        
        # Trigger CH re-election in cluster 0 using LEACH
        self._select_new_ch_leach(cluster_id=0)
    
    def _select_new_ch_leach(self, cluster_id, alpha=0.6, beta=0.4):
        """
        Select new CH using LEACH algorithm: max(α·E_residual + β·RSSI_avg)
        
        Args:
            cluster_id: Which cluster to elect new CH for
            alpha: Weight for energy component (default 0.6)
            beta: Weight for RSSI component (default 0.4)
        """
        if self.uavs is None or self.p is None:
            return
        
        print(f"Electing new CH for Cluster {cluster_id} using LEACH algorithm...")
        print(f"   Formula: Score = {alpha}*E_residual + {beta}*RSSI_avg")
        
        # Find old CH and all non-CH drones in the cluster
        old_ch = None
        candidates = []
        for uav in self.uavs:
            if uav.cluster_id == cluster_id:
                if uav.is_ch:
                    old_ch = uav
                else:
                    candidates.append(uav)
        
        if not candidates:
            print(f"   WARNING: No candidates available for CH election")
            return
        
        # Calculate LEACH scores for each candidate
        print(f"\n   Candidate Scores:")
        best_score = -1
        new_ch = None
        
        for uav in candidates:
            # Get energy and RSSI from drone data (normalized 0-1)
            energy = getattr(uav, 'energy_residual', 0.8)  # Default if not set
            rssi = getattr(uav, 'rssi_avg', 0.75)  # Default if not set
            
            score = alpha * energy + beta * rssi
            
            print(f"   Drone {uav.drone_id}: E={energy:.3f}, RSSI={rssi:.3f} -> Score={score:.3f}")
            
            if score > best_score:
                best_score = score
                new_ch = uav
        
        if new_ch is None:
            print(f"   WARNING: CH election failed")
            return
        
        # Update old CH (if exists)
        if old_ch:
            old_ch.is_ch = False
            old_ch.set_color(CLUSTER_COLORS[cluster_id])
            print(f"\n   Old CH: Drone {old_ch.drone_id} -> Regular drone")
        
        # Update new CH
        new_ch.is_ch = True
        new_ch.is_frozen = False  # Unfreeze if was frozen
        new_ch.participation_level = 1.0
        
        # Set golden color for new CH - use global constant
        new_ch.set_color(COLOR_NEW_CH)
        
        print(f"   New CH elected: Drone {new_ch.drone_id}")
        print(f"      E_residual={getattr(new_ch, 'energy_residual', 0.8):.3f}")
        print(f"      RSSI_avg={getattr(new_ch, 'rssi_avg', 0.75):.3f}")
        print(f"      Final Score={best_score:.3f}")
        print(f"   New CH is now golden and stationary\n")
    
    def _visualize_round(self, server_round, phase, line_type='local'):
        """Animate UAVs and show parameter passing lines"""
        if self.uavs is None or self.p is None:
            return
        
        # Update UAV states based on phase
        cluster_0_drones = []
        for uav in self.uavs:
            if uav.cluster_id == 0:
                cluster_0_drones.append(uav)
                if phase == 'detection':
                    uav.participation_level = 0.0
                    uav.is_frozen = True  # Freeze movement
                    if uav.is_ch:
                        uav.is_compromised = True
                        uav.set_color(COLOR_COMPROMISED)
                elif phase == 'continuity':
                    rounds_since = server_round - self.compromise_round
                    if rounds_since == self.detection_rounds:
                        uav.participation_level = 0.3
                        # 30% of drones resume
                        if uav.id % 10 < 3:
                            uav.is_frozen = False
                    elif rounds_since == self.detection_rounds + 1:
                        uav.participation_level = 0.7
                        # 70% of drones resume
                        if uav.id % 10 < 7:
                            uav.is_frozen = False
                    else:
                        uav.participation_level = 1.0
                        uav.is_frozen = False  # All resume
                else:
                    # Normal or stabilization
                    uav.participation_level = 1.0
                    uav.is_frozen = False
            else:
                uav.participation_level = 1.0
        
        # Animate for a few frames
        for frame in range(15):
            for uav in self.uavs:
                uav.update_position()
            
            # Draw parameter lines
            if frame % 5 == 0:  # Draw lines every 5 frames
                if line_type == 'local':
                    # Local aggregation: drones to CH
                    for uav in self.uavs:
                        if not uav.is_ch and uav.participation_level > 0:
                            # Skip CH0 during detection
                            if uav.cluster_id == 0 and phase == 'detection':
                                continue
                            
                            ch = next((u for u in self.uavs if u.is_ch and u.cluster_id == uav.cluster_id), None)
                            if ch:
                                # Color based on participation
                                if uav.participation_level == 1.0:
                                    color = [0, 1, 0]  # Green - full
                                elif uav.participation_level == 0.7:
                                    color = [1, 1, 0]  # Yellow - 70%
                                elif uav.participation_level == 0.3:
                                    color = [1, 0.5, 0]  # Orange - 30%
                                else:
                                    continue
                                
                                self.p.addUserDebugLine(
                                    uav.position, ch.position,
                                    lineColorRGB=color, lineWidth=2, lifeTime=0.3
                                )
                elif line_type == 'global':
                    # Global aggregation: CHs to CH1 (skip CH0 during detection)
                    ch1 = next((u for u in self.uavs if u.is_ch and u.cluster_id == 1), None)
                    for uav in self.uavs:
                        if uav.is_ch and uav.cluster_id != 1:
                            # Skip CH0 during detection
                            if uav.cluster_id == 0 and phase == 'detection':
                                continue
                            
                            if ch1:
                                color = [1, 1, 0] if uav.cluster_id == 0 else [0, 0.5, 1]
                                self.p.addUserDebugLine(
                                    uav.position, ch1.position,
                                    lineColorRGB=color, lineWidth=3, lifeTime=0.3
                                )
            
            # Update camera views every few frames
            if frame % 5 == 0:
                update_camera_windows(self.uavs)
            
            self.p.stepSimulation()
            time.sleep(0.01)
    
    def aggregate_fit(self, server_round, results, failures):
        round_start_time = time.time()
        
        if len(results) == 0:
            return None, {}
        
        # Determine phase for visualization
        if server_round < self.compromise_round:
            phase = 'normal'
        elif server_round < self.compromise_round + self.detection_rounds:
            phase = 'detection'
        elif server_round < self.compromise_round + self.detection_rounds + self.continuity_rounds:
            phase = 'continuity'
        else:
            phase = 'stabilization'
        
        # Show LOCAL round visualization (clients -> CHs)
        print(f"\n{'='*100}")
        print(f"ROUND {server_round}/{30} - Phase: {phase.upper()}")
        print(f"{'='*100}")
        
        # Show selected clients
        selected_clients = []
        for client_proxy, fit_res in results:
            cid = int(fit_res.metrics.get('cluster_id', 0))
            client_id = fit_res.metrics.get('client_id', '?')
            selected_clients.append(f"C{cid}-D{client_id}")
        print(f"   Selected {len(results)} clients: {', '.join(selected_clients[:10])}{'...' if len(selected_clients) > 10 else ''}")
        
        print(f"   LOCAL aggregation (drones -> cluster heads)...")
        self._visualize_round(server_round, phase, line_type='local')
        
        # Check for compromise - PAUSE FOR ANIMATION
        if server_round == self.compromise_round and not self.ch_compromised:
            print(f"\n{'!'*80}")
            print(f"ATTACK IN PROGRESS...")
            print(f"{'!'*80}\n")
            
            # Show attacker compromising CH0
            self._show_compromise_animation()
            
            self.ch_compromised = True
            self.compromise_detected_round = server_round
            self.recovery_phase = 'detection'
            print(f"\nCH0 COMPROMISE DETECTED AT ROUND {server_round}")
            print(f"   Cluster 0 will be isolated for {self.detection_rounds} rounds")
            self.recovery_log.append({'round': server_round, 'event': 'CH0_COMPROMISE', 'time': time.time() - round_start_time})
        
        # Extract client results
        triples = []
        for client_proxy, fit_res in results:
            nds = fl.common.parameters_to_ndarrays(fit_res.parameters)
            weight = fit_res.num_examples
            cluster_id = int(fit_res.metrics.get('cluster_id', 0))
            client_id = int(fit_res.metrics.get('client_id', 0))
            energy = float(fit_res.metrics.get('energy_residual', 0.85))
            rssi = float(fit_res.metrics.get('rssi_avg', 0.80))
            
            # Update UAV object with energy and RSSI values
            if self.uavs:
                for uav in self.uavs:
                    if uav.drone_id == client_id and uav.cluster_id == cluster_id:
                        uav.energy_residual = energy
                        uav.rssi_avg = rssi
                        break
            
            triples.append((nds, weight, cluster_id, fit_res.metrics))
        
        # Update recovery phase
        if self.ch_compromised:
            rounds_since = server_round - self.compromise_detected_round
            participation_fraction = self._get_participation_fraction(rounds_since)
            
            if rounds_since < self.detection_rounds:
                self.recovery_phase = 'detection'
                print(f"   Cluster 0 ISOLATED (detection phase {rounds_since}/{self.detection_rounds})")
            elif rounds_since < self.detection_rounds + self.continuity_rounds:
                if self.recovery_phase != 'continuity':
                    self.recovery_phase = 'continuity'
                    print(f"\nCONTINUITY PHASE STARTED AT ROUND {server_round}")
                
                # Show gradual recovery
                if rounds_since == self.detection_rounds:
                    print(f"   30% of Cluster 0 drones participating")
                elif rounds_since == self.detection_rounds + 1:
                    print(f"   70% of Cluster 0 drones participating")
                else:
                    print(f"   100% of Cluster 0 drones participating")
            else:
                if self.recovery_phase != 'stabilization':
                    self.recovery_phase = 'stabilization'
                    print(f"STABILIZATION PHASE AT ROUND {server_round}")
        
        # Cluster aggregation with participation control
        cluster_to_pairs = {}
        participating_clusters = set()
        actually_participating_clients = 0  # Track actual participating clients
        
        for nds, w, cid, metrics in triples:
            if self.ch_compromised and cid == 0:
                rounds_since = server_round - self.compromise_detected_round
                participation_fraction = self._get_participation_fraction(rounds_since)
                
                if participation_fraction == 0:
                    continue  # Skip during detection
            
            cluster_to_pairs.setdefault(cid, []).append((nds, w))
            actually_participating_clients += 1  # Count this client as participating
        
        # Aggregate within clusters
        cluster_params = {}
        for cid, pairs in cluster_to_pairs.items():
            if pairs:
                cluster_params[cid] = self._ndarrays_weighted_average(pairs)
                participating_clusters.add(cid)
        
        # Freeze Cluster 0 params at compromise
        if server_round == self.compromise_round and 0 in cluster_params:
            self._frozen_cluster_params = [arr.copy() for arr in cluster_params[0]]
            print(f"  Froze Cluster 0 parameters at round {server_round}")
        
        if not cluster_params:
            return None, {}
        
        # Show GLOBAL round visualization (CHs -> CH1)
        print(f"   GLOBAL aggregation (cluster heads -> CH1)...")
        self._visualize_round(server_round, phase, line_type='global')
        
        # Global aggregation
        global_pairs = [(cluster_params[cid], 1.0) for cid in cluster_params.keys()]
        averaged = self._ndarrays_weighted_average(global_pairs)
        aggregated_params = fl.common.ndarrays_to_parameters(averaged)
        
        # Integrated testing
        test_results = {}
        for cid in range(3):
            if cid not in self.test_data:
                continue
            
            # Use frozen params for C0 during detection
            if (cid == 0 and 
                self.recovery_phase == 'detection' and 
                self._frozen_cluster_params is not None):
                test_params = self._frozen_cluster_params
            elif cid in cluster_params:
                test_params = cluster_params[cid]
            else:
                continue
            
            temp_model = self.model_class(self.in_dims, self.n_classes, dropout=0.1)
            temp_model.build_all(input_dim)
            temp_model.set_weights(test_params)
            
            task_metrics = {}
            for task in ['traffic', 'duration', 'bandwidth']:
                if task not in self.test_data[cid]:
                    continue
                X_test, y_test = self.test_data[cid][task]
                X_tensor = tf.convert_to_tensor(X_test, dtype=tf.float32)
                logits = temp_model.predict_task(X_tensor, task=task, training=False)
                preds = tf.argmax(logits, axis=1).numpy()
                acc = float(np.mean(preds == y_test))
                task_metrics[f'{task}_accuracy'] = acc
            
            test_results[cid] = task_metrics
        
        round_duration = time.time() - round_start_time
        
        self.cluster_test_accuracies.append({
            'round': server_round,
            'accuracies': test_results,
            'participating_clusters': list(participating_clusters),
            'phase': self.recovery_phase,
            'duration': round_duration,
            'num_clients': actually_participating_clients,  # Use actual participating count
            'selected_clients': len(results)  # Total selected (for KPI tracking)
        })
        
        # Print results with KPIs
        print(f"\n   Round Results:")
        for cid in sorted(test_results.keys()):
            traffic_acc = test_results[cid].get('traffic_accuracy', 0)
            duration_acc = test_results[cid].get('duration_accuracy', 0)
            bandwidth_acc = test_results[cid].get('bandwidth_accuracy', 0)
            print(f"      Cluster {cid}: Traffic={traffic_acc:.4f} | Duration={duration_acc:.4f} | Bandwidth={bandwidth_acc:.4f}")
        
        print(f"   Round Duration: {round_duration:.2f}s | Participating: {actually_participating_clients}/{len(results)} clients")
        print(f"{'='*100}\n")
        
        return aggregated_params, {}

# ============================================================================
# PYBULLET VISUALIZATION
# ============================================================================

class UAV:
    """UAV with orbital motion"""
    
    def __init__(self, uav_id, cluster_id, position, is_ch=False):
        self.id = uav_id
        self.drone_id = uav_id  # Alias for LEACH algorithm
        self.cluster_id = cluster_id
        self.position = np.array(position, dtype=float)
        self.is_ch = is_ch
        self.is_compromised = False
        self.is_frozen = False  # Stop movement when compromised
        self.body_id = None
        self.participation_level = 1.0
        
        # Energy and RSSI for LEACH algorithm (will be updated from fit() results)
        self.energy_residual = 0.85  # Initial energy level
        self.rssi_avg = 0.80  # Initial RSSI
        
        # Orbital parameters
        if not is_ch:
            self.orbit_radius = np.random.uniform(3, 6)
            self.orbit_speed = np.random.uniform(0.3, 0.6)
            self.orbit_angle = np.random.uniform(0, 2 * np.pi)
            self.orbit_height_offset = np.random.uniform(-2, 2)
            self.ch_position = CLUSTER_POSITIONS[cluster_id]
    
    def update_position(self):
        """Update drone position (orbital motion for members)"""
        # CHs don't move, and frozen drones don't move
        if self.is_ch or self.is_frozen:
            return
        
        # Only update if orbital attributes exist (regular drones)
        if not hasattr(self, 'orbit_angle'):
            return
        
        # Orbit around CH
        self.orbit_angle += self.orbit_speed * 0.05
        self.position[0] = self.ch_position[0] + self.orbit_radius * np.cos(self.orbit_angle)
        self.position[1] = self.ch_position[1] + self.orbit_radius * np.sin(self.orbit_angle)
        self.position[2] = self.ch_position[2] + self.orbit_height_offset
        
        if self.body_id is not None:
            p.resetBasePositionAndOrientation(
                self.body_id,
                self.position,
                p.getQuaternionFromEuler([0, 0, 0])
            )
    
    def set_color(self, color):
        if self.body_id is not None:
            p.changeVisualShape(self.body_id, -1, rgbaColor=color)
            try:
                num_joints = p.getNumJoints(self.body_id)
                for j in range(num_joints):
                    p.changeVisualShape(self.body_id, j, rgbaColor=color)
            except:
                pass

def draw_parameter_lines(uavs, phase, current_round):
    """Draw lines showing parameter passing"""
    lines = []
    
    # Members → CH (within cluster)
    for uav in uavs:
        if not uav.is_ch and uav.participation_level > 0:
            ch = [u for u in uavs if u.is_ch and u.cluster_id == uav.cluster_id][0]
            line = p.addUserDebugLine(
                uav.position,
                ch.position,
                lineColorRGB=[1, 1, 0],
                lineWidth=1,
                lifeTime=0.5
            )
            lines.append(line)
    
    # CH → Global aggregator (CH1)
    if phase != 'detection' or current_round < COMPROMISE_ROUND + 1:
        ch1 = [u for u in uavs if u.is_ch and u.cluster_id == 1][0]
        for uav in uavs:
            if uav.is_ch and uav.cluster_id != 1 and uav.participation_level > 0:
                line = p.addUserDebugLine(
                    uav.position,
                    ch1.position,
                    lineColorRGB=[0, 1, 1],
                    lineWidth=2,
                    lifeTime=0.5
                )
                lines.append(line)
    
    return lines

# ============================================================================
# CAMERA SYSTEM
# ============================================================================

def create_cluster_camera_views(uavs, cluster_id):
    """Create camera view focused on a cluster head"""
    # Find the cluster head
    ch = next((u for u in uavs if u.is_ch and u.cluster_id == cluster_id), None)
    if not ch:
        return None
    
    # Camera parameters - much closer for better view
    ch_pos = ch.position
    camera_distance = 8  # Closer distance
    camera_height = 5    # Height above CH
    
    # Camera looks at CH from an angle
    angle = cluster_id * (2 * np.pi / 3)  # Different angle for each camera
    camera_pos = ch_pos + np.array([
        camera_distance * np.cos(angle),
        camera_distance * np.sin(angle),
        camera_height
    ])
    
    # View matrix (camera looking at CH)
    view_matrix = p.computeViewMatrix(
        cameraEyePosition=camera_pos.tolist(),
        cameraTargetPosition=ch_pos.tolist(),
        cameraUpVector=[0, 0, 1]
    )
    
    # Projection matrix with better FOV
    proj_matrix = p.computeProjectionMatrixFOV(
        fov=75,  # Wider field of view
        aspect=4.0/3.0,
        nearVal=0.1,
        farVal=100
    )
    
    # Capture image with higher resolution
    width, height = 640, 480
    img_arr = p.getCameraImage(
        width, height,
        viewMatrix=view_matrix,
        projectionMatrix=proj_matrix,
        renderer=p.ER_BULLET_HARDWARE_OPENGL
    )
    
    # Extract RGB image (index 2 is RGB data)
    rgb_array = np.array(img_arr[2], dtype=np.uint8).reshape(height, width, 4)[:, :, :3]
    
    return rgb_array

def update_camera_windows(uavs):
    """Update all 3 cluster camera windows"""
    try:
        for cluster_id in range(3):
            img = create_cluster_camera_views(uavs, cluster_id)
            if img is not None:
                # Convert RGB to BGR for OpenCV
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                
                # Find cluster head for status
                ch = next((u for u in uavs if u.is_ch and u.cluster_id == cluster_id), None)
                status = "COMPROMISED" if (ch and ch.is_compromised) else "ACTIVE"
                color = (0, 0, 255) if status == "COMPROMISED" else (0, 255, 0)
                
                # Add text overlay with black background for visibility
                cv2.rectangle(img_bgr, (0, 0), (250, 70), (0, 0, 0), -1)
                cv2.putText(img_bgr, f"Cluster {cluster_id} CH View", (10, 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(img_bgr, f"Status: {status}", (10, 55),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Show window
                window_name = f"Cluster {cluster_id} Camera (Zoomed)"
                cv2.imshow(window_name, img_bgr)
        
        cv2.waitKey(1)  # Process window events
    except Exception as e:
        print(f"Camera error: {e}")  # Debug output

# ============================================================================
# KPI TRACKER
# ============================================================================

class KPITracker:
    """Track KPIs during training"""
    
    def __init__(self):
        self.round_times = []
        self.start_time = time.time()
    
    def print_round_kpis(self, round_num, phase, accuracies, duration):
        """Print KPIs for current round"""
        phase_icons = {'normal': '✅', 'detection': '🔧', 'continuity': '📊', 'stabilization': '🔄'}
        icon = phase_icons.get(phase, '📍')
        
        print(f"\n{icon} ROUND {round_num}/{TOTAL_ROUNDS} | Phase: {phase.upper() if phase else 'NORMAL'} | Duration: {duration:.1f}s")
        print("-" * 100)
        
        for cid in sorted(accuracies.keys()):
            acc = accuracies[cid]
            frozen = " [FROZEN]" if phase == 'detection' and cid == 0 else ""
            print(f"  Cluster {cid}: Traffic={acc['traffic_accuracy']:.4f} | "
                  f"Duration={acc['duration_accuracy']:.4f} | "
                  f"Bandwidth={acc['bandwidth_accuracy']:.4f}{frozen}")
        
        cumulative = time.time() - self.start_time
        print(f"  Cumulative Time: {cumulative:.1f}s ({cumulative/60:.1f} min)")

# ============================================================================
# MAIN SIMULATION
# ============================================================================

def main():
    """Main simulation loop"""
    
    # Initialize PyBullet
    print("\nInitializing PyBullet GUI...")
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf", [0, 0, 0])
    
    # Main camera view (overview)
    p.resetDebugVisualizerCamera(
        cameraDistance=50,
        cameraYaw=45,
        cameraPitch=-20,
        cameraTargetPosition=[0, 0, 25]
    )
    
    # Add text labels for cluster views
    print("   Setting up camera views for each cluster...")
    cluster_view_text_ids = []
    for i in range(3):
        text_pos = CLUSTER_POSITIONS[i] + np.array([0, 0, 12])
        text_id = p.addUserDebugText(
            f"Cluster {i}\n(Press {i+1} for zoom)",
            text_pos,
            textColorRGB=[1, 1, 1],
            textSize=1.5
        )
        cluster_view_text_ids.append(text_id)
    
    print("   Press 1/2/3 for zoomed cluster views, 0 for overview")
    
    # Create UAVs
    print("\nCreating UAV swarm...")
    uavs = []
    
    for cluster_id in range(NUM_CLUSTERS):
        cluster_center = CLUSTER_POSITIONS[cluster_id]
        
        for local_id in range(UAVS_PER_CLUSTER):
            uav_id = cluster_id * UAVS_PER_CLUSTER + local_id
            is_ch = (local_id == 0)
            
            if is_ch:
                position = cluster_center
            else:
                angle = (local_id / UAVS_PER_CLUSTER) * 2 * np.pi
                radius = 5
                offset = np.array([
                    radius * np.cos(angle),
                    radius * np.sin(angle),
                    np.random.uniform(-2, 2)
                ])
                position = cluster_center + offset
            
            uav = UAV(uav_id, cluster_id, position, is_ch)
            
            # Load drone
            if os.path.exists(DRONE_URDF):
                uav.body_id = p.loadURDF(DRONE_URDF, position,
                                         p.getQuaternionFromEuler([0, 0, 0]),
                                         useFixedBase=True,
                                         globalScaling=3.0)
            else:
                collision = p.createCollisionShape(p.GEOM_SPHERE, radius=0.9)
                visual = p.createVisualShape(p.GEOM_SPHERE, radius=0.9)
                uav.body_id = p.createMultiBody(0.1, collision, visual, position)
            
            color = COLOR_CH if is_ch else CLUSTER_COLORS[cluster_id]
            uav.set_color(color)
            
            uavs.append(uav)
    
    print(f"Created {len(uavs)} UAVs")
    
    # Setup Flower
    print("\nSetting up Flower federated learning...")
    
    def client_fn(context: Context):
        client_id = int(context.node_id)
        drone_data = drone_clients[client_id % len(drone_clients)]
        in_dims = {'all': input_dim}  # Wrap scalar in dict
        model = FedMTLModel(in_dims, n_classes, dropout=0.1)
        model.build_all(input_dim)
        return DroneClient(drone_data, model, {
            'local_epochs': 1,  # Match notebook (1 epoch per round)
            'batch_size': 32,
            'lr': 1e-3
        }).to_client()
    
    # Create strategy
    in_dims = {'all': input_dim}  # Wrap scalar in dict
    global_model = FedMTLModel(in_dims, n_classes, dropout=0.1)
    global_model.build_all(input_dim)
    
    # Adjust min_fit_clients to available CPU cores (8) to avoid Ray actor bottleneck
    MIN_FIT_CLIENTS = 8
    
    strategy = IntegratedHierarchicalCHStrategy(
        test_data=test_data_per_cluster,
        model_class=FedMTLModel,
        in_dims=in_dims,
        n_classes=n_classes,
        compromise_round=COMPROMISE_ROUND,
        uavs=uavs,  # Pass UAVs for visualization
        p=p,  # Pass PyBullet instance
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=MIN_FIT_CLIENTS,  # Match CPU cores instead of all clients
        min_available_clients=TOTAL_UAVS,
        min_evaluate_clients=MIN_FIT_CLIENTS,
        initial_parameters=fl.common.ndarrays_to_parameters(global_model.get_weights())
    )
    
    # KPI tracker
    kpi_tracker = KPITracker()
    
    print(f"\n{'='*100}")
    print(f"STARTING REAL-TIME TRANSIENT TRAINING (30 ROUNDS)")
    print(f"{'='*100}")
    print(f"CH0 Compromise: Round {COMPROMISE_ROUND}")
    print(f"D&R-E: Rounds {COMPROMISE_ROUND+1}-{COMPROMISE_ROUND+DETECTION_ROUNDS}")
    print(f"Continuity: Rounds {COMPROMISE_ROUND+DETECTION_ROUNDS+1}-{COMPROMISE_ROUND+DETECTION_ROUNDS+CONTINUITY_ROUNDS}")
    print(f"{'='*100}\n")
    
    # Test initial model accuracy (Round 0 - before any training)
    print(f"Testing initial model accuracy (Round 0 - untrained)...")
    initial_test_results = {}
    for cid in range(3):
        if cid not in test_data_per_cluster:
            continue
        
        temp_model = FedMTLModel(in_dims, n_classes, dropout=0.1)
        temp_model.build_all(input_dim)
        temp_model.set_weights(global_model.get_weights())
        
        task_metrics = {}
        for task in ['traffic', 'duration', 'bandwidth']:
            if task not in test_data_per_cluster[cid]:
                continue
            X_test_data, y_test_data = test_data_per_cluster[cid][task]
            X_tensor = tf.convert_to_tensor(X_test_data, dtype=tf.float32)
            logits = temp_model.predict_task(X_tensor, task=task, training=False)
            preds = tf.argmax(logits, axis=1).numpy()
            acc = float(np.mean(preds == y_test_data))
            task_metrics[f'{task}_accuracy'] = acc
        
        initial_test_results[cid] = task_metrics
    
    print(f"Initial Accuracy (Untrained Model):")
    for cid in sorted(initial_test_results.keys()):
        traffic_acc = initial_test_results[cid].get('traffic_accuracy', 0)
        duration_acc = initial_test_results[cid].get('duration_accuracy', 0)
        bandwidth_acc = initial_test_results[cid].get('bandwidth_accuracy', 0)
        print(f"   Cluster {cid}: Traffic={traffic_acc:.4f} | Duration={duration_acc:.4f} | Bandwidth={bandwidth_acc:.4f}")
    print(f"{'='*100}\n")
    
    # Store initial results as round 0
    strategy.cluster_test_accuracies.append({
        'round': 0,
        'accuracies': initial_test_results,
        'participating_clusters': [0, 1, 2],
        'phase': None,
        'duration': 0.0,
        'num_clients': 0,
        'selected_clients': 0
    })
    
    # Start all 30 rounds in a single simulation
    kpi_tracker = KPITracker()
    
    print(f"Starting Flower federated learning ({TOTAL_ROUNDS} rounds)...")
    print(f"   Using {MIN_FIT_CLIENTS} clients per round (out of {TOTAL_UAVS} available)\n")
    
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=TOTAL_UAVS,
        config=fl.server.ServerConfig(num_rounds=TOTAL_ROUNDS),
        strategy=strategy,
        client_resources={"num_cpus": 1}
    )
    
    # Summary
    total_time = time.time() - kpi_tracker.start_time
    print(f"\n{'='*100}")
    print(f"TRAINING COMPLETE - {len(strategy.cluster_test_accuracies)} rounds")
    print(f"   Total Time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"   Avg Time/Round: {total_time/TOTAL_ROUNDS:.1f}s")
    print(f"{'='*100}\n")
    
    # Print comprehensive summary table
    print(f"\n{'='*100}")
    print(f"COMPREHENSIVE RESULTS TABLE")
    print(f"{'='*100}")
    print(f"{'Round':<7} {'Phase':<15} {'C0 Traffic':<12} {'C1 Traffic':<12} {'C2 Traffic':<12} {'Duration':<10} {'Participate':<12}")
    print(f"{'-'*100}")
    
    for round_data in strategy.cluster_test_accuracies:
        round_num = round_data['round']
        phase = round_data['phase'] or 'normal'
        accuracies = round_data['accuracies']
        duration = round_data.get('duration', 0)
        num_clients = round_data.get('num_clients', 0)
        selected_clients = round_data.get('selected_clients', num_clients)
        
        c0_acc = accuracies.get(0, {}).get('traffic_accuracy', 0)
        c1_acc = accuracies.get(1, {}).get('traffic_accuracy', 0)
        c2_acc = accuracies.get(2, {}).get('traffic_accuracy', 0)
        
        phase_str = phase.upper()[:13]
        participate_str = f"{num_clients}/{selected_clients}"
        print(f"{round_num:<7} {phase_str:<15} {c0_acc:<12.4f} {c1_acc:<12.4f} {c2_acc:<12.4f} {duration:<10.2f} {participate_str:<12}")
    
    print(f"{'-'*100}")
    print(f"{'TOTAL':<7} {'':<15} {'':<12} {'':<12} {'':<12} {total_time:<10.1f} {'':<12}")
    print(f"{'='*100}\n")
    
    # Statistics with comprehensive KPIs
    print(f"KEY STATISTICS AND KPIs:")
    print(f"{'-'*100}")
    
    c0_before = [r['accuracies'].get(0, {}).get('traffic_accuracy', 0) 
                 for r in strategy.cluster_test_accuracies if r['round'] < COMPROMISE_ROUND]
    c0_during = [r['accuracies'].get(0, {}).get('traffic_accuracy', 0) 
                 for r in strategy.cluster_test_accuracies 
                 if COMPROMISE_ROUND <= r['round'] < COMPROMISE_ROUND + DETECTION_ROUNDS]
    c0_after = [r['accuracies'].get(0, {}).get('traffic_accuracy', 0) 
                for r in strategy.cluster_test_accuracies if r['round'] >= COMPROMISE_ROUND + DETECTION_ROUNDS]
    
    # TIER 1: Learning Performance
    print(f"\n[TIER 1: Learning Performance]")
    if c0_before:
        print(f"   C0 Avg Accuracy (Before Compromise): {np.mean(c0_before):.4f}")
    if c0_during:
        print(f"   C0 Avg Accuracy (During D&R-E): {np.mean(c0_during):.4f} (using frozen params)")
        accuracy_degradation = np.mean(c0_before) - np.mean(c0_during) if c0_before else 0
        print(f"   Accuracy Degradation: {accuracy_degradation:.4f} ({accuracy_degradation*100:.1f}%)")
    if c0_after:
        print(f"   C0 Avg Accuracy (After Recovery): {np.mean(c0_after):.4f}")
        if c0_before:
            recovery_rate = (np.mean(c0_after) / np.mean(c0_before)) * 100
            print(f"   Recovery Rate: {recovery_rate:.1f}%")
    
    # TIER 1: Timing
    print(f"\n[TIER 1: Timing]")
    avg_round_time = total_time / TOTAL_ROUNDS
    print(f"   Average Round Time: {avg_round_time:.2f}s")
    print(f"   Total Simulation Time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"   Convergence Rounds: {TOTAL_ROUNDS}")
    
    # TIER 2: Attack Impact
    print(f"\n[TIER 2: Attack Impact]")
    print(f"   Compromise Round: {COMPROMISE_ROUND}")
    print(f"   Detection Duration: {DETECTION_ROUNDS} rounds ({DETECTION_ROUNDS * avg_round_time:.1f}s)")
    print(f"   Continuity Duration: {CONTINUITY_ROUNDS} rounds ({CONTINUITY_ROUNDS * avg_round_time:.1f}s)")
    recovery_time = (DETECTION_ROUNDS + CONTINUITY_ROUNDS) * avg_round_time
    print(f"   Total Recovery Time: {recovery_time:.1f}s")
    
    # TIER 2: Cluster Health
    print(f"\n[TIER 2: Cluster Health]")
    total_participating = sum(r.get('num_clients', 0) for r in strategy.cluster_test_accuracies)
    total_selected = sum(r.get('selected_clients', r.get('num_clients', 0)) for r in strategy.cluster_test_accuracies)
    overall_participation_rate = (total_participating / total_selected * 100) if total_selected > 0 else 0
    print(f"   Overall Participation Rate: {overall_participation_rate:.1f}% ({total_participating}/{total_selected} client-rounds)")
    
    # Calculate isolation impact
    isolated_rounds = [r for r in strategy.cluster_test_accuracies if r.get('phase') == 'detection']
    if isolated_rounds:
        avg_isolated_clients = np.mean([r.get('num_clients', 0) for r in isolated_rounds])
        avg_isolated_selected = np.mean([r.get('selected_clients', r.get('num_clients', 0)) for r in isolated_rounds])
        isolation_impact = ((avg_isolated_selected - avg_isolated_clients) / avg_isolated_selected * 100) if avg_isolated_selected > 0 else 0
        print(f"   Isolation Impact: {isolation_impact:.1f}% ({avg_isolated_clients:.0f}/{avg_isolated_selected:.0f} clients during detection)")
    
    print(f"{'='*100}\n")
    
    # Animate final visualization with trained results
    print("Visualization running with 3 camera windows...")
    print("   Main PyBullet window + 3 OpenCV camera windows (Cluster 0, 1, 2)")
    print("   Press Ctrl+C to exit")
    
    # Create initial camera windows
    print("\nOpening camera windows...")
    update_camera_windows(uavs)
    
    try:
        frame_count = 0
        while True:
            for uav in uavs:
                uav.update_position()
            
            # Update camera views every 10 frames
            if frame_count % 10 == 0:
                update_camera_windows(uavs)
            
            p.stepSimulation()
            frame_count += 1
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nClosing camera windows...")
        cv2.destroyAllWindows()
        print("Exiting...")
    
    p.disconnect()

if __name__ == "__main__":
    # Import Ray for Flower
    import ray
    if not ray.is_initialized():
        ray.init(num_cpus=8, include_dashboard=False)
    
    main()
    
    ray.shutdown()
