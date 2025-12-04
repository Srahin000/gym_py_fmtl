"""
Gym-PyBullet Integrated Simulation
Implements the complete integrated training+testing approach with:
- 15 drones per cluster (45 total)
- Per-cluster testing (equal + Dirichlet splits)
- CH1 as global aggregator
- CH0 compromise scenarios
- KPI tracking
- Model checkpointing every round
- NO delays for faster simulation
"""

import numpy as np
import pickle
import os
import sys
import tensorflow as tf
from tensorflow import keras
import flwr as fl
from flwr.common import Context
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import time
from collections import defaultdict

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class SimulationConfig:
    """Configuration for gym simulation"""
    # Drone configuration
    n_drones_per_cluster: int = 15
    n_clusters: int = 3
    total_drones: int = 45
    
    # Training parameters
    local_epochs: int = 1
    batch_size: int = 32
    lr: float = 1e-3
    dropout: float = 0.1
    
    # FL parameters
    global_aggregator_cluster: int = 1
    client_frac: float = 1.0
    
    # CH Compromise
    detection_rounds: int = 7
    continuity_rounds: int = 3
    alpha_energy: float = 0.6
    beta_rssi: float = 0.4
    
    # Data paths
    preprocessed_data_path: str = 'trained_models/preprocessed_test_data.pkl'
    
    # Model save paths
    convergence_save_dir: str = 'trained_models/gym_convergence_integrated'
    transient_save_dir: str = 'trained_models/gym_transient_integrated'
    
    # Scenarios
    convergence_rounds: int = 125
    convergence_compromise_round: int = 111
    transient_rounds: int = 30
    transient_compromise_round: int = 11

CFG = SimulationConfig()

print("="*80)
print("🚁 GYM-PYBULLET: INTEGRATED TRAINING+TESTING SIMULATION")
print("="*80)
print(f"Configuration:")
print(f"  Total Drones: {CFG.total_drones} ({CFG.n_clusters} clusters × {CFG.n_drones_per_cluster} drones)")
print(f"  Global Aggregator: Cluster {CFG.global_aggregator_cluster}")
print(f"  Client Fraction: {CFG.client_frac * 100}%")
print(f"  Local Epochs: {CFG.local_epochs}")
print("="*80)

# ============================================================================
# LOAD PREPROCESSED DATA
# ============================================================================

print("\n📂 Loading preprocessed test data...")
with open(CFG.preprocessed_data_path, 'rb') as f:
    preprocessed_data = pickle.load(f)

X_traffic_test = preprocessed_data['X_traffic']
X_duration_test = preprocessed_data['X_duration']
X_bandwidth_test = preprocessed_data['X_bandwidth']
y_traffic_test = preprocessed_data['y_traffic']
y_duration_test = preprocessed_data['y_duration']
y_bandwidth_test = preprocessed_data['y_bandwidth']

input_dim = preprocessed_data['input_dim']
n_classes_traffic = len(np.unique(y_traffic_test))
n_classes_duration = len(np.unique(y_duration_test))
n_classes_bandwidth = len(np.unique(y_bandwidth_test))

print(f"✓ Test data loaded:")
print(f"  Input dimension: {input_dim}")
print(f"  Traffic classes: {n_classes_traffic}")
print(f"  Duration classes: {n_classes_duration}")
print(f"  Bandwidth classes: {n_classes_bandwidth}")
print(f"  Test samples: {len(y_traffic_test)}")

# ============================================================================
# MULTI-TASK MODEL
# ============================================================================

class FedMTLModel(keras.Model):
    """Multi-Task Learning Model for Traffic, Duration, Bandwidth"""
    
    def __init__(self, in_dims, n_classes, dropout=0.1):
        super().__init__()
        self.in_dims = in_dims
        self.n_classes = n_classes
        self.dropout_rate = dropout
        
        # Shared layers
        self.shared1 = keras.layers.Dense(256, activation='relu', name='shared1')
        self.shared_dropout1 = keras.layers.Dropout(dropout)
        self.shared2 = keras.layers.Dense(128, activation='relu', name='shared2')
        self.shared_dropout2 = keras.layers.Dropout(dropout)
        
        # Task-specific heads
        self.traffic_head = keras.layers.Dense(n_classes['traffic'], activation='softmax', name='traffic_head')
        self.duration_head = keras.layers.Dense(n_classes['duration'], activation='softmax', name='duration_head')
        self.bandwidth_head = keras.layers.Dense(n_classes['bandwidth'], activation='softmax', name='bandwidth_head')
    
    def call(self, inputs, task='traffic', training=False):
        x = self.shared1(inputs)
        x = self.shared_dropout1(x, training=training)
        x = self.shared2(x)
        x = self.shared_dropout2(x, training=training)
        
        if task == 'traffic':
            return self.traffic_head(x)
        elif task == 'duration':
            return self.duration_head(x)
        elif task == 'bandwidth':
            return self.bandwidth_head(x)
        else:
            raise ValueError(f"Unknown task: {task}")
    
    def build_all(self, max_dim):
        """Build all layers with dummy input"""
        dummy = tf.zeros((1, max_dim))
        _ = self(dummy, task='traffic')
        _ = self(dummy, task='duration')
        _ = self(dummy, task='bandwidth')

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class DroneData:
    """Data for a single drone"""
    drone_id: int
    cluster_id: int
    ds: Dict[str, Tuple[np.ndarray, np.ndarray]]
    energy_residual: float = 0.85
    rssi_avg: float = 0.80

@dataclass
class UAVMetrics:
    """UAV context metrics for CH selection"""
    client_id: int
    cluster_id: int
    energy_residual: float
    rssi_avg: float
    num_examples: int
    param_change: float = 0.01

# ============================================================================
# DATA PARTITIONING
# ============================================================================

def create_per_cluster_test_data_equal(test_data, n_clusters=3):
    """Create per-cluster test data with EQUAL split"""
    cluster_test_data = {}
    
    for task in ['traffic', 'duration', 'bandwidth']:
        if task not in test_data:
            continue
        
        X_test, y_test = test_data[task]
        n_samples = len(X_test)
        samples_per_cluster = n_samples // n_clusters
        
        for cluster_id in range(n_clusters):
            if cluster_id not in cluster_test_data:
                cluster_test_data[cluster_id] = {}
            
            start_idx = cluster_id * samples_per_cluster
            end_idx = start_idx + samples_per_cluster if cluster_id < n_clusters - 1 else n_samples
            
            cluster_test_data[cluster_id][task] = (
                X_test[start_idx:end_idx],
                y_test[start_idx:end_idx]
            )
    
    return cluster_test_data

def create_per_cluster_test_data_dirichlet(test_data, n_clusters=3, alpha=0.4, seed=42):
    """Create per-cluster test data with DIRICHLET (non-IID) split"""
    np.random.seed(seed)
    cluster_test_data = {}
    
    for task in ['traffic', 'duration', 'bandwidth']:
        if task not in test_data:
            continue
        
        X_test, y_test = test_data[task]
        unique_labels = np.unique(y_test)
        cluster_indices = [[] for _ in range(n_clusters)]
        
        for label in unique_labels:
            label_indices = np.where(y_test == label)[0]
            n_label_samples = len(label_indices)
            
            proportions = np.random.dirichlet([alpha] * n_clusters)
            proportions = (proportions * n_label_samples).astype(int)
            proportions[-1] = n_label_samples - proportions[:-1].sum()
            
            start = 0
            for cluster_id in range(n_clusters):
                end = start + proportions[cluster_id]
                cluster_indices[cluster_id].extend(label_indices[start:end])
                start = end
        
        for cluster_id in range(n_clusters):
            if cluster_id not in cluster_test_data:
                cluster_test_data[cluster_id] = {}
            
            indices = cluster_indices[cluster_id]
            cluster_test_data[cluster_id][task] = (
                X_test[indices],
                y_test[indices]
            )
    
    return cluster_test_data

def build_drone_partitions(verbose=False):
    """Create drone data partitions (equal split for training)"""
    # Load training data from notebook or use preprocessed
    # For simulation, we'll use simplified partitioning
    drones = []
    
    for cluster_id in range(CFG.n_clusters):
        for drone_idx in range(CFG.n_drones_per_cluster):
            drone_id = cluster_id * CFG.n_drones_per_cluster + drone_idx
            
            # Create dummy training data (in practice, load from saved partitions)
            dummy_ds = {
                'traffic': (np.random.randn(50, input_dim), np.random.randint(0, n_classes_traffic, 50)),
                'duration': (np.random.randn(50, input_dim), np.random.randint(0, n_classes_duration, 50)),
                'bandwidth': (np.random.randn(50, input_dim), np.random.randint(0, n_classes_bandwidth, 50)),
            }
            
            drone = DroneData(
                drone_id=drone_id,
                cluster_id=cluster_id,
                ds=dummy_ds,
                energy_residual=np.random.uniform(0.7, 0.95),
                rssi_avg=np.random.uniform(0.75, 0.90)
            )
            drones.append(drone)
    
    if verbose:
        print(f"\n✓ Created {len(drones)} drone partitions:")
        for cid in range(CFG.n_clusters):
            cluster_drones = [d for d in drones if d.cluster_id == cid]
            print(f"  Cluster {cid}: {len(cluster_drones)} drones")
    
    return drones

# ============================================================================
# FLOWER CLIENT
# ============================================================================

class GymDroneClient(fl.client.NumPyClient):
    """Flower client for gym simulation drones"""
    
    def __init__(self, model, drone_data, cfg, cluster_id, drone_id):
        self.model = model
        self.drone_data = drone_data
        self.cfg = cfg
        self.cluster_id = cluster_id
        self.drone_id = drone_id
    
    def get_parameters(self, config):
        return self.model.get_weights()
    
    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        
        # Train on all tasks
        total_samples = 0
        task_losses = {}
        
        for task in ['traffic', 'duration', 'bandwidth']:
            X, y = self.drone_data[task]
            total_samples += len(X)
            
            # Compile with task-specific loss
            self.model.compile(
                optimizer=keras.optimizers.Adam(self.cfg.lr),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Train (NO DELAY)
            history = self.model.fit(
                X, y,
                epochs=self.cfg.local_epochs,
                batch_size=self.cfg.batch_size,
                verbose=0
            )
            task_losses[f'{task}_loss'] = float(history.history['loss'][-1])
        
        # Simulate UAV metrics
        energy_residual = self.drone_data.energy_residual if hasattr(self.drone_data, 'energy_residual') else np.random.uniform(0.7, 0.95)
        rssi_avg = self.drone_data.rssi_avg if hasattr(self.drone_data, 'rssi_avg') else np.random.uniform(0.75, 0.90)
        
        return self.model.get_weights(), total_samples, {
            'cluster_id': self.cluster_id,
            'client_id': self.drone_id,
            'energy_residual': energy_residual,
            'rssi_avg': rssi_avg,
            'num_examples': total_samples,
            **task_losses
        }
    
    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        
        # Evaluate on all tasks
        total_loss = 0.0
        total_acc = 0.0
        total_samples = 0
        
        for task in ['traffic', 'duration', 'bandwidth']:
            X, y = self.drone_data[task]
            total_samples += len(X)
            
            logits = self.model(X, task=task, training=False)
            preds = tf.argmax(logits, axis=1).numpy()
            acc = float(np.mean(preds == y))
            total_acc += acc * len(X)
        
        avg_acc = total_acc / total_samples if total_samples > 0 else 0.0
        
        return 0.0, total_samples, {'accuracy': avg_acc}

# ============================================================================
# INTEGRATED STRATEGY
# ============================================================================

class IntegratedGymStrategy(fl.server.strategy.FedAvg):
    """
    Integrated strategy for gym simulation with:
    - Hierarchical aggregation (CH0/CH2 → CH1 → broadcast)
    - Per-cluster testing (equal + Dirichlet)
    - CH compromise recovery
    - Model checkpointing every round
    """
    
    def __init__(
        self,
        test_data_equal=None,
        test_data_dirichlet=None,
        model_class=None,
        in_dims=None,
        n_classes=None,
        max_dim=None,
        compromise_round=None,
        compromised_cluster=0,
        global_aggregator_cluster=1,
        drone_list=None,
        detection_rounds=7,
        continuity_rounds=3,
        alpha_energy=0.6,
        beta_rssi=0.4,
        save_dir=None,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.test_data_equal = test_data_equal
        self.test_data_dirichlet = test_data_dirichlet
        self.model_class = model_class
        self.in_dims = in_dims
        self.n_classes = n_classes
        self.max_dim = max_dim
        self.save_dir = save_dir
        
        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
            print(f"💾 Model checkpoints will be saved to: {self.save_dir}")
        
        # CH compromise parameters
        self.compromise_round = compromise_round
        self.compromised_cluster = compromised_cluster
        self.global_aggregator_cluster = global_aggregator_cluster
        self.drone_list = drone_list or []
        self.detection_rounds = detection_rounds
        self.continuity_rounds = continuity_rounds
        self.alpha_energy = alpha_energy
        self.beta_rssi = beta_rssi
        
        # State
        self.ch_compromised = False
        self.compromise_detected_round = None
        self.recovery_phase = None
        self.cluster_uav_metrics = defaultdict(list)
        self.cluster_test_accuracies_by_round = []
        self.recovery_log = []
    
    def _get_participation_fraction(self, server_round, cluster_id):
        """Get participation fraction for gradual re-entry"""
        if not self.ch_compromised or cluster_id != self.compromised_cluster:
            return 1.0
        
        if self.recovery_phase == 'detection':
            return 0.0
        elif self.recovery_phase == 'continuity':
            rounds_since_detection = server_round - self.compromise_detected_round
            if rounds_since_detection <= self.detection_rounds + 1:
                return 0.3
            elif rounds_since_detection <= self.detection_rounds + 2:
                return 0.7
            else:
                return 1.0
        elif self.recovery_phase == 'complete':
            return 1.0
        
        return 1.0
    
    def aggregate_fit(self, server_round, results, failures):
        """Hierarchical aggregation + testing"""
        if len(results) == 0:
            return None, {}
        
        # Step 1: Check for CH compromise
        if (self.compromise_round is not None and 
            server_round == self.compromise_round and 
            not self.ch_compromised):
            
            self.ch_compromised = True
            self.compromise_detected_round = server_round
            self.recovery_phase = 'detection'
            print(f"\n🔴 CH COMPROMISE DETECTED: Round {server_round}, Cluster {self.compromised_cluster}")
        
        # Update recovery phase
        if self.ch_compromised:
            rounds_since = server_round - self.compromise_detected_round
            if rounds_since <= self.detection_rounds:
                self.recovery_phase = 'detection'
            elif rounds_since <= self.detection_rounds + self.continuity_rounds:
                self.recovery_phase = 'continuity'
            else:
                self.recovery_phase = 'complete'
        
        # Step 2: Extract results by cluster
        triples = []
        for client_proxy, fit_res in results:
            nds = fl.common.parameters_to_ndarrays(fit_res.parameters)
            weight = fit_res.num_examples
            cluster_id = int(fit_res.metrics.get('cluster_id', 0))
            triples.append((nds, weight, cluster_id, fit_res.metrics))
        
        # Step 3: Filter by participation
        participating_clusters = set()
        filtered_triples = []
        for nds, weight, cid, metrics in triples:
            frac = self._get_participation_fraction(server_round, cid)
            if np.random.random() < frac:
                participating_clusters.add(cid)
                filtered_triples.append((nds, weight, cid, metrics))
        
        # Step 4: Intra-cluster aggregation
        cluster_params = {}
        cluster_weights = {}
        
        for cid in participating_clusters:
            cluster_results = [(nds, w) for nds, w, c, _ in filtered_triples if c == cid]
            if cluster_results:
                aggregated = self._ndarrays_weighted_average(cluster_results)
                cluster_params[cid] = aggregated
                cluster_weights[cid] = sum(w for _, w in cluster_results)
        
        # Step 5: Global aggregation at CH1
        if self.global_aggregator_cluster in cluster_params:
            global_pairs = []
            if 0 in cluster_params and 0 != self.global_aggregator_cluster:
                global_pairs.append((cluster_params[0], cluster_weights[0]))
            if 2 in cluster_params and 2 != self.global_aggregator_cluster:
                global_pairs.append((cluster_params[2], cluster_weights[2]))
            global_pairs.append((cluster_params[self.global_aggregator_cluster], 
                                cluster_weights[self.global_aggregator_cluster]))
            
            averaged = self._ndarrays_weighted_average(global_pairs)
            aggregated_params = fl.common.ndarrays_to_parameters(averaged)
        else:
            # Fallback
            global_pairs = [(cluster_params[cid], cluster_weights[cid]) for cid in cluster_params.keys()]
            averaged = self._ndarrays_weighted_average(global_pairs)
            aggregated_params = fl.common.ndarrays_to_parameters(averaged)
        
        # Step 6: Integrated testing
        test_results = {'equal': {}, 'dirichlet': {}}
        
        if self.test_data_equal is not None and server_round > 0:
            for cid, params in cluster_params.items():
                if cid not in self.test_data_equal:
                    continue
                
                temp_model = self.model_class(self.in_dims, self.n_classes, dropout=0.1)
                temp_model.build_all(self.max_dim)
                temp_model.set_weights(params)
                task_metrics = {}
                
                for task in ['traffic', 'duration', 'bandwidth']:
                    if task not in self.test_data_equal[cid]:
                        continue
                    X_test, y_test = self.test_data_equal[cid][task]
                    X_test_tensor = tf.convert_to_tensor(X_test, dtype=tf.float32)
                    logits = temp_model(X_test_tensor, task=task, training=False)
                    preds = tf.argmax(logits, axis=1).numpy()
                    acc = float(np.mean(preds == y_test))
                    task_metrics[f'{task}_accuracy'] = acc
                
                test_results['equal'][int(cid)] = task_metrics
        
        if self.test_data_dirichlet is not None and server_round > 0:
            for cid, params in cluster_params.items():
                if cid not in self.test_data_dirichlet:
                    continue
                
                temp_model = self.model_class(self.in_dims, self.n_classes, dropout=0.1)
                temp_model.build_all(self.max_dim)
                temp_model.set_weights(params)
                task_metrics = {}
                
                for task in ['traffic', 'duration', 'bandwidth']:
                    if task not in self.test_data_dirichlet[cid]:
                        continue
                    X_test, y_test = self.test_data_dirichlet[cid][task]
                    X_test_tensor = tf.convert_to_tensor(X_test, dtype=tf.float32)
                    logits = temp_model(X_test_tensor, task=task, training=False)
                    preds = tf.argmax(logits, axis=1).numpy()
                    acc = float(np.mean(preds == y_test))
                    task_metrics[f'{task}_accuracy'] = acc
                
                test_results['dirichlet'][int(cid)] = task_metrics
        
        # Store results
        if test_results['equal'] or test_results['dirichlet']:
            self.cluster_test_accuracies_by_round.append({
                'round': int(server_round),
                'equal_split': test_results['equal'],
                'dirichlet_split': test_results['dirichlet'],
                'participating_clusters': list(participating_clusters),
                'recovery_phase': self.recovery_phase
            })
            
            # Print progress
            if 1 in test_results['equal']:
                metrics_equal = test_results['equal'][1]
                metrics_dirichlet = test_results.get('dirichlet', {}).get(1, {})
                phase_icon = {
                    None: '📍', 'detection': '🔧', 'continuity': '📊', 
                    'complete': '✅', 'normal': '✅'
                }
                print(f"{phase_icon.get(self.recovery_phase, '📍')} Round {server_round:3d}")
                print(f"   Equal     | Traffic: {metrics_equal.get('traffic_accuracy', 0):.4f} | "
                      f"Duration: {metrics_equal.get('duration_accuracy', 0):.4f} | "
                      f"Bandwidth: {metrics_equal.get('bandwidth_accuracy', 0):.4f}")
                if metrics_dirichlet:
                    print(f"   Dirichlet | Traffic: {metrics_dirichlet.get('traffic_accuracy', 0):.4f} | "
                          f"Duration: {metrics_dirichlet.get('duration_accuracy', 0):.4f} | "
                          f"Bandwidth: {metrics_dirichlet.get('bandwidth_accuracy', 0):.4f}")
        
        # Calculate aggregate metrics
        accs = [m.get('accuracy', 0) for _, _, _, m in triples if m]
        avg_acc = float(np.mean(accs)) if accs else 0.0
        
        # Save checkpoint
        if self.save_dir and server_round > 0:
            save_path = os.path.join(self.save_dir, f'model_round_{server_round}.pkl')
            checkpoint = {
                'round': server_round,
                'global_params': averaged,
                'cluster_params': cluster_params,
                'cluster_weights': cluster_weights,
                'participating_clusters': list(participating_clusters),
                'recovery_phase': self.recovery_phase,
                'ch_compromised': self.ch_compromised,
                'test_results': test_results,
                'avg_accuracy': avg_acc
            }
            with open(save_path, 'wb') as f:
                pickle.dump(checkpoint, f)
            
            if server_round % 10 == 0 or server_round <= 5:
                print(f"   💾 Checkpoint saved: {save_path}")
        
        return aggregated_params, {'accuracy': avg_acc}
    
    def _ndarrays_weighted_average(self, pairs):
        """Weighted average of numpy arrays"""
        if not pairs:
            return []
        
        total_weight = sum(w for _, w in pairs)
        weighted_params = [
            np.sum([p[i] * w for p, w in pairs], axis=0) / total_weight
            for i in range(len(pairs[0][0]))
        ]
        return weighted_params

# ============================================================================
# SIMULATION EXECUTION
# ============================================================================

def run_convergence_scenario():
    """Run 125-round convergence scenario"""
    print("\n" + "="*80)
    print("🚀 CONVERGENCE SCENARIO (125 rounds)")
    print("="*80)
    print(f"📍 CH0 compromise at round {CFG.convergence_compromise_round}")
    print("="*80)
    
    # Build drone partitions
    drones = build_drone_partitions(verbose=True)
    
    # Create test data
    test_data_global = {
        'traffic': (X_traffic_test, y_traffic_test),
        'duration': (X_duration_test, y_duration_test),
        'bandwidth': (X_bandwidth_test, y_bandwidth_test)
    }
    
    cluster_test_equal = create_per_cluster_test_data_equal(test_data_global, n_clusters=3)
    cluster_test_dirichlet = create_per_cluster_test_data_dirichlet(test_data_global, n_clusters=3, alpha=0.4)
    
    print(f"\n✓ Test data partitioned:")
    print(f"  Equal split: {len(cluster_test_equal)} clusters")
    print(f"  Dirichlet split: {len(cluster_test_dirichlet)} clusters")
    
    # Model dimensions
    max_dim = input_dim
    in_dims = {'traffic': max_dim, 'duration': max_dim, 'bandwidth': max_dim}
    n_classes = {'traffic': n_classes_traffic, 'duration': n_classes_duration, 'bandwidth': n_classes_bandwidth}
    
    # Client function
    def client_fn(context: Context) -> fl.client.Client:
        tf.random.set_seed(42)
        drone_idx = hash(context.node_id) % len(drones)
        drone = drones[drone_idx]
        
        model = FedMTLModel(in_dims, n_classes, dropout=CFG.dropout)
        model.build_all(max_dim)
        
        numpy_client = GymDroneClient(
            model=model,
            drone_data=drone.ds,
            cfg=CFG,
            cluster_id=drone.cluster_id,
            drone_id=drone.drone_id
        )
        
        return numpy_client.to_client()
    
    # Create global model
    global_model = FedMTLModel(in_dims, n_classes, dropout=CFG.dropout)
    global_model.build_all(max_dim)
    
    # Aggregation function
    def aggregate_metrics(metrics):
        aggregated = {}
        for num_examples, client_metrics in metrics:
            for metric_name, metric_value in client_metrics.items():
                if metric_name not in aggregated:
                    aggregated[metric_name] = []
                aggregated[metric_name].append(metric_value)
        for metric_name in aggregated:
            aggregated[metric_name] = np.mean(aggregated[metric_name])
        return aggregated
    
    # Create strategy
    strategy = IntegratedGymStrategy(
        test_data_equal=cluster_test_equal,
        test_data_dirichlet=cluster_test_dirichlet,
        model_class=FedMTLModel,
        in_dims=in_dims,
        n_classes=n_classes,
        max_dim=max_dim,
        compromise_round=CFG.convergence_compromise_round,
        compromised_cluster=0,
        global_aggregator_cluster=CFG.global_aggregator_cluster,
        drone_list=drones,
        detection_rounds=CFG.detection_rounds,
        continuity_rounds=CFG.continuity_rounds,
        alpha_energy=CFG.alpha_energy,
        beta_rssi=CFG.beta_rssi,
        save_dir=CFG.convergence_save_dir,
        fraction_fit=CFG.client_frac,
        fraction_evaluate=CFG.client_frac,
        min_fit_clients=10,
        min_available_clients=len(drones),
        min_evaluate_clients=10,
        initial_parameters=fl.common.ndarrays_to_parameters(global_model.get_weights()),
        fit_metrics_aggregation_fn=aggregate_metrics,
        evaluate_metrics_aggregation_fn=aggregate_metrics,
    )
    
    # Run simulation
    start_time = time.time()
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=len(drones),
        config=fl.server.ServerConfig(num_rounds=CFG.convergence_rounds),
        strategy=strategy,
    )
    elapsed = time.time() - start_time
    
    print(f"\n✅ Convergence scenario completed in {elapsed:.2f}s")
    print(f"   Results saved to: {CFG.convergence_save_dir}")
    
    return history, strategy

def run_transient_scenario():
    """Run 30-round transient scenario"""
    print("\n" + "="*80)
    print("🚀 TRANSIENT SCENARIO (30 rounds)")
    print("="*80)
    print(f"📍 CH0 compromise at round {CFG.transient_compromise_round}")
    print("="*80)
    
    # Build drone partitions
    drones = build_drone_partitions(verbose=True)
    
    # Create test data
    test_data_global = {
        'traffic': (X_traffic_test, y_traffic_test),
        'duration': (X_duration_test, y_duration_test),
        'bandwidth': (X_bandwidth_test, y_bandwidth_test)
    }
    
    cluster_test_equal = create_per_cluster_test_data_equal(test_data_global, n_clusters=3)
    cluster_test_dirichlet = create_per_cluster_test_data_dirichlet(test_data_global, n_clusters=3, alpha=0.4)
    
    # Model dimensions
    max_dim = input_dim
    in_dims = {'traffic': max_dim, 'duration': max_dim, 'bandwidth': max_dim}
    n_classes = {'traffic': n_classes_traffic, 'duration': n_classes_duration, 'bandwidth': n_classes_bandwidth}
    
    # Client function
    def client_fn(context: Context) -> fl.client.Client:
        tf.random.set_seed(42)
        drone_idx = hash(context.node_id) % len(drones)
        drone = drones[drone_idx]
        
        model = FedMTLModel(in_dims, n_classes, dropout=CFG.dropout)
        model.build_all(max_dim)
        
        numpy_client = GymDroneClient(
            model=model,
            drone_data=drone.ds,
            cfg=CFG,
            cluster_id=drone.cluster_id,
            drone_id=drone.drone_id
        )
        
        return numpy_client.to_client()
    
    # Create global model
    global_model = FedMTLModel(in_dims, n_classes, dropout=CFG.dropout)
    global_model.build_all(max_dim)
    
    # Aggregation function
    def aggregate_metrics(metrics):
        aggregated = {}
        for num_examples, client_metrics in metrics:
            for metric_name, metric_value in client_metrics.items():
                if metric_name not in aggregated:
                    aggregated[metric_name] = []
                aggregated[metric_name].append(metric_value)
        for metric_name in aggregated:
            aggregated[metric_name] = np.mean(aggregated[metric_name])
        return aggregated
    
    # Create strategy
    strategy = IntegratedGymStrategy(
        test_data_equal=cluster_test_equal,
        test_data_dirichlet=cluster_test_dirichlet,
        model_class=FedMTLModel,
        in_dims=in_dims,
        n_classes=n_classes,
        max_dim=max_dim,
        compromise_round=CFG.transient_compromise_round,
        compromised_cluster=0,
        global_aggregator_cluster=CFG.global_aggregator_cluster,
        drone_list=drones,
        detection_rounds=CFG.detection_rounds,
        continuity_rounds=CFG.continuity_rounds,
        alpha_energy=CFG.alpha_energy,
        beta_rssi=CFG.beta_rssi,
        save_dir=CFG.transient_save_dir,
        fraction_fit=CFG.client_frac,
        fraction_evaluate=CFG.client_frac,
        min_fit_clients=10,
        min_available_clients=len(drones),
        min_evaluate_clients=10,
        initial_parameters=fl.common.ndarrays_to_parameters(global_model.get_weights()),
        fit_metrics_aggregation_fn=aggregate_metrics,
        evaluate_metrics_aggregation_fn=aggregate_metrics,
    )
    
    # Run simulation
    start_time = time.time()
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=len(drones),
        config=fl.server.ServerConfig(num_rounds=CFG.transient_rounds),
        strategy=strategy,
    )
    elapsed = time.time() - start_time
    
    print(f"\n✅ Transient scenario completed in {elapsed:.2f}s")
    print(f"   Results saved to: {CFG.transient_save_dir}")
    
    return history, strategy

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Gym-PyBullet Integrated Simulation')
    parser.add_argument('--scenario', type=str, choices=['convergence', 'transient', 'both'], 
                        default='both', help='Scenario to run')
    args = parser.parse_args()
    
    if args.scenario in ['convergence', 'both']:
        history_conv, strategy_conv = run_convergence_scenario()
        print(f"\n✅ Convergence test accuracies saved: {len(strategy_conv.cluster_test_accuracies_by_round)} rounds")
    
    if args.scenario in ['transient', 'both']:
        history_trans, strategy_trans = run_transient_scenario()
        print(f"\n✅ Transient test accuracies saved: {len(strategy_trans.cluster_test_accuracies_by_round)} rounds")
    
    print("\n" + "="*80)
    print("🎉 GYM SIMULATION COMPLETE")
    print("="*80)
