"""
COMPLETE UAV SWARM FEDERATED LEARNING SIMULATION WITH PYBULLET + REAL-TIME KPIs
================================================================================

Full PyBullet GUI simulation with comprehensive terminal KPI display:
- 45 UAVs (15 per cluster) performing federated learning
- CH0 compromise at round 25 (out of 50 total)
- Real-time terminal display of all KPIs
- Visual GUI showing the entire process
"""

import pybullet as p
import pybullet_data
import numpy as np
import time
import sys
import os
from datetime import datetime
from collections import defaultdict

# Get paths to drone models
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'gym_pybullet_drones', 'assets')
DRONE_URDF = os.path.join(ASSETS_DIR, 'cf2x.urdf')

# Configuration
NUM_CLUSTERS = 3
UAVS_PER_CLUSTER = 15  # 15 UAVs per cluster (45 total)
TOTAL_UAVS = NUM_CLUSTERS * UAVS_PER_CLUSTER
TOTAL_ROUNDS = 50
COMPROMISE_ROUND = 25
DETECTION_ROUNDS = 7
CONTINUITY_ROUNDS = 3

# Cluster positions (closer together)
CLUSTER_POSITIONS = [
    np.array([-20.0, 0.0, 30.0]),  # Cluster 0 (left)
    np.array([0.0, 0.0, 30.0]),     # Cluster 1 (center)
    np.array([20.0, 0.0, 30.0])     # Cluster 2 (right)
]

# Colors (RGBA)
CLUSTER_COLORS = {
    0: [0.2, 1.0, 0.2, 1.0],   # Green
    1: [0.2, 0.5, 1.0, 1.0],   # Blue
    2: [0.8, 0.2, 1.0, 1.0]    # Purple
}
COLOR_CH = [1.0, 0.84, 0.0, 1.0]               # Gold
COLOR_CUAV = [1.0, 0.0, 0.0, 1.0]              # Red
COLOR_COMPROMISED = [0.3, 0.3, 0.3, 0.7]       # Dark gray

CLUSTER_COLORS_DEGRADED = {
    0: [0.1, 0.5, 0.1, 1.0],
    1: [0.1, 0.3, 0.6, 1.0],
    2: [0.4, 0.1, 0.5, 1.0]
}
CLUSTER_COLORS_RECOVERING = {
    0: [0.15, 0.75, 0.15, 1.0],
    1: [0.15, 0.4, 0.8, 1.0],
    2: [0.6, 0.15, 0.75, 1.0]
}

# ============================================================================
# KPI TRACKING CLASS
# ============================================================================

class RealTimeKPITracker:
    """Track and display all KPIs in terminal during simulation"""
    
    def __init__(self):
        self.round_data = []
        self.start_time = None
        self.round_start_time = None
        
        # Metrics storage
        self.accuracy_history = {
            'equal': {0: [], 1: [], 2: []},
            'dirichlet': {0: [], 1: [], 2: []}
        }
        self.participation_history = {0: [], 1: [], 2: []}
        self.communication_cost = []
        self.model_divergence = []
        self.recovery_events = []
        
        # Model parameters
        self.model_size_kb = 246.42
        self.bandwidth_mbps = 100
        
    def start_experiment(self):
        """Start experiment timing"""
        self.start_time = time.time()
        print("\n" + "="*100)
        print("🚁 UAV SWARM FEDERATED LEARNING - REAL-TIME KPI MONITORING")
        print("="*100)
        print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Configuration: {TOTAL_UAVS} UAVs | {NUM_CLUSTERS} Clusters | {TOTAL_ROUNDS} Rounds")
        print(f"CH Compromise: Round {COMPROMISE_ROUND}")
        print("="*100 + "\n")
        
    def start_round(self, round_num):
        """Start tracking a new round"""
        self.round_start_time = time.time()
        self.current_round = round_num
        
    def end_round(self, round_num, accuracies, participation, phase):
        """End round and display comprehensive KPIs"""
        round_duration = time.time() - self.round_start_time
        cumulative_time = time.time() - self.start_time
        
        # Store data
        self.accuracy_history['equal'][0].append(accuracies['equal'][0]['traffic'])
        self.accuracy_history['equal'][1].append(accuracies['equal'][1]['traffic'])
        self.accuracy_history['equal'][2].append(accuracies['equal'][2]['traffic'])
        
        self.accuracy_history['dirichlet'][0].append(accuracies['dirichlet'][0]['traffic'])
        self.accuracy_history['dirichlet'][1].append(accuracies['dirichlet'][1]['traffic'])
        self.accuracy_history['dirichlet'][2].append(accuracies['dirichlet'][2]['traffic'])
        
        self.participation_history[0].append(participation[0])
        self.participation_history[1].append(participation[1])
        self.participation_history[2].append(participation[2])
        
        # Calculate communication cost
        active_clusters = sum(1 for p in participation.values() if p > 0)
        comm_cost = self._calculate_comm_cost(active_clusters)
        self.communication_cost.append(comm_cost)
        
        # Calculate model divergence (simplified)
        if round_num > 1:
            divergence = self._calculate_divergence(accuracies)
            self.model_divergence.append(divergence)
        
        # Display comprehensive KPIs
        self._display_round_kpis(round_num, round_duration, cumulative_time, 
                                 accuracies, participation, phase, comm_cost)
        
    def _calculate_comm_cost(self, active_clusters):
        """Calculate communication cost in MB"""
        # Members → CH
        members_to_ch = active_clusters * UAVS_PER_CLUSTER * self.model_size_kb / 1024
        # CH → Global
        ch_to_global = active_clusters * self.model_size_kb / 1024
        # Global → CH
        global_to_ch = active_clusters * self.model_size_kb / 1024
        # CH → Members
        ch_to_members = active_clusters * UAVS_PER_CLUSTER * self.model_size_kb / 1024
        
        return members_to_ch + ch_to_global + global_to_ch + ch_to_members
    
    def _calculate_divergence(self, accuracies):
        """Calculate model divergence across clusters"""
        traffic_accs = [accuracies['equal'][i]['traffic'] for i in range(3)]
        return np.std(traffic_accs)
    
    def _display_round_kpis(self, round_num, duration, cumulative, 
                            accuracies, participation, phase, comm_cost):
        """Display comprehensive KPIs in terminal"""
        
        # Determine phase icon
        phase_icons = {
            'normal': '✅',
            'detection': '🔧',
            'continuity': '📊',
            'stabilization': '🔄'
        }
        icon = phase_icons.get(phase, '📍')
        
        # Clear and print header
        print("\n" + "="*100)
        print(f"{icon} ROUND {round_num}/{TOTAL_ROUNDS} - Phase: {phase.upper()}")
        print("="*100)
        
        # Timing metrics
        print(f"\n⏱️  TIMING METRICS")
        print(f"   Round Duration: {duration:.2f}s")
        print(f"   Cumulative Time: {cumulative:.1f}s ({cumulative/60:.1f} min)")
        print(f"   Avg Round Time: {cumulative/round_num:.2f}s")
        
        # Accuracy metrics (Equal Split)
        print(f"\n📊 ACCURACY METRICS - EQUAL SPLIT")
        print(f"   {'Cluster':<10} {'Traffic':<12} {'Duration':<12} {'Bandwidth':<12} {'Trend':<10}")
        print(f"   {'-'*60}")
        for cid in range(3):
            acc = accuracies['equal'][cid]
            trend = self._get_trend(self.accuracy_history['equal'][cid])
            frozen_marker = " [FROZEN]" if phase == 'detection' and cid == 0 else ""
            print(f"   Cluster {cid}   {acc['traffic']:.4f}       "
                  f"{acc['duration']:.4f}       {acc['bandwidth']:.4f}       "
                  f"{trend}{frozen_marker}")
        
        # Accuracy metrics (Dirichlet Split)
        print(f"\n📊 ACCURACY METRICS - DIRICHLET SPLIT")
        print(f"   {'Cluster':<10} {'Traffic':<12} {'Duration':<12} {'Bandwidth':<12} {'Trend':<10}")
        print(f"   {'-'*60}")
        for cid in range(3):
            acc = accuracies['dirichlet'][cid]
            trend = self._get_trend(self.accuracy_history['dirichlet'][cid])
            frozen_marker = " [FROZEN]" if phase == 'detection' and cid == 0 else ""
            print(f"   Cluster {cid}   {acc['traffic']:.4f}       "
                  f"{acc['duration']:.4f}       {acc['bandwidth']:.4f}       "
                  f"{trend}{frozen_marker}")
        
        # Participation rates
        print(f"\n👥 CLUSTER PARTICIPATION")
        print(f"   {'Cluster':<10} {'Rate':<10} {'UAVs Active':<15} {'Status':<20}")
        print(f"   {'-'*60}")
        for cid in range(3):
            rate = participation[cid]
            uavs_active = int(rate * UAVS_PER_CLUSTER)
            status = self._get_participation_status(rate, cid, phase)
            print(f"   Cluster {cid}   {rate:.1%}      {uavs_active}/{UAVS_PER_CLUSTER}          {status}")
        
        # Communication metrics
        print(f"\n📡 COMMUNICATION METRICS")
        total_comm = sum(self.communication_cost)
        print(f"   Round Comm Cost: {comm_cost:.2f} MB")
        print(f"   Cumulative Comm: {total_comm:.2f} MB")
        print(f"   Model Size: {self.model_size_kb:.2f} KB")
        print(f"   Bandwidth: {self.bandwidth_mbps} Mbps")
        
        # Model convergence metrics
        if len(self.model_divergence) > 0:
            current_divergence = self.model_divergence[-1]
            avg_divergence = np.mean(self.model_divergence)
            print(f"\n🎯 MODEL CONVERGENCE")
            print(f"   Current Divergence (std): {current_divergence:.6f}")
            print(f"   Average Divergence: {avg_divergence:.6f}")
            print(f"   Convergence Quality: {self._get_convergence_quality(current_divergence)}")
        
        # Recovery status (if in recovery phase)
        if phase != 'normal':
            self._display_recovery_status(round_num, phase)
        
        print("="*100)
        
    def _get_trend(self, history):
        """Get accuracy trend arrow"""
        if len(history) < 2:
            return "→"
        diff = history[-1] - history[-2]
        if abs(diff) < 0.001:
            return "→"  # Flat
        elif diff > 0:
            return "↑"  # Improving
        else:
            return "↓"  # Degrading
    
    def _get_participation_status(self, rate, cluster_id, phase):
        """Get participation status string"""
        if phase == 'detection' and cluster_id == 0:
            return "🔴 ISOLATED (D&R-E)"
        elif phase == 'continuity' and cluster_id == 0:
            if rate < 0.5:
                return "🟡 GRADUAL RE-ENTRY (30%)"
            elif rate < 0.9:
                return "🟡 GRADUAL RE-ENTRY (70%)"
            else:
                return "🟢 FULL RESTORATION"
        elif rate == 1.0:
            return "🟢 ACTIVE"
        else:
            return f"🟡 PARTIAL ({rate:.0%})"
    
    def _get_convergence_quality(self, divergence):
        """Assess convergence quality"""
        if divergence < 0.01:
            return "🟢 EXCELLENT"
        elif divergence < 0.05:
            return "🟡 GOOD"
        elif divergence < 0.1:
            return "🟠 MODERATE"
        else:
            return "🔴 POOR"
    
    def _display_recovery_status(self, round_num, phase):
        """Display recovery phase information"""
        print(f"\n🔧 RECOVERY STATUS")
        
        if phase == 'detection':
            rounds_in_phase = (round_num - COMPROMISE_ROUND)
            rounds_remaining = DETECTION_ROUNDS - rounds_in_phase
            print(f"   Phase: Detection & Re-Election")
            print(f"   Progress: {rounds_in_phase}/{DETECTION_ROUNDS} rounds")
            print(f"   Remaining: {rounds_remaining} rounds")
            print(f"   CH0 Status: 🔴 OFFLINE (0% participation)")
            
        elif phase == 'continuity':
            rounds_in_phase = (round_num - COMPROMISE_ROUND - DETECTION_ROUNDS)
            print(f"   Phase: Continuity & Gradual Re-entry")
            print(f"   Progress: {rounds_in_phase}/{CONTINUITY_ROUNDS} rounds")
            print(f"   CH0 Status: 🟡 GRADUAL RECOVERY")
            
        elif phase == 'stabilization':
            rounds_since_recovery = (round_num - COMPROMISE_ROUND - DETECTION_ROUNDS - CONTINUITY_ROUNDS)
            print(f"   Phase: Stabilization")
            print(f"   Rounds Since Recovery: {rounds_since_recovery}")
            print(f"   CH0 Status: 🟢 FULLY RESTORED")
    
    def print_final_summary(self):
        """Print final experiment summary"""
        total_time = time.time() - self.start_time
        total_comm = sum(self.communication_cost)
        
        print("\n\n" + "="*100)
        print("📊 FINAL EXPERIMENT SUMMARY")
        print("="*100)
        
        print(f"\n⏱️  TIMING")
        print(f"   Total Duration: {total_time:.1f}s ({total_time/60:.1f} min)")
        print(f"   Total Rounds: {TOTAL_ROUNDS}")
        print(f"   Avg Round Time: {total_time/TOTAL_ROUNDS:.2f}s")
        
        print(f"\n📡 COMMUNICATION")
        print(f"   Total Data Transferred: {total_comm:.2f} MB")
        print(f"   Avg Per Round: {total_comm/TOTAL_ROUNDS:.2f} MB")
        
        print(f"\n📊 FINAL ACCURACIES (EQUAL SPLIT)")
        for cid in range(3):
            final_acc = self.accuracy_history['equal'][cid][-1]
            improvement = final_acc - self.accuracy_history['equal'][cid][0]
            print(f"   Cluster {cid}: {final_acc:.4f} (Δ {improvement:+.4f})")
        
        print(f"\n📊 FINAL ACCURACIES (DIRICHLET SPLIT)")
        for cid in range(3):
            final_acc = self.accuracy_history['dirichlet'][cid][-1]
            improvement = final_acc - self.accuracy_history['dirichlet'][cid][0]
            print(f"   Cluster {cid}: {final_acc:.4f} (Δ {improvement:+.4f})")
        
        print(f"\n🎯 CONVERGENCE")
        if len(self.model_divergence) > 0:
            final_divergence = self.model_divergence[-1]
            avg_divergence = np.mean(self.model_divergence)
            print(f"   Final Divergence: {final_divergence:.6f}")
            print(f"   Average Divergence: {avg_divergence:.6f}")
        
        print(f"\n🔧 RECOVERY PERFORMANCE")
        print(f"   Compromise Round: {COMPROMISE_ROUND}")
        print(f"   Detection Duration: {DETECTION_ROUNDS} rounds")
        print(f"   Continuity Duration: {CONTINUITY_ROUNDS} rounds")
        print(f"   Total Recovery Time: {DETECTION_ROUNDS + CONTINUITY_ROUNDS} rounds")
        
        # CH0 recovery analysis
        if len(self.accuracy_history['equal'][0]) >= COMPROMISE_ROUND:
            acc_before = self.accuracy_history['equal'][0][COMPROMISE_ROUND-1]
            acc_after = self.accuracy_history['equal'][0][-1]
            recovery_rate = (acc_after - acc_before) / acc_before * 100
            print(f"   CH0 Recovery: {acc_before:.4f} → {acc_after:.4f} ({recovery_rate:+.1f}%)")
        
        print("="*100)
        print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*100 + "\n")

# ============================================================================
# UAV CLASS
# ============================================================================

class UAV:
    def __init__(self, uav_id, cluster_id, position, is_ch=False):
        self.id = uav_id
        self.cluster_id = cluster_id
        self.position = np.array(position, dtype=float)
        self.is_ch = is_ch
        self.is_alive = True
        self.is_compromised = False
        self.body_id = None
        self.participation_level = 1.0
        
        # Orbital flight parameters (smaller orbits for closer spacing)
        self.orbit_radius = np.random.uniform(3, 6)  # Closer to CH
        self.orbit_speed = np.random.uniform(0.15, 0.35)
        self.orbit_angle = np.random.uniform(0, 2 * np.pi)
        self.orbit_height_offset = np.random.uniform(-2, 2)  # Less vertical variation
        
    def set_color(self, color):
        """Update UAV visual color"""
        if self.body_id is not None:
            p.changeVisualShape(self.body_id, -1, rgbaColor=color)
            try:
                num_joints = p.getNumJoints(self.body_id)
                for joint_idx in range(num_joints):
                    p.changeVisualShape(self.body_id, joint_idx, rgbaColor=color)
            except:
                pass

class CUAVAttacker:
    def __init__(self, spawn_position, target_position):
        self.position = np.array(spawn_position, dtype=float)
        self.target = np.array(target_position, dtype=float)
        self.speed = 2.0
        self.body_id = None
        self.active = True
        
    def move_toward_target(self, dt=1.0):
        """Move toward target CH"""
        direction = self.target - self.position
        distance = np.linalg.norm(direction)
        
        if distance < 0.5:
            return True  # Reached target
        
        direction = direction / distance
        self.position += direction * self.speed * dt
        
        if self.body_id is not None:
            p.resetBasePositionAndOrientation(
                self.body_id,
                self.position,
                p.getQuaternionFromEuler([0, 0, 0])
            )
        
        return False

# ============================================================================
# SIMULATION FUNCTIONS
# ============================================================================

def simulate_federated_learning_round(round_num, uavs, phase, kpi_tracker):
    """
    Simulate one round of federated learning with KPI tracking
    
    Returns accuracies and participation data
    """
    # Determine participation based on phase
    participation = {0: 1.0, 1: 1.0, 2: 1.0}
    
    if phase == 'detection':
        participation[0] = 0.0  # CH0 offline
    elif phase == 'continuity':
        rounds_since_detection = round_num - COMPROMISE_ROUND - DETECTION_ROUNDS
        if rounds_since_detection == 0:
            participation[0] = 0.3
        elif rounds_since_detection == 1:
            participation[0] = 0.7
        else:
            participation[0] = 1.0
    
    # Simulate accuracies (simplified - use your trained model results here)
    # For demonstration, using synthetic data
    base_acc = 0.4 + (round_num / TOTAL_ROUNDS) * 0.4  # Linear improvement 0.4 → 0.8
    
    accuracies = {
        'equal': {},
        'dirichlet': {}
    }
    
    for cid in range(3):
        # Add cluster-specific variation
        cluster_variation = np.random.uniform(-0.02, 0.02)
        
        # Simulate frozen accuracy for CH0 during detection
        if cid == 0 and phase == 'detection':
            # Use accuracy from compromise round (frozen)
            if round_num == COMPROMISE_ROUND + 1:
                frozen_acc = base_acc
            else:
                frozen_acc = accuracies['equal'].get(0, {}).get('traffic', base_acc)
            
            accuracies['equal'][cid] = {
                'traffic': frozen_acc,
                'duration': frozen_acc * 0.95,
                'bandwidth': frozen_acc * 0.90
            }
            accuracies['dirichlet'][cid] = {
                'traffic': frozen_acc * 0.98,
                'duration': frozen_acc * 0.93,
                'bandwidth': frozen_acc * 0.88
            }
        else:
            # Normal accuracy progression
            acc = base_acc + cluster_variation
            accuracies['equal'][cid] = {
                'traffic': min(0.95, acc),
                'duration': min(0.93, acc * 0.95),
                'bandwidth': min(0.90, acc * 0.90)
            }
            accuracies['dirichlet'][cid] = {
                'traffic': min(0.93, acc * 0.98),
                'duration': min(0.91, acc * 0.93),
                'bandwidth': min(0.88, acc * 0.88)
            }
    
    return accuracies, participation

def determine_phase(round_num):
    """Determine current recovery phase"""
    if round_num < COMPROMISE_ROUND:
        return 'normal'
    elif round_num < COMPROMISE_ROUND + DETECTION_ROUNDS:
        return 'detection'
    elif round_num < COMPROMISE_ROUND + DETECTION_ROUNDS + CONTINUITY_ROUNDS:
        return 'continuity'
    else:
        return 'stabilization'

# ============================================================================
# MAIN SIMULATION
# ============================================================================

def main():
    """Main simulation with KPI tracking"""
    
    # Initialize KPI tracker
    kpi_tracker = RealTimeKPITracker()
    kpi_tracker.start_experiment()
    
    # Initialize PyBullet
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    
    # Load environment
    p.loadURDF("plane.urdf", [0, 0, 0])
    
    # Create UAVs
    uavs = []
    for cluster_id in range(NUM_CLUSTERS):
        cluster_center = CLUSTER_POSITIONS[cluster_id]
        
        for local_id in range(UAVS_PER_CLUSTER):
            uav_id = cluster_id * UAVS_PER_CLUSTER + local_id
            is_ch = (local_id == 0)  # First UAV in each cluster is CH
            
            # Position: CH at center, members orbit around (closer)
            if is_ch:
                position = cluster_center
            else:
                angle = (local_id / UAVS_PER_CLUSTER) * 2 * np.pi
                radius = 5  # Closer orbit radius
                offset = np.array([
                    radius * np.cos(angle),
                    radius * np.sin(angle),
                    np.random.uniform(-2, 2)  # Less vertical spread
                ])
                position = cluster_center + offset
            
            uav = UAV(uav_id, cluster_id, position, is_ch)
            
            # Load drone model (3x larger)
            if os.path.exists(DRONE_URDF):
                uav.body_id = p.loadURDF(DRONE_URDF, position, 
                                         p.getQuaternionFromEuler([0, 0, 0]),
                                         useFixedBase=True,
                                         globalScaling=3.0)  # 3x size
            else:
                # Fallback to sphere (3x larger)
                collision_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=0.9)  # 3x radius
                visual_shape = p.createVisualShape(p.GEOM_SPHERE, radius=0.9)
                uav.body_id = p.createMultiBody(0.1, collision_shape, visual_shape, position)
            
            # Set initial color
            color = COLOR_CH if is_ch else CLUSTER_COLORS[cluster_id]
            uav.set_color(color)
            
            uavs.append(uav)
    
    print(f"✓ Created {len(uavs)} UAVs in {NUM_CLUSTERS} clusters")
    
    # Camera setup (closer view)
    p.resetDebugVisualizerCamera(
        cameraDistance=40,  # Closer camera
        cameraYaw=0,
        cameraPitch=-30,
        cameraTargetPosition=[0, 0, 30]
    )
    
    # Create CUAV (will spawn at compromise round)
    cuav = None
    
    # Main simulation loop
    for round_num in range(1, TOTAL_ROUNDS + 1):
        kpi_tracker.start_round(round_num)
        
        # Determine phase
        phase = determine_phase(round_num)
        
        # Spawn CUAV at compromise round (larger size)
        if round_num == COMPROMISE_ROUND and cuav is None:
            spawn_pos = CLUSTER_POSITIONS[0] + np.array([-15, 0, 10])  # Closer spawn
            target_pos = CLUSTER_POSITIONS[0]
            cuav = CUAVAttacker(spawn_pos, target_pos)
            
            collision_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=1.5)  # 3x larger
            visual_shape = p.createVisualShape(p.GEOM_SPHERE, radius=1.5, rgbaColor=COLOR_CUAV)
            cuav.body_id = p.createMultiBody(0.1, collision_shape, visual_shape, cuav.position)
            
            print(f"\n🔴 CUAV SPAWNED AT ROUND {round_num}!")
        
        # Move CUAV toward CH0
        if cuav and cuav.active:
            reached = cuav.move_toward_target()
            if reached:
                # Compromise CH0
                ch0 = uavs[0]  # CH0 is first UAV in cluster 0
                ch0.is_compromised = True
                ch0.set_color(COLOR_COMPROMISED)
                cuav.active = False
                print(f"🔴 CH0 COMPROMISED AT ROUND {round_num}!")
        
        # Update UAV colors based on phase
        for uav in uavs:
            if uav.is_compromised:
                continue  # Keep compromised color
            
            if uav.cluster_id == 0 and phase == 'detection':
                # Cluster 0 degraded during detection
                color = CLUSTER_COLORS_DEGRADED[0] if not uav.is_ch else COLOR_CH
                uav.set_color(color)
            elif uav.cluster_id == 0 and phase == 'continuity':
                # Cluster 0 recovering
                color = CLUSTER_COLORS_RECOVERING[0] if not uav.is_ch else COLOR_CH
                uav.set_color(color)
            else:
                # Normal colors
                color = COLOR_CH if uav.is_ch else CLUSTER_COLORS[uav.cluster_id]
                uav.set_color(color)
        
        # Simulate federated learning round
        accuracies, participation = simulate_federated_learning_round(
            round_num, uavs, phase, kpi_tracker
        )
        
        # Update KPI tracker
        kpi_tracker.end_round(round_num, accuracies, participation, phase)
        
        # Visualization update (step physics)
        p.stepSimulation()
        time.sleep(0.1)  # Slow down visualization
    
    # Print final summary
    kpi_tracker.print_final_summary()
    
    # Keep window open
    print("\n✅ Simulation complete! Press Ctrl+C to exit...")
    try:
        while True:
            p.stepSimulation()
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n👋 Exiting...")
    
    p.disconnect()

if __name__ == "__main__":
    main()
