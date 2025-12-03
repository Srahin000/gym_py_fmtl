"""
Complete Convergence Test with 4-Tier Communication Tracking and PyBullet Visualization
Simulates hierarchical federated learning with CUAV attack, D&R-E, and recovery phases

Each UAV performs REAL network classification using trained_models/hierarchical_equal/
"""

import numpy as np
import sys
import os
from typing import Dict, List, Tuple

# Add cesnet_zoo_clean to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class CommunicationTracker:
    """
    Tracks all 4 tiers of hierarchical communication with byte-level precision
    Tier 1: Members → CH (local aggregation)
    Tier 2: CH → CH* (global aggregation at CH1)
    Tier 3: CH* → CH (broadcast global model)
    Tier 4: CH → Members (distribute to cluster)
    """
    
    def __init__(self, model_size_kb: float = 246.42):
        self.model_size_kb = model_size_kb
        self.model_size_bytes = model_size_kb * 1024
        
        # Communication logs
        self.tier1_log = []  # Members → CH
        self.tier2_log = []  # CH → CH*
        self.tier3_log = []  # CH* → CH
        self.tier4_log = []  # CH → Members
        
        # Model version tracking
        self.model_versions = {}  # {entity_id: 'R<round>_<stage>'}
        
        # Statistics
        self.total_bytes = 0
        self.round_stats = []
        
        print(f"✓ Communication tracker initialized: Model size = {model_size_kb:.2f} KB")
    
    def log_member_to_ch(self, round_num: int, cluster_id: int, member_ids: List[int],
                         model_version: str):
        """
        Tier 1: Log members sending models to their cluster head
        
        Args:
            round_num: Current training round
            cluster_id: Cluster ID (0, 1, 2)
            member_ids: List of member UAV IDs sending models
            model_version: Model version being sent (e.g., 'R110_local_train')
        """
        num_members = len(member_ids)
        bytes_transferred = num_members * self.model_size_bytes
        
        self.tier1_log.append({
            'round': round_num,
            'cluster_id': cluster_id,
            'num_members': num_members,
            'bytes': bytes_transferred,
            'model_version': model_version,
        })
        
        # Update model versions for CH (receives aggregated)
        ch_id = f'CH{cluster_id}'
        self.model_versions[ch_id] = f'R{round_num}_cluster_agg'
        
        self.total_bytes += bytes_transferred
        
        return bytes_transferred
    
    def log_ch_to_global(self, round_num: int, ch_ids: List[int], is_compromised: Dict[int, bool]):
        """
        Tier 2: Log cluster heads sending to global aggregator (CH*)
        
        Args:
            round_num: Current training round
            ch_ids: List of CH cluster IDs participating (e.g., [0, 1, 2] or [1, 2] during D&R-E)
            is_compromised: Dict mapping cluster_id to whether it's compromised
        """
        num_chs = len(ch_ids)
        bytes_transferred = num_chs * self.model_size_bytes
        
        self.tier2_log.append({
            'round': round_num,
            'ch_ids': ch_ids,
            'num_chs': num_chs,
            'bytes': bytes_transferred,
            'compromised_status': {cid: is_compromised.get(cid, False) for cid in ch_ids},
        })
        
        # Update CH* model version
        self.model_versions['CH*'] = f'R{round_num}_global_agg'
        
        self.total_bytes += bytes_transferred
        
        return bytes_transferred
    
    def log_global_to_ch(self, round_num: int, ch_ids: List[int]):
        """
        Tier 3: Log global aggregator broadcasting to cluster heads
        
        Args:
            round_num: Current training round
            ch_ids: List of CH cluster IDs receiving the global model
        """
        num_chs = len(ch_ids)
        bytes_transferred = num_chs * self.model_size_bytes
        
        self.tier3_log.append({
            'round': round_num,
            'ch_ids': ch_ids,
            'num_chs': num_chs,
            'bytes': bytes_transferred,
        })
        
        # Update CH model versions
        for cid in ch_ids:
            self.model_versions[f'CH{cid}'] = f'R{round_num}_global_agg'
        
        self.total_bytes += bytes_transferred
        
        return bytes_transferred
    
    def log_ch_to_members(self, round_num: int, cluster_id: int, member_ids: List[int]):
        """
        Tier 4: Log cluster head distributing global model to members
        
        Args:
            round_num: Current training round
            cluster_id: Cluster ID
            member_ids: List of member UAV IDs receiving the model
        """
        num_members = len(member_ids)
        bytes_transferred = num_members * self.model_size_bytes
        
        self.tier4_log.append({
            'round': round_num,
            'cluster_id': cluster_id,
            'num_members': num_members,
            'bytes': bytes_transferred,
        })
        
        # Update member model versions
        for mid in member_ids:
            self.model_versions[f'C{cluster_id}_M{mid}'] = f'R{round_num}_global_agg'
        
        self.total_bytes += bytes_transferred
        
        return bytes_transferred
    
    def log_members_keep_old_model(self, round_num: int, cluster_id: int, old_model_version: str,
                                     reason: str = "D&R-E exclusion"):
        """
        Special log: Members in compromised cluster retain old model
        
        Args:
            round_num: Current training round
            cluster_id: Cluster ID
            old_model_version: Model version being retained (e.g., 'R110_global_agg')
            reason: Why they're keeping the old model
        """
        print(f"  📌 [Round {round_num}] Cluster {cluster_id} members RETAIN {old_model_version} ({reason})")
    
    def finalize_round(self, round_num: int, phase: str, cluster0_participating: bool):
        """
        Finalize communication statistics for a round
        
        Args:
            round_num: Round number
            phase: Current phase
            cluster0_participating: Whether Cluster 0 participated
        """
        # Calculate bytes for this round
        round_bytes = 0
        for log in [self.tier1_log, self.tier2_log, self.tier3_log, self.tier4_log]:
            for entry in log:
                if entry['round'] == round_num:
                    round_bytes += entry['bytes']
        
        self.round_stats.append({
            'round': round_num,
            'phase': phase,
            'bytes': round_bytes,
            'cluster0_participating': cluster0_participating,
        })
        
        return round_bytes
    
    def print_round_summary(self, round_num: int, detailed: bool = False):
        """Print communication summary for a specific round"""
        print(f"\n{'=' * 80}")
        print(f"ROUND {round_num} COMMUNICATION")
        print(f"{'=' * 80}")
        
        round_bytes = 0
        
        # Tier 1: Members → CH
        tier1_entries = [e for e in self.tier1_log if e['round'] == round_num]
        for entry in tier1_entries:
            mb = entry['bytes'] / (1024 * 1024)
            print(f"  Tier 1: {entry['num_members']} members → CH{entry['cluster_id']}: {mb:.2f} MB")
            round_bytes += entry['bytes']
        
        # Tier 2: CH → CH*
        tier2_entries = [e for e in self.tier2_log if e['round'] == round_num]
        for entry in tier2_entries:
            mb = entry['bytes'] / (1024 * 1024)
            compromised_info = ""
            if any(entry['compromised_status'].values()):
                compromised_chs = [cid for cid, comp in entry['compromised_status'].items() if comp]
                compromised_info = f" ⚠️ CH{compromised_chs[0]} COMPROMISED"
            print(f"  Tier 2: {entry['num_chs']} CHs → CH* (Global): {mb:.2f} MB{compromised_info}")
            round_bytes += entry['bytes']
        
        # Tier 3: CH* → CH
        tier3_entries = [e for e in self.tier3_log if e['round'] == round_num]
        for entry in tier3_entries:
            mb = entry['bytes'] / (1024 * 1024)
            print(f"  Tier 3: CH* → {entry['num_chs']} CHs: {mb:.2f} MB")
            round_bytes += entry['bytes']
        
        # Tier 4: CH → Members
        tier4_entries = [e for e in self.tier4_log if e['round'] == round_num]
        for entry in tier4_entries:
            mb = entry['bytes'] / (1024 * 1024)
            print(f"  Tier 4: CH{entry['cluster_id']} → {entry['num_members']} members: {mb:.2f} MB")
            round_bytes += entry['bytes']
        
        total_mb = round_bytes / (1024 * 1024)
        print(f"\n  Total this round: {total_mb:.2f} MB")
    
    def print_final_summary(self):
        """Print complete communication summary"""
        print(f"\n{'=' * 80}")
        print("FINAL COMMUNICATION SUMMARY")
        print(f"{'=' * 80}")
        
        total_gb = self.total_bytes / (1024 * 1024 * 1024)
        avg_per_round_mb = (self.total_bytes / len(self.round_stats)) / (1024 * 1024) if self.round_stats else 0
        
        print(f"Model size: {self.model_size_kb:.2f} KB")
        print(f"Total rounds: {len(self.round_stats)}")
        print(f"Total data transferred: {total_gb:.2f} GB")
        print(f"Average per round: {avg_per_round_mb:.2f} MB")
        
        # Break down by phase
        phase_stats = {}
        for stat in self.round_stats:
            phase = stat['phase']
            if phase not in phase_stats:
                phase_stats[phase] = {'rounds': 0, 'bytes': 0}
            phase_stats[phase]['rounds'] += 1
            phase_stats[phase]['bytes'] += stat['bytes']
        
        print(f"\nPhase breakdown:")
        for phase, stats in sorted(phase_stats.items()):
            gb = stats['bytes'] / (1024 * 1024 * 1024)
            print(f"  {phase:15s}: {stats['rounds']:3d} rounds, {gb:6.2f} GB")


def run_convergence_with_visualization():
    """
    Main function: Run convergence scenario with full communication tracking and visualization
    UAVs perform REAL network classification using trained FMTL models
    """
    print("=" * 80)
    print("CONVERGENCE SCENARIO WITH CUAV ATTACK")
    print("Hierarchical Federated Multi-Task Learning")
    print("=" * 80)
    
    # Initialize REAL model inference engine
    inference_engine = None
    try:
        from fmtl_visualization.inference import FMTLInferenceEngine, create_inference_engine
        
        # Get paths relative to this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(script_dir, 'trained_models', 'hierarchical_equal')
        data_path = os.path.join(script_dir, 'datasets', 'local_cache', 'dataset_12500_samples_65_features.csv')
        
        print(f"\n🔧 Initializing FMTL Inference Engine...")
        print(f"   Models: {models_dir}")
        print(f"   Data: {data_path}")
        
        if os.path.exists(models_dir):
            inference_engine = create_inference_engine(models_dir, data_path)
            print("✓ Real model inference enabled - UAVs will perform actual network classification!")
        else:
            print(f"⚠️ Models directory not found: {models_dir}")
            print("   Using simulated accuracy values")
    except ImportError as e:
        print(f"⚠️ Inference engine not available: {e}")
        print("   Using simulated accuracy values")
    
    # Try to import visualization (optional)
    viz = None
    try:
        from fmtl_visualization import FMTLVisualization, ScenarioConfig
        
        config = ScenarioConfig(scenario_type='convergence_compromise')
        viz = FMTLVisualization(config, use_gui=True, capture_frames=False)
        print("✓ PyBullet visualization enabled")
    except ImportError as e:
        print(f"⚠️ Visualization not available: {e}")
        print("  Running in communication-tracking-only mode")
    
    # Initialize communication tracker
    tracker = CommunicationTracker(model_size_kb=246.42)
    
    # Simulation parameters
    num_clusters = 3
    members_per_cluster = 200
    total_rounds = 125
    
    # Attack timeline (convergence scenario)
    compromise_round = 111
    detection_round = 112
    dre_duration = 7  # Rounds 112-118
    continuity_start = detection_round + dre_duration  # Round 119
    continuity_duration = 3  # Rounds 119-121
    
    # Track last known good model round for D&R-E
    last_good_model_round = compromise_round - 1  # Round 110
    
    # Run simulation
    for round_num in range(1, total_rounds + 1):
        # Determine phase
        if round_num < compromise_round:
            phase = 'NORMAL'
        elif round_num == compromise_round:
            phase = 'COMPROMISED'
        elif detection_round <= round_num < detection_round + dre_duration:
            phase = 'DRE'
        elif continuity_start <= round_num < continuity_start + continuity_duration:
            phase = 'CONTINUITY'
        else:
            phase = 'STABILIZATION' if round_num < continuity_start + continuity_duration + 4 else 'NORMAL'
        
        # Determine Cluster 0 participation
        if phase in ['COMPROMISED', 'DRE']:
            cluster0_participation = 0.0
            cluster0_offline = True
        elif phase == 'CONTINUITY':
            rounds_into_continuity = round_num - continuity_start
            if rounds_into_continuity == 0:
                cluster0_participation = 0.3
            elif rounds_into_continuity == 1:
                cluster0_participation = 0.7
            else:
                cluster0_participation = 1.0
            cluster0_offline = False
        else:
            cluster0_participation = 1.0
            cluster0_offline = False
        
        # TIER 1: Members → CH (for each cluster)
        for cluster_id in range(num_clusters):
            if cluster_id == 0 and cluster0_offline:
                # Cluster 0 offline during D&R-E
                continue
            
            if cluster_id == 0:
                num_participating = int(members_per_cluster * cluster0_participation)
            else:
                num_participating = members_per_cluster
            
            if num_participating > 0:
                member_ids = list(range(num_participating))
                tracker.log_member_to_ch(
                    round_num, cluster_id, member_ids,
                    model_version=f'R{round_num}_local_train'
                )
        
        # TIER 2: CH → CH* (Global Aggregator at CH1)
        if cluster0_offline:
            # Only CH1 and CH2 send to CH*
            participating_chs = [1, 2]
        else:
            # All CHs participate
            participating_chs = [0, 1, 2]
        
        is_compromised = {0: (round_num == compromise_round), 1: False, 2: False}
        tracker.log_ch_to_global(round_num, participating_chs, is_compromised)
        
        if round_num == compromise_round:
            print(f"\n{'=' * 80}")
            print(f"[Round {round_num}] Tier 2: CH → CH* (Global Aggregator)")
            print(f"  ⚠️ COMPROMISED: CH0 sending poisoned model!")
            print(f"{'=' * 80}")
        
        # TIER 3: CH* → CH (Broadcast global model)
        if cluster0_offline:
            # CH* excludes compromised CH0
            receiving_chs = [1, 2]
            print(f"\n[Round {round_num}] Tier 3: CH* → CH")
            print(f"  ⚠️ D&R-E Phase: Excluding compromised CH0")
        else:
            receiving_chs = [0, 1, 2]
        
        tracker.log_global_to_ch(round_num, receiving_chs)
        
        # TIER 4: CH → Members (Each CH distributes to its members)
        for cluster_id in range(num_clusters):
            if cluster_id == 0 and cluster0_offline:
                # Members retain old model
                tracker.log_members_keep_old_model(
                    round_num, cluster_id,
                    old_model_version=f'R{last_good_model_round}_global_agg',
                    reason="D&R-E phase - CH0 offline"
                )
                continue
            
            if cluster_id == 0:
                num_receiving = int(members_per_cluster * cluster0_participation)
            else:
                num_receiving = members_per_cluster
            
            if num_receiving > 0:
                member_ids = list(range(num_receiving))
                tracker.log_ch_to_members(round_num, cluster_id, member_ids)
        
        # Finalize round
        round_bytes = tracker.finalize_round(round_num, phase, not cluster0_offline)
        
        # ─────────────────────────────────────────────────────────────────────────
        # REAL MODEL INFERENCE: Get accuracies from trained models
        # ─────────────────────────────────────────────────────────────────────────
        if inference_engine is not None:
            # Update inference engine with D&R-E state
            accuracies = inference_engine.update_round(
                round_num,
                compromised_cluster=0 if phase in ['COMPROMISED', 'DRE'] else None,
                dre_phase=(phase == 'DRE'),
                old_model_round=last_good_model_round
            )
            
            # Print real accuracies
            if round_num % 10 == 0 or round_num in [111, 112, 119]:
                print(f"\n📊 [Round {round_num}] Real Model Accuracies (from trained_models):")
                print(f"   Traffic:   {accuracies['traffic']:.4f}")
                print(f"   Duration:  {accuracies['duration']:.4f}")
                print(f"   Bandwidth: {accuracies['bandwidth']:.4f}")
                
                # Show model version for sample UAVs
                print(f"   Model versions:")
                print(f"     UAV 0 (C0): {inference_engine.get_model_version(0)}")
                print(f"     UAV 200 (C1): {inference_engine.get_model_version(200)}")
                print(f"     UAV 400 (C2): {inference_engine.get_model_version(400)}")
        else:
            # Fallback to simulated accuracies (dummy values)
            base_acc = 0.70 + min(round_num, 90) * 0.002
            accuracies = {
                'traffic': base_acc + np.random.uniform(-0.02, 0.02),
                'duration': base_acc - 0.03 + np.random.uniform(-0.02, 0.02),
                'bandwidth': base_acc + 0.03 + np.random.uniform(-0.02, 0.02),
            }
        
        comm_stats = {
            'round_data': round_bytes / (1024 * 1024),  # MB
            'total_data': tracker.total_bytes / (1024 * 1024 * 1024),  # GB
        }
        
        # Update visualization if available
        if viz:
            viz.update_round(round_num, accuracies, comm_stats)
        
        # Print detailed logs for key rounds
        if round_num in [111, 112, 119, 122] or round_num % 25 == 0:
            tracker.print_round_summary(round_num)
    
    # Final summary
    print(f"\n{'=' * 80}")
    print("SIMULATION COMPLETE")
    print(f"{'=' * 80}")
    tracker.print_final_summary()
    
    if viz:
        viz.close()



if __name__ == '__main__':
    try:
        run_convergence_with_visualization()
    except KeyboardInterrupt:
        print("\n⚠️ Simulation interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
