"""
KPI Tracker for PyBullet FMTL Simulation

Mirrors the notebook's ComprehensiveKPITracker for real-time metric collection
during physical UAV simulation.
"""

import time
import psutil
import json
import os
import numpy as np
from dataclasses import dataclass, asdict
from typing import Optional, Dict, List


@dataclass
class RoundKPI:
    """KPI snapshot for a single round"""
    round: int
    phase: str  # 'normal', 'attack', 'dre', 'continuity', 'stabilization'
    duration_sec: float
    cumulative_sec: float
    cpu_percent: float
    memory_mb: float
    accuracies: Dict[str, float]  # {'traffic', 'duration', 'bandwidth', 'global'}
    participation: Dict[str, float]  # {'C0', 'C1', 'C2'}
    communication_bytes: Dict[str, int]  # {'tier1', 'tier2', 'tier3', 'tier4', 'total'}
    divergence: Optional[float] = None
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return asdict(self)


class ComprehensiveKPITracker:
    """
    Comprehensive KPI Tracker for PyBullet FMTL Simulation
    
    Tracks all metrics from TIER 1 and TIER 2 categories:
    - Learning Performance (round duration, accuracies, convergence)
    - Model Architecture & Resources (CPU, memory, inference latency)
    - Communication Efficiency (bytes per tier, per phase)
    - Attack Impact & Recovery (attack timing, degradation, restoration)
    - Cluster Health & Participation (per-cluster participation rates)
    - CH Selection & Load (re-election timing, CH characteristics)
    """
    
    def __init__(self, save_dir: str, model_size_bytes: int = 278100):
        """
        Initialize KPI tracker
        
        Args:
            save_dir: Directory to save KPI snapshots
            model_size_bytes: Size of model in bytes (default ~278 KB from study)
        """
        self.save_dir = save_dir
        self.model_size_bytes = model_size_bytes
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Timing state
        self._exp_start = None
        self._round_start = None
        self.cumulative_sec = 0.0
        
        # Round history
        self.round_history: List[RoundKPI] = []
        
        # Attack tracking
        self.attack_info = {
            'start_round': None,
            'detected_round': None,
            'pre_attack_accuracy': None
        }
        
        # Convergence tracking
        self.convergence_round = None
        self.convergence_time_sec = None
        
        # CH selection tracking
        self.ch_reelection_events = []
        
        # Process handle for resource monitoring
        self._process = psutil.Process()
    
    def start_experiment(self):
        """Mark the start of the experiment"""
        self._exp_start = time.time()
        print(f"✅ KPI Tracker: Experiment started at {time.ctime()}")
    
    def start_round(self):
        """Mark the start of a training round"""
        self._round_start = time.time()
    
    def measure_computational_load(self) -> tuple[float, float]:
        """
        Measure current CPU and memory usage
        
        Returns:
            (cpu_percent, memory_mb)
        """
        try:
            mem_mb = self._process.memory_info().rss / (1024 * 1024)
            cpu = psutil.cpu_percent(interval=0.1)
            return cpu, mem_mb
        except Exception as e:
            print(f"⚠️ KPI Tracker: Failed to measure computational load: {e}")
            return 0.0, 0.0
    
    def compute_communication_bytes(
        self, 
        active_clusters: List[int],
        clients_per_cluster: int = 200,
        participation_rates: Optional[Dict[int, float]] = None
    ) -> Dict[str, int]:
        """
        Compute communication bytes for this round using hierarchical formula
        
        Formula from study:
        - Tier 1 (Members → CH): clients_per_cluster × ω per active cluster
        - Tier 2 (CH → CH*): num_active_CHs × ω
        - Tier 3 (CH* → CH): num_active_CHs × ω
        - Tier 4 (CH → Members): clients_per_cluster × ω per active cluster
        
        Args:
            active_clusters: List of active cluster IDs (e.g., [0, 1, 2] or [1, 2] during D&R-E)
            clients_per_cluster: Number of clients per cluster
            participation_rates: Optional dict of {cluster_id: participation_rate} for partial participation
        
        Returns:
            Dict with tier-wise and total bytes
        """
        ω = self.model_size_bytes
        
        # Default full participation
        if participation_rates is None:
            participation_rates = {cid: 1.0 for cid in active_clusters}
        
        # Tier 1: Members → CH (upload)
        tier1 = sum(
            int(clients_per_cluster * participation_rates.get(cid, 1.0) * ω)
            for cid in active_clusters
        )
        
        # Tier 2: CH → CH* (CH to global aggregator, assuming CH1 is aggregator)
        # Only CHs that are active send to aggregator
        num_active_chs = len(active_clusters)
        tier2 = num_active_chs * ω
        
        # Tier 3: CH* → CH (global aggregator sends back to CHs)
        tier3 = num_active_chs * ω
        
        # Tier 4: CH → Members (download)
        tier4 = sum(
            int(clients_per_cluster * participation_rates.get(cid, 1.0) * ω)
            for cid in active_clusters
        )
        
        total = tier1 + tier2 + tier3 + tier4
        
        return {
            'tier1': tier1,
            'tier2': tier2,
            'tier3': tier3,
            'tier4': tier4,
            'total': total
        }
    
    def check_convergence(self, window_size: int = 5, variance_threshold: float = 0.01) -> bool:
        """
        Check if model has converged based on accuracy variance
        
        Args:
            window_size: Number of recent rounds to check
            variance_threshold: Maximum variance for convergence
        
        Returns:
            True if converged
        """
        if len(self.round_history) < window_size:
            return False
        
        recent_global_acc = [
            kpi.accuracies.get('global', 0.0)
            for kpi in self.round_history[-window_size:]
        ]
        
        variance = np.var(recent_global_acc)
        
        if variance < variance_threshold and self.convergence_round is None:
            self.convergence_round = self.round_history[-1].round
            self.convergence_time_sec = self.round_history[-1].cumulative_sec
            print(f"🎯 KPI Tracker: Convergence detected at round {self.convergence_round} "
                  f"(variance={variance:.6f})")
            return True
        
        return False
    
    def end_round(
        self,
        round_num: int,
        phase: str,
        accuracies: Dict[str, float],
        participation: Dict[str, float],
        communication_bytes: Dict[str, int],
        divergence: Optional[float] = None
    ):
        """
        Record metrics at the end of a training round
        
        Args:
            round_num: Current round number
            phase: 'normal', 'attack', 'dre', 'continuity', 'stabilization'
            accuracies: Dict with 'traffic', 'duration', 'bandwidth', 'global'
            participation: Dict with 'C0', 'C1', 'C2' participation rates
            communication_bytes: Dict with tier-wise and total bytes
            divergence: Optional L2 norm divergence metric
        """
        # Calculate round duration
        dur = time.time() - self._round_start if self._round_start else 0.0
        self.cumulative_sec += dur
        
        # Measure computational load
        cpu, mem_mb = self.measure_computational_load()
        
        # Create KPI snapshot
        kpi = RoundKPI(
            round=round_num,
            phase=phase,
            duration_sec=dur,
            cumulative_sec=self.cumulative_sec,
            cpu_percent=cpu,
            memory_mb=mem_mb,
            accuracies=accuracies,
            participation=participation,
            communication_bytes=communication_bytes,
            divergence=divergence
        )
        
        self.round_history.append(kpi)
        
        # Check for convergence
        self.check_convergence()
        
        # Save snapshot
        self.save_snapshot(round_num, kpi)
        
        # Print progress
        if round_num % 10 == 0 or round_num == 1:
            print(f"[Round {round_num:3d}] Phase: {phase:12s} | "
                  f"Traffic: {accuracies.get('traffic', 0):.4f} | "
                  f"Duration: {accuracies.get('duration', 0):.4f} | "
                  f"Bandwidth: {accuracies.get('bandwidth', 0):.4f} | "
                  f"Comm: {communication_bytes['total']/1e6:.2f} MB")
    
    def save_snapshot(self, round_num: int, kpi: RoundKPI):
        """Save individual round KPI snapshot to JSON"""
        path = os.path.join(self.save_dir, f'round_{round_num}.json')
        try:
            with open(path, 'w') as f:
                json.dump(kpi.to_dict(), f, indent=2)
        except Exception as e:
            print(f"⚠️ KPI Tracker: Failed to save snapshot for round {round_num}: {e}")
    
    def save_summary(self):
        """Save consolidated summary of all KPIs"""
        summary_path = os.path.join(self.save_dir, 'kpis_summary.json')
        
        summary = {
            'experiment_info': {
                'total_rounds': len(self.round_history),
                'total_time_sec': self.cumulative_sec,
                'convergence_round': self.convergence_round,
                'convergence_time_sec': self.convergence_time_sec,
                'model_size_bytes': self.model_size_bytes,
                'model_size_kb': self.model_size_bytes / 1024
            },
            'attack_info': self.attack_info,
            'ch_reelection_events': self.ch_reelection_events,
            'round_history': [kpi.to_dict() for kpi in self.round_history]
        }
        
        try:
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"✅ KPI Tracker: Summary saved to {summary_path}")
        except Exception as e:
            print(f"⚠️ KPI Tracker: Failed to save summary: {e}")
    
    def record_attack_start(self, round_num: int, pre_attack_accuracy: Optional[float] = None):
        """Record when attack starts"""
        self.attack_info['start_round'] = round_num
        self.attack_info['pre_attack_accuracy'] = pre_attack_accuracy
        print(f"⚔️ KPI Tracker: Attack started at round {round_num}")
    
    def record_attack_detected(self, round_num: int):
        """Record when attack is detected"""
        self.attack_info['detected_round'] = round_num
        detection_time = round_num - self.attack_info['start_round']
        print(f"🔍 KPI Tracker: Attack detected at round {round_num} "
              f"({detection_time} rounds after start)")
    
    def record_ch_reelection(
        self,
        round_num: int,
        election_time_sec: float,
        new_ch_id: int,
        new_ch_energy: Optional[float] = None,
        new_ch_rssi: Optional[float] = None
    ):
        """Record CH re-election event"""
        event = {
            'round': round_num,
            'election_time_sec': election_time_sec,
            'new_ch_id': new_ch_id,
            'new_ch_energy': new_ch_energy,
            'new_ch_rssi': new_ch_rssi
        }
        self.ch_reelection_events.append(event)
        print(f"👑 KPI Tracker: CH re-election at round {round_num} → CH{new_ch_id} "
              f"(took {election_time_sec*1000:.2f}ms)")
    
    def get_summary_stats(self) -> Dict:
        """Get summary statistics for reporting"""
        if not self.round_history:
            return {}
        
        # Extract metrics
        round_durations = [kpi.duration_sec for kpi in self.round_history]
        cpu_percentages = [kpi.cpu_percent for kpi in self.round_history]
        memory_mbs = [kpi.memory_mb for kpi in self.round_history]
        total_comm_bytes = sum(kpi.communication_bytes['total'] for kpi in self.round_history)
        
        # Final accuracies
        final_kpi = self.round_history[-1]
        
        return {
            'total_rounds': len(self.round_history),
            'total_time_sec': self.cumulative_sec,
            'avg_round_duration_sec': np.mean(round_durations),
            'convergence_round': self.convergence_round,
            'convergence_time_sec': self.convergence_time_sec,
            'final_accuracies': final_kpi.accuracies,
            'total_communication_bytes': total_comm_bytes,
            'total_communication_gb': total_comm_bytes / 1e9,
            'avg_cpu_percent': np.mean(cpu_percentages),
            'avg_memory_mb': np.mean(memory_mbs),
            'attack_start_round': self.attack_info['start_round'],
            'attack_detected_round': self.attack_info['detected_round'],
            'ch_reelections': len(self.ch_reelection_events)
        }
    
    def print_summary(self):
        """Print formatted summary of KPIs"""
        stats = self.get_summary_stats()
        
        print("\n" + "=" * 80)
        print("KPI TRACKER SUMMARY")
        print("=" * 80)
        
        print(f"\n📊 EXPERIMENT INFO")
        print(f"  Total Rounds: {stats.get('total_rounds', 0)}")
        print(f"  Total Time: {stats.get('total_time_sec', 0):.2f}s")
        print(f"  Avg Round Duration: {stats.get('avg_round_duration_sec', 0):.3f}s")
        print(f"  Convergence Round: {stats.get('convergence_round', 'N/A')}")
        
        print(f"\n🎯 FINAL ACCURACIES")
        final_acc = stats.get('final_accuracies', {})
        print(f"  Traffic: {final_acc.get('traffic', 0):.4f}")
        print(f"  Duration: {final_acc.get('duration', 0):.4f}")
        print(f"  Bandwidth: {final_acc.get('bandwidth', 0):.4f}")
        print(f"  Global: {final_acc.get('global', 0):.4f}")
        
        print(f"\n📡 COMMUNICATION")
        print(f"  Total: {stats.get('total_communication_gb', 0):.2f} GB")
        
        print(f"\n💻 RESOURCES")
        print(f"  Avg CPU: {stats.get('avg_cpu_percent', 0):.1f}%")
        print(f"  Avg Memory: {stats.get('avg_memory_mb', 0):.1f} MB")
        
        if stats.get('attack_start_round'):
            print(f"\n⚔️ ATTACK INFO")
            print(f"  Attack Start: Round {stats['attack_start_round']}")
            print(f"  Detection: Round {stats.get('attack_detected_round', 'N/A')}")
        
        if stats.get('ch_reelections', 0) > 0:
            print(f"\n👑 CH RE-ELECTIONS")
            print(f"  Total: {stats['ch_reelections']}")
        
        print("\n" + "=" * 80)
