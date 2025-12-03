"""
KPI Helper Functions for FMTL Simulation

Provides utility functions for:
- Model evaluation with real inference
- Participation rate computation
- Communication overhead calculation
- Divergence metrics
"""

import numpy as np
from typing import Dict, List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .inference import FMTLInferenceEngine


def evaluate_model_accuracies(
    inference_engine,  # Type: FMTLInferenceEngine
    round_num: int,
    phase: str,
    compromised_cluster: int = None
) -> Dict[str, float]:
    """
    Evaluate model accuracies using real inference engine
    
    Args:
        inference_engine: FMTL inference engine with loaded models
        round_num: Current training round
        phase: Current phase ('Normal', 'Compromised', 'D&R-E', etc.)
        compromised_cluster: Cluster ID that is compromised (if any)
    
    Returns:
        Dictionary with task accuracies
    """
    # Determine if we're in D&R-E phase
    dre_phase = (phase == 'D&R-E')
    old_model_round = 110 if dre_phase else None  # Pre-attack model
    
    # Update inference engine for this round
    accuracies = inference_engine.update_round(
        round_num,
        compromised_cluster=compromised_cluster,
        dre_phase=dre_phase,
        old_model_round=old_model_round
    )
    
    return accuracies


def compute_participation_rates(
    round_num: int,
    phase: str,
    num_clusters: int = 3,
    uavs_per_cluster: int = 200
) -> Dict[str, float]:
    """
    Compute participation rates for each cluster based on phase
    
    Phase participation rules:
    - Normal (1-110): All 100%
    - Compromised (111): C0=0%, C1=100%, C2=100%
    - D&R-E (112-118): C0=30%, C1=100%, C2=100%
    - Continuity (119-121): C0=70%, C1=100%, C2=100%
    - Stabilization (122+): All 100%
    
    Args:
        round_num: Current round number
        phase: Current phase name
        num_clusters: Number of clusters
        uavs_per_cluster: UAVs per cluster
    
    Returns:
        Dictionary with participation rates: {'C0': 1.0, 'C1': 1.0, 'C2': 1.0}
    """
    participation = {}
    
    # Define participation based on phase
    if phase == 'Normal':
        # All clusters 100%
        for i in range(num_clusters):
            participation[f'C{i}'] = 1.0
    
    elif phase == 'Compromised':
        # C0 excluded (0%), others 100%
        participation['C0'] = 0.0
        for i in range(1, num_clusters):
            participation[f'C{i}'] = 1.0
    
    elif phase == 'D&R-E':
        # C0 at 30%, others 100%
        participation['C0'] = 0.3
        for i in range(1, num_clusters):
            participation[f'C{i}'] = 1.0
    
    elif phase == 'Continuity':
        # C0 at 70%, others 100%
        participation['C0'] = 0.7
        for i in range(1, num_clusters):
            participation[f'C{i}'] = 1.0
    
    else:  # Stabilization and beyond
        # All clusters back to 100%
        for i in range(num_clusters):
            participation[f'C{i}'] = 1.0
    
    return participation


def compute_communication_overhead(
    round_num: int,
    phase: str,
    participation: Dict[str, float],
    model_size_bytes: int = 246 * 1024,  # 246 KB per model
    num_clusters: int = 3,
    uavs_per_cluster: int = 200
) -> int:
    """
    Compute total communication bytes for the round
    
    Communication patterns:
    - Member → CH: Each participating member uploads model (model_size_bytes)
    - CH → Global: Each CH uploads aggregated model
    - Global → CH: Global model broadcasted to all CHs
    - CH → Members: CH broadcasts to all members in cluster
    
    Args:
        round_num: Current round
        phase: Current phase
        participation: Dict of participation rates per cluster
        model_size_bytes: Size of one model in bytes
        num_clusters: Number of clusters
        uavs_per_cluster: UAVs per cluster
    
    Returns:
        Total communication bytes for this round
    """
    total_bytes = 0
    
    # Calculate participating members per cluster
    participating_members = {}
    for cluster_id in range(num_clusters):
        cluster_key = f'C{cluster_id}'
        rate = participation.get(cluster_key, 1.0)
        participating_members[cluster_id] = int(uavs_per_cluster * rate)
    
    # Member → CH uploads
    for cluster_id, n_members in participating_members.items():
        total_bytes += n_members * model_size_bytes
    
    # CH → Global uploads (3 CHs upload to global)
    total_bytes += num_clusters * model_size_bytes
    
    # Global → CH broadcasts (global sends to 3 CHs)
    total_bytes += num_clusters * model_size_bytes
    
    # CH → Members broadcasts (each CH broadcasts to all members, not just participating)
    total_bytes += num_clusters * uavs_per_cluster * model_size_bytes
    
    return total_bytes


def compute_model_divergence(
    round_num: int,
    phase: str,
    compromised_cluster: int = 0
) -> float:
    """
    Compute model divergence metric
    
    During D&R-E phase, compromised cluster uses old model (R110)
    while others use current model, causing divergence.
    
    Divergence formula:
    - Normal: 0.0 (all models aligned)
    - Compromised: 0.0 (C0 excluded, no divergence)
    - D&R-E: Simulated divergence based on version difference
    - Continuity: Decreasing as C0 reintegrates
    - Stabilization: Back to 0.0
    
    Args:
        round_num: Current round
        phase: Current phase
        compromised_cluster: ID of compromised cluster
    
    Returns:
        Divergence metric (0.0 = aligned, higher = more divergence)
    """
    if phase == 'Normal' or phase == 'Stabilization':
        return 0.0
    
    elif phase == 'Compromised':
        # C0 excluded, no divergence among participating clusters
        return 0.0
    
    elif phase == 'D&R-E':
        # C0 uses old model (R110), others use current (R112-118)
        # Simulate divergence based on round difference
        old_round = 110
        round_diff = round_num - old_round
        
        # Divergence increases with round difference
        # Scale: ~0.1 to 0.3 based on how far behind
        divergence = min(0.05 * round_diff, 0.35)
        return divergence
    
    elif phase == 'Continuity':
        # C0 reintegrating, divergence decreasing
        # Simulate gradual convergence
        # Continuity spans rounds 119-121
        if round_num == 119:
            return 0.20  # Still some divergence
        elif round_num == 120:
            return 0.10
        else:  # 121
            return 0.05
    
    else:
        return 0.0


def determine_phase(round_num: int) -> Tuple[str, int]:
    """
    Determine simulation phase based on round number
    
    Phase timeline:
    - Normal: 1-110
    - Compromised: 111 (attack begins, C0 excluded)
    - D&R-E: 112-118 (detection & response, C0 at 30%)
    - Continuity: 119-121 (C0 at 70%)
    - Stabilization: 122-125 (C0 at 100%)
    
    Args:
        round_num: Current round number
    
    Returns:
        Tuple of (phase_name, compromised_cluster_id or None)
    """
    if round_num <= 110:
        return ('Normal', None)
    elif round_num == 111:
        return ('Compromised', 0)
    elif 112 <= round_num <= 118:
        return ('D&R-E', 0)
    elif 119 <= round_num <= 121:
        return ('Continuity', 0)
    else:  # 122+
        return ('Stabilization', None)


def format_bytes(bytes_val: int) -> str:
    """Format bytes as human-readable string"""
    if bytes_val < 1024:
        return f"{bytes_val} B"
    elif bytes_val < 1024 ** 2:
        return f"{bytes_val / 1024:.2f} KB"
    elif bytes_val < 1024 ** 3:
        return f"{bytes_val / 1024**2:.2f} MB"
    else:
        return f"{bytes_val / 1024**3:.2f} GB"


def format_accuracy(accuracy: float) -> str:
    """Format accuracy as percentage string"""
    return f"{accuracy * 100:.2f}%"


# Example usage
if __name__ == '__main__':
    print("=" * 80)
    print("Testing KPI Helper Functions")
    print("=" * 80)
    
    # Test phase determination
    test_rounds = [50, 110, 111, 115, 120, 125]
    print("\nPhase Determination:")
    for r in test_rounds:
        phase, comp_cluster = determine_phase(r)
        print(f"  Round {r:3d}: {phase:15s} (compromised: {comp_cluster})")
    
    # Test participation computation
    print("\nParticipation Rates:")
    for r in [110, 111, 115, 120, 125]:
        phase, _ = determine_phase(r)
        participation = compute_participation_rates(r, phase)
        print(f"  Round {r:3d} ({phase:15s}): {participation}")
    
    # Test communication overhead
    print("\nCommunication Overhead:")
    for r in [110, 111, 115, 120]:
        phase, _ = determine_phase(r)
        participation = compute_participation_rates(r, phase)
        bytes_val = compute_communication_overhead(r, phase, participation)
        print(f"  Round {r:3d}: {format_bytes(bytes_val)}")
    
    # Test divergence
    print("\nModel Divergence:")
    for r in [110, 111, 115, 120, 125]:
        phase, comp = determine_phase(r)
        div = compute_model_divergence(r, phase, comp if comp is not None else 0)
        print(f"  Round {r:3d} ({phase:15s}): {div:.4f}")
