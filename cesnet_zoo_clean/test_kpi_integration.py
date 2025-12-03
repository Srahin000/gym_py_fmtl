"""
Quick test of KPI integration with simulation
Runs only 5 rounds to verify KPI tracking works correctly
"""

import numpy as np
import sys
import os

# Add cesnet_zoo_clean to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fmtl_visualization.kpi import ComprehensiveKPITracker
from fmtl_visualization.kpi_helpers import (
    compute_participation_rates,
    compute_model_divergence,
    compute_communication_overhead,
    determine_phase
)
from fmtl_visualization.inference import create_inference_engine


def test_kpi_integration():
    """Test KPI tracking with a few simulation rounds"""
    
    print("=" * 80)
    print("KPI INTEGRATION TEST")
    print("=" * 80)
    
    # Initialize KPI tracker
    script_dir = os.path.dirname(os.path.abspath(__file__))
    kpi_output_dir = os.path.join(script_dir, 'trained_models', 'kpi_test_output')
    
    kpi_tracker = ComprehensiveKPITracker(save_dir=kpi_output_dir)
    
    # Start experiment
    kpi_tracker.start_experiment()
    print("✓ KPI tracker initialized")
    
    # Initialize inference engine
    models_dir = os.path.join(script_dir, 'trained_models', 'hierarchical_equal')
    data_path = os.path.join(script_dir, 'datasets', 'local_cache', 'dataset_12500_samples_65_features.csv')
    
    inference_engine = None
    if os.path.exists(models_dir):
        try:
            inference_engine = create_inference_engine(models_dir, data_path)
            print("✓ Inference engine initialized")
        except Exception as e:
            print(f"⚠️ Inference engine failed: {e}")
    
    # Test rounds: 1, 110, 111, 115, 120
    test_rounds = [1, 110, 111, 115, 120]
    
    for round_num in test_rounds:
        print(f"\n{'─' * 80}")
        print(f"Testing Round {round_num}")
        print(f"{'─' * 80}")
        
        # Start round timing
        kpi_tracker.start_round()
        
        # Determine phase
        phase, comp_cluster = determine_phase(round_num)
        print(f"  Phase: {phase} (compromised cluster: {comp_cluster})")
        
        # Record attack events
        if round_num == 111:
            kpi_tracker.record_attack_start(round_num)
        elif round_num == 115:  # Detection at round 112, but test at 115
            pass
        
        # Get accuracies from inference or use dummy
        if inference_engine:
            try:
                dre_phase = (phase == 'D&R-E')
                accuracies = inference_engine.update_round(
                    round_num,
                    compromised_cluster=comp_cluster,
                    dre_phase=dre_phase,
                    old_model_round=110
                )
            except Exception as e:
                print(f"  ⚠️ Inference failed: {e}")
                accuracies = {
                    'traffic': 0.70 + np.random.uniform(-0.02, 0.02),
                    'duration': 0.68 + np.random.uniform(-0.02, 0.02),
                    'bandwidth': 0.73 + np.random.uniform(-0.02, 0.02),
                }
        else:
            accuracies = {
                'traffic': 0.70 + np.random.uniform(-0.02, 0.02),
                'duration': 0.68 + np.random.uniform(-0.02, 0.02),
                'bandwidth': 0.73 + np.random.uniform(-0.02, 0.02),
            }
        
        print(f"  Accuracies: Traffic={accuracies['traffic']:.4f}, "
              f"Duration={accuracies['duration']:.4f}, Bandwidth={accuracies['bandwidth']:.4f}")
        
        # Compute participation
        participation = compute_participation_rates(round_num, phase)
        print(f"  Participation: {participation}")
        
        # Compute communication
        communication_bytes_total = compute_communication_overhead(round_num, phase, participation)
        communication_bytes = {
            'tier1_member_to_ch': int(communication_bytes_total * 0.4),  # Approximate distribution
            'tier2_ch_to_global': int(communication_bytes_total * 0.1),
            'tier3_global_to_ch': int(communication_bytes_total * 0.1),
            'tier4_ch_to_member': int(communication_bytes_total * 0.4),
            'total': communication_bytes_total
        }
        print(f"  Communication: {communication_bytes['total'] / (1024**2):.2f} MB")
        
        # Compute divergence
        divergence = compute_model_divergence(round_num, phase, comp_cluster if comp_cluster is not None else 0)
        print(f"  Divergence: {divergence:.4f}")
        
        # End round and save
        kpi_tracker.end_round(
            round_num=round_num,
            phase=phase,
            accuracies=accuracies,
            participation=participation,
            communication_bytes=communication_bytes,
            divergence=divergence
        )
        
        print(f"  ✓ KPI snapshot saved")
    
    # Save summary
    print(f"\n{'=' * 80}")
    print("Saving KPI Summary")
    print(f"{'=' * 80}")
    
    kpi_tracker.save_summary()
    
    print(f"\n✅ KPI test complete!")
    print(f"   Output dir: {kpi_output_dir}")
    print(f"   Rounds tracked: {len(kpi_tracker.round_history)}")
    print(f"   Attack info: {kpi_tracker.attack_info}")
    
    # Verify files exist
    import json
    summary_file = os.path.join(kpi_output_dir, 'kpi_summary.json')
    if os.path.exists(summary_file):
        with open(summary_file, 'r') as f:
            summary = json.load(f)
        print(f"\n📊 Summary Preview:")
        print(f"   Rounds: {summary['metadata']['total_rounds']}")
        print(f"   Duration: {summary['metadata']['total_duration_seconds']:.2f}s")
        print(f"   Attack info: start={summary.get('attack_start_round', 'N/A')}, "
              f"detected={summary.get('attack_detected_round', 'N/A')}")


if __name__ == '__main__':
    test_kpi_integration()
