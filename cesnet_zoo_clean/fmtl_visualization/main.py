"""
Main integration module for FMTL visualization
Connects PyBullet visualization to FMTL training loop
"""

import time
from typing import Dict, Optional
from .scenario import ScenarioConfig, Scenario
from .scene import FMTLScene
from .attack import CUAVAttacker
from .ch_election import CHElection
from .hud import HUDOverlay
from .frame_capture import FrameCapture


class FMTLVisualization:
    """
    Main class that integrates PyBullet visualization with FMTL training
    This is the bridge between logical training and physical visualization
    """
    
    def __init__(self, config: Optional[ScenarioConfig] = None, use_gui: bool = True,
                 capture_frames: bool = False):
        if config is None:
            config = ScenarioConfig()
        
        self.config = config
        self.scenario = Scenario(config)
        self.scene = FMTLScene(config, use_gui=use_gui)
        self.attacker = CUAVAttacker(config, self.scene)
        self.ch_election = CHElection(config, self.scene)
        self.hud = HUDOverlay()
        
        self.capture_frames = capture_frames
        self.frame_capture = None
        if capture_frames:
            self.frame_capture = FrameCapture()
        
        self.current_round = 0
        self.phase_transitions = []
        
        print("=" * 80)
        print("FMTL HIERARCHICAL FEDERATION VISUALIZATION")
        print(f"Scenario: {config.scenario_type}")
        print(f"Total UAVs: {config.num_clusters * config.uavs_per_cluster}")
        print(f"Clusters: {config.num_clusters}")
        print("=" * 80)
    
    def update_round(self, round_num: int, accuracies: Optional[Dict[str, float]] = None,
                     communication_stats: Optional[Dict[str, any]] = None):
        """
        Update visualization for a new training round
        This is called once per federated learning round
        
        Args:
            round_num: Current training round number
            accuracies: Per-task test accuracies {'traffic': 0.xx, 'duration': 0.xx, 'bandwidth': 0.xx}
            communication_stats: Communication data transfer stats
        """
        self.current_round = round_num
        self.scenario.update(round_num)
        status = self.scenario.get_status()
        
        # Track phase transitions
        if self.phase_transitions and self.phase_transitions[-1]['phase'] != status['phase']:
            self.phase_transitions.append({'round': round_num, 'phase': status['phase']})
        elif not self.phase_transitions:
            self.phase_transitions.append({'round': round_num, 'phase': status['phase']})
        
        # ===== ATTACK SEQUENCE =====
        should_compromise = self.attacker.update(round_num)
        if should_compromise:
            # Visual: CH0 turns dark red, cluster breaks formation
            self.scene.mark_ch_compromised(0)
            self.scene.break_cluster_formation(0)
            self.hud.show_attack_warning()
            
            # Draw red communication lines in Cluster 0
            cluster0_members = self.scene.get_cluster_members(0)
            ch0_id = self.scene.cluster_heads[0]
            red_connections = [(ch0_id, mid, 'red') for mid in cluster0_members[:10]]  # Sample
            self.scene.draw_communication_lines(red_connections)
            
            print(f"\n💥 [Round {round_num}] CH0 COMPROMISED - Attack successful!")
        
        # ===== DETECTION & D&R-E PHASE =====
        if round_num == self.config.detection_round:
            self.hud.show_detection_message()
            print(f"\n🚨 [Round {round_num}] Attack detected - Starting D&R-E phase")
            print(f"   Cluster 0 will be OFFLINE for {self.config.dre_duration} rounds")
        
        # During D&R-E: Cluster 0 is isolated (no communication lines)
        if status['cluster0_offline']:
            # Draw only Cluster 1 <-> Cluster 2 <-> CH1 lines
            ch1_id = self.scene.cluster_heads[1]
            ch2_id = self.scene.cluster_heads[2]
            active_connections = [(ch1_id, ch2_id, 'green')]
            self.scene.draw_communication_lines(active_connections)
        
        # ===== CH RE-ELECTION =====
        if round_num == self.config.detection_round + self.config.dre_duration:
            print(f"\n🔄 [Round {round_num}] End of D&R-E - Electing new CH0...")
            new_ch_id = self.ch_election.elect_new_ch(0)
            if new_ch_id:
                self.ch_election.install_new_ch(0, new_ch_id)
                print(f"   ✓ New CH0 installed: UAV {new_ch_id}")
        
        # ===== CONTINUITY PHASE =====
        if status['phase'] == 'CONTINUITY':
            participation = status['cluster0_participation']
            self.scene.restore_cluster_formation(0, participation)
            
            # Visual feedback for recovery
            if round_num == self.config.detection_round + self.config.dre_duration:
                self.hud.show_recovery_message(participation)
                print(f"\n♻️ [Round {round_num}] Cluster 0 CONTINUITY phase started")
            
            print(f"   Cluster 0 participation: {participation:.0%}")
            
            # Draw partial connections for recovering cluster
            ch0_id = self.scene.cluster_heads[0]
            cluster0_members = self.scene.get_cluster_members(0)
            active_members = cluster0_members[:int(len(cluster0_members) * participation)]
            yellow_connections = [(ch0_id, mid, 'yellow') for mid in active_members[:10]]
            self.scene.draw_communication_lines(yellow_connections)
        
        # ===== STABILIZATION & NORMAL =====
        if status['phase'] in ['STABILIZATION', 'NORMAL'] and not status['cluster0_offline']:
            if status['cluster0_participation'] >= 1.0:
                # Full green connections
                ch0_id = self.scene.cluster_heads[0]
                ch1_id = self.scene.cluster_heads[1]
                ch2_id = self.scene.cluster_heads[2]
                
                green_connections = [
                    (ch0_id, ch1_id, 'green'),
                    (ch1_id, ch2_id, 'green'),
                ]
                self.scene.draw_communication_lines(green_connections)
        
        # ===== UPDATE HUD =====
        cluster_statuses = {
            0: self._get_cluster_status_string(0, status),
            1: "Online (100%)",
            2: "Online (100%)",
        }
        
        self.hud.update(
            round_num=round_num,
            phase=status['phase'],
            accuracies=accuracies,
            cluster_statuses=cluster_statuses,
            communication_stats=communication_stats
        )
        
        # ===== CAPTURE FRAME =====
        if self.capture_frames and self.frame_capture:
            self.frame_capture.capture_frame(save_to_disk=True)
        
        # Step physics
        self.scene.step()
        time.sleep(0.01)  # Small delay for visualization
    
    def _get_cluster_status_string(self, cluster_id: int, status: Dict) -> str:
        """Generate status string for a cluster"""
        if cluster_id == 0:
            if status['ch0_compromised']:
                return "⚠️ COMPROMISED"
            elif status['cluster0_offline']:
                return "❌ OFFLINE (D&R-E)"
            elif status['phase'] == 'CONTINUITY':
                return f"♻️ Recovering ({status['cluster0_participation']:.0%})"
            elif status['phase'] == 'STABILIZATION':
                return "✅ Stabilizing"
            else:
                return "✅ Online (100%)"
        else:
            return "✅ Online (100%)"
    
    def run_full_simulation(self, max_rounds: int = None):
        """
        Run complete simulation standalone (no external training loop)
        Useful for testing visualization only
        """
        if max_rounds is None:
            max_rounds = self.config.total_rounds
        
        print(f"\n🚀 Running standalone simulation for {max_rounds} rounds...")
        
        for round_num in range(1, max_rounds + 1):
            # Dummy accuracies (in real use, these come from training)
            accuracies = {
                'traffic': 0.85 + (round_num / max_rounds) * 0.10,
                'duration': 0.82 + (round_num / max_rounds) * 0.12,
                'bandwidth': 0.88 + (round_num / max_rounds) * 0.08,
            }
            
            # Dummy communication stats
            comm_stats = {
                'round_data': 297.15 if not self.scenario.config.is_cluster0_offline(round_num) else 198.10,
                'total_data': (round_num * 297.15) / 1024,  # Convert to GB
            }
            
            self.update_round(round_num, accuracies, comm_stats)
            
            if round_num % 10 == 0:
                status = self.scenario.get_status()
                print(f"  Round {round_num}/{max_rounds}: {status['phase']}")
        
        print("\n✓ Simulation complete!")
        self._print_summary()
    
    def _print_summary(self):
        """Print simulation summary"""
        print("\n" + "=" * 80)
        print("SIMULATION SUMMARY")
        print("=" * 80)
        print(f"Total rounds: {self.current_round}")
        print(f"\nPhase transitions:")
        for transition in self.phase_transitions:
            print(f"  Round {transition['round']}: {transition['phase']}")
    
    def export_attack_video(self, output_filename: str = "attack_sequence.gif"):
        """Export attack sequence as GIF"""
        if self.frame_capture and self.frame_capture.frames:
            self.frame_capture.export_gif(output_filename)
        else:
            print("⚠️ No frames captured. Enable capture_frames=True")
    
    def close(self):
        """Clean up and close visualization"""
        self.scene.close()
        print("\n✓ Visualization closed")


# Convenience function for quick testing
def run_visualization_demo(scenario_type: str = 'convergence_compromise'):
    """
    Quick demo of the visualization system
    
    Args:
        scenario_type: 'convergence_compromise' or 'transient_compromise'
    """
    config = ScenarioConfig(scenario_type=scenario_type)
    viz = FMTLVisualization(config, use_gui=True, capture_frames=False)
    
    try:
        viz.run_full_simulation()
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
    finally:
        viz.close()


if __name__ == '__main__':
    # Run demo
    run_visualization_demo('convergence_compromise')
