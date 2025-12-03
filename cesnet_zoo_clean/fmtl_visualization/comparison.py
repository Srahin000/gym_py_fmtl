"""
Comparison view for convergence vs transient scenarios
Side-by-side visualization
"""

import pybullet as p
from .scenario import ScenarioConfig, Scenario
from .scene import FMTLScene
from .attack import CUAVAttacker
from .ch_election import CHElection
from .hud import HUDOverlay


class ComparisonView:
    """
    Manages side-by-side comparison of two scenarios
    Left: Convergence compromise (attack at round 111)
    Right: Transient compromise (attack at round 11)
    """
    
    def __init__(self, use_gui: bool = True):
        # Create two separate PyBullet instances would require separate processes
        # For simplicity, we'll run them sequentially
        
        self.convergence_config = ScenarioConfig(scenario_type='convergence_compromise')
        self.transient_config = ScenarioConfig(scenario_type='transient_compromise')
        
        self.use_gui = use_gui
        
        print("=" * 80)
        print("COMPARISON MODE: Convergence vs Transient Compromise")
        print("=" * 80)
    
    def run_convergence_scenario(self, max_rounds: int = 125):
        """Run convergence compromise scenario"""
        print("\n" + "=" * 80)
        print("SCENARIO 1: CONVERGENCE COMPROMISE (Attack at Round 111)")
        print("=" * 80)
        
        scenario = Scenario(self.convergence_config)
        scene = FMTLScene(self.convergence_config, use_gui=self.use_gui)
        attacker = CUAVAttacker(self.convergence_config, scene)
        ch_election = CHElection(self.convergence_config, scene)
        hud = HUDOverlay()
        
        results = []
        
        for round_num in range(1, max_rounds + 1):
            scenario.update(round_num)
            status = scenario.get_status()
            
            # Update attacker
            should_compromise = attacker.update(round_num)
            if should_compromise:
                scene.mark_ch_compromised(0)
                scene.break_cluster_formation(0)
                hud.show_attack_warning()
            
            # Handle D&R-E phase
            if round_num == self.convergence_config.detection_round:
                hud.show_detection_message()
            
            # Handle CH re-election
            if round_num == self.convergence_config.detection_round + self.convergence_config.dre_duration:
                new_ch_id = ch_election.elect_new_ch(0)
                ch_election.install_new_ch(0, new_ch_id)
            
            # Handle continuity phase
            if status['phase'] == 'CONTINUITY':
                participation = status['cluster0_participation']
                scene.restore_cluster_formation(0, participation)
                if round_num == self.convergence_config.detection_round + self.convergence_config.dre_duration:
                    hud.show_recovery_message(participation)
            
            # Update HUD
            cluster_statuses = {
                0: f"{'OFFLINE' if status['cluster0_offline'] else 'Online'} ({status['cluster0_participation']:.0%})",
                1: "Online (100%)",
                2: "Online (100%)",
            }
            
            hud.update(
                round_num=round_num,
                phase=status['phase'],
                cluster_statuses=cluster_statuses
            )
            
            scene.step()
            
            results.append(status)
            
            if round_num % 10 == 0:
                print(f"  Round {round_num}: {status['phase']}, Cluster 0: {status['cluster0_participation']:.0%}")
        
        scene.close()
        
        print("\n✓ Convergence scenario complete")
        return results
    
    def run_transient_scenario(self, max_rounds: int = 125):
        """Run transient compromise scenario"""
        print("\n" + "=" * 80)
        print("SCENARIO 2: TRANSIENT COMPROMISE (Attack at Round 11)")
        print("=" * 80)
        
        scenario = Scenario(self.transient_config)
        scene = FMTLScene(self.transient_config, use_gui=self.use_gui)
        attacker = CUAVAttacker(self.transient_config, scene)
        ch_election = CHElection(self.transient_config, scene)
        hud = HUDOverlay()
        
        results = []
        
        for round_num in range(1, max_rounds + 1):
            scenario.update(round_num)
            status = scenario.get_status()
            
            # Update attacker
            should_compromise = attacker.update(round_num)
            if should_compromise:
                scene.mark_ch_compromised(0)
                scene.break_cluster_formation(0)
                hud.show_attack_warning()
            
            # Handle D&R-E phase
            if round_num == self.transient_config.detection_round:
                hud.show_detection_message()
            
            # Handle CH re-election
            if round_num == self.transient_config.detection_round + self.transient_config.dre_duration:
                new_ch_id = ch_election.elect_new_ch(0)
                ch_election.install_new_ch(0, new_ch_id)
            
            # Handle continuity phase
            if status['phase'] == 'CONTINUITY':
                participation = status['cluster0_participation']
                scene.restore_cluster_formation(0, participation)
                if round_num == self.transient_config.detection_round + self.transient_config.dre_duration:
                    hud.show_recovery_message(participation)
            
            # Update HUD
            cluster_statuses = {
                0: f"{'OFFLINE' if status['cluster0_offline'] else 'Online'} ({status['cluster0_participation']:.0%})",
                1: "Online (100%)",
                2: "Online (100%)",
            }
            
            hud.update(
                round_num=round_num,
                phase=status['phase'],
                cluster_statuses=cluster_statuses
            )
            
            scene.step()
            
            results.append(status)
            
            if round_num % 10 == 0:
                print(f"  Round {round_num}: {status['phase']}, Cluster 0: {status['cluster0_participation']:.0%}")
        
        scene.close()
        
        print("\n✓ Transient scenario complete")
        return results
    
    def run_both(self, max_rounds: int = 125):
        """Run both scenarios sequentially"""
        conv_results = self.run_convergence_scenario(max_rounds)
        trans_results = self.run_transient_scenario(max_rounds)
        
        return {
            'convergence': conv_results,
            'transient': trans_results,
        }
