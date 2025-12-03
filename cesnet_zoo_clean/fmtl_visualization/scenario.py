"""
Scenario configuration for FMTL simulations
Supports convergence and transient compromise scenarios
"""

from dataclasses import dataclass
from typing import Literal


@dataclass
class ScenarioConfig:
    """Configuration for FMTL attack scenarios"""
    
    # Scenario type
    scenario_type: Literal['convergence_compromise', 'transient_compromise'] = 'convergence_compromise'
    
    # Cluster configuration
    num_clusters: int = 3
    uavs_per_cluster: int = 200
    
    # Cluster positions in 3D space (x, y, z)
    cluster_positions: dict = None
    
    # Communication
    communication_radius: float = 50.0  # meters
    jamming_radius: float = 10.0  # meters for CUAV attack
    
    # Timing - Convergence scenario (default)
    convergence_round: int = 90
    compromise_round: int = 111
    detection_round: int = 112
    dre_duration: int = 7  # D&R-E phase rounds (112-118)
    continuity_duration: int = 3  # Continuity phase (119-121)
    stabilization_duration: int = 4  # Stabilization (122-125)
    total_rounds: int = 125
    
    # CH election weights
    alpha_energy: float = 0.6  # Weight for residual energy
    beta_rssi: float = 0.4  # Weight for RSSI
    
    # Visual settings
    uav_radius: float = 0.5
    ch_radius: float = 1.0
    cuav_radius: float = 0.7
    
    # Colors (RGBA)
    cluster_colors: list = None
    ch_color: tuple = (1.0, 0.84, 0.0, 1.0)  # Gold
    compromised_color: tuple = (0.5, 0.0, 0.0, 1.0)  # Dark red
    cuav_color: tuple = (1.0, 0.0, 0.0, 1.0)  # Red
    
    def __post_init__(self):
        if self.cluster_positions is None:
            self.cluster_positions = {
                0: (-60, 0, 30),
                1: (0, 0, 30),
                2: (60, 0, 30),
            }
        
        if self.cluster_colors is None:
            self.cluster_colors = [
                (0.2, 0.6, 1.0, 1.0),  # Cluster 0: Blue
                (0.2, 1.0, 0.4, 1.0),  # Cluster 1: Green
                (1.0, 0.6, 0.2, 1.0),  # Cluster 2: Orange
            ]
        
        # Adjust timing for transient scenario
        if self.scenario_type == 'transient_compromise':
            self.compromise_round = 11
            self.detection_round = 12
            # D&R-E: 12-18 (7 rounds)
            # Continuity: 19-21 (3 rounds)
            # Still converges around 90, but starts recovery early
    
    def get_phase(self, round_num: int) -> str:
        """Get current phase based on round number"""
        if round_num < self.compromise_round:
            return 'NORMAL'
        elif round_num == self.compromise_round:
            return 'COMPROMISED'
        elif self.detection_round <= round_num < self.detection_round + self.dre_duration:
            return 'DRE'
        elif round_num < self.detection_round + self.dre_duration + self.continuity_duration:
            return 'CONTINUITY'
        elif round_num < self.detection_round + self.dre_duration + self.continuity_duration + self.stabilization_duration:
            return 'STABILIZATION'
        else:
            return 'NORMAL'
    
    def get_cluster0_participation(self, round_num: int) -> float:
        """Get Cluster 0 participation percentage for current round"""
        phase = self.get_phase(round_num)
        
        if phase in ['NORMAL', 'STABILIZATION']:
            return 1.0
        elif phase in ['COMPROMISED', 'DRE']:
            return 0.0
        elif phase == 'CONTINUITY':
            # 30% -> 70% -> 100%
            continuity_start = self.detection_round + self.dre_duration
            rounds_into_continuity = round_num - continuity_start
            
            if rounds_into_continuity == 0:
                return 0.3
            elif rounds_into_continuity == 1:
                return 0.7
            else:
                return 1.0
        
        return 1.0
    
    def is_ch0_compromised(self, round_num: int) -> bool:
        """Check if CH0 is compromised at this round"""
        return self.compromise_round <= round_num < self.detection_round + self.dre_duration
    
    def is_cluster0_offline(self, round_num: int) -> bool:
        """Check if Cluster 0 is offline (during D&R-E)"""
        return self.detection_round <= round_num < self.detection_round + self.dre_duration


class Scenario:
    """Manages scenario execution and state transitions"""
    
    def __init__(self, config: ScenarioConfig):
        self.config = config
        self.current_round = 0
        self.ch0_compromised = False
        self.cluster0_offline = False
        self.new_ch0_elected = False
    
    def update(self, round_num: int):
        """Update scenario state for current round"""
        self.current_round = round_num
        self.ch0_compromised = self.config.is_ch0_compromised(round_num)
        self.cluster0_offline = self.config.is_cluster0_offline(round_num)
        
        # Check if new CH0 should be elected (end of D&R-E)
        if round_num == self.config.detection_round + self.config.dre_duration:
            self.new_ch0_elected = True
    
    def get_status(self) -> dict:
        """Get current scenario status"""
        return {
            'round': self.current_round,
            'phase': self.config.get_phase(self.current_round),
            'ch0_compromised': self.ch0_compromised,
            'cluster0_offline': self.cluster0_offline,
            'cluster0_participation': self.config.get_cluster0_participation(self.current_round),
            'new_ch0_elected': self.new_ch0_elected,
        }
