"""
Cluster Head Election using LEACH-inspired formula
Context-aware CH selection based on residual energy and RSSI
"""

import numpy as np
from typing import Dict, Tuple


class CHElection:
    """Manages cluster head re-election after compromise"""
    
    def __init__(self, config, scene):
        self.config = config
        self.scene = scene
        self.alpha = config.alpha_energy  # Weight for energy
        self.beta = config.beta_rssi  # Weight for RSSI
    
    def calculate_rssi(self, uav1_id: int, uav2_id: int) -> float:
        """
        Calculate RSSI between two UAVs based on distance
        RSSI (dBm) = -10 * n * log10(distance) + A
        where n = path loss exponent (2 for free space), A = reference RSSI at 1m
        """
        uav1 = self.scene.uavs[uav1_id]
        uav2 = self.scene.uavs[uav2_id]
        
        distance = np.linalg.norm(uav1.position - uav2.position)
        
        if distance < 0.1:
            distance = 0.1  # Avoid log(0)
        
        A = -30  # Reference RSSI at 1m (dBm)
        n = 2.0  # Path loss exponent
        rssi = -10 * n * np.log10(distance) + A
        
        return rssi
    
    def update_rssi_for_cluster(self, cluster_id: int):
        """Update RSSI values for all UAVs in a cluster"""
        members = [uid for uid, uav in self.scene.uavs.items() 
                   if uav.cluster_id == cluster_id]
        
        for uav_id in members:
            uav = self.scene.uavs[uav_id]
            uav.rssi_to_neighbors = {}
            
            for neighbor_id in members:
                if neighbor_id != uav_id:
                    rssi = self.calculate_rssi(uav_id, neighbor_id)
                    uav.rssi_to_neighbors[neighbor_id] = rssi
    
    def calculate_avg_rssi(self, uav_id: int) -> float:
        """Calculate average RSSI for a UAV to all its neighbors"""
        uav = self.scene.uavs[uav_id]
        
        if not uav.rssi_to_neighbors:
            return -100.0  # Very poor signal if no neighbors
        
        avg_rssi = np.mean(list(uav.rssi_to_neighbors.values()))
        return avg_rssi
    
    def elect_new_ch(self, cluster_id: int) -> int:
        """
        Elect new cluster head for a cluster using LEACH formula:
        Score = α * E_residual + β * RSSI_avg
        
        Returns: UAV ID of new cluster head
        """
        # Update RSSI for all UAVs in cluster
        self.update_rssi_for_cluster(cluster_id)
        
        # Get all candidates (non-compromised UAVs in cluster)
        candidates = [uid for uid, uav in self.scene.uavs.items() 
                      if uav.cluster_id == cluster_id and 
                      not uav.is_compromised and 
                      uav.is_alive]
        
        if not candidates:
            print(f"⚠️ No valid candidates for CH election in Cluster {cluster_id}")
            return None
        
        # Calculate scores for all candidates
        scores = {}
        for uav_id in candidates:
            uav = self.scene.uavs[uav_id]
            
            # Normalize energy (0-100 -> 0-1)
            norm_energy = uav.residual_energy / 100.0
            
            # Normalize RSSI (-100 to -30 dBm -> 0 to 1)
            avg_rssi = self.calculate_avg_rssi(uav_id)
            norm_rssi = (avg_rssi + 100) / 70.0  # Map -100 to 0, -30 to 1
            norm_rssi = np.clip(norm_rssi, 0, 1)
            
            # Calculate weighted score
            score = self.alpha * norm_energy + self.beta * norm_rssi
            scores[uav_id] = score
            
            print(f"  Candidate UAV {uav_id}: E={norm_energy:.3f}, RSSI={norm_rssi:.3f}, Score={score:.3f}")
        
        # Select UAV with highest score
        new_ch_id = max(scores, key=scores.get)
        new_ch_score = scores[new_ch_id]
        
        print(f"✓ New CH{cluster_id} elected: UAV {new_ch_id} (score={new_ch_score:.3f})")
        
        return new_ch_id
    
    def install_new_ch(self, cluster_id: int, new_ch_id: int):
        """Install a new cluster head visually and logically"""
        # Get old CH
        old_ch_id = self.scene.cluster_heads[cluster_id]
        
        # Update old CH to regular member
        if old_ch_id in self.scene.uavs:
            old_uav = self.scene.uavs[old_ch_id]
            old_uav.is_cluster_head = False
            old_uav.is_compromised = False
            # Change to regular cluster color
            self.scene.update_uav_color(old_ch_id, self.config.cluster_colors[cluster_id])
        
        # Update new CH
        new_uav = self.scene.uavs[new_ch_id]
        new_uav.is_cluster_head = True
        new_uav.is_compromised = False
        
        # Change to gold
        self.scene.update_uav_color(new_ch_id, self.config.ch_color)
        
        # Update cluster heads mapping
        self.scene.cluster_heads[cluster_id] = new_ch_id
        
        print(f"✓ CH{cluster_id} transitioned: UAV {old_ch_id} -> UAV {new_ch_id}")
    
    def simulate_energy_consumption(self, cluster_id: int, round_num: int):
        """Simulate energy consumption for UAVs in a cluster"""
        for uav_id, uav in self.scene.uavs.items():
            if uav.cluster_id == cluster_id and uav.is_alive:
                # CHs consume more energy
                if uav.is_cluster_head:
                    consumption = np.random.uniform(0.3, 0.5)
                else:
                    consumption = np.random.uniform(0.1, 0.2)
                
                uav.residual_energy = max(0, uav.residual_energy - consumption)
                
                if uav.residual_energy <= 0:
                    uav.is_alive = False
