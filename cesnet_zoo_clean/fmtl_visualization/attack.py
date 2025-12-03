"""
CUAV (Compromised UAV) Attacker
Simulates red CUAV flying toward CH0 and compromising it
"""

import numpy as np
from typing import Tuple
from .scenario import ScenarioConfig


class CUAVAttacker:
    """Manages CUAV attack visualization and logic"""
    
    def __init__(self, config: ScenarioConfig, scene):
        self.config = config
        self.scene = scene
        
        # CUAV state
        self.cuav_body_id = None
        self.current_position = None
        self.target_position = None
        self.is_active = False
        self.attack_started = False
        self.attack_completed = False
        
        # Target is CH0
        self.target_cluster = 0
        self.target_ch_id = None
        
        # Attack parameters
        self.approach_speed = 2.0  # meters per step
        self.start_distance = 100.0  # Start 100m away from cluster
    
    def spawn_cuav(self):
        """Spawn the red CUAV at the edge of the environment"""
        import pybullet as p
        
        # Get CH0 position
        self.target_ch_id = self.scene.cluster_heads[self.target_cluster]
        ch0_pos = self.scene.uavs[self.target_ch_id].position
        
        # Start position: 100m away from CH0, same height
        direction = np.array([-1, 0, 0])  # Approach from the left
        self.current_position = ch0_pos + direction * self.start_distance
        self.target_position = ch0_pos.copy()
        
        # Create red CUAV sphere
        visual_shape = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=self.config.cuav_radius,
            rgbaColor=self.config.cuav_color
        )
        collision_shape = p.createCollisionShape(
            shapeType=p.GEOM_SPHERE,
            radius=self.config.cuav_radius
        )
        
        self.cuav_body_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=tuple(self.current_position)
        )
        
        self.is_active = True
        print(f"🔴 CUAV spawned at {self.current_position}, targeting CH0 at {ch0_pos}")
    
    def update(self, round_num: int) -> bool:
        """
        Update CUAV position and check for compromise
        Returns True if CH0 should be compromised this round
        """
        import pybullet as p
        
        if not self.is_active:
            # Check if attack should start
            attack_start_round = self.config.compromise_round - 10  # Start 10 rounds before
            if round_num == attack_start_round:
                self.spawn_cuav()
                self.attack_started = True
            return False
        
        if self.attack_completed:
            return False
        
        # Move CUAV toward CH0
        ch0_pos = self.scene.uavs[self.target_ch_id].position
        self.target_position = ch0_pos.copy()
        
        direction = self.target_position - self.current_position
        distance = np.linalg.norm(direction)
        
        if distance > self.config.jamming_radius:
            # Keep moving toward target
            direction_normalized = direction / distance
            step = direction_normalized * self.approach_speed
            self.current_position += step
            
            # Update PyBullet position
            p.resetBasePositionAndOrientation(
                self.cuav_body_id,
                tuple(self.current_position),
                [0, 0, 0, 1]
            )
            
            return False
        else:
            # Within jamming radius - compromise CH0
            if not self.attack_completed:
                self.attack_completed = True
                print(f"💥 CUAV reached jamming distance ({distance:.2f}m) - CH0 COMPROMISED!")
                return True
            
            return False
    
    def remove(self):
        """Remove CUAV from scene"""
        import pybullet as p
        
        if self.cuav_body_id is not None:
            p.removeBody(self.cuav_body_id)
            self.cuav_body_id = None
            self.is_active = False
    
    def get_position(self) -> Tuple[float, float, float]:
        """Get current CUAV position"""
        if self.current_position is not None:
            return tuple(self.current_position)
        return None
    
    def get_distance_to_target(self) -> float:
        """Get distance from CUAV to CH0"""
        if self.is_active and self.target_ch_id is not None:
            ch0_pos = self.scene.uavs[self.target_ch_id].position
            distance = np.linalg.norm(self.current_position - ch0_pos)
            return distance
        return float('inf')
