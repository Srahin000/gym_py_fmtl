"""
PyBullet scene for 600 UAVs in 3 clusters
Handles UAV spawning, cluster formation, and state management
"""

import pybullet as p
import numpy as np
from typing import Dict, List, Tuple
from .scenario import ScenarioConfig


class UAVMetadata:
    """Metadata for a single UAV"""
    def __init__(self, uav_id: int, cluster_id: int, position: Tuple[float, float, float],
                 is_cluster_head: bool = False):
        self.uav_id = uav_id
        self.cluster_id = cluster_id
        self.position = np.array(position)
        self.is_cluster_head = is_cluster_head
        self.is_alive = True
        self.is_compromised = False
        self.residual_energy = 100.0  # Initial energy (%)
        self.rssi_to_neighbors = {}  # {neighbor_id: rssi_value}
        self.communication_radius = 50.0
        self.body_id = None  # PyBullet body ID
        self.current_color = None


class FMTLScene:
    """Main PyBullet scene for FMTL visualization"""
    
    def __init__(self, config: ScenarioConfig, use_gui: bool = True):
        self.config = config
        self.use_gui = use_gui
        
        # PyBullet connection
        if use_gui:
            self.physics_client = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        else:
            self.physics_client = p.connect(p.DIRECT)
        
        # Setup scene
        p.setGravity(0, 0, 0)  # No gravity for UAVs
        p.setRealTimeSimulation(0)
        
        # Camera setup
        p.resetDebugVisualizerCamera(
            cameraDistance=150,
            cameraYaw=0,
            cameraPitch=-30,
            cameraTargetPosition=[0, 0, 30]
        )
        
        # UAV storage
        self.uavs: Dict[int, UAVMetadata] = {}
        self.cluster_heads: Dict[int, int] = {}  # {cluster_id: uav_id}
        self.communication_lines = []  # Visual communication lines
        
        # Initialize clusters
        self._spawn_clusters()
    
    def _spawn_clusters(self):
        """Spawn all UAVs in 3 clusters"""
        global_uav_id = 0
        
        for cluster_id in range(self.config.num_clusters):
            cluster_center = self.config.cluster_positions[cluster_id]
            
            # Spawn cluster members in a formation
            for local_id in range(self.config.uavs_per_cluster):
                # Arrange in a grid pattern within cluster
                grid_size = int(np.ceil(np.sqrt(self.config.uavs_per_cluster)))
                row = local_id // grid_size
                col = local_id % grid_size
                
                # Offset from cluster center
                offset_x = (col - grid_size / 2) * 4
                offset_y = (row - grid_size / 2) * 4
                offset_z = np.random.uniform(-2, 2)
                
                position = (
                    cluster_center[0] + offset_x,
                    cluster_center[1] + offset_y,
                    cluster_center[2] + offset_z
                )
                
                # First UAV in each cluster is the CH
                is_ch = (local_id == 0)
                
                uav = UAVMetadata(
                    uav_id=global_uav_id,
                    cluster_id=cluster_id,
                    position=position,
                    is_cluster_head=is_ch
                )
                
                # Create visual representation
                if is_ch:
                    # Cluster head: larger gold sphere
                    visual_shape = p.createVisualShape(
                        shapeType=p.GEOM_SPHERE,
                        radius=self.config.ch_radius,
                        rgbaColor=self.config.ch_color
                    )
                    collision_shape = p.createCollisionShape(
                        shapeType=p.GEOM_SPHERE,
                        radius=self.config.ch_radius
                    )
                    self.cluster_heads[cluster_id] = global_uav_id
                else:
                    # Regular member: small colored sphere
                    visual_shape = p.createVisualShape(
                        shapeType=p.GEOM_SPHERE,
                        radius=self.config.uav_radius,
                        rgbaColor=self.config.cluster_colors[cluster_id]
                    )
                    collision_shape = p.createCollisionShape(
                        shapeType=p.GEOM_SPHERE,
                        radius=self.config.uav_radius
                    )
                
                body_id = p.createMultiBody(
                    baseMass=0,  # Static for now
                    baseCollisionShapeIndex=collision_shape,
                    baseVisualShapeIndex=visual_shape,
                    basePosition=position
                )
                
                uav.body_id = body_id
                uav.current_color = self.config.ch_color if is_ch else self.config.cluster_colors[cluster_id]
                self.uavs[global_uav_id] = uav
                
                global_uav_id += 1
        
        print(f"✓ Spawned {len(self.uavs)} UAVs in {self.config.num_clusters} clusters")
        print(f"  Cluster heads: {self.cluster_heads}")
    
    def update_uav_color(self, uav_id: int, color: Tuple[float, float, float, float]):
        """Change the color of a UAV"""
        uav = self.uavs[uav_id]
        if uav.body_id is not None:
            p.changeVisualShape(uav.body_id, -1, rgbaColor=color)
            uav.current_color = color
    
    def update_uav_position(self, uav_id: int, position: Tuple[float, float, float]):
        """Move a UAV to a new position"""
        uav = self.uavs[uav_id]
        if uav.body_id is not None:
            p.resetBasePositionAndOrientation(
                uav.body_id,
                position,
                [0, 0, 0, 1]
            )
            uav.position = np.array(position)
    
    def mark_ch_compromised(self, cluster_id: int):
        """Mark a cluster head as compromised (turn dark red)"""
        ch_id = self.cluster_heads[cluster_id]
        self.uavs[ch_id].is_compromised = True
        self.update_uav_color(ch_id, self.config.compromised_color)
        print(f"⚠️ CH{cluster_id} (UAV {ch_id}) marked as COMPROMISED")
    
    def restore_ch(self, cluster_id: int):
        """Restore cluster head to normal (gold)"""
        ch_id = self.cluster_heads[cluster_id]
        self.uavs[ch_id].is_compromised = False
        self.update_uav_color(ch_id, self.config.ch_color)
    
    def break_cluster_formation(self, cluster_id: int):
        """Visually break cluster formation (UAVs drift)"""
        for uav_id, uav in self.uavs.items():
            if uav.cluster_id == cluster_id and not uav.is_cluster_head:
                # Add random drift
                drift = np.random.uniform(-3, 3, size=3)
                new_pos = uav.position + drift
                self.update_uav_position(uav_id, tuple(new_pos))
    
    def restore_cluster_formation(self, cluster_id: int, participation: float = 1.0):
        """Restore cluster formation with given participation rate"""
        cluster_center = self.config.cluster_positions[cluster_id]
        
        # Get all non-CH members in this cluster
        members = [uid for uid, uav in self.uavs.items() 
                   if uav.cluster_id == cluster_id and not uav.is_cluster_head]
        
        # Restore only a fraction based on participation
        num_to_restore = int(len(members) * participation)
        to_restore = members[:num_to_restore]
        
        grid_size = int(np.ceil(np.sqrt(self.config.uavs_per_cluster)))
        
        for idx, uav_id in enumerate(to_restore):
            local_id = idx + 1  # Skip CH position
            row = local_id // grid_size
            col = local_id % grid_size
            
            offset_x = (col - grid_size / 2) * 4
            offset_y = (row - grid_size / 2) * 4
            offset_z = np.random.uniform(-2, 2)
            
            position = (
                cluster_center[0] + offset_x,
                cluster_center[1] + offset_y,
                cluster_center[2] + offset_z
            )
            
            self.update_uav_position(uav_id, position)
    
    def draw_communication_lines(self, connections: List[Tuple[int, int, str]]):
        """
        Draw communication lines between UAVs
        connections: [(uav_id1, uav_id2, color), ...]
        color: 'green', 'red', 'yellow'
        """
        # Remove old lines
        for line_id in self.communication_lines:
            p.removeUserDebugItem(line_id)
        self.communication_lines = []
        
        color_map = {
            'green': [0, 1, 0],
            'red': [1, 0, 0],
            'yellow': [1, 1, 0],
            'gold': [1, 0.84, 0],
        }
        
        for uav1_id, uav2_id, color_name in connections:
            if uav1_id in self.uavs and uav2_id in self.uavs:
                pos1 = self.uavs[uav1_id].position
                pos2 = self.uavs[uav2_id].position
                line_color = color_map.get(color_name, [1, 1, 1])
                
                line_id = p.addUserDebugLine(
                    pos1, pos2,
                    lineColorRGB=line_color,
                    lineWidth=2
                )
                self.communication_lines.append(line_id)
    
    def get_cluster_members(self, cluster_id: int) -> List[int]:
        """Get all UAV IDs in a cluster (excluding CH)"""
        return [uid for uid, uav in self.uavs.items() 
                if uav.cluster_id == cluster_id and not uav.is_cluster_head]
    
    def get_all_states(self) -> Dict:
        """Get current state of all UAVs"""
        return {
            uid: {
                'position': uav.position.tolist(),
                'cluster_id': uav.cluster_id,
                'is_ch': uav.is_cluster_head,
                'is_alive': uav.is_alive,
                'is_compromised': uav.is_compromised,
                'energy': uav.residual_energy,
            }
            for uid, uav in self.uavs.items()
        }
    
    def step(self):
        """Advance physics simulation one step"""
        p.stepSimulation()
    
    def close(self):
        """Clean up PyBullet connection"""
        p.disconnect(self.physics_client)
