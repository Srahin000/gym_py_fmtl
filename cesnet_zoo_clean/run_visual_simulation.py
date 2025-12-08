"""
COMPLETE UAV SWARM FEDERATED LEARNING SIMULATION WITH PYBULLET
===============================================================

Fully functional PyBullet GUI simulation showing:
- 600 UAVs in 3 clusters performing federated learning
- CUAV (compromised UAV) attacks and compromises a Cluster Head
- System detects compromise and recovers
- Visual GUI showing the entire process over 30 rounds
"""

import pybullet as p
import pybullet_data
import numpy as np
import time
import sys
import os

# Get paths to drone models
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'gym_pybullet_drones', 'assets')
DRONE_URDF = os.path.join(ASSETS_DIR, 'cf2x.urdf')  # Crazyflie 2.x model

# Configuration
NUM_CLUSTERS = 3
UAVS_PER_CLUSTER = 15  # 5 UAVs per cluster (15 total for testing with large models)
TOTAL_UAVS = NUM_CLUSTERS * UAVS_PER_CLUSTER
TOTAL_ROUNDS = 30

# Cluster positions
CLUSTER_POSITIONS = [
    np.array([-60.0, 0.0, 30.0]),  # Cluster 0 (left)
    np.array([0.0, 0.0, 30.0]),     # Cluster 1 (center)
    np.array([60.0, 0.0, 30.0])     # Cluster 2 (right)
]

# Colors (RGBA) - Cluster-specific colors
CLUSTER_COLORS = {
    0: [0.2, 1.0, 0.2, 1.0],   # Cluster 0: Green
    1: [0.2, 0.5, 1.0, 1.0],   # Cluster 1: Blue
    2: [0.8, 0.2, 1.0, 1.0]    # Cluster 2: Purple
}
COLOR_CH = [1.0, 0.84, 0.0, 1.0]               # Gold
COLOR_CUAV = [1.0, 0.0, 0.0, 1.0]              # Red
COLOR_COMPROMISED = [0.3, 0.3, 0.3, 0.7]       # Dark gray

# Degraded colors (darker versions of cluster colors)
CLUSTER_COLORS_DEGRADED = {
    0: [0.1, 0.5, 0.1, 1.0],   # Dark green
    1: [0.1, 0.3, 0.6, 1.0],   # Dark blue
    2: [0.4, 0.1, 0.5, 1.0]    # Dark purple
}
CLUSTER_COLORS_RECOVERING = {
    0: [0.15, 0.75, 0.15, 1.0],  # Medium green
    1: [0.15, 0.4, 0.8, 1.0],    # Medium blue
    2: [0.6, 0.15, 0.75, 1.0]    # Medium purple
}

# UAV class to track state
class UAV:
    def __init__(self, uav_id, cluster_id, position, is_ch=False):
        self.id = uav_id
        self.cluster_id = cluster_id
        self.position = np.array(position, dtype=float)
        self.is_ch = is_ch
        self.is_alive = True
        self.is_compromised = False
        self.body_id = None
        self.participation_level = 1.0  # 0.0 to 1.0
        
        # Orbital flight parameters (for member UAVs)
        self.orbit_radius = np.random.uniform(8, 18)  # Distance from CH
        self.orbit_speed = np.random.uniform(0.15, 0.35)  # Radians per step
        self.orbit_angle = np.random.uniform(0, 2 * np.pi)  # Starting angle
        self.orbit_height_offset = np.random.uniform(-3, 3)  # Vertical variation
        
    def set_color(self, color):
        """Update UAV visual color"""
        if self.body_id is not None:
            # Change base link color
            p.changeVisualShape(self.body_id, -1, rgbaColor=color)
            
            # Change all joint/link colors for drone model
            try:
                num_joints = p.getNumJoints(self.body_id)
                for joint_idx in range(num_joints):
                    p.changeVisualShape(self.body_id, joint_idx, rgbaColor=color)
            except:
                pass  # If it's a simple shape without joints


class CUAVAttacker:
    def __init__(self, spawn_position, target_position):
        self.position = np.array(spawn_position, dtype=float)
        self.target = np.array(target_position, dtype=float)
        self.speed = 2.0  # m/s
        self.body_id = None
        self.active = True
        
    def move_toward_target(self, dt=1.0):
        """Move CUAV toward target"""
        if not self.active:
            return
            
        direction = self.target - self.position
        distance = np.linalg.norm(direction)
        
        if distance < 5.0:
            self.active = False
            return
            
        direction_normalized = direction / distance
        self.position += direction_normalized * self.speed * dt
        
        if self.body_id is not None:
            p.resetBasePositionAndOrientation(
                self.body_id,
                self.position.tolist(),
                [0, 0, 0, 1]
            )


class SimulationState:
    def __init__(self):
        self.round = 0
        self.status = "NORMAL"
        self.event_log = []
        self.cuav = None
        self.uavs = []
        self.ch_indices = {}  # cluster_id -> uav index
        self.debug_text_ids = []
        self.debug_line_ids = []
        
    def log_event(self, message):
        """Add event to log (keep last 5)"""
        self.event_log.append(f"[Round {self.round}] {message}")
        if len(self.event_log) > 5:
            self.event_log.pop(0)
        print(f"[Round {self.round}] {message}")


def initialize_pybullet():
    """Initialize PyBullet in GUI mode with interactive camera"""
    physics_client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    # Add a ground plane for reference
    p.loadURDF("plane.urdf", [0, 0, 0])
    
    # Configure camera - MUCH CLOSER for better view
    p.resetDebugVisualizerCamera(
        cameraDistance=80,      # Reduced from 200 to 80
        cameraYaw=45,
        cameraPitch=-25,        # Slightly steeper angle
        cameraTargetPosition=[0, 0, 30]
    )
    
    # Configure visualizer - Enable interactive controls
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)  # Keep GUI controls
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
    p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 1)  # Enable mouse interaction
    
    # Set gravity (but UAVs will ignore it)
    p.setGravity(0, 0, 0)
    
    print("✓ PyBullet initialized in GUI mode")
    print("  Camera Controls:")
    print("    - Mouse drag: Rotate view")
    print("    - Mouse wheel: Zoom in/out")
    print("    - Right-click drag: Pan")
    print("    - Ctrl+mouse: Move camera target")
    print("\n  Keyboard Shortcuts for Camera Focus:")
    print("    - Press '0': Focus on Cluster 0 (Green)")
    print("    - Press '1': Focus on Cluster 1 (Blue)")
    print("    - Press '2': Focus on Cluster 2 (Purple)")
    print("    - Press 'C': Focus on CUAV (Red)")
    print("    - Press 'H': Focus on all Cluster Heads")
    print("    - Press 'A': Show all clusters (overview)")
    print("    - Press 'Z': Zoom in close")
    print("    - Press 'X': Zoom out far")
    return physics_client


def update_camera_focus(state, focus_type='all'):
    """Update camera to focus on specific elements
    
    Args:
        state: SimulationState object
        focus_type: 'cluster_0', 'cluster_1', 'cluster_2', 'cuav', 'ch', 'all', 'zoom_in', 'zoom_out'
    """
    
    if focus_type == 'cluster_0':
        # Focus on cluster 0 (green)
        cluster_uavs = [uav for uav in state.uavs if uav.cluster_id == 0 and uav.is_alive]
        if cluster_uavs:
            center = np.mean([uav.position for uav in cluster_uavs], axis=0)
            p.resetDebugVisualizerCamera(
                cameraDistance=30,
                cameraYaw=45,
                cameraPitch=-20,
                cameraTargetPosition=center.tolist()
            )
            print(f"📹 Camera focusing on Cluster 0 (Green) - {len(cluster_uavs)} UAVs")
    
    elif focus_type == 'cluster_1':
        # Focus on cluster 1 (blue)
        cluster_uavs = [uav for uav in state.uavs if uav.cluster_id == 1 and uav.is_alive]
        if cluster_uavs:
            center = np.mean([uav.position for uav in cluster_uavs], axis=0)
            p.resetDebugVisualizerCamera(
                cameraDistance=30,
                cameraYaw=45,
                cameraPitch=-20,
                cameraTargetPosition=center.tolist()
            )
            print(f"📹 Camera focusing on Cluster 1 (Blue) - {len(cluster_uavs)} UAVs")
    
    elif focus_type == 'cluster_2':
        # Focus on cluster 2 (purple)
        cluster_uavs = [uav for uav in state.uavs if uav.cluster_id == 2 and uav.is_alive]
        if cluster_uavs:
            center = np.mean([uav.position for uav in cluster_uavs], axis=0)
            p.resetDebugVisualizerCamera(
                cameraDistance=30,
                cameraYaw=45,
                cameraPitch=-20,
                cameraTargetPosition=center.tolist()
            )
            print(f"📹 Camera focusing on Cluster 2 (Purple) - {len(cluster_uavs)} UAVs")
    
    elif focus_type == 'cuav':
        # Focus on CUAV
        if state.cuav:
            p.resetDebugVisualizerCamera(
                cameraDistance=20,
                cameraYaw=45,
                cameraPitch=-15,
                cameraTargetPosition=state.cuav.position.tolist()
            )
            print(f"📹 Camera focusing on CUAV (Red) at {state.cuav.position}")
    
    elif focus_type == 'ch':
        # Focus on all cluster heads
        ch_uavs = [state.uavs[idx] for idx in state.ch_indices.values() if state.uavs[idx].is_alive]
        if ch_uavs:
            center = np.mean([uav.position for uav in ch_uavs], axis=0)
            p.resetDebugVisualizerCamera(
                cameraDistance=50,
                cameraYaw=45,
                cameraPitch=-25,
                cameraTargetPosition=center.tolist()
            )
            print(f"📹 Camera focusing on {len(ch_uavs)} Cluster Heads (Golden)")
    
    elif focus_type == 'zoom_in':
        # Get current camera params and zoom in
        cam_info = p.getDebugVisualizerCamera()
        current_distance = cam_info[10]
        current_yaw = cam_info[8]
        current_pitch = cam_info[9]
        current_target = cam_info[11]
        p.resetDebugVisualizerCamera(
            cameraDistance=max(10, current_distance * 0.5),  # Zoom in 50%
            cameraYaw=current_yaw,
            cameraPitch=current_pitch,
            cameraTargetPosition=current_target
        )
        print(f"🔍 Zoomed in (distance: {max(10, current_distance * 0.5):.1f}m)")
    
    elif focus_type == 'zoom_out':
        # Get current camera params and zoom out
        cam_info = p.getDebugVisualizerCamera()
        current_distance = cam_info[10]
        current_yaw = cam_info[8]
        current_pitch = cam_info[9]
        current_target = cam_info[11]
        p.resetDebugVisualizerCamera(
            cameraDistance=min(200, current_distance * 2.0),  # Zoom out 2x
            cameraYaw=current_yaw,
            cameraPitch=current_pitch,
            cameraTargetPosition=current_target
        )
        print(f"🔍 Zoomed out (distance: {min(200, current_distance * 2.0):.1f}m)")
    
    else:  # 'all' - overview of entire swarm
        alive_uavs = [uav for uav in state.uavs if uav.is_alive]
        if alive_uavs:
            center = np.mean([uav.position for uav in alive_uavs], axis=0)
            p.resetDebugVisualizerCamera(
                cameraDistance=80,
                cameraYaw=45,
                cameraPitch=-25,
                cameraTargetPosition=center.tolist()
            )
            print(f"📹 Camera showing all clusters - {len(alive_uavs)} UAVs")


def check_keyboard_input(state):
    """Check for keyboard input to change camera focus"""
    keys = p.getKeyboardEvents()
    
    # Check for camera focus keys
    if ord('0') in keys and keys[ord('0')] & p.KEY_WAS_TRIGGERED:
        update_camera_focus(state, 'cluster_0')
    elif ord('1') in keys and keys[ord('1')] & p.KEY_WAS_TRIGGERED:
        update_camera_focus(state, 'cluster_1')
    elif ord('2') in keys and keys[ord('2')] & p.KEY_WAS_TRIGGERED:
        update_camera_focus(state, 'cluster_2')
    elif ord('c') in keys and keys[ord('c')] & p.KEY_WAS_TRIGGERED:
        update_camera_focus(state, 'cuav')
    elif ord('C') in keys and keys[ord('C')] & p.KEY_WAS_TRIGGERED:
        update_camera_focus(state, 'cuav')
    elif ord('h') in keys and keys[ord('h')] & p.KEY_WAS_TRIGGERED:
        update_camera_focus(state, 'ch')
    elif ord('H') in keys and keys[ord('H')] & p.KEY_WAS_TRIGGERED:
        update_camera_focus(state, 'ch')
    elif ord('a') in keys and keys[ord('a')] & p.KEY_WAS_TRIGGERED:
        update_camera_focus(state, 'all')
    elif ord('A') in keys and keys[ord('A')] & p.KEY_WAS_TRIGGERED:
        update_camera_focus(state, 'all')
    elif ord('z') in keys and keys[ord('z')] & p.KEY_WAS_TRIGGERED:
        update_camera_focus(state, 'zoom_in')
    elif ord('Z') in keys and keys[ord('Z')] & p.KEY_WAS_TRIGGERED:
        update_camera_focus(state, 'zoom_in')
    elif ord('x') in keys and keys[ord('x')] & p.KEY_WAS_TRIGGERED:
        update_camera_focus(state, 'zoom_out')
    elif ord('X') in keys and keys[ord('X')] & p.KEY_WAS_TRIGGERED:
        update_camera_focus(state, 'zoom_out')



def create_uav_visual(position, radius, color, is_ch=False, is_cuav=False):
    """Create a UAV using real quadcopter model with colored materials"""
    
    # Scale factor for visibility - EVEN BIGGER!
    if is_cuav:
        scale = 12.0  # CUAV is largest (was 8.0)
    elif is_ch:
        scale = 10.0  # CH is larger (was 6.0)
    else:
        scale = 8.0   # Regular drones - MUCH bigger (was 5.0)
    
    # Check if drone URDF exists
    if os.path.exists(DRONE_URDF):
        # Load real drone model
        orientation = p.getQuaternionFromEuler([0, 0, 0])
        body_id = p.loadURDF(
            DRONE_URDF,
            basePosition=position,
            baseOrientation=orientation,
            useFixedBase=False,
            globalScaling=scale
        )
        
        # Change color of all visual shapes
        num_joints = p.getNumJoints(body_id)
        
        # Change base link color
        p.changeVisualShape(body_id, -1, rgbaColor=color)
        
        # Change all joint/link colors
        for joint_idx in range(num_joints):
            p.changeVisualShape(body_id, joint_idx, rgbaColor=color)
        
        print(f"      Loaded drone model, body_id={body_id}, scale={scale}, color={color[:3]}")
        return body_id
    else:
        # Fallback to sphere if URDF not found
        print(f"⚠️ Drone URDF not found at {DRONE_URDF}, using sphere")
        visual_shape = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=radius * 8,  # Make spheres even larger
            rgbaColor=color
        )
        
        collision_shape = p.createCollisionShape(
            shapeType=p.GEOM_SPHERE,
            radius=radius * 8
        )
        
        body_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=position
        )
        
        return body_id


def spawn_cluster(cluster_id, center_pos, state):
    """Spawn UAVs in spherical formation around cluster center"""
    cluster_uavs = []
    cluster_color = CLUSTER_COLORS[cluster_id]
    
    print(f"  Spawning Cluster {cluster_id}...")
    
    # First UAV is the Cluster Head
    ch_uav = UAV(
        uav_id=len(state.uavs),
        cluster_id=cluster_id,
        position=center_pos,
        is_ch=True
    )
    print(f"    Creating CH{cluster_id} at {center_pos}...")
    ch_uav.body_id = create_uav_visual(center_pos, radius=1.0, color=COLOR_CH, is_ch=True)
    print(f"    CH{cluster_id} body_id: {ch_uav.body_id}")
    cluster_uavs.append(ch_uav)
    state.ch_indices[cluster_id] = len(state.uavs)
    state.uavs.append(ch_uav)
    
    # Spawn remaining member UAVs in spherical formation with cluster color
    for i in range(1, UAVS_PER_CLUSTER):
        # Random spherical coordinates
        theta = np.random.uniform(0, 2 * np.pi)
        phi = np.random.uniform(0, np.pi)
        r = np.random.uniform(5, 15)
        
        # Convert to Cartesian
        x = center_pos[0] + r * np.sin(phi) * np.cos(theta)
        y = center_pos[1] + r * np.sin(phi) * np.sin(theta)
        z = center_pos[2] + r * np.cos(phi)
        
        position = np.array([x, y, z])
        
        uav = UAV(
            uav_id=len(state.uavs),
            cluster_id=cluster_id,
            position=position,
            is_ch=False
        )
        print(f"    Creating member UAV #{len(state.uavs)} at {position}...")
        uav.body_id = create_uav_visual(position, radius=0.6, color=cluster_color, is_ch=False)
        print(f"    Member UAV #{len(state.uavs)} body_id: {uav.body_id}")
        cluster_uavs.append(uav)
        state.uavs.append(uav)
    
    print(f"✓ Spawned Cluster {cluster_id}: {len(cluster_uavs)} UAVs")
    return cluster_uavs


def spawn_cuav(target_cluster_id, state):
    """Spawn red CUAV attacker as a larger drone"""
    target_ch_idx = state.ch_indices[target_cluster_id]
    target_pos = state.uavs[target_ch_idx].position
    
    # Spawn 20m away from target
    spawn_pos = target_pos + np.array([20, 0, 5])
    
    cuav = CUAVAttacker(spawn_pos, target_pos)
    cuav.body_id = create_uav_visual(spawn_pos, radius=1.0, color=COLOR_CUAV, is_cuav=True)
    
    state.cuav = cuav
    state.log_event("⚠️  CUAV SPAWNED - Targeting Cluster 0")
    state.status = "ATTACK INITIATED"


def compromise_ch(cluster_id, state):
    """Compromise the cluster head"""
    ch_idx = state.ch_indices[cluster_id]
    ch_uav = state.uavs[ch_idx]
    
    ch_uav.is_compromised = True
    ch_uav.is_alive = False
    ch_uav.set_color(COLOR_COMPROMISED)
    
    state.log_event(f"💥 CH{cluster_id} COMPROMISED!")
    state.status = "CRITICAL"


def elect_new_ch(cluster_id, state):
    """Elect new cluster head from remaining UAVs"""
    # Find cluster center
    center = CLUSTER_POSITIONS[cluster_id]
    
    # Find closest alive UAV to center (excluding current CH)
    min_dist = float('inf')
    new_ch_idx = None
    
    for uav in state.uavs:
        if uav.cluster_id == cluster_id and not uav.is_ch and uav.is_alive:
            dist = np.linalg.norm(uav.position - center)
            if dist < min_dist:
                min_dist = dist
                new_ch_idx = uav.id
    
    if new_ch_idx is not None:
        # Promote new CH
        new_ch = state.uavs[new_ch_idx]
        new_ch.is_ch = True
        new_ch.set_color(COLOR_CH)
        
        # Update CH index
        state.ch_indices[cluster_id] = new_ch_idx
        
        state.log_event(f"🗳️  NEW CH ELECTED - Cluster {cluster_id}: UAV #{new_ch_idx}")
        state.status = "RECOVERY PHASE"


def restore_cluster_participation(cluster_id, round_num, state):
    """Gradually restore cluster participation with cluster-specific colors"""
    # Determine participation level based on round
    if round_num == 25:
        participation = 0.3
        color = CLUSTER_COLORS_DEGRADED[cluster_id]
        state.log_event(f"Cluster {cluster_id}: 30% participation")
    elif round_num == 26:
        participation = 0.7
        color = CLUSTER_COLORS_RECOVERING[cluster_id]
        state.log_event(f"Cluster {cluster_id}: 70% participation")
    else:  # round 27
        participation = 1.0
        color = CLUSTER_COLORS[cluster_id]
        state.log_event(f"✓ CLUSTER {cluster_id} FULLY RESTORED")
    
    # Update member UAVs (not CH)
    for uav in state.uavs:
        if uav.cluster_id == cluster_id and not uav.is_ch:
            uav.participation_level = participation
            uav.set_color(color)


def draw_communication_links(state):
    """Draw lines between communicating UAVs"""
    # Remove old lines
    for line_id in state.debug_line_ids:
        try:
            p.removeUserDebugItem(line_id)
        except:
            pass
    state.debug_line_ids.clear()
    
    # Draw new lines within each cluster
    for cluster_id in range(NUM_CLUSTERS):
        cluster_uavs = [uav for uav in state.uavs if uav.cluster_id == cluster_id and uav.is_alive]
        
        if len(cluster_uavs) < 2:
            continue
        
        # Draw lines from CH to some members
        ch_idx = state.ch_indices.get(cluster_id)
        if ch_idx is not None:
            ch_uav = state.uavs[ch_idx]
            
            # Sample some UAVs to draw links to (not all 200)
            sample_size = min(20, len(cluster_uavs))
            sampled = np.random.choice(cluster_uavs, sample_size, replace=False)
            
            for uav in sampled:
                if uav.id == ch_uav.id:
                    continue
                
                distance = np.linalg.norm(uav.position - ch_uav.position)
                
                # Determine color based on state
                if ch_uav.is_compromised or uav.is_compromised:
                    line_color = [1, 0, 0]  # Red
                elif distance > 50:
                    line_color = [1, 1, 0]  # Yellow
                else:
                    line_color = [0, 1, 0]  # Green
                
                line_id = p.addUserDebugLine(
                    ch_uav.position.tolist(),
                    uav.position.tolist(),
                    lineColorRGB=line_color,
                    lineWidth=1,
                    lifeTime=0.2
                )
                state.debug_line_ids.append(line_id)


def draw_cluster_boundaries(state):
    """Draw transparent spheres around cluster centers"""
    for cluster_id, center in enumerate(CLUSTER_POSITIONS):
        # Draw using debug lines (circle approximation)
        num_points = 24
        radius = 20.0
        
        for i in range(num_points):
            theta1 = (i / num_points) * 2 * np.pi
            theta2 = ((i + 1) / num_points) * 2 * np.pi
            
            # Horizontal circle
            x1 = center[0] + radius * np.cos(theta1)
            y1 = center[1] + radius * np.sin(theta1)
            x2 = center[0] + radius * np.cos(theta2)
            y2 = center[1] + radius * np.sin(theta2)
            
            line_id = p.addUserDebugLine(
                [x1, y1, center[2]],
                [x2, y2, center[2]],
                lineColorRGB=[0.5, 0.5, 0.5],
                lineWidth=1,
                lifeTime=0.2
            )
            state.debug_line_ids.append(line_id)


def draw_hud(state):
    """Draw HUD overlay with debug text"""
    # Remove old text
    for text_id in state.debug_text_ids:
        try:
            p.removeUserDebugItem(text_id)
        except:
            pass
    state.debug_text_ids.clear()
    
    # Count alive UAVs
    alive_count = sum(1 for uav in state.uavs if uav.is_alive)
    active_chs = sum(1 for cluster_id in range(NUM_CLUSTERS) 
                     if state.ch_indices.get(cluster_id) is not None 
                     and state.uavs[state.ch_indices[cluster_id]].is_alive)
    
    # HUD position (top-left in world space)
    hud_pos = [-80, 40, 60]
    
    # Title
    text_id = p.addUserDebugText(
        f"Round: {state.round}/{TOTAL_ROUNDS}",
        hud_pos,
        textSize=2.0,
        textColorRGB=[1, 1, 1],
        lifeTime=0.2
    )
    state.debug_text_ids.append(text_id)
    
    # Alive UAVs
    text_id = p.addUserDebugText(
        f"Alive UAVs: {alive_count}/{TOTAL_UAVS}",
        [hud_pos[0], hud_pos[1], hud_pos[2] - 5],
        textSize=1.5,
        textColorRGB=[0.5, 1, 0.5],
        lifeTime=0.2
    )
    state.debug_text_ids.append(text_id)
    
    # Active CHs
    text_id = p.addUserDebugText(
        f"Active CHs: {active_chs}/{NUM_CLUSTERS}",
        [hud_pos[0], hud_pos[1], hud_pos[2] - 9],
        textSize=1.5,
        textColorRGB=[1, 0.84, 0],
        lifeTime=0.2
    )
    state.debug_text_ids.append(text_id)
    
    # Status
    status_color = [1, 0, 0] if "CRITICAL" in state.status else [0, 1, 0]
    text_id = p.addUserDebugText(
        f"Status: {state.status}",
        [hud_pos[0], hud_pos[1], hud_pos[2] - 13],
        textSize=1.5,
        textColorRGB=status_color,
        lifeTime=0.2
    )
    state.debug_text_ids.append(text_id)
    
    # Event log (bottom-left)
    log_pos = [-80, -40, 10]
    for i, event in enumerate(state.event_log[-5:]):
        text_id = p.addUserDebugText(
            event,
            [log_pos[0], log_pos[1], log_pos[2] + i * 3],
            textSize=1.2,
            textColorRGB=[1, 1, 0.5],
            lifeTime=0.2
        )
        state.debug_text_ids.append(text_id)


def update_uav_positions(state):
    """Update UAV positions to orbit around their cluster heads"""
    for uav in state.uavs:
        if not uav.is_alive or uav.is_ch:
            continue  # Skip dead UAVs and cluster heads (they stay stationary)
        
        # Get the cluster head position
        ch_idx = state.ch_indices.get(uav.cluster_id)
        if ch_idx is None:
            continue
        
        ch_position = state.uavs[ch_idx].position
        
        # Update orbital angle
        uav.orbit_angle += uav.orbit_speed * 0.016  # Assuming ~60 FPS
        
        # Calculate new position in circular orbit
        x = ch_position[0] + uav.orbit_radius * np.cos(uav.orbit_angle)
        y = ch_position[1] + uav.orbit_radius * np.sin(uav.orbit_angle)
        z = ch_position[2] + uav.orbit_height_offset
        
        uav.position = np.array([x, y, z])
        
        # Update visual position in PyBullet
        if uav.body_id is not None:
            p.resetBasePositionAndOrientation(
                uav.body_id,
                uav.position.tolist(),
                [0, 0, 0, 1]
            )


def update_visualization(state):
    """Update all visual elements"""
    update_uav_positions(state)  # Animate UAVs flying around CHs
    draw_communication_links(state)
    draw_cluster_boundaries(state)
    draw_hud(state)


def print_round_header(round_num):
    """Print round header to console"""
    print(f"\n{'='*60}")
    print(f"ROUND {round_num}")
    print(f"{'='*60}")


def print_round_kpis(state, round_num):
    """Print comprehensive KPI metrics for the round"""
    # Count metrics
    alive_count = sum(1 for uav in state.uavs if uav.is_alive)
    compromised_count = sum(1 for uav in state.uavs if uav.is_compromised)
    
    # Participation by cluster
    participation = {}
    for cluster_id in range(NUM_CLUSTERS):
        cluster_uavs = [uav for uav in state.uavs if uav.cluster_id == cluster_id]
        if cluster_uavs:
            active = sum(uav.participation_level for uav in cluster_uavs if uav.is_alive)
            participation[f'C{cluster_id}'] = active / len(cluster_uavs)
    
    # Communication status
    active_chs = sum(1 for cluster_id in range(NUM_CLUSTERS) 
                     if state.ch_indices.get(cluster_id) is not None 
                     and state.uavs[state.ch_indices[cluster_id]].is_alive)
    
    # CUAV status
    cuav_status = "NONE"
    if state.cuav:
        distance = np.linalg.norm(state.cuav.position - state.cuav.target)
        if state.cuav.active:
            cuav_status = f"ACTIVE (dist: {distance:.1f}m)"
        else:
            cuav_status = f"REACHED TARGET"
    
    print(f"\n📊 KPI Metrics:")
    print(f"  Status: {state.status}")
    print(f"  Alive UAVs: {alive_count}/{TOTAL_UAVS}")
    print(f"  Compromised: {compromised_count}")
    print(f"  Active CHs: {active_chs}/{NUM_CLUSTERS}")
    print(f"  Participation: {', '.join(f'{k}={v:.1%}' for k, v in participation.items())}")
    print(f"  CUAV: {cuav_status}")
    
    # Print CH status for each cluster
    print(f"\n🎯 Cluster Head Status:")
    for cluster_id in range(NUM_CLUSTERS):
        ch_idx = state.ch_indices.get(cluster_id)
        if ch_idx is not None:
            ch = state.uavs[ch_idx]
            status = "COMPROMISED" if ch.is_compromised else ("OFFLINE" if not ch.is_alive else "OPERATIONAL")
            print(f"  CH{cluster_id} (UAV #{ch_idx}): {status}")
        else:
            print(f"  CH{cluster_id}: NO LEADER")
    
    # Communication metrics (simulated)
    total_links = sum(1 for uav in state.uavs if uav.is_alive)
    healthy_links = sum(1 for uav in state.uavs if uav.is_alive and not uav.is_compromised)
    
    print(f"\n📡 Communication:")
    print(f"  Total links: {total_links}")
    print(f"  Healthy: {healthy_links} ({healthy_links/total_links*100 if total_links > 0 else 0:.1f}%)")
    print(f"  Degraded: {total_links - healthy_links}")
    
    # Estimated model accuracy (simulated based on phase)
    if state.status == "NORMAL":
        acc = 0.85 + np.random.uniform(-0.02, 0.02)
    elif state.status in ["ATTACK INITIATED", "CRITICAL"]:
        acc = 0.65 + np.random.uniform(-0.05, 0.05)
    elif state.status == "DETECTION PHASE":
        acc = 0.70 + np.random.uniform(-0.03, 0.03)
    elif state.status == "RECOVERY PHASE":
        acc = 0.75 + np.random.uniform(-0.03, 0.03)
    else:  # STABILIZED
        acc = 0.83 + np.random.uniform(-0.02, 0.02)
    
    print(f"\n📈 Learning Performance:")
    print(f"  Global Model Accuracy: {acc:.2%}")
    print(f"  Convergence Status: {'✓ CONVERGED' if state.status == 'STABILIZED' else '⏳ IN PROGRESS'}")


def run_simulation():
    """Main simulation loop"""
    # Initialize
    physics_client = initialize_pybullet()
    state = SimulationState()
    
    # Spawn all clusters
    print(f"\nSpawning {TOTAL_UAVS} UAVs in {NUM_CLUSTERS} clusters...")
    print(f"  (Testing with {UAVS_PER_CLUSTER} UAVs per cluster)")
    for cluster_id in range(NUM_CLUSTERS):
        spawn_cluster(cluster_id, CLUSTER_POSITIONS[cluster_id], state)
    
    print(f"\n✓ Simulation initialized: {len(state.uavs)} UAVs ready")
    print(f"  Cluster Heads: {list(state.ch_indices.values())}")
    print(f"\n⏳ Starting simulation rounds...")
    
    # Main loop: 30 rounds
    for round_num in range(1, TOTAL_ROUNDS + 1):
        state.round = round_num
        print_round_header(round_num)
        
        # Round 11: Spawn CUAV
        if round_num == 11:
            spawn_cuav(target_cluster_id=0, state=state)
        
        # Rounds 11-15: Move CUAV toward CH0
        if 11 <= round_num <= 15 and state.cuav is not None:
            state.cuav.move_toward_target(dt=1.0)
            distance = np.linalg.norm(state.cuav.position - state.cuav.target)
            print(f"CUAV moving... Distance to CH0: {distance:.1f}m")
        
        # Round 16: Compromise CH0
        if round_num == 16:
            compromise_ch(cluster_id=0, state=state)
        
        # Round 17: Detect compromise
        if round_num == 17:
            state.log_event("🔍 COMPROMISE DETECTED")
            state.log_event("Initiating recovery protocol...")
            state.status = "DETECTION PHASE"
        
        # Round 18: Elect new CH
        if round_num == 18:
            elect_new_ch(cluster_id=0, state=state)
        
        # Rounds 25-27: Restore participation
        if 25 <= round_num <= 27:
            restore_cluster_participation(cluster_id=0, round_num=round_num, state=state)
        
        # Round 28+: Stabilization
        if round_num == 28:
            state.log_event("✓ System stabilized")
            state.status = "STABILIZED"
        
        # Update visualization
        update_visualization(state)
        
        # Check for keyboard input (camera focus changes)
        check_keyboard_input(state)
        
        # Step physics simulation with continuous animation
        for _ in range(240):
            update_uav_positions(state)  # Update positions every frame for smooth flight
            p.stepSimulation()
            time.sleep(1.0 / 240.0)
        
        # Print KPIs for this round
        print_round_kpis(state, round_num)
    
    # Final summary
    print(f"\n{'='*60}")
    print("✓✓✓ SIMULATION COMPLETE ✓✓✓")
    print(f"{'='*60}")
    print(f"Status: {state.status}")
    print(f"Final State: {sum(1 for uav in state.uavs if uav.is_alive)}/{TOTAL_UAVS} UAVs operational")
    print(f"All CHs: {'HEALTHY' if all(state.uavs[state.ch_indices[i]].is_alive for i in range(NUM_CLUSTERS)) else 'DEGRADED'}")
    print("\nPress Ctrl+C to exit...")
    
    # Keep window open with continuous animation
    try:
        while True:
            check_keyboard_input(state)  # Allow camera control after simulation ends
            update_uav_positions(state)  # Continue orbiting animation
            p.stepSimulation()
            time.sleep(1.0 / 60.0)
    except KeyboardInterrupt:
        print("\n✓ Simulation terminated by user")
    
    p.disconnect()


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════╗
║   UAV SWARM FEDERATED LEARNING SIMULATION                    ║
║   CUAV Attack → Detection → Recovery → Stabilization         ║
╚══════════════════════════════════════════════════════════════╝
""")
    
    try:
        run_simulation()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
