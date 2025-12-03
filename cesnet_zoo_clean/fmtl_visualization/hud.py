"""
HUD (Heads-Up Display) Overlay for PyBullet
Shows round info, phase, accuracies, and cluster status
"""

import pybullet as p
from typing import Dict, Optional


class HUDOverlay:
    """Manages on-screen text display for simulation status"""
    
    def __init__(self):
        self.text_items = []  # List of PyBullet debug text IDs
        self.current_data = {}
    
    def update(self, round_num: int, phase: str, accuracies: Optional[Dict[str, float]] = None,
               cluster_statuses: Optional[Dict[int, str]] = None,
               communication_stats: Optional[Dict[str, any]] = None):
        """
        Update HUD with current simulation data
        
        Args:
            round_num: Current training round
            phase: Current phase (NORMAL, COMPROMISED, DRE, CONTINUITY, STABILIZATION)
            accuracies: Dict with keys 'traffic', 'duration', 'bandwidth'
            cluster_statuses: Dict with cluster status strings
            communication_stats: Dict with data transfer info
        """
        # Remove old text
        self.clear()
        
        # Position for text (top-left corner)
        text_position = [-50, -30, 50]
        line_height = 3
        
        # Build HUD content
        lines = []
        
        # Round and Phase
        phase_colors = {
            'NORMAL': [0, 1, 0],  # Green
            'COMPROMISED': [1, 0.5, 0],  # Orange
            'DRE': [1, 0, 0],  # Red
            'CONTINUITY': [1, 1, 0],  # Yellow
            'STABILIZATION': [0, 1, 1],  # Cyan
        }
        phase_color = phase_colors.get(phase, [1, 1, 1])
        
        lines.append(f"=== FMTL HIERARCHICAL FEDERATION ===")
        lines.append(f"Round: {round_num}")
        lines.append(f"Phase: {phase}")
        lines.append("")
        
        # Accuracies
        if accuracies:
            lines.append("Per-Task Test Accuracies:")
            lines.append(f"  Traffic:    {accuracies.get('traffic', 0):.2%}")
            lines.append(f"  Duration:   {accuracies.get('duration', 0):.2%}")
            lines.append(f"  Bandwidth:  {accuracies.get('bandwidth', 0):.2%}")
            lines.append("")
        
        # Cluster Statuses
        if cluster_statuses:
            lines.append("Cluster Status:")
            for cluster_id in sorted(cluster_statuses.keys()):
                status = cluster_statuses[cluster_id]
                lines.append(f"  Cluster {cluster_id}: {status}")
            lines.append("")
        
        # Communication Stats
        if communication_stats:
            lines.append("Communication:")
            if 'round_data' in communication_stats:
                lines.append(f"  This round: {communication_stats['round_data']:.2f} MB")
            if 'total_data' in communication_stats:
                lines.append(f"  Total: {communication_stats['total_data']:.2f} GB")
        
        # Render text lines
        for i, line in enumerate(lines):
            pos = [text_position[0], text_position[1], text_position[2] - i * line_height]
            
            # Use phase color for phase line, white for others
            if "Phase:" in line:
                color = phase_color
            else:
                color = [1, 1, 1]
            
            text_id = p.addUserDebugText(
                text=line,
                textPosition=pos,
                textColorRGB=color,
                textSize=1.2
            )
            self.text_items.append(text_id)
        
        self.current_data = {
            'round': round_num,
            'phase': phase,
            'accuracies': accuracies,
            'cluster_statuses': cluster_statuses,
            'communication_stats': communication_stats,
        }
    
    def clear(self):
        """Remove all text from display"""
        for text_id in self.text_items:
            p.removeUserDebugItem(text_id)
        self.text_items = []
    
    def add_message(self, message: str, duration: int = 60, color: list = None):
        """
        Add a temporary message to the display
        
        Args:
            message: Text to display
            duration: Number of steps to show (approximate)
            color: RGB color [r, g, b], defaults to yellow
        """
        if color is None:
            color = [1, 1, 0]
        
        # Display in center-top
        pos = [0, 0, 60]
        
        text_id = p.addUserDebugText(
            text=message,
            textPosition=pos,
            textColorRGB=color,
            textSize=2.0,
            lifeTime=duration / 240.0  # PyBullet time is in seconds, assuming 240 Hz
        )
        # Don't add to text_items since it auto-expires
    
    def show_attack_warning(self):
        """Show special warning for attack event"""
        self.add_message("⚠️ CUAV ATTACK IN PROGRESS ⚠️", duration=120, color=[1, 0, 0])
    
    def show_detection_message(self):
        """Show message for attack detection"""
        self.add_message("🚨 ATTACK DETECTED - INITIATING D&R-E 🚨", duration=120, color=[1, 0.5, 0])
    
    def show_recovery_message(self, participation: float):
        """Show cluster recovery progress"""
        msg = f"♻️ Cluster 0 Recovery: {participation:.0%} Participation"
        self.add_message(msg, duration=60, color=[0, 1, 1])
