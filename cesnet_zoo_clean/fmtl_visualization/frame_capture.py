"""
Frame Capture for creating GIFs and videos
Export attack sequences and key moments
"""

import pybullet as p
import numpy as np
from PIL import Image
import os
from typing import List, Optional


class FrameCapture:
    """Captures frames from PyBullet for export"""
    
    def __init__(self, output_dir: str = "./frames", width: int = 1280, height: int = 720):
        self.output_dir = output_dir
        self.width = width
        self.height = height
        self.frames = []
        self.frame_count = 0
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"✓ Frame capture initialized: {width}x{height} -> {output_dir}")
    
    def capture_frame(self, save_to_disk: bool = False) -> np.ndarray:
        """
        Capture current frame from PyBullet
        
        Args:
            save_to_disk: If True, save frame as PNG
        
        Returns:
            RGB image as numpy array
        """
        # Get camera view matrix
        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[0, 0, 30],
            distance=150,
            yaw=0,
            pitch=-30,
            roll=0,
            upAxisIndex=2
        )
        
        # Get projection matrix
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=self.width / self.height,
            nearVal=0.1,
            farVal=300.0
        )
        
        # Render image
        (_, _, px, _, _) = p.getCameraImage(
            width=self.width,
            height=self.height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )
        
        # Convert to RGB numpy array
        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = rgb_array[:, :, :3]  # Remove alpha channel
        
        # Optionally save
        if save_to_disk:
            img = Image.fromarray(rgb_array)
            filename = os.path.join(self.output_dir, f"frame_{self.frame_count:05d}.png")
            img.save(filename)
            self.frame_count += 1
        
        self.frames.append(rgb_array)
        return rgb_array
    
    def capture_attack_sequence(self, scene, attacker, start_round: int, duration: int = 20):
        """
        Capture sequence of frames during attack
        
        Args:
            scene: FMTLScene object
            attacker: CUAVAttacker object
            start_round: Round to start capturing
            duration: Number of steps to capture
        """
        print(f"📹 Capturing attack sequence: {duration} frames")
        
        attack_frames = []
        for step in range(duration):
            # Update attacker
            attacker.update(start_round + step)
            
            # Capture frame
            frame = self.capture_frame(save_to_disk=True)
            attack_frames.append(frame)
            
            # Step simulation
            scene.step()
        
        print(f"✓ Captured {len(attack_frames)} frames")
        return attack_frames
    
    def export_gif(self, output_filename: str, fps: int = 10, frames: Optional[List[np.ndarray]] = None):
        """
        Export captured frames as GIF
        
        Args:
            output_filename: Output GIF filename
            fps: Frames per second
            frames: List of frames to export (default: all captured)
        """
        if frames is None:
            frames = self.frames
        
        if not frames:
            print("⚠️ No frames to export")
            return
        
        # Convert to PIL Images
        pil_frames = [Image.fromarray(frame) for frame in frames]
        
        # Save as GIF
        output_path = os.path.join(self.output_dir, output_filename)
        pil_frames[0].save(
            output_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=int(1000 / fps),
            loop=0
        )
        
        print(f"✓ GIF exported: {output_path} ({len(frames)} frames @ {fps} fps)")
    
    def export_video(self, output_filename: str, fps: int = 30, frames: Optional[List[np.ndarray]] = None):
        """
        Export captured frames as MP4 video (requires opencv)
        
        Args:
            output_filename: Output MP4 filename
            fps: Frames per second
            frames: List of frames to export (default: all captured)
        """
        try:
            import cv2
        except ImportError:
            print("⚠️ OpenCV not installed. Install with: pip install opencv-python")
            return
        
        if frames is None:
            frames = self.frames
        
        if not frames:
            print("⚠️ No frames to export")
            return
        
        # Setup video writer
        output_path = os.path.join(self.output_dir, output_filename)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (self.width, self.height))
        
        # Write frames
        for frame in frames:
            # Convert RGB to BGR for OpenCV
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(bgr_frame)
        
        out.release()
        print(f"✓ Video exported: {output_path} ({len(frames)} frames @ {fps} fps)")
    
    def clear_frames(self):
        """Clear stored frames from memory"""
        self.frames = []
        self.frame_count = 0
    
    def capture_comparison_frame(self, scene1, scene2, labels: tuple = ("Convergence", "Transient")):
        """
        Capture side-by-side comparison frame from two scenes
        
        Args:
            scene1: First FMTLScene
            scene2: Second FMTLScene
            labels: Tuple of labels for each scene
        """
        # TODO: Implement side-by-side capture
        # This would require rendering two different camera views
        pass
