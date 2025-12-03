"""
FMTL Visualization Package for PyBullet
Hierarchical Federated Multi-Task Learning with CUAV Attack Simulation
"""

from .scenario import Scenario, ScenarioConfig
from .scene import FMTLScene
from .attack import CUAVAttacker
from .ch_election import CHElection
from .hud import HUDOverlay
from .comparison import ComparisonView
from .frame_capture import FrameCapture
from .inference import ModelLoader, UAVInference, FMTLInferenceEngine, create_inference_engine

__all__ = [
    'Scenario',
    'ScenarioConfig',
    'FMTLScene',
    'CUAVAttacker',
    'CHElection',
    'HUDOverlay',
    'ComparisonView',
    'FrameCapture',
    'ModelLoader',
    'UAVInference',
    'FMTLInferenceEngine',
    'create_inference_engine',
]

__version__ = '1.0.0'
