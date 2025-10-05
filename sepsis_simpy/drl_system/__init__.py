"""
DRL (Deep Reinforcement Learning) System Package
Contains Sepsis-Aware DRL agents for intelligent edge/cloud offloading decisions
"""

from .single_edge_drl_agent import SepsisAwareDRLAgent

# Backward compatibility
SingleEdgeDRLAgent = SepsisAwareDRLAgent

__all__ = [
    'SepsisAwareDRLAgent',
    'SingleEdgeDRLAgent'  # For backward compatibility
]