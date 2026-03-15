"""Multitimescale XOR synthetic task.

This module is intentionally thin: the actual batch generator lives in
`src.common.long_term_mem_driver` (adapted from Origin/DH-SNN-main).
"""

from src.common.long_term_mem_driver import generate_multiscale_xor_batch

__all__ = ["generate_multiscale_xor_batch"]
