"""
    Definitions for configurations.
"""

from dataclasses import dataclass
from typing import Optional, List
from simple_parsing.helpers import Serializable, field, choice


@dataclass
class ExperimentConfig(Serializable):
    """
    Basic experimental configuration.
    """
    submission: str
    """Name for submission attempt."""
    methods: List[str]
    """Methods to use for watermark removal. Applied sequentially."""
    track: str = field(choice(["black", "biege", "test"]))
    """Which track to target?"""
    aggregation: str = field(choice(["mean", "random"]))
    """Aggregation method (valid if attack generates multiple perturbations)."""
    skip_zip: Optional[bool] = False
    """Skip zipping the submission folder?"""

    def __post_init__(self):
       if len(self.methods) == 0:
           raise ValueError("At least one method should be specified.")


@dataclass
class AttackConfig(Serializable):
    """
    Configuration for the attack.
    """
    eps: float = 1/255
    """Limit on perturbation size (Linf norm)"""
    n_iters: int = 50
    """Number of iterations to run for attack"""
    step_size_alpha: float = 10.0
    """Step-size to use for the attack"""
