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
    gpus_list: Optional[List[int]] = field(default_factory=lambda: [0, 1])
    """List of available GPUs to use (set to [0, 1])"""
    attack_config: Optional["AttackConfig"] = None

    def __post_init__(self):
       if len(self.methods) == 0:
           raise ValueError("At least one method should be specified.")


@dataclass
class AttackConfig(Serializable):
    """
    Configuration for the attack.
    """
    num_transformations: Optional[int] = 20
    """Number of transformations to apply to the image"""
    eps_list: Optional[List[float]] = field(default_factory=lambda: [1/255, 1/255, 1/255])
    """Limit on perturbation size (Linf norm) for each rinsing round"""
    n_iters: Optional[List[int]] = field(default_factory=lambda: [5, 5, 5])
    """Number of iterations to run for attack, for each rinsing round"""
    step_size_alpha: Optional[float] = 1.0
    """Step-size to use for the attack"""
    decoding_models: Optional[List[str]] = field(default_factory=lambda: ["stabilityai/stable-diffusion-2-1", "openai/consistency-decoder", "stabilityai/stable-diffusion-2-1"])
