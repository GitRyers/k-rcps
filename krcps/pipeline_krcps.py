from typing import Callable, Iterable
import torch
from .calibration import _calibrate_k_rcps
from diffusers import DiffusionPipeline


class KRCPSPipeline(DiffusionPipeline):
    def __init__(self, unet, scheduler):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)

    @torch.no_grad()
    def __call__(self, 
        cal_set: torch.Tensor,
        I: Callable[[torch.Tensor], torch.Tensor],
        bound_name: str,
        epsilon: float,
        delta: float,
        lambda_max: float,
        stepsize: torch.Tensor,
        k: int,
        membership_name: str,
        n_opt: int,
        prob_size: float,
        gamma: Iterable[float],
    ):
        return _calibrate_k_rcps(cal_set, I, bound_name, epsilon, delta, lambda_max, stepsize, 
                          k, membership_name, n_opt, prob_size, gamma) 