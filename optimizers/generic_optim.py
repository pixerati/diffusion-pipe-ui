# Implementation taken from https://github.com/timmytonga/sn-sm
# Modified to automatically do Kahan summation for bfloat16 parameters.
# Made resuming from checkpoint work, but ONLY for the svd case.
# Muon method taken from: https://github.com/KellerJordan/Muon

from typing import Callable, Iterable, Tuple
import math
from .projectors.svd_projector import SVDProjector
from .projectors.uniform_projector import UniformProjector  # get random subset
from .projectors.topk_norm_projector import TopKNormProjector  # topk indices

import torch
from torch.optim import Optimizer

from transformers.utils.versions import require_version


NS_STEPS = 5


def has_inf_or_nan(x):
    s = x.sum()
    return s.isinf() or s.isnan()


def get_and_update_subset_norm_denom(group, state, grad, beta2):
    # First, compute subset norm if applicable
    if "subset_size" in group:
        if group.get('correct_dim', False):
            reduce_fn = torch.mean
        else:
            reduce_fn = torch.sum
        if group["subset_size"] == "heuristics":  # heuristics
            if "reduce_dim" not in state:
                state["reduce_dim"] = 0 if grad.shape[0] >= grad.shape[1] else 1
            second_moment_update = reduce_fn(grad ** 2, dim=(1 - state["reduce_dim"]), keepdim=True)
        else:  # it is an int
            assert group["subset_size"] != 0, f"Subset size should not be 0."
            if "subset_shape" not in state:
                numel = grad.numel()
                if group["subset_size"] > 0:
                    reduce_size = closest_smaller_divisor_of_n_to_k(numel, group["subset_size"])
                else:  # default is sqrt
                    div = abs(int(group["subset_size"]))
                    reduce_size = closest_smaller_divisor_of_n_to_k(numel, int(math.sqrt(numel) / div))
                state["subset_shape"] = (numel // reduce_size, reduce_size)
            reshaped_grad = grad.view(state["subset_shape"])
            second_moment_update = reduce_fn(reshaped_grad ** 2, dim=1, keepdim=True)
    else:  # standard EMA
        second_moment_update = grad ** 2

    # Initialization
    if "exp_avg_sq" not in state:
        state["exp_avg_sq"] = torch.zeros_like(second_moment_update)
    exp_avg_sq = state["exp_avg_sq"]

    # Second moment term update
    if beta2 < 1:  # EMA
        exp_avg_sq.mul_(beta2).add_(second_moment_update, alpha=1.0 - beta2)
    else:  # AdaGrad
        exp_avg_sq.add_(second_moment_update)
    return exp_avg_sq.sqrt().add_(group["eps"])


def get_and_update_subspace_momentum(group, state, p):
    grad = p.grad
    beta1, beta2 = group["betas"]

    # Projection for compressing momentum term
    if "rank" in group:
        proj_grad = get_projected_grad(group, state, p)
    else:  # if not SM or module is not set then it's just standard momentum
        proj_grad = grad

    # Init
    if "exp_avg" not in state:
        state["exp_avg"] = torch.zeros_like(proj_grad)
    # Momentum term
    exp_avg = state["exp_avg"]

    # reset exp_avg state when we update as default
    if ("rank" in group and state["step"] > 1 and state["step"] % group["update_proj_gap"] == 0):
        if "overlap_state" not in group:
            state["exp_avg"] = torch.zeros_like(proj_grad)
        # else we overlap the momentum update where we don't need to do anything

    # Subspace momentum and orthogonal SGD
    if "rank" in group:
        exp_avg.mul_(beta1).add_(proj_grad, alpha=(1.0 - beta1))
        orth_comp = grad - state["projector"].project_back(proj_grad)
        numerator = state["projector"].project_back(exp_avg) + orth_comp
    else:  # just normal full momentum
        exp_avg.mul_(beta1).add_(proj_grad, alpha=(1.0 - beta1))
        numerator = exp_avg

    return numerator


def get_projected_grad(group, state, p):
    if "projector" not in state:
        state["projector"] = get_projector(group, p)
    return state["projector"].project(p.grad)


def get_projector(group, p):
    rank = group["rank"]
    update_proj_gap = group.get("update_proj_gap", 1)
    projector_type = group.get("projector_type", "svd")
    
    if projector_type == "svd":
        return SVDProjector(rank, update_proj_gap)
    elif projector_type == "uniform":
        return UniformProjector(rank, update_proj_gap)
    elif projector_type == "topk_norm":
        return TopKNormProjector(rank, update_proj_gap)
    else:
        raise ValueError(f"Unknown projector type: {projector_type}")


def closest_smaller_divisor_of_n_to_k(n: int, k: int) -> int:
    """
    Find the closest smaller divisor of n that is less than or equal to k.
    """
    if k >= n:
        return n
    
    # Find all divisors of n
    divisors = []
    for i in range(1, int(math.sqrt(n)) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    
    # Sort divisors and find the largest one <= k
    divisors.sort()
    for divisor in reversed(divisors):
        if divisor <= k:
            return divisor
    
    return 1


def zeropower_via_newtonschulz5(G, steps: int):
    """
    Compute G^(-1/2) using Newton-Schulz iteration.
    """
    if steps <= 0:
        return G
    
    # Initialize
    Y = G
    Z = torch.eye(G.shape[0], device=G.device, dtype=G.dtype)
    
    for _ in range(steps):
        Y_new = 0.5 * (3 * Z - Y @ Z @ Z)
        Z_new = 0.5 * (3 * Y - Z @ Y @ Y)
        Y = Y_new
        Z = Z_new
    
    return Y


class GenericOptim(Optimizer):
    """
    Generic optimizer that supports various optimization methods including
    Subspace Momentum (SM), AdaGrad, and standard momentum methods.
    """
    
    def __init__(
            self,
            params: Iterable,
            lr: float = 1e-3,
            # set beta2 = 1 to use AdaGrad-style accumulation
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
            momentum_type: str = "ema",
            second_moment_type: str = "ema",
            correct_dim=False,
            cpu_offload=False,
            muon=False,
            adamuon=False,
            compile=False,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            correct_bias=correct_bias,
            momentum_type=momentum_type,
            second_moment_type=second_moment_type,
            correct_dim=correct_dim,
            cpu_offload=cpu_offload,
            muon=muon,
            adamuon=adamuon,
            compile=compile,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable = None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('GenericOptim does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Compute step size
                step_size = group['lr']
                if group['correct_bias'] and beta2 != 1:
                    bias_correction2 = 1 - beta2 ** state['step']
                    step_size = step_size * math.sqrt(bias_correction2)

                # Compute denominator
                denom = get_and_update_subset_norm_denom(group, state, grad, beta2)

                # Compute numerator
                numerator = get_and_update_subspace_momentum(group, state, p)

                # Update parameters
                p.addcdiv_(numerator, denom, value=-step_size)

                # Weight decay
                if group['weight_decay'] != 0:
                    p.add_(p, alpha=-group['lr'] * group['weight_decay'])

        return loss

    def get_numerator(self, group, state, p, state_device):
        """Get the numerator for the update step."""
        grad = p.grad
        beta1, beta2 = group["betas"]

        # Projection for compressing momentum term
        if "rank" in group:
            proj_grad = get_projected_grad(group, state, p)
        else:  # if not SM or module is not set then it's just standard momentum
            proj_grad = grad

        # Init
        if "exp_avg" not in state:
            state["exp_avg"] = torch.zeros_like(proj_grad)
        # Momentum term
        exp_avg = state["exp_avg"]

        # reset exp_avg state when we update as default
        if ("rank" in group and state["step"] > 1 and state["step"] % group["update_proj_gap"] == 0):
            if "overlap_state" not in group:
                state["exp_avg"] = torch.zeros_like(proj_grad)
            # else we overlap the momentum update where we don't need to do anything

        # Subspace momentum and orthogonal SGD
        if "rank" in group:
            exp_avg.mul_(beta1).add_(proj_grad, alpha=(1.0 - beta1))
            orth_comp = grad - state["projector"].project_back(proj_grad)
            numerator = state["projector"].project_back(exp_avg) + orth_comp
        else:  # just normal full momentum
            exp_avg.mul_(beta1).add_(proj_grad, alpha=(1.0 - beta1))
            numerator = exp_avg

        return numerator

    def get_denominator(self, group, state, grad, state_device):
        """Get the denominator for the update step."""
        return get_and_update_subset_norm_denom(group, state, grad, group["betas"][1])

    @torch.no_grad()
    def check_params(self):
        """Check if all parameters are valid."""
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    if has_inf_or_nan(p.grad):
                        raise RuntimeError("Found inf or nan in gradients")

    def load_state_dict(self, sd):
        """Load state dict with validation."""
        # Validate that the state_dict is from a GenericOptim optimizer
        if 'param_groups' not in sd:
            raise ValueError("state_dict is not from a GenericOptim optimizer")
        
        super().load_state_dict(sd) 