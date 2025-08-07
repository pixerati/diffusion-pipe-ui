# Copied from AI Toolkit.
# I added Kahan summation for bfloat16 parameters.

# MIT License

# Copyright (c) 2024 Ostris, LLC

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


from typing import List
import torch
from optimizers.optimizer_utils import Auto8bitTensor, copy_stochastic, stochastic_grad_accummulation
from optimum.quanto import QBytesTensor
import random


class Automagic(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr=1e-6, # lr is start lr
        min_lr=1e-7,
        max_lr=1e-3,
        lr_bump=1e-6, # amount to bump the lr when adjusting
        eps=(1e-30, 1e-3),
        clip_threshold=1.0,
        beta2=0.999,
        weight_decay=0.0,
        do_paramiter_swapping=False,
        paramiter_swapping_factor=0.1,
    ):
        self.lr = lr
        if self.lr > 1e-3:
            print(f"Warning! Start lr is very high: {self.lr}. Forcing to 1e-6. this does not work like prodigy")
            self.lr = 1e-6
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.lr_bump = lr_bump

        defaults = {
            "lr": lr,
            "eps": eps,
            "clip_threshold": clip_threshold,
            "beta2": beta2,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

        self.base_lrs: List[float] = [
            lr for group in self.param_groups
        ]

        self.is_stochastic_rounding_accumulation = False

        # setup stochastic grad accum hooks
        # for group in self.param_groups:
        #     for param in group['params']:
        #         if param.requires_grad and param.dtype != torch.float32:
        #             self.is_stochastic_rounding_accumulation = True
        #             param.register_post_accumulate_grad_hook(
        #                 stochastic_grad_accummulation
        #             )

        self.do_paramiter_swapping = do_paramiter_swapping
        self.paramiter_swapping_factor = paramiter_swapping_factor
        self._total_paramiter_size = 0
        # count total paramiters
        for group in self.param_groups:
            for param in group['params']:
                self._total_paramiter_size += torch.numel(param)
        # pretty print total paramiters with comma seperation
        print(f"Total training paramiters: {self._total_paramiter_size:,}")

        # needs to be enabled to count paramiters
        if self.do_paramiter_swapping:
            self.enable_paramiter_swapping(self.paramiter_swapping_factor)

    def enable_paramiter_swapping(self, paramiter_swapping_factor=0.1):
        self.do_paramiter_swapping = True
        self.paramiter_swapping_factor = paramiter_swapping_factor
        # call it an initial time
        self.swap_paramiters()

    def swap_paramiters(self):
        if not self.do_paramiter_swapping:
            return
        # randomly swap some paramiters
        for group in self.param_groups:
            for param in group['params']:
                if param.requires_grad:
                    # randomly swap some paramiters
                    if random.random() < self.paramiter_swapping_factor:
                        # swap with a random other paramiter
                        other_group = random.choice(self.param_groups)
                        other_param = random.choice(other_group['params'])
                        if other_param.requires_grad and other_param is not param:
                            # swap the paramiters
                            param.data, other_param.data = other_param.data, param.data

    @staticmethod
    def _get_lr(param_group, param_state):
        return param_group['lr']

    def _get_group_lr(self, group):
        return group['lr']

    @staticmethod
    def _rms(tensor):
        return tensor.norm(2) / (tensor.numel() ** 0.5)

    @staticmethod
    def _approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col):
        r_factor = (exp_avg_sq_row / exp_avg_sq_row.mean(dim=-1, keepdim=True)).rsqrt_()
        c_factor = exp_avg_sq_col.rsqrt()
        return torch.mul(r_factor.unsqueeze(-1), c_factor.unsqueeze(-2))

    def step_hook(self):
        # called after step
        pass

    def get_learning_rates(self):
        return [group['lr'] for group in self.param_groups]

    def get_avg_learning_rate(self):
        lrs = self.get_learning_rates()
        return sum(lrs) / len(lrs)

    @torch.no_grad()
    def step(self, closure=None):
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

                # Perform stepweight decay
                if group['weight_decay'] != 0:
                    p.data.mul_(1 - group['lr'] * group['weight_decay'])

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('Automagic does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(group['beta2']).add_(grad, alpha=1 - group['beta2'])
                exp_avg_sq.mul_(group['beta2']).addcmul_(grad, grad, value=1 - group['beta2'])

                # Compute the RMS
                rms = self._rms(exp_avg_sq)

                # Compute the adaptive learning rate
                lr = group['lr']
                if rms > group['clip_threshold']:
                    lr = lr * group['clip_threshold'] / rms

                # Clamp the learning rate
                lr = max(group['min_lr'], min(group['max_lr'], lr))

                # Update the learning rate
                group['lr'] = lr

                # Update the parameter
                p.add_(exp_avg, alpha=-lr)

        self.step_hook()
        return loss

    def initialize_state(self, p):
        """Initialize state for a parameter."""
        if p not in self.state:
            self.state[p] = {}
        state = self.state[p]
        if 'step' not in state:
            state['step'] = 0
        if 'exp_avg' not in state:
            state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
        if 'exp_avg_sq' not in state:
            state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        state_dict['lr'] = self.lr
        state_dict['min_lr'] = self.min_lr
        state_dict['max_lr'] = self.max_lr
        state_dict['lr_bump'] = self.lr_bump
        return state_dict

    def load_state_dict(self, state_dict, strict=True):
        # Validate that the state_dict is from an Automagic optimizer
        if 'lr' not in state_dict:
            raise ValueError("state_dict is not from an Automagic optimizer")
        
        self.lr = state_dict['lr']
        self.min_lr = state_dict.get('min_lr', 1e-7)
        self.max_lr = state_dict.get('max_lr', 1e-3)
        self.lr_bump = state_dict.get('lr_bump', 1e-6)
        
        super().load_state_dict(state_dict, strict) 