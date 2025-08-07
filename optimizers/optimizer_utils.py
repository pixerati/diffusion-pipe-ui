# Copied from AI Toolkit.

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


import torch
from torch import Tensor
from typing import Optional
from optimum.quanto import QBytesTensor


def compute_scale_for_dtype(tensor, dtype):
    """
    Compute appropriate scale for the given tensor and target dtype.

    Args:
        tensor: Input tensor to be quantized
        dtype: Target dtype for quantization
    Returns:
        Appropriate scale factor for the quantization
    """
    if dtype == torch.int8:
        abs_max = torch.max(torch.abs(tensor))
        return abs_max / 127.0 if abs_max > 0 else 1.0
    elif dtype == torch.uint8:
        max_val = torch.max(tensor)
        min_val = torch.min(tensor)
        range_val = max_val - min_val
        return range_val / 255.0 if range_val > 0 else 1.0
    elif dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        # For float8, we typically want to preserve the magnitude of the values
        # while fitting within the representable range of the format
        abs_max = torch.max(torch.abs(tensor))
        if dtype == torch.float8_e4m3fn:
            # e4m3fn has range [-448, 448] with no infinities
            max_representable = 448.0
        else:  # torch.float8_e5m2
            # e5m2 has range [-57344, 57344] with infinities
            max_representable = 57344.0

        return abs_max / max_representable if abs_max > 0 else 1.0
    else:
        raise ValueError(f"Unsupported dtype for quantization: {dtype}")

def quantize_tensor(tensor, dtype):
    """
    Quantize a floating-point tensor to the target dtype with appropriate scaling.

    Args:
        tensor: Input tensor (float)
        dtype: Target dtype for quantization
    Returns:
        quantized_data: Quantized tensor
        scale: Scale factor used
    """
    scale = compute_scale_for_dtype(tensor, dtype)

    if dtype == torch.int8:
        quantized_data = torch.clamp(torch.round(tensor / scale), -128, 127).to(dtype)
    elif dtype == torch.uint8:
        quantized_data = torch.clamp(torch.round(tensor / scale), 0, 255).to(dtype)
    elif dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        # For float8, we scale and then cast directly to the target type
        # The casting operation will handle the appropriate rounding
        scaled_tensor = tensor / scale
        quantized_data = scaled_tensor.to(dtype)
    else:
        raise ValueError(f"Unsupported dtype for quantization: {dtype}")

    return quantized_data, scale


def update_parameter(target, result_float):
    """
    Updates a parameter tensor, handling both regular torch.Tensor and QBytesTensor cases
    with proper rescaling for quantized tensors.

    Args:
        target: The parameter to update (either torch.Tensor or QBytesTensor)
        result_float: The new values to assign (torch.Tensor)
    """
    if isinstance(target, QBytesTensor):
        # For QBytesTensor, we need to handle the quantization properly
        # First, we need to get the scale from the target tensor
        scale = target.scale
        # Then we quantize the result_float using the same scale
        quantized_result, _ = quantize_tensor(result_float, target.dtype)
        # Finally, we update the target with the quantized result
        target.data = quantized_result
    else:
        # For regular torch.Tensor, we can directly assign
        target.data = result_float


def get_format_params(dtype: torch.dtype) -> tuple[int, int]:
    """
    Get the format parameters for a given dtype.
    
    Args:
        dtype: The dtype to get format parameters for
        
    Returns:
        tuple: (mantissa_bits, exponent_bits)
    """
    if dtype == torch.float8_e4m3fn:
        return 3, 4
    elif dtype == torch.float8_e5m2:
        return 2, 5
    else:
        raise ValueError(f"Unsupported dtype for format params: {dtype}")


def copy_stochastic(
    target: torch.Tensor,
    source: torch.Tensor,
    eps: Optional[float] = None
) -> None:
    """
    Copy source tensor to target tensor with stochastic rounding for low precision.
    
    Args:
        target: Target tensor to copy to
        source: Source tensor to copy from
        eps: Epsilon for stochastic rounding (optional)
    """
    if target.dtype == source.dtype:
        target.data = source.data
        return

    # For different dtypes, we need to handle quantization
    if target.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        # For float8, we use stochastic rounding
        if eps is None:
            eps = 1e-6
        
        # Add small random noise for stochastic rounding
        noise = torch.rand_like(source) * eps
        source_with_noise = source + noise
        
        # Quantize to target dtype
        quantized, scale = quantize_tensor(source_with_noise, target.dtype)
        target.data = quantized
    else:
        # For other dtypes, use regular casting
        target.data = source.data.to(target.dtype)


class Auto8bitTensor:
    """
    A wrapper for 8-bit tensors that automatically handles quantization and dequantization.
    """
    
    def __init__(self, data: Tensor, *args, **kwargs):
        self.data = data
        self.scale = kwargs.get('scale', 1.0)
        self.dtype = data.dtype
        
    def dequantize(self) -> Tensor:
        """Dequantize the tensor back to float32."""
        if self.dtype in (torch.int8, torch.uint8):
            return self.data.float() * self.scale
        else:
            return self.data.float()
    
    def to(self, *args, **kwargs):
        # Handle the dtype argument whether it's positional or keyword
        if len(args) > 0 and isinstance(args[0], torch.dtype):
            dtype = args[0]
            args = args[1:]
        else:
            dtype = kwargs.get('dtype', self.dtype)
            kwargs.pop('dtype', None)
        
        if dtype == torch.float32:
            return self.dequantize()
        else:
            # For other dtypes, we need to requantize
            dequantized = self.dequantize()
            quantized, scale = quantize_tensor(dequantized, dtype)
            return Auto8bitTensor(quantized, scale=scale)
    
    def state_dict(self):
        return {
            'data': self.data,
            'scale': self.scale,
            'dtype': self.dtype
        }
    
    def _load_from_state_dict(self, state_dict):
        self.data = state_dict['data']
        self.scale = state_dict['scale']
        self.dtype = state_dict['dtype']
    
    def __str__(self):
        return f"Auto8bitTensor(data={self.data}, scale={self.scale}, dtype={self.dtype})"


def stochastic_grad_accummulation(param):
    """
    Stochastic gradient accumulation hook for parameters.
    
    Args:
        param: The parameter to apply stochastic rounding to
    """
    if param.grad is not None and param.dtype != torch.float32:
        # Apply stochastic rounding to gradients
        eps = 1e-6
        noise = torch.rand_like(param.grad) * eps
        param.grad = param.grad + noise 