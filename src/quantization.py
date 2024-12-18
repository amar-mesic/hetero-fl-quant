import random
import numpy as np
import torch

# 1) Scalar quantization: approach where model weights or updates are scaled and rounded to discrete levels
# Source: https://ieeexplore.ieee.org/document/9054168

def scalar_quantize(weights, bitwidth=8):
    # Define the quantization levels
    if isinstance(weights, np.ndarray):
        weights = torch.tensor(weights)

    # Define the quantization levels
    levels = 2 ** bitwidth
    scale = (weights.max() - weights.min()) / (levels - 1)
    zero_point = -weights.min() / scale
    quantized = torch.round((weights / scale) + zero_point)
    return quantized, scale, zero_point

def scalar_dequantize(quantized, scale, min_val):
    return quantized * scale + min_val


# 2) Lattice-based quantization: approach using predefined lattice for quantization
# Source: https://ieeexplore.ieee.org/document/9054168
from scipy.spatial import distance

def lattice_quantize(weights, lattice):
    quantized = [lattice[np.argmin(distance.cdist([w], lattice))] for w in weights]
    return np.array(quantized)


# 3) Probabilistic quantization: approach using stochastic rounding to achieve unbiased quantization.
# Source: https://ieeexplore.ieee.org/document/9054168

def probabilistic_quantize(weights, bitwidth=8):
    levels = 2 ** bitwidth
    scale = (weights.max() - weights.min()) / (levels - 1)
    normalized_weights = (weights - weights.min()) / scale
    lower = np.floor(normalized_weights)
    upper = np.ceil(normalized_weights)
    probabilities = normalized_weights - lower
    quantized = np.where(np.random.rand(*weights.shape) < probabilities, upper, lower)
    return quantized.astype(np.int32), scale, weights.min()

# 4) Dithered quantization: approach that adds random noise to mitigate quantization bias.
# Source: https://ieeexplore.ieee.org/document/9054168

def dithered_quantize(weights, bitwidth=8, dither_scale=0.01):
    levels = 2 ** bitwidth
    scale = (weights.max() - weights.min()) / (levels - 1)
    dither = np.random.uniform(-dither_scale, dither_scale, weights.shape)
    quantized = np.round((weights - weights.min() + dither) / scale).astype(np.int32)
    return quantized, scale, weights.min()


# 5) Kurtosis Regularization (KURE): approach adding regularization to increase robustness against quantization
# Source: https://arxiv.org/abs/2206.10844

def kurtosis_regularization(weights, target_kurtosis=1.8):
    mean = weights.mean()
    std = weights.std()
    kurtosis = ((weights - mean) ** 4).mean() / (std ** 4)
    return (kurtosis - target_kurtosis) ** 2


# 6) Pseudo-Quantization Noise (APQN)
# Source: https://arxiv.org/abs/2206.10844

def add_pseudo_quantization_noise(weights, delta):
    noise = torch.empty_like(weights).uniform_(-delta / 2, delta / 2)
    return weights + noise

# 7) Quantize weights using Straight-Through Estimator (STE) for QAT
# Source: https://arxiv.org/abs/2206.10844

def quantize_ste(weights, step_size, min_val, max_val):
    quantized = torch.clamp((weights / step_size).round(), min_val, max_val)
    return quantized * step_size

# 8) Multi-Bit Quantization (MQAT)
# Source: https://arxiv.org/abs/2206.10844

def quantize_multi_bit(weights, bit_width):
    step_size = (weights.max() - weights.min()) / (2 ** bit_width - 1)
    quantized = torch.clamp((weights / step_size).round(), -(2 ** (bit_width - 1)), (2 ** (bit_width - 1)) - 1)
    return quantized * step_size