from .cortex import NeuralCortex
from .image_utils import generate_letter_image, letter_to_tensor, load_image_tensor
from .gaussian import gaussian_kernel_2d, gaussian_blur_torch, img_to_encoder_torch
from .grid import generate_hex_grid
from .particles import propagate_particles, resample_particles, sample_positions
from .glm import GLMContext, make_glm_context, glm_rates_np, glm_rates_torch_batch, precompute_spatial_kernels
from .decoder import Er, Ep, Er_per_sample, adam_update_A, compute_gauss_newton_hessian, update_hessian
from .animation import create_animation

NeuralEncoder = NeuralCortex

__all__ = [
    "NeuralCortex",
    "NeuralEncoder",
    "generate_letter_image",
    "letter_to_tensor",
    "load_image_tensor",
    "gaussian_kernel_2d",
    "gaussian_blur_torch",
    "img_to_encoder_torch",
    "generate_hex_grid",
    "propagate_particles",
    "resample_particles",
    "sample_positions",
    "GLMContext",
    "make_glm_context",
    "glm_rates_np",
    "glm_rates_torch_batch",
    "precompute_spatial_kernels",
    "Er",
    "Ep",
    "Er_per_sample",
    "adam_update_A",
    "compute_gauss_newton_hessian",
    "update_hessian",
    "create_animation",
]
