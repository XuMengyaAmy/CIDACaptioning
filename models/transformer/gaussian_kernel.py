'''
Paper : Class-Incremental Domain Adaptation with Smoothing and Calibration for Surgical Report Generation
'''
import math
import torch
import torch.nn as nn

# 1D gaussian kernel 
def get_gaussian_filter(kernel_sizex=3, kernel_sizey=1, sigma=2, channels=3):
    kernel_size = max(kernel_sizex, kernel_sizey)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()

    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()
    mean = (kernel_size - 1)/2.
    variance = sigma**2.
    xy_grid = torch.sum((xy_grid[:kernel_size,:kernel_size,:] - mean)**2., dim=-1)

    # Calculate the 1-dimensional gaussian kernel
    gaussian_kernel = (1./((math.sqrt(2.*math.pi)*sigma))) * \
                        torch.exp(-1* (xy_grid[int(kernel_size/2)]) / (2*variance))

    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1)

    padding = 1 if kernel_size==3 else 2 if kernel_size == 5 else 0
    gaussian_filter = nn.Conv1d(in_channels=channels, out_channels=channels,
                                kernel_size=kernel_size, groups=channels,
                                bias=False, padding=padding)
    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False 
    return gaussian_filter



