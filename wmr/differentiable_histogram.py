"""
    Differentiable histogram computation in Pych.
    Borrowed from https://github.com/Yukun-Huang/pytorch-differentiable-histogram
"""
import torch as ch
import gc
from typing import Optional


def differentiable_histogram(input: ch.Tensor,
                             bins: int = 100,
                             min: Optional[float] = None,
                             max: Optional[float] = None) -> ch.Tensor:
    """ Compute a differentiable histogram of a tensor. 
        The function returns a tensor of shape (batch_size, n_channels, bins) where each value represents the count of values in the input tensor that fall into the corresponding bin.
        Args:
            input (ch.Tensor): Input tensor of shape (batch_size, n_channels, ...)
            bins (int): Number of bins in the histogram.
            min (Optional[float]): Minimum value of the histogram (inclusive). If None, the minimum value of the input tensor is used.
            max (Optional[float]): Maximum value of the histogram (inclusive). If None, the maximum value of the input tensor is used.
        Returns:
            hist (ch.Tensor): Histogram tensor of shape (batch_size, n_channels, bins)
    """
    # Ensure the input tensor has at least 2 dimensions
    assert input.ndim >= 2
    input = input.view(input.shape[0], input.shape[1], -1)
    batch_size, n_channels, n_values = input.shape

    # Compute the minimum and maximum values of the input tensor
    with ch.no_grad():
        if min is None:
            min = input.min().item()
        if max is None:
            max = input.max().item()
    
    # Initialize the histogram tensor
    hist = ch.zeros(batch_size, n_channels, bins).to(input.device)

    # Create a table of bin edges
    delta = (max - min) / bins
    BIN_Table = ch.arange(start=0, end=bins+1, step=1) * delta

    # Iterate over each bin
    for dim in range(1, bins-1):
        h_curr = BIN_Table[dim].item()
        h_last = BIN_Table[dim - 1].item()
        h_next = BIN_Table[dim + 1].item()

        # Create masks for values falling into the current bin
        mask_last = ((h_last <= input) & (input < h_curr)).float()
        mask_next = ((h_curr <= input) & (input <= h_next)).float()

        # Accumulate histogram values for the current bin
        hist[:, :, dim] += ch.sum(((input - h_last) * mask_last).view(batch_size, n_channels, -1), dim=-1)
        hist[:, :, dim] += ch.sum(((h_next - input) * mask_next).view(batch_size, n_channels, -1), dim=-1)
    
    # Handle the first bin
    mask = (input < BIN_Table[1]).float()
    hist[:, :, 0] += ch.sum(((BIN_Table[1] - input) * mask).view(batch_size, n_channels, -1), dim=-1)

    # Handle the last bin
    mask = (input >= BIN_Table[bins-1]).float()
    hist[:, :, bins-1] += ch.sum(((input - BIN_Table[bins-1]) * mask).view(batch_size, n_channels, -1), dim=-1)

    # Divide by the bin width
    hist = hist / delta

    # Normalize the histogram without flow of gradients
    with ch.no_grad():
        hist.data /= hist.data.sum(dim=-1, keepdim=True) * n_values
    # hist = hist / detached_denom * n_values

    return hist
