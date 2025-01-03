import torch
import math


def log_multivariate_normal_pdf(samples, means, stds):
    """
    Calculate the log probability density of samples under a multivariate normal distribution.

    Args:
    samples (torch.Tensor): Tensor of shape (B, N) representing the sample points.
    means (torch.Tensor): Tensor of shape (B, N) representing the means of the multivariate normal distribution.
    stds (torch.Tensor): Tensor of shape (B, N) representing the standard deviations of the multivariate normal distribution.

    Returns:
    torch.Tensor: Tensor of shape (B,) representing the log probability densities of the samples.
    """
    B, N = samples.shape
    var = stds ** 2
    log_det_var = torch.sum(torch.log(var), dim=1)
    log_norm_const = -0.5 * \
        (N * torch.log(torch.tensor(2 * math.pi)) + log_det_var)
    diff = samples - means
    log_exponent = -0.5 * torch.sum((diff ** 2) / var, dim=1)
    log_pdf = log_norm_const + log_exponent
    return log_pdf


def stable_pdf_ratio(samples, means1, stds1, means2, stds2):
    """
    Calculate the stable ratio of the probability densities of two multivariate normal distributions.

    Args:
    samples (torch.Tensor): Tensor of shape (B, N) representing the sample points.
    means1 (torch.Tensor): Tensor of shape (B, N) representing the means of the first multivariate normal distribution.
    stds1 (torch.Tensor): Tensor of shape (B, N) representing the standard deviations of the first multivariate normal distribution.
    means2 (torch.Tensor): Tensor of shape (B, N) representing the means of the second multivariate normal distribution.
    stds2 (torch.Tensor): Tensor of shape (B, N) representing the standard deviations of the second multivariate normal distribution.

    Returns:
    torch.Tensor: Tensor of shape (B,) representing the ratio of the probability densities of the samples.
    """
    log_pdf1 = log_multivariate_normal_pdf(samples, means1, stds1)
    log_pdf2 = log_multivariate_normal_pdf(samples, means2, stds2)
    log_ratio = log_pdf1 - log_pdf2
    ratio = torch.exp(log_ratio)
    return ratio


# Example usage
B, N = 5, 61
samples = torch.randn((8, 61))
sample2 = torch.randn((8, 61))
means1 = torch.rand((8, 61))
means2 = torch.rand((8, 61))
stds1 = torch.rand((8, 61))
stds2 = torch.rand((8, 61))
ratio_values = stable_pdf_ratio(samples, means1, stds1, means2, stds2)
print(ratio_values)
