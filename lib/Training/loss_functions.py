import torch
import torch.nn as nn


def loss_function(output, target):
    class_loss = nn.CrossEntropyLoss(reduction='sum')

    cl = class_loss(output, target) / torch.numel(target)
    return cl


def loss_function_joint(classification, target, recon, inp):
    class_loss = nn.CrossEntropyLoss(reduction='sum')
    recon_loss = nn.BCEWithLogitsLoss(reduction='sum')

    cl = class_loss(classification, target) / torch.numel(target)
    rl = recon_loss(recon, inp) / torch.numel(inp)

    return cl, rl


def var_loss_function(output_samples, target, mu, std, device):
    """
    Computes the loss function consisting of a KL term between approximate posterior and prior and the loss for the
    generative classifier. The number of variational samples is one per default, as specified in the command line parser
    and typically is how VAE models and also our unified model is trained.
    We have added the option to flexibly work with an arbitrary amount of samples.

    Parameters:
        output_samples (torch.Tensor): Mini-batch of var_sample many classification prediction values.
        target (torch.Tensor): Classification targets for each element in the mini-batch.
        mu (torch.Tensor): Encoder (recognition model's) mini-batch of mean vectors.
        std (torch.Tensor): Encoder (recognition model's) mini-batch of standard deviation vectors.
        device (str): Device for computation.

    Returns:
        float: normalized classification loss
        float: normalized KL divergence
    """
    class_loss = nn.CrossEntropyLoss(reduction='sum')

    # Place-holders for the final loss values over all latent space samples
    cl_losses = torch.zeros(output_samples.size(0)).to(device)

    # numerical value for stability of log computation
    eps = 1e-8

    # loop through each sample for each input and calculate the correspond loss. Normalize the losses.
    for i in range(output_samples.size(0)):
        cl_losses[i] = class_loss(output_samples[i], target) / torch.numel(target)

    # average the loss over all samples per input
    cl = torch.mean(cl_losses, dim=0)

    # Compute the KL divergence, normalized by latent dimensionality
    kld = -0.5 * torch.sum(1 + torch.log(eps + std ** 2) - (mu ** 2) - (std ** 2)) / torch.numel(mu)

    return cl, kld


def var_loss_function_joint(output_samples_classification, target, output_samples_recon, inp, mu, std, device):
    recon_loss = nn.BCEWithLogitsLoss(reduction='sum')
    class_loss = nn.CrossEntropyLoss(reduction='sum')

    # Place-holders for the final loss values over all latent space samples
    recon_losses = torch.zeros(output_samples_recon.size(0)).to(device)
    cl_losses = torch.zeros(output_samples_classification.size(0)).to(device)

    # numerical value for stability of log computation
    eps = 1e-8

    # loop through each sample for each input and calculate the correspond loss. Normalize the losses.
    for i in range(output_samples_classification.size(0)):
        cl_losses[i] = class_loss(output_samples_classification[i], target) / torch.numel(target)
        recon_losses[i] = recon_loss(output_samples_recon[i], inp) / torch.numel(inp)

    # average the loss over all samples per input
    cl = torch.mean(cl_losses, dim=0)
    rl = torch.mean(recon_losses, dim=0)

    # Compute the KL divergence, normalized by latent dimensionality
    kld = -0.5 * torch.sum(1 + torch.log(eps + std ** 2) - (mu ** 2) - (std ** 2)) / torch.numel(mu)

    return cl, rl, kld
