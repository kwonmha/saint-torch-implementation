
import copy

import numpy as np
import torch
from torch import nn


def get_mask(seq_len, device):
    mask = torch.from_numpy(np.triu(np.ones((seq_len, seq_len)), k=1).astype('bool')).to(device)
    return mask


def position_embedding(bs, seq_len, dim_model):
    # Return the position embedding for the whole sequence
    pe_array = np.array([[_position_encoding(pos, dim_model) for pos in range(seq_len)]] * bs)
    return torch.from_numpy(pe_array).float()


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


# Experimental functions
def _reparameterize(mu, logvar):
    """
    Reparameterization trick to sample from N(mu, var) from
    N(0,1).
    :param mu: (Tensor) Mean of the latent Gaussian [B x N x D]
    :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x N x D]
    :return: (Tensor) [B x D]
    """
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return eps * std + mu


def calculate_latent(mu, logvar):
    z = _reparameterize(mu, logvar)
    return z, mu, logvar


def _position_encoding(pos, dim_model):
    # Encode one position with sin and cos
    # Attention Is All You Need uses positional sines, SAINT paper does not specify
    pos_enc = np.zeros(dim_model)
    for i in range(0, dim_model, 2):
        pos_enc[i] = np.sin(pos / (10000 ** (2 * i / dim_model)))
        pos_enc[i + 1] = np.cos(pos / (10000 ** (2 * i / dim_model)))
    return pos_enc
