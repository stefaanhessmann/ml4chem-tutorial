"""Small local helpers for the ML4Chem tutorial notebook.

Everything computational comes from ``schnetpack``; this module only holds the
glue between the library and the notebook: building the static batch layout
that sampling runs on, and adapting a SchNetPack batch-dict model to the
sampler's tensor-level ``model(x, t, cond)`` contract. Rendering lives in
``viz``.
"""

import torch

from schnetpack import properties

__all__ = [
    "fully_connected_batch",
    "make_model_fn",
    "to_device",
]


def to_device(batch, device):
    """Move every tensor of a SchNetPack batch dict onto ``device``.

    Transforms run in the dataloader, i.e. on CPU, so a batch always arrives
    on the host; this is the one hop to wherever the model lives. Non-tensor
    entries are passed through untouched.
    """
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def fully_connected_batch(numbers, n_mol):
    """A static batch layout for sampling: ``n_mol`` copies of one
    composition, each molecule internally fully connected.

    During sampling the atoms move at every step, but with a large cutoff
    the neighbor list never needs rebuilding: the cutoff function already
    downweights pairs by distance, and atoms of different molecules are
    simply never connected (block-diagonal pair indices).
    """
    n_at = len(numbers)
    i, j = torch.meshgrid(torch.arange(n_at), torch.arange(n_at), indexing="ij")
    mask = i != j
    offsets = torch.arange(n_mol).repeat_interleave(int(mask.sum())) * n_at
    return {
        properties.Z: torch.tensor(numbers).repeat(n_mol),
        properties.idx_i: i[mask].repeat(n_mol) + offsets,
        properties.idx_j: j[mask].repeat(n_mol) + offsets,
        properties.offsets: torch.zeros(int(mask.sum()) * n_mol, 3),
        properties.idx_m: torch.arange(n_mol).repeat_interleave(n_at),
        properties.n_atoms: torch.full((n_mol,), n_at),
    }


def make_model_fn(model, static_batch, output_key):
    """Adapt a SchNetPack model to the sampler's ``(x, t, cond)`` contract."""

    def model_fn(x, t, cond=None):
        inputs = dict(static_batch)
        inputs[properties.R] = x
        if t is not None:
            inputs["t"] = t
        return model(inputs)[output_key]

    return model_fn
