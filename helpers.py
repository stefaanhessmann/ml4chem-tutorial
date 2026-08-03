"""Small local helpers for the ML4Chem tutorial notebook.

Everything computational comes from ``schnetpack``; this module only holds the
glue between the library and the notebook: adapting a SchNetPack batch-dict
model to the sampler's tensor-level ``model(x, t, cond)`` contract. Rendering
lives in ``viz``.
"""

import torch

from schnetpack import properties

__all__ = [
    "make_model_fn",
    "recording_model_fn",
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


def make_model_fn(model, static_batch, output_key):
    """Adapt a SchNetPack model to the sampler's ``(x, t, cond)`` contract."""

    def model_fn(x, t, cond=None):
        inputs = dict(static_batch)
        inputs[properties.R] = x
        if t is not None:
            inputs["t"] = t
        return model(inputs)[output_key]

    return model_fn


def recording_model_fn(model_fn):
    """Wrap ``model_fn`` so every state it is asked about is kept.

    A sampler hands back the structure it ended on and nothing else — the
    frames along the way are not its to return. They do not have to be: every
    step passes its current state through the model, so wrapping the *model*
    records the trajectory without the sampler knowing. That is why this works
    for any of them, the one you write in section 5 included.

    Returns ``(wrapped, frames)``. ``frames`` fills as sampling runs and ends
    one short — the last state the model saw is not the last state of the run —
    so append the sampler's return value to finish the trajectory.
    """
    frames = []

    def wrapped(x, t, cond=None):
        frames.append(x.detach().clone())
        return model_fn(x, t, cond)

    return wrapped, frames
