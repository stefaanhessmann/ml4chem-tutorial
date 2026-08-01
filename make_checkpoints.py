"""Train the tutorial checkpoint from scratch.

Produces ``checkpoints/gpff.pt`` — the time-free GPFF model with
perturbation-alignment coupling that sections 5-8 of the notebook define,
train and sample. The notebook loads it so sampling works instantly; run this
script to regenerate it.

The architectures here MUST match the ones assembled in ``notebook.py`` —
the checkpoints are plain ``state_dict``s.

Usage (from this directory, in the ``schnetpack`` conda env)::

    python make_checkpoints.py [n_steps]
"""

import os
import sys
from itertools import cycle

import numpy as np
import torch
from ase.io import read

import schnetpack.nn as snn
import schnetpack.transform as trn
from schnetpack.data import ASEAtomsData, AtomsLoader
from schnetpack.generative import (
    VE,
    Diffuse,
    PermutationCoupling,
    PseudoForceParametrization,
)
from schnetpack.model import (
    AtomwiseVector,
    NeuralNetworkPotential,
    PaiNN,
    PairwiseDistances,
)

HERE = os.path.dirname(os.path.abspath(__file__))
XYZ = os.path.join(HERE, "data", "qm9_c4h4n2o2.xyz")
DB = os.path.join(HERE, "data", "qm9_c4h4n2o2.db")

SIGMA_MIN = 0.05
SIGMA_MAX = 10.0  # ~ largest pairwise distance in the dataset (7.7 A) + margin
CUTOFF = 30.0  # must cover the *noised* structures, not just the clean ones
BATCH_SIZE = 10
LEARNING_RATE = 5e-4
N_STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 5000


def build_db():
    if os.path.exists(DB):
        return
    mols = read(XYZ, index=":")
    db = ASEAtomsData.create(DB, distance_unit="Ang", property_unit_dict={"energy": "eV"})
    db.add_systems(
        property_list=[{"energy": np.array([m.info["energy_U0"]])} for m in mols],
        atoms_list=mols,
    )


def make_loader(process, parametrization, label_key):
    dataset = ASEAtomsData(
        DB,
        load_properties=[],
        transforms=[
            trn.SubtractCenterOfGeometry(),
            Diffuse(process, parametrization, label_key=label_key, time_key="t"),
            trn.MatScipyNeighborList(cutoff=CUTOFF),
            trn.CastTo32(),
        ],
    )
    return AtomsLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


def gpff_network(output_key):
    """The time-free vector head of notebook section 5 — no TimeConditioning."""
    return NeuralNetworkPotential(
        representation=PaiNN(
            n_atom_basis=32,
            n_interactions=3,
            radial_basis=snn.GaussianRBF(n_rbf=30, cutoff=CUTOFF),
            cutoff_fn=snn.CosineCutoff(CUTOFF),
        ),
        input_modules=[PairwiseDistances()],
        output_modules=[AtomwiseVector(n_in=32, n_layers=1, output_key=output_key)],
    )


def train(model, loader, loss_fn, tag):
    """The notebook's training loop, run to ``N_STEPS``.

    Iterating the loader afresh every epoch is load-bearing: the transforms
    re-run, so each epoch sees new (t, noise) draws. Caching the batches
    instead (``itertools.cycle``) freezes those draws and the model memorizes
    ~19 fixed noised batches rather than learning the denoising field.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    history, step = [], 0
    while step < N_STEPS:
        for batch in loader:
            step += 1
            loss = loss_fn(model(batch), batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step == 1 or step % 100 == 0:
                history.append((step, loss.item()))
                print(f"[{tag}] step {step:5d}   loss {loss.item():.4f}", flush=True)
            if step >= N_STEPS:
                break
    return history


def main():
    build_db()
    os.makedirs(os.path.join(HERE, "checkpoints"), exist_ok=True)

    # --- GPFF with perturbation-alignment coupling ------------------------- #
    torch.manual_seed(0)
    process = VE(sigma_min=SIGMA_MIN, sigma_max=SIGMA_MAX, coupling=PermutationCoupling())
    loader = make_loader(process, PseudoForceParametrization(), label_key="pseudo_force")
    model = gpff_network("pseudo_force_pred")

    def gpff_loss(pred, batch):
        # 1/sigma^2 undoes the target's sigma-scaling (making the objective
        # noise matching); the clip keeps nearly-clean samples from dominating.
        weight = (1.0 / process.sigma(batch["t"]) ** 2).clamp(max=1.0)
        diff = pred["pseudo_force_pred"] - batch["pseudo_force"]
        return (weight[:, None] * diff**2).mean()

    history = train(model, loader, gpff_loss, "gpff")
    torch.save(
        {"state_dict": model.state_dict(), "history": history},
        os.path.join(HERE, "checkpoints", "gpff.pt"),
    )
    print("checkpoint written.")


if __name__ == "__main__":
    main()
