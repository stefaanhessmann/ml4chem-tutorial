# ML4Chem hands-on tutorial — SchNetPack: from force fields to generative models

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/stefaanhessmann/ml4chem-tutorial/blob/main/notebook.ipynb)

**Students: click the badge.** It opens your own copy in Google Colab — nothing
to install, and everyone gets a separate machine. Run the first cell and work
downwards.

A single [marimo](https://marimo.io) notebook. It starts where SchNetPack
starts — your own data in an ASE database, transforms, batches — and then
builds a generative model out of the same pieces a machine-learned force field
is made of: noising structures and computing labels as a dataloader transform
(`Diffuse`), a `NeuralNetworkPotential` with a per-atom vector head, a plain
PyTorch training loop, and two samplers — ancestral sampling and direct
denoising — to generate new structures. Ends with two hands-on tasks, both
steering a sampler without retraining the model: shape-guided direct denoising
(generate at a prescribed ratio of principal-axis variances) and
scaffold-conditioned generation (keep an amide anchor, generate the molecule
around it).

The model is **GPFF** (generative pseudo-force field), whose target is a pseudo
force — which makes it, structurally, an ordinary force field applied to noised
structures, with no time conditioning anywhere.

## Contents

```
ML4Chem-tutorial/
├── notebook.py            ← the tutorial (marimo notebook)
├── data/
│   └── qm9_c4h4n2o2.xyz   ← 181 QM9 isomers of C4H4N2O2, with U0 energies [eV]
├── checkpoints/
│   ├── gpff.pt            ← the section-6 model (1.7M params, 12k steps on the 181
│   │                        isomers) and its loss history. That cell loads it
│   │                        unless you set RETRAIN = True
│   └── gpff_big.pt        ← GPFF trained on all of QM9 (5.1M params); section 7's
│                            USE_BIG_MODEL switch swaps it in for comparison
├── viz.py                 ← 3D trajectory viewer (3Dmol.js grid, shared slider,
│                            optional per-atom vector arrows)
├── helpers.py             ← notebook glue: static sampling batches and the
│                            batch-dict → model(x, t) adapter
├── assets/3Dmol-min.js    ← vendored viewer library (offline, no CDN)
└── make_colab.py          ← derives notebook.ipynb for Colab (see below)
```

The notebook creates `data/qm9_c4h4n2o2.db` (a SchNetPack database) from the
xyz file on first run.

## Setup

Python ≥ 3.12 (what SchNetPack requires); everything runs on CPU:

```bash
conda create -n ml4chem python=3.12
conda activate ml4chem
pip install "git+https://github.com/atomistic-machine-learning/schnetpack.git@sh/v3"
pip install marimo matplotlib scipy rdkit
```

## Run

From this folder:

```bash
marimo edit notebook.py     # interactive (recommended for the tutorial)
marimo run notebook.py      # read-only app view
```

The section-6 model ships as a checkpoint, so that cell loads it and the whole
notebook runs in a couple of minutes. Set `RETRAIN = True` there to train it
yourself instead — 12000 steps, ~15 min on a GPU and considerably longer on a
CPU — which overwrites `checkpoints/gpff.pt` when it finishes.

**Two models, deliberately.** Section 6's is teaching-sized, trained with
GPFF's recipe — geometric VE process over σ ∈ [0.05, 30] Å, log-normal
σ-focused timestep sampling — on the 181 isomers, and section 7 samples with it
by default. For comparison, section 7's `USE_BIG_MODEL` switch swaps in
`gpff_big.pt`: the same target, process and training density at research scale
(5.1M parameters, all of QM9), converted from the GPFF QM9 training run by
`convert_gpff_seed999.py` (a dev script, not shipped). Section 8's two tasks
always use the research-scale model, whatever that switch says — steering a
sampler is only legible when the model underneath is not the bottleneck.

## Colab

The same notebook runs in Google Colab, where students need no local install at
all — and, because Colab opens the badge link as a *fresh copy* on a *fresh
VM*, a whole class can work at once without sharing anything.

`notebook.py` stays the source of truth; `notebook.ipynb` is **generated** —
never hand-edit it. After changing the notebook, rebuild and commit both:

```bash
python make_colab.py --repo https://github.com/stefaanhessmann/ml4chem-tutorial \
                     --schnetpack ~/projects/schnetpack
```

That builds `schnetpack-*.whl` from your checkout, exports via marimo, swaps the
conda instructions above for a setup cell that clones this repo and installs
that wheel, and writes `notebook.ipynb`.

The wheel is why students install in seconds: `pip install git+https://...`
would have each of them clone ~66 MB of history and build it, where the wheel is
~330 KB and pure Python. Being a file rather than a ref, it also pins the class
to one exact build — firmer than a tag, which can be moved.

Nothing else differs between the two. `viz.py` builds one self-contained page
either way and only its last step branches on the frontend, so the viewers,
the scrubber and §8's task movies behave the same in both. The one
rule that branch enforces: every page is sealed in its own iframe, because the
pages name their controls with fixed ids (`slider`, `grid`, `play`) and a
Jupyter document renders every output into one shared DOM — two unsealed pages
there and the second one's script dies on a redeclared `const`, leaving a live
slider above an empty grid.
