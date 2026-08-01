# ML4Chem hands-on tutorial — SchNetPack: from force fields to generative models

A single [marimo](https://marimo.io) notebook. It starts where SchNetPack
starts — your own data in an ASE database, transforms, batches — and then
builds a generative model out of the same pieces a machine-learned force field
is made of: noising structures and computing labels as a dataloader transform
(`Diffuse`), a `NeuralNetworkPotential` with a per-atom vector head, a plain
PyTorch training loop, and direct denoising to generate new structures. Ends
with a hands-on task: a variance-conditioned direct-denoising sampler.

The model is **GPFF** (Gaussian pseudo-force field), whose target is a pseudo
force — which makes it, structurally, an ordinary force field applied to noised
structures, with no time conditioning anywhere.

## Contents

```
ML4Chem-tutorial/
├── notebook.py            ← the tutorial (marimo notebook)
├── data/
│   └── qm9_c4h4n2o2.xyz   ← 181 QM9 isomers of C4H4N2O2, with U0 energies [eV]
├── checkpoints/
│   └── gpff.pt            ← trained GPFF model (sections 6-8)
├── viz.py                 ← 3D trajectory viewer (3Dmol.js grid, shared slider,
│                            optional per-atom vector arrows)
├── helpers.py             ← notebook glue: static sampling batches and the
│                            batch-dict → model(x, t) adapter
├── assets/3Dmol-min.js    ← vendored viewer library (offline, no CDN)
├── make_checkpoints.py    ← regenerates the checkpoint (~20 min on CPU)
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
pip install marimo matplotlib scipy
```

## Run

From this folder:

```bash
marimo edit notebook.py     # interactive (recommended for the tutorial)
marimo run notebook.py      # read-only app view
```

The training cell loads the shipped checkpoint by default; set `RETRAIN = True`
there (or run `python make_checkpoints.py`) to train from scratch.

## Colab

The same notebook runs in Google Colab, where students need no local install at
all. `notebook.py` stays the source of truth; the `.ipynb` is **generated** —
never hand-edit it:

```bash
python make_colab.py --repo https://github.com/<owner>/<bundle> --pin <tag>
```

That exports via marimo, swaps the conda instructions above for a setup cell
that pip-installs SchNetPack at `<tag>` and clones the bundle, and writes
`notebook.ipynb`. Commit it, and link students at:

```
https://colab.research.google.com/github/<owner>/<bundle>/blob/main/notebook.ipynb
```

Pin a **tag**, not a branch — a branch moves under a class mid-course.

Nothing else differs between the two. `viz.py` builds one self-contained page
either way and only its last step branches on the frontend, so the viewers,
the scrubber and §8's collapsible solution behave the same in both. The one
rule that branch enforces: every page is sealed in its own iframe, because the
pages name their controls with fixed ids (`slider`, `grid`, `play`) and a
Jupyter document renders every output into one shared DOM — two unsealed pages
there and the second one's script dies on a redeclared `const`, leaving a live
slider above an empty grid.
