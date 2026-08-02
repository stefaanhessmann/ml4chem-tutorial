import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # SchNetPack: from force fields to generative models

    **ML4Chem hands-on tutorial.** From loading your own dataset to
    *generating* molecular structures with
    [SchNetPack](https://github.com/atomistic-machine-learning/schnetpack)
    and its new `schnetpack.generative` subpackage.

    **Contents**

    1. **Setup**
    2. **Introduction** — SchNetPack, force fields, and why generative models
    3. **Using your own data** — ASE databases, transforms, batches
    4. **Generative models in SchNetPack** — the roadmap
    5. **Data augmentation** — noising structures, defining labels
    6. **Model architecture and training** — a force field on noised structures
    7. **Sampling** — direct denoising, ancestral sampling, and checking what we made
    8. **Your tasks** — steering the sampler: shape guidance and scaffolds

    ## 1. Setup

    Everything you need ships in one folder:

    ```
    ML4Chem-tutorial/
    ├── notebook.py            ← this tutorial
    ├── data/
    │   └── qm9_c4h4n2o2.xyz   ← the dataset: 181 small molecules from QM9
    ├── checkpoints/
    │   ├── gpff.pt            ← the §6 model; that cell loads it unless RETRAIN = True
    │   └── gpff_big.pt        ← a GPFF at research scale (all of QM9); §7's switch swaps it in
    ├── helpers.py             ← small glue code (batch adapters for sampling)
    ├── viz.py                 ← 3D molecule viewer
    └── assets/3Dmol-min.js    ← vendored viewer library (works offline)
    ```

    Create an environment, install SchNetPack from the tutorial branch, and
    start the notebook:

    ```bash
    conda create -n ml4chem python=3.12
    conda activate ml4chem
    pip install "git+https://github.com/atomistic-machine-learning/schnetpack.git@sh/v3"
    pip install marimo matplotlib scipy rdkit
    marimo edit notebook.py
    ```

    This is a [marimo](https://marimo.io) notebook: cells form a dependency
    graph and re-run automatically when something they use changes. Every cell
    is plain Python — everything here works the same in a script.

    **Hardware.** The first code cell sets `DEVICE` to a GPU when one is
    available and to the CPU otherwise; the model and every batch follow it,
    and nothing below is device-specific. Everything here is comfortable on
    either: §6 ships its trained model, so the one genuinely GPU-hungry step —
    training it yourself, with `RETRAIN = True` — is opt-in. That one is
    roughly ten minutes on a GPU and much longer on a CPU.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Introduction

    **SchNetPack** is an open-source toolbox for atomistic machine learning.
    It is best known for **machine-learned force fields (MLFFs)**: neural
    networks such as SchNet, PaiNN and SO3net that predict energies, forces
    and other properties from atomic positions — plus the data pipelines to
    train them and interfaces to run molecular dynamics with them. You bring
    structures, it learns the potential energy surface.

    **Generative models** answer a different question. A force field evaluates
    structures it is *given*; a generative model *produces* structures — it
    learns the distribution behind a dataset's geometries so that new,
    plausible ones can be drawn from it. Diffusion-type models have become the
    leading approach to this, and SchNetPack 3 ships their building blocks in
    `schnetpack.generative`.

    **The two are close cousins.** A diffusion model's denoising network takes
    a (noised) structure and predicts a 3-vector per atom — architecturally
    that is exactly a force field of the non-energy-conserving kind. **GPFF**,
    the Gaussian pseudo-force field, makes the kinship literal: the quantity it
    learns *is* a pseudo force, pointing every atom back toward the clean
    structure. The theory of diffusion models and GPFF was covered in the
    lecture; this notebook is the hands-on counterpart — we build the training
    data, the network and the sampler with SchNetPack, and generate molecules.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Using your own data in SchNetPack

    SchNetPack reads training data from an **ASE-backed SQLite database**.
    Whatever format your structures are in, the first step is always the same:
    put them in a db. Our raw data is an xyz file with all **181 isomers of
    C₄H₄N₂O₂** from QM9 — one fixed composition, so the generative model later
    only has to learn *where the atoms go*, not which atoms to place.

    Two calls build the database: `ASEAtomsData.create` declares the stored
    properties with their units, `add_systems` fills it with `ase.Atoms`
    objects. We store the U0 energies that come with the xyz — this tutorial
    never uses them, but a database declares its properties up front, and real
    datasets carry them.
    """)
    return


@app.cell
def _():
    import os

    import numpy as np
    import torch
    from ase.io import read

    from schnetpack.data import ASEAtomsData, AtomsLoader

    # Everything downstream follows this one line: the GPU when this machine —
    # or this Colab runtime — has one, the CPU otherwise.
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else "."
    XYZ_FILE = os.path.join(HERE, "data", "qm9_c4h4n2o2.xyz")
    DB_PATH = os.path.join(HERE, "data", "qm9_c4h4n2o2.db")

    molecules = read(XYZ_FILE, index=":")  # a list of ase.Atoms
    numbers = molecules[0].get_atomic_numbers().tolist()  # every isomer: same composition

    if not os.path.exists(DB_PATH):
        db = ASEAtomsData.create(
            DB_PATH, distance_unit="Ang", property_unit_dict={"energy": "eV"}
        )
        db.add_systems(
            atoms_list=molecules,
            property_list=[
                {"energy": np.array([m.info["energy_U0"]])} for m in molecules
            ],
        )
    f"{len(molecules)} × {molecules[0].get_chemical_formula()} → {DB_PATH} · running on {DEVICE}"
    return ASEAtomsData, AtomsLoader, DB_PATH, DEVICE, HERE, numbers, os, torch


@app.cell
def _(mo):
    mo.md(r"""
    ### Transforms and batches

    A dataset item is a **dict of tensors** — positions `R`, atomic numbers
    `Z`, ... — keyed by the names in `schnetpack.properties`. That dict is the
    universal interface: every SchNetPack model consumes it, and everything we
    build below writes into it.

    **Transforms** are per-structure preprocessing steps owned by the dataset;
    they run each time an item is loaded. Here: center each molecule
    (`SubtractCenterOfGeometry`), build its neighbor list
    (`MatScipyNeighborList` — with a 10 Å cutoff every atom sees every other),
    and cast to float32 (`CastTo32`).

    **Batches are not padded.** `AtomsLoader` concatenates the atoms of all
    molecules along one axis and records in `idx_m` which atom belongs to
    which molecule — a batch of 8 twelve-atom molecules is one `(96, 3)`
    position tensor.
    """)
    return


@app.cell
def _(ASEAtomsData, AtomsLoader, DB_PATH):
    import schnetpack.transform as trn
    from schnetpack import properties

    dataset = ASEAtomsData(
        DB_PATH,
        load_properties=[],  # skip the stored energies — not needed here
        transforms=[
            trn.SubtractCenterOfGeometry(),
            trn.MatScipyNeighborList(cutoff=10.0),  # fully connects a molecule
            trn.CastTo32(),
        ],
    )
    loader = AtomsLoader(dataset, batch_size=8, shuffle=False)
    batch = next(iter(loader))
    {
        "R": tuple(batch[properties.R].shape),
        "Z": tuple(batch[properties.Z].shape),
        "idx_m": batch[properties.idx_m].tolist()[:14] + ["..."],
    }
    return batch, dataset, properties, trn


@app.cell
def _(batch):
    import viz

    # the loaded batch in 3D — drag to rotate
    viz.show_batch(batch, cell_px=170)
    return (viz,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Generative models in SchNetPack: the roadmap

    From the code perspective, a diffusion-based generative model consists of
    **three parts**, and each part gets one section:

    | | part | what it is | where |
    |---|---|---|---|
    | a | **data augmentation** | the forward process: noising structures and computing labels, inside the dataloader | §5 |
    | b | **model architecture** | an MLFF-shaped network applied to noised structures | §6 |
    | c | **sampling** | iterating the trained model from noise to structures | §7 |

    **Training** (§6) is where a and b meet; after sampling we **validate the
    generated structures** (§7). The goal, stated plainly: train GPFF on our
    181-molecule QM9 slice and generate new C₄H₄N₂O₂ geometries.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Data augmentation: making training data from noise

    A force field trains on labels the dataset ships — energies, forces. A
    diffusion model **manufactures its own training data**: take a clean
    structure, blend noise into it, and ask the network for the way back.
    `schnetpack.generative` frames the blending as an **interpolation**
    between the data $x_0$ and an endpoint $x_1$ drawn from noise:

    $$x_t = a(t)\,x_0 + b(t)\,x_1,$$

    with $a(0) = 1, b(0) = 0$ (the data) and $b(1) = 1$ (pure noise). A
    `Process` object owns the schedules $a$, $b$, the resulting noise level
    $\sigma(t)$, and the endpoint draw.

    ### Generating noisy structures

    The two standard processes, both in the library:

    - **`VP`** (variance preserving — the DDPM family): the data is scaled
      away ($a \to 0$) while noise of fixed scale blends in; the total
      variance stays constant.
    - **`VE`** (variance exploding — score matching): the data is never scaled
      ($a \equiv 1$); noise whose scale grows geometrically from
      $\sigma_\text{min}$ to $\sigma_\text{max}$ is *added* until it drowns
      the structure. $\sigma_\text{max}$ must at least match the data scale —
      rule of thumb: the largest pairwise distance, ~7.7 Å here. Too small and
      the endpoint still remembers the data; too large and training wastes
      capacity — and *neither failure is loud*. We take **30 Å**, which is
      what the ready-trained generation model of §7 uses: staying on one
      process keeps every model in this notebook interchangeable.

    For all illustrations we use a single molecule: an elongated, open-chain
    isomer whose shape is easy to track through the noise. Below, that chain
    under both processes with the same underlying noise draw (slider = $t$):
    VP shrinks it into a small fixed-size cloud, VE leaves it in place and
    buries it under a 30 Å one.
    """)
    return


@app.cell
def _(AtomsLoader, batch, dataset, properties, torch, viz):
    from schnetpack.generative import VE, VP

    # the process every later section shares
    SIGMA_MIN, SIGMA_MAX = 0.05, 30.0  # as the §7 generation model was trained
    process = VE(sigma_min=SIGMA_MIN, sigma_max=SIGMA_MAX)
    vp = VP(scale=float(batch[properties.R].std()))  # VP wants the data scale

    CHAIN_IDX = 90  # the most elongated open-chain isomer of the dataset
    chain = next(iter(AtomsLoader(dataset, sampler=[CHAIN_IDX])))
    x0 = chain[properties.R]  # the structure every illustration below noises

    # a trajectory is just interpolate() evaluated along a grid of times
    torch.manual_seed(3)
    t_noise = torch.linspace(0.0, 1.0, 13)
    z_noise = torch.randn_like(x0)  # shared draw — only its scale differs
    frames_vp = [vp.interpolate(x0, vp.prior.std * z_noise, t) for t in t_noise]
    frames_ve = [
        process.interpolate(x0, process.prior.std * z_noise, t) for t in t_noise
    ]

    # ghost_id=0: the clean chain stays faintly in place behind xₜ
    viz.show_trajectory(
        {"VP": frames_vp, "VE": frames_ve},
        chain,
        times=t_noise.tolist(),
        start=True,
        end=True,
        ghost_id=0,
        panel_labels=("x₀ (data)", "xₜ", "x₁ (prior)"),
    )
    return SIGMA_MAX, chain, process, x0


@app.cell
def _(mo):
    mo.md(r"""
    ### Defining the labels: what should the network predict?

    Noised structures are only half of the training data — the other half is
    the **label**, and choosing it is the second axis: the `Parametrization`.
    GPFF's choice is the **pseudo force**

    $$F = 2\,(x_0 - x_t), \qquad \mathrm{RMS}(F) = 2\,\sigma(t):$$

    direction and distance straight back to the clean structure, so getting
    home is one addition, $\hat x_0 = x_t + F/2$. And because its magnitude
    *carries* the noise level, a GPFF network needs **no time input at all** —
    remember this for §6. The second identity is that fact read backwards: a
    trained GPFF can be *asked* how noisy a structure is, by measuring how long
    the arrows are that it draws on it.

    The lecture introduced the alternatives, and the library has them too:
    `EpsParametrization` predicts the unit noise that was mixed in (same
    typical arrow length at every noise level — the DDPM convention), and
    `ScoreParametrization` the score, whose magnitude runs like $1/\sigma$.

    Below, the same noising path three times, with the **training target drawn
    as an arrow on every atom** — eps, score, pseudo force. The eps arrows
    keep their size everywhere; the score arrows explode as $\sigma \to 0$;
    the pseudo-force arrows are long in deep noise and shrink to nothing as
    the structure comes home, which is $\mathrm{RMS}(F) = 2\sigma$ made
    visible. That last row is drawn at **half length**, as $F/2 = x_0 - x_t$:
    the arrow is then the offset itself, and its tip is where the atom is
    headed.
    """)
    return


@app.cell
def _(chain, process, torch, viz, x0):
    from schnetpack.generative import (
        EpsParametrization,
        PseudoForceParametrization,
        ScoreParametrization,
    )

    force_param = PseudoForceParametrization()  # F = 2 (x0 - x_t) — GPFF's target
    eps_param = EpsParametrization()
    score_param = ScoreParametrization()

    # one path, three targets: `target` turns the same (x0, x1, t) into
    # whichever field the network is asked to predict
    torch.manual_seed(3)
    t_param = torch.linspace(1.0, 0.2, 13)
    x1_param = process.prior.sample_like(x0)
    ts_param = [torch.full((len(x0),), float(ti)) for ti in t_param]
    frames_param = [process.interpolate(x0, x1_param, t) for t in ts_param]
    targets = {
        # the pseudo force is drawn at half length: F/2 = x0 - x_t is the
        # offset itself, so each arrow lands exactly on the clean structure
        name: [scale * p.target(process, x0, x1_param, t) for t in ts_param]
        for name, p, scale in (
            ("eps target", eps_param, 1.0),
            ("score target", score_param, 1.0),
            ("pseudo-force target (F/2)", force_param, 0.5),
        )
    }

    viz.show_frames(
        {name: frames_param for name in targets},
        chain,
        n_frames=5,
        times=t_param.tolist(),
        vectors=targets,
    )
    return (force_param,)


@app.cell
def _(mo):
    mo.md(r"""
    Two further axes exist that this section does not vary: the **coupling**
    (how the drawn $(x_0, x_1)$ pairs are matched up) and the **prior** (what
    $x_1$ is drawn from) — equally swappable constructor arguments. The prior
    comes back in §8b, where replacing it is half the task; the coupling stays
    for the `schnetpack.generative` API docs.

    ### Wrapping it into a transform

    Building diffusion training data is preprocessing — so it is a
    **transform** like the ones in §3. `Diffuse(process, parametrization)`
    runs the forward process inside the dataloader: per structure it draws a
    time, noises the positions, and writes the label into the item dict.

    *Where along the path* those draws land is its own choice — the **time
    sampler**. Uniform $t$ on a geometric schedule is *log-uniform* in
    $\sigma$: equal weight to every decade from 0.05 to 30 Å, so over a third
    of the budget goes above 3 Å, where the target is nearly the endpoint
    itself and there is little to learn. `LogNormalSigmaTimes` states the density
    where the statement means something — in $\sigma$:
    $\log\sigma \sim \mathcal N(-0.7,\, 1.2^2)$, a median of ~0.5 Å with most
    of the mass between 0.15 and 1.7 Å — exactly the band where bonds live and
    denoising is genuinely hard. `truncate=True` redraws the few samples that
    land outside the schedule's range instead of piling them onto its ends.
    This is GPFF's own training density (and EDM's, for images).

    The process does the converting, through `t_of_sigma`. Here that map is
    closed-form and the arithmetic is pretty: on a geometric schedule $t$ is
    *affine* in $\log\sigma$, so a log-normal over $\sigma$ is exactly a
    normal over $t$ — $t \sim \mathcal N(0.36,\, 0.19^2)$ for these numbers.

    Only the transform order needs thought:

    1. `SubtractCenterOfGeometry` — diffusion lives in the centered frame,
       and the prior draws its endpoints there too (`GaussianPrior` centers
       per molecule by default): a translation-invariant network could never
       predict a displacement of a whole structure, so an off-center endpoint
       would be unlearnable noise in every label;
    2. `Diffuse` — overwrites `R` with $x_t$, writes the label
       `"pseudo_force"` and the time `"t"`;
    3. `AllToAllNeighborList` — **after** noising, and every pair is a
       neighbor. A *distance*-based list is the wrong tool here: one built on
       $x_t$ at one noise level is wrong at another, and a cutoff wide enough
       for a fully noised cloud (~90 Å across) would return every pair anyway
       — at the cost of searching for them. Saying it directly is cheaper and
       stays correct at every $t$. Pairs the model's cutoff function
       downweights to zero cost nothing;
    4. `CastTo32`.

    An ordinary MSE against `"pseudo_force"` is then the whole training
    objective — no special training loop anywhere.
    """)
    return


@app.cell
def _(ASEAtomsData, AtomsLoader, DB_PATH, force_param, process, trn):
    from schnetpack.generative import Diffuse, LogNormalSigmaTimes

    CUTOFF = 150.0  # must cover *noised* structures — clouds ~90 Å across

    # train mostly around half an Ångström of displacement — GPFF's density
    t_sampler = LogNormalSigmaTimes(process, mean=-0.7, std=1.2, truncate=True)

    diffused = ASEAtomsData(
        DB_PATH,
        load_properties=[],
        transforms=[
            trn.SubtractCenterOfGeometry(),
            # the same schedule the frames above walked, sampled where it helps
            Diffuse(
                process,
                force_param,
                t_sampler=t_sampler,
                label_key="pseudo_force",
                time_key="t",
            ),
            trn.AllToAllNeighborList(),
            trn.CastTo32(),
        ],
    )
    # a small draw from the pipeline — only so the picture below stays a
    # picture; a hundred viewers on one page is not one
    peek = next(iter(AtomsLoader(diffused, batch_size=10, shuffle=True)))
    {key: tuple(peek[key].shape) for key in ("_positions", "pseudo_force", "t")}
    return CUTOFF, diffused, peek


@app.cell
def _(mo):
    mo.md(r"""
    Those batches *are* the training set of our generative model — so look at
    one. Below, ten structures drawn from the same loader: each at its own
    randomly drawn time, captioned with its noise level, with the
    **pseudo-force label drawn as an arrow on every atom** — again at half
    length, $F/2$, so each arrow ends where its atom belongs.

    Read it as a difficulty gradient. At $\sigma \lesssim 0.5$ Å the molecule
    is intact and the arrows are tiny corrections; at $\sigma$ of several
    Ångström there is no molecule left and the arrows span the whole cloud —
    the label's scale runs with $\sigma$, which is exactly what the loss in
    §6 will have to compensate. And notice what the time sampler did: most
    draws sit below ~2 Å, where denoising is hard but learnable — the rare
    deep-noise structure is in the mix, just no longer the bulk of it.
    """)
    return


@app.cell
def _(peek, process, properties, viz):
    # one box per structure of the batch, captioned with its own noise level
    sigma_peek = process.sigma(peek["t_structure"])
    viz.show_trajectory(
        [peek[properties.R]],
        peek,
        # F/2 = x0 - x_t, so each arrow ends on the clean structure
        vectors=[peek["pseudo_force"] / 2],
        titles=[f"σ = {float(s):.2f} Å" for s in sigma_peek],
        cell_px=170,
        zoom=1.0,
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. Model architecture

    A diffusion denoiser uses the **same architecture as an MLFF** — of the
    *non-energy-conserving* kind: instead of predicting one energy per
    molecule and differentiating it, the network reads out a 3-vector per atom
    directly. Diffusion models generally need one extra ingredient, time
    conditioning, because the same noised geometry means a different noise
    target at a different $t$. **GPFF doesn't**: the pseudo force's own
    magnitude carries the noise level, so the network here is *exactly* an
    ordinary force field.

    `NeuralNetworkPotential` assembles a model from three stages, each a plain
    `nn.Module` acting on the batch dict:

    1. **Input modules** — geometric preparation: `PairwiseDistances` turns
       positions + neighbor list into distance vectors.
    2. **Representation** — message passing from atomic numbers and distances
       to per-atom features. **PaiNN** is *equivariant*: besides scalars it
       carries vector features that rotate with the molecule — which is what
       lets a head output a well-behaved vector per atom. (SchNet and SO3net
       are drop-in alternatives.)
    3. **Output modules** — the head. `AtomwiseVector` reads PaiNN's vector
       features and predicts a 3-vector per atom. (An energy model would end
       in `Atomwise` instead: a scalar per atom, summed per molecule.)

    The cutoff is 150 Å because this force field sees *noised* structures — a
    fully noised cloud is ~90 Å across. That number then decides a second one.
    A `GaussianRBF` spreads its basis functions evenly over the cutoff, so 100
    of them across 150 Å put one every 1.5 Å — wider than a bond, which leaves
    a 1.0 Å contact and a 1.4 Å bond with 98% identical embeddings, and a
    denoiser that cannot tell a clash from a bond will happily generate both.
    600 puts one every 0.25 Å. `norm_epsilon=1` is the other concession to
    noised inputs: PaiNN normalizes each pair direction as
    $r_{ij}/(d_{ij} + 1)$ rather than $r_{ij}/d_{ij}$, which stays finite when
    two atoms of a noise cloud land on top of each other.
    """)
    return


@app.cell
def _(CUTOFF, DEVICE, torch):
    import schnetpack.nn as snn
    from schnetpack.model import (
        AtomwiseVector,
        NeuralNetworkPotential,
        PaiNN,
        PairwiseDistances,
    )

    torch.manual_seed(0)
    gpff_net = NeuralNetworkPotential(
        representation=PaiNN(
            n_atom_basis=128,
            n_interactions=4,
            radial_basis=snn.GaussianRBF(n_rbf=600, cutoff=CUTOFF),
            cutoff_fn=snn.CosineCutoff(CUTOFF),
            norm_epsilon=1.0,  # dir_ij = r_ij / (d_ij + 1): smooth at d → 0
        ),
        input_modules=[PairwiseDistances()],
        output_modules=[
            AtomwiseVector(n_in=128, n_layers=3, output_key="pseudo_force_pred")
        ],
    ).to(DEVICE)
    f"GPFF model: {sum(p.numel() for p in gpff_net.parameters()):,} parameters on {DEVICE}"
    return (
        AtomwiseVector,
        NeuralNetworkPotential,
        PaiNN,
        PairwiseDistances,
        gpff_net,
        snn,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ### Training

    Model and data augmentation meet in an ordinary PyTorch loop: pull a batch
    from an `AtomsLoader` over §5's diffused dataset, compare the model's
    prediction against the `"pseudo_force"` label, step the optimizer. Nothing
    in the loop knows it is training a generative model — the transforms did
    that part.

    The loader is worth one look, because the obvious way to build it is slow.
    181 structures at batch 64 is under three batches per epoch, and a
    dataloader spends its first batches filling a prefetch queue that it then
    throws away at the epoch boundary — so with tiny epochs the workers never
    get ahead and the GPU waits. Handing it a `RandomSampler` with
    `replacement=True` and `num_samples = BATCH * STEPS` turns the whole run
    into *one* epoch of 12000 batches, drawn with replacement. The workers
    pipeline, preprocessing hides behind the backward pass, and the training
    loop becomes a single `for` over the loader.

    The one line worth arguing about is the **loss weight**
    $\min(1/\sigma(t)^2, w_\text{max})$. The $1/\sigma^2$ undoes the label's
    $\sigma$-scaling, so that what is being minimized is the *relative* error
    at every noise level rather than the absolute one; without it the deep-noise
    samples, whose labels are tens of Ångström long, drown out everything else.
    The ceiling then decides how much of the small-$\sigma$ end survives, and
    it is easy to set too low: at $w_\text{max} = 1$ it binds for 71% of the
    draws, which flattens the weight back to "every sample counts the same"
    across the whole band where bond lengths are decided. At 100 the weight is
    the honest $1/\sigma^2$ almost everywhere.

    Two things about the loop's shape. **Every step sees fresh
    $(t, \text{noise})$ draws**, because the transforms re-run on every item
    the loader hands out — the dataset is effectively infinite, and a model
    that saw a fixed set of noised structures (say, from an
    `itertools.cycle` over a cached list of batches) would memorize them
    instead of learning the denoising field. And the weights that get used
    downstream are an **exponential moving average** of the ones the optimizer
    visited: a few thousand steps is a noisy place to stop, and the averaged
    model samples visibly better than whichever iterate happened to be last.

    The cell below **loads** that run by default, from
    `checkpoints/gpff.pt` — twelve thousand steps is a quarter of an hour on a
    laptop GPU and considerably worse on a CPU, which is a long time to sit and
    watch. Set `RETRAIN = True` to run it yourself instead; the loop is right
    there, and it overwrites the checkpoint when it finishes. Either way the
    curve below is real: the checkpoint stores its loss history alongside its
    weights. Don't over-read it — it plateaus well above zero because every
    noise level keeps an irreducible error, and that is healthy.
    """)
    return


@app.cell
def _(AtomsLoader, DEVICE, HERE, diffused, gpff_net, os, process, torch):
    import matplotlib.pyplot as plt
    from tqdm.auto import tqdm

    from helpers import to_device

    CKPT = os.path.join(HERE, "checkpoints", "gpff.pt")
    RETRAIN = False  # True: run the loop below instead of loading the checkpoint

    STEPS, BATCH, LR_START, LR_END, EMA_DECAY = 12000, 64, 1e-3, 1e-5, 0.999

    def gpff_loss(pred, inputs):
        # 1/sigma^2 undoes the label's sigma-scaling; the ceiling keeps the
        # small-sigma end — where bonds are decided — from being flattened away
        weight = (1.0 / process.sigma(inputs["t"]) ** 2).clamp(max=100.0)
        diff = pred["pseudo_force_pred"] - inputs["pseudo_force"]
        return (weight[:, None] * diff**2).mean()

    if os.path.exists(CKPT) and not RETRAIN:
        # the checkpoint carries its loss curve as well as its weights, so the
        # plot below is the real one from the run that produced them
        ckpt_state = torch.load(CKPT, weights_only=True, map_location=DEVICE)
        gpff_net.load_state_dict(ckpt_state["state_dict"])
        history = [tuple(h) for h in ckpt_state["history"]]
    else:
        # the whole run as one epoch of STEPS batches, drawn with replacement —
        # which is what lets the workers stay ahead of the GPU (see above)
        train_loader = AtomsLoader(
            diffused,
            batch_size=BATCH,
            sampler=torch.utils.data.RandomSampler(
                diffused, replacement=True, num_samples=BATCH * STEPS
            ),
            num_workers=4,
            persistent_workers=True,
        )

        optimizer = torch.optim.Adam(gpff_net.parameters(), lr=LR_START)
        # decay the step size geometrically from LR_START to LR_END across the
        # run — so the last steps only polish
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=(LR_END / LR_START) ** (1 / STEPS)
        )
        # the running average of the weights, which is what samples at the end
        ema = {k: v.detach().clone().float() for k, v in gpff_net.state_dict().items()}

        history = []
        steps = tqdm(train_loader, desc="step", unit="it", total=STEPS)
        for step, train_batch in enumerate(steps):
            train_batch = to_device(train_batch, DEVICE)  # transforms ran on CPU
            loss = gpff_loss(gpff_net(train_batch), train_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            with torch.no_grad():
                for key, value in gpff_net.state_dict().items():
                    ema[key].mul_(EMA_DECAY).add_(value.float(), alpha=1.0 - EMA_DECAY)

            history.append((step, loss.item()))
            if step % 50 == 0:
                steps.set_postfix(
                    loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.1e}"
                )

        gpff_net.load_state_dict({key: v.to(DEVICE) for key, v in ema.items()})
        torch.save({"state_dict": gpff_net.state_dict(), "history": history}, CKPT)

    gpff_model = gpff_net.eval()  # downstream cells use the *trained* model

    loss_fig, loss_ax = plt.subplots(figsize=(6, 3))
    loss_ax.plot(*zip(*history), lw=0.7, alpha=0.8)
    loss_ax.set_yscale("log")
    loss_ax.set_xlabel("step")
    loss_ax.set_ylabel("weighted pseudo-force MSE")
    loss_ax.grid(alpha=0.3)
    return gpff_model, plt, to_device


@app.cell
def _(mo):
    mo.md(r"""
    ## 7. Sampling

    ### The model that generates

    Everything from here on — both samplers and your tasks in §8 — runs on the
    model §6 just trained, a **teaching-sized** network on 181 structures.
    Denoising has to be accurate exactly where geometry is decided, at
    $\sigma \lesssim 0.3$ Å where bond lengths live — which is precisely
    where the time sampler concentrated its training. How well that worked is
    what the validation cells below measure; don't expect perfection from a
    model this size trained on 181 molecules. Roughly half of what it draws is
    a chemically valid molecule, and essentially all of it is sane geometry —
    which, at this scale, is the honest answer.

    For comparison the bundle also ships `checkpoints/gpff_big.pt`: a GPFF
    trained the same way — same pseudo-force target, same VE process, same
    $\sigma$-focused time sampling — but at research scale: **5.1M
    parameters, trained on all ~130k molecules of QM9**. The assembly below is
    the same `NeuralNetworkPotential` of §6 — same process, same cutoff, same
    radial basis — with
    `USE_BIG_MODEL` swaps it into every sampling and validation cell that
    follows. Flip it after one pass through §7: how far the numbers move is
    the most honest measure of what scale buys.
    """)
    return


@app.cell
def _(
    AtomwiseVector,
    CUTOFF,
    DEVICE,
    HERE,
    NeuralNetworkPotential,
    PaiNN,
    PairwiseDistances,
    os,
    snn,
    torch,
):
    big_net = NeuralNetworkPotential(
        representation=PaiNN(
            n_atom_basis=256,
            n_interactions=4,
            radial_basis=snn.GaussianRBF(n_rbf=600, cutoff=CUTOFF),
            cutoff_fn=snn.CosineCutoff(CUTOFF),
            norm_epsilon=1.0,  # this run normalized pair directions as r / (d + 1)
        ),
        input_modules=[PairwiseDistances()],
        output_modules=[
            AtomwiseVector(n_in=256, n_layers=3, output_key="pseudo_force_pred")
        ],
    ).to(DEVICE)
    big_net.load_state_dict(
        torch.load(
            os.path.join(HERE, "checkpoints", "gpff_big.pt"),
            weights_only=True,
            map_location=DEVICE,
        )["state_dict"]
    )
    big_model = big_net.eval()

    f"generation model: {sum(p.numel() for p in big_net.parameters()):,} parameters on {DEVICE}"
    return (big_model,)


@app.cell
def _(big_model, gpff_model):
    USE_BIG_MODEL = False  # True: sample with the research-scale QM9 model

    # the model that generates from here on; both were trained against the
    # same process, so nothing else changes with it
    gen_model = big_model if USE_BIG_MODEL else gpff_model
    return (gen_model,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Ancestral sampling

    The **classical** route to a sample — the reverse process of the lecture,
    and what every time-conditioned diffusion model uses — walks a
    *prescribed* ladder of noise levels
    $\sigma_N > \sigma_{N-1} > \dots > \sigma_0$ and takes one exact step
    down each rung.

    SchNetPack assembles that from four parts, which is the same decomposition
    as §5's — process and parametrization, plus two that only sampling needs:

    | part | what it decides | here |
    |---|---|---|
    | `process` | the noise schedule $\sigma(t)$ | the `VE` of §5 |
    | `parametrization` | what the network's output means | pseudo force |
    | `integrator` | how one step down the ladder is taken | `Ancestral` |
    | `grid` | where the rungs sit | uniform in $t$ (the default) |

    `Ancestral` takes the exact step: it converts the prediction into an
    estimate $\hat x_0$, then draws from the closed-form Gaussian for
    $x_{\sigma_{k-1}}$ given $x_{\sigma_k}$ and $\hat x_0$ — no approximation
    beyond the model's own error. On a VE-family process that reduces to the
    familiar score-form update. And because VE's $\sigma$ grows
    *geometrically* in $t$, a uniform grid already gives the geometric ladder
    score matching wants — rungs that bunch up where $\sigma$ is small — so no
    schedule code is needed. (Warping the grid is still an option, and the
    `grid` slot is where it would go; see the outro.)

    Two pieces of glue from `helpers.py`. `fully_connected_batch` lays out 8
    copies of our composition as one flat batch — the topology the model needs
    (`Z`, `idx_m`, and a neighbor list) for molecules that don't exist yet.
    `make_model_fn` adapts our batch-dict model to the plain-tensor contract
    `model(x, t, cond)` the sampler expects, binding everything except the
    positions. The neighbor list is static: atoms move at every step, but the
    cutoff function handles the changing distances, so nothing is rebuilt.

    A third, `recording_model_fn`, is what makes the viewer below a **movie**
    rather than a still. A sampler returns the structure it ended on and
    nothing else — but every step passes its state through the model, so
    wrapping the model captures the whole run without the sampler knowing.
    One box per molecule, as before — they just move now. Scrub the slider;
    the frames carry the grid's own times, so the caption reads out the rung
    each one sits on, $t = 1$ down to $0$: the ladder comes down *gradually*,
    and the structure only appears over the last handful of rungs.

    One thing to expect on frame 0. The prior is $\sigma_\text{max} = 30$ Å, so
    the starting cloud is ~80 Å across against a ~3 Å molecule — a 25× range no
    single camera can hold. The view is framed on the finished structure and
    pulled back (`zoom=0.35`), so the earliest frames spill past the edges and
    the atoms fly in from outside. That gap *is* the scale the model closes.
    """)
    return


@app.cell
def _(
    DEVICE,
    force_param,
    gen_model,
    numbers,
    process,
    properties,
    to_device,
    torch,
    viz,
):
    from helpers import fully_connected_batch, make_model_fn, recording_model_fn
    from schnetpack.generative import Sampler
    from schnetpack.generative.integrators import Ancestral

    N_LADDER = 64

    torch.manual_seed(2)
    # 8 molecules-to-be, laid out where the model lives
    sampling_batch = to_device(fully_connected_batch(numbers, n_mol=8), DEVICE)
    model_fn = make_model_fn(gen_model, sampling_batch, "pseudo_force_pred")
    n_total = int(sampling_batch[properties.n_atoms].sum())

    # process + parametrization as before, plus the two sampling-only parts;
    # `grid` is left at its default (uniform in t = geometric in sigma on VE)
    ancestral = Sampler(process, force_param, integrator=Ancestral())

    # the sampler returns the final structure only; wrapping the model keeps
    # every state it was asked about, which is the trajectory
    watched_anc, ancestral_frames = recording_model_fn(model_fn)
    with torch.no_grad():
        x_ancestral = ancestral.sample(
            watched_anc,
            shape=(n_total, 3),
            n_steps=N_LADDER,
            context=sampling_batch,
            device=DEVICE,
        )
    ancestral_frames.append(x_ancestral)

    # this sampler *does* have a time grid, so the frames can be captioned with
    # it: the rungs the ladder actually stepped through, t = 1 down to t = 0
    ladder_t = ancestral.grid(ancestral.t_max, ancestral.t_min, N_LADDER)

    # zoom < 1 pulls the camera back: it frames the final molecule, and the
    # first frames are a wide noise cloud that should not fly off the panel
    viz.show_trajectory(
        ancestral_frames,
        sampling_batch,
        times=ladder_t.tolist(),
        zoom=0.35,
        cell_px=170,
        frame_ms=120,  # 65 frames — play them faster than the default
    )
    return (
        fully_connected_batch,
        make_model_fn,
        model_fn,
        n_total,
        recording_model_fn,
        sampling_batch,
        x_ancestral,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ### Direct denoising

    GPFF's own sampler is unusual: no schedule, no time grid, nothing that
    tracks how noisy the iterate is. It can afford that because the pseudo
    force is the way home in *one* step, $\hat x_0 = x + F/2$. Taken from pure
    noise that single jump lands on the model's *conditional mean* over every
    structure that could hide under it — a blob, not a molecule — so sampling
    iterates instead: denoise, re-noise a little, repeat.

    That loop is `DirectDenoisingSampler`. At iteration $k$ of $N$ it injects
    $\lambda\,(1 - k/N)$ of noise before the model call, decaying to zero. It
    never asks what noise level its iterate sits at, which is only possible
    because the model doesn't either.

    $\lambda$ deserves a moment, because $\lambda = 0$ — the plain repeated
    jump, no injection at all — looks like the natural choice and is the wrong
    one. With nothing put back the loop is a deterministic fixed-point
    iteration, and it converges to whatever the model's map happens to
    attract: for a live-trained model that means atoms stranded off the
    structure, and occasionally a run that leaves the finite numbers behind
    altogether. The injection keeps the iterate inside the band the model was
    trained on, and the decay walks it down from there. Here $\lambda = 1$,
    and on the validation counts below it is the difference between a third of
    the samples passing and nearly all of them.

    Compare the two samplers on cost. The ladder above took **64** model
    calls; this loop takes **60**. They are close here because this model is
    small and wants the iterations — the loop's budget is a dial rather than a
    schedule, and a stronger model gets away with far fewer (flip
    `USE_BIG_MODEL` and 15 is plenty). Both are `schnetpack.generative`
    one-liners over the same trained model, so swapping them is a one-line
    experiment.

    Watch the difference in the movie. Where the ladder descended gradually
    and only revealed a structure near the bottom, direct denoising is at
    molecular size after two or three calls and spends everything after that
    tidying up; the last frame is the sample. Same model, same starting
    noise — only the route differs.
    """)
    return


@app.cell
def _(
    DEVICE,
    force_param,
    model_fn,
    n_total,
    process,
    recording_model_fn,
    sampling_batch,
    torch,
    viz,
):
    from schnetpack.generative import DirectDenoisingSampler

    torch.manual_seed(2)
    direct = DirectDenoisingSampler(process, force_param, stochastic_lambda=1.0)

    # draw the start from the prior — a 30 Å cloud per molecule — then run
    # the denoising loop
    x_init = direct.prior.sample((n_total, 3), device=DEVICE, context=sampling_batch)

    watched_fn, direct_frames = recording_model_fn(model_fn)
    with torch.no_grad():
        x_direct = direct.denoise(watched_fn, x_init, n_steps=60)
    direct_frames.append(x_direct)  # ...plus the structure it ended on

    viz.show_trajectory(
        direct_frames, sampling_batch, zoom=0.35, cell_px=170, frame_ms=120
    )
    return DirectDenoisingSampler, x_direct


@app.cell
def _(mo):
    mo.md(r"""
    ### Did we actually make molecules?

    A rendered batch can look right and still hide broken geometry — so turn
    "it looks like molecules" into numbers. The simplest checks live on each
    structure's nearest-neighbor distances: two atoms closer than **0.7 Å**
    are fused (a *clash*), an atom whose nearest neighbor is beyond **2.5 Å**
    isn't bonded to anything (a *stray*). Running the same checks on a clean
    dataset batch calibrates them: real bond lengths here are ~1.0–1.5 Å, and
    the dataset scores zero on both counts by construction.

    Both samplers are scored, against the dataset row. Being able to say *how
    many* is the point: that number is what tells you whether a change to the
    model, the process or the sampler actually helped — and, one flip of
    `USE_BIG_MODEL` later, it is what shows what research scale changes.
    """)
    return


@app.cell
def _(batch, properties, sampling_batch, torch, x_ancestral, x_direct):
    def geometry_checks(x, layout):
        """Nearest-neighbor distances per molecule: min = closest pair, max = loneliest atom."""
        rows = []
        for m in range(int(layout[properties.n_atoms].shape[0])):
            pos = x[layout[properties.idx_m] == m]
            eye = torch.eye(len(pos), device=x.device)
            dist = torch.cdist(pos, pos) + eye * 1e6  # mask self-pairs
            nearest = dist.min(dim=1).values
            rows.append((float(nearest.min()), float(nearest.max())))
        return rows

    def geometry_summary(x, layout):
        rows = geometry_checks(x, layout)
        min_pairs, max_gaps = [r[0] for r in rows], [r[1] for r in rows]
        return {
            "all finite": bool(torch.isfinite(x).all()),
            "clashes (pair < 0.7 Å)": sum(mp < 0.7 for mp in min_pairs),
            "strays (gap > 2.5 Å)": sum(g > 2.5 for g in max_gaps),
            "closest pair [Å]": f"{min(min_pairs):.2f} … {max(min_pairs):.2f}",
            "largest gap [Å]": f"{min(max_gaps):.2f} … {max(max_gaps):.2f}",
        }

    {
        "dataset (reference)": geometry_summary(batch[properties.R], batch),
        "ancestral (8)": geometry_summary(x_ancestral, sampling_batch),
        "direct denoising (8)": geometry_summary(x_direct, sampling_batch),
    }
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### From geometry to chemistry

    Distance checks ask whether the geometry is *sane* — they never ask
    whether it is a *molecule*. RDKit can: `rdDetermineBonds` infers a bond
    graph from nothing but the coordinates, and the structure counts as
    **chemically valid** only if that graph works out — every valence
    satisfied as a neutral molecule, no unpaired electrons left over, and
    everything in one connected piece. A structure that passes earns a
    **SMILES** string: the molecule's identity, independent of coordinates.

    That string is what makes this more than a stricter filter. The *valid
    fraction* is the standard headline metric for molecular generative
    models, and SMILES says *which* molecule each sample is — compare them
    against the dataset's to see whether the model reproduced a training
    isomer or found a new one. The dataset batch calibrates the check once
    more: relaxed QM9 structures pass it.

    It is a *strict* judge — a single fused pair already sinks a sample —
    which is what makes it the honest headline number. Watching it is how you
    know a bigger model or a better sampler actually helped: flip
    `USE_BIG_MODEL` and see what a network trained on all of QM9 does to it.
    """)
    return


@app.cell
def _(batch, properties, sampling_batch, x_ancestral, x_direct):
    from ase.data import chemical_symbols
    from rdkit import Chem
    from rdkit.Chem import rdDetermineBonds
    from rdkit.rdBase import BlockLogs

    def rdkit_verdict(Z, pos):
        """(valid, smiles) for one structure, bonds inferred from coordinates."""
        xyz = [str(len(Z)), ""] + [
            f"{chemical_symbols[int(z)]} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}"
            for z, p in zip(Z.tolist(), pos.tolist())
        ]
        try:
            with BlockLogs():  # silence RDKit's complaints about broken samples
                mol = Chem.MolFromXYZBlock("\n".join(xyz))
                rdDetermineBonds.DetermineBonds(
                    mol, charge=0, allowChargedFragments=False, embedChiral=True
                )
                if any(a.GetNumRadicalElectrons() for a in mol.GetAtoms()):
                    return False, ""
                smiles = Chem.CanonSmiles(Chem.MolToSmiles(mol))
                return smiles != "" and "." not in smiles, smiles
        except Exception:  # no consistent bond graph exists for these positions
            return False, ""

    def chemistry_summary(x, layout):
        idx_m = layout[properties.idx_m]
        verdicts = [
            rdkit_verdict(layout[properties.Z][idx_m == m], x[idx_m == m])
            for m in range(int(layout[properties.n_atoms].shape[0]))
        ]
        valid = [s for ok, s in verdicts if ok]
        return {"valid": f"{len(valid)}/{len(verdicts)}", "SMILES": sorted(valid)}

    {
        "dataset (reference)": chemistry_summary(batch[properties.R], batch),
        "ancestral (8)": chemistry_summary(x_ancestral, sampling_batch),
        "direct denoising (8)": chemistry_summary(x_direct, sampling_batch),
    }
    return (rdkit_verdict,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 8. Your tasks: steering the sampler

    Both samplers of §7 draw from the *whole* distribution the model learned:
    hand them noise and they hand back some molecule. Neither takes an
    instruction — and nearly every real use of a generative model is one. Make
    it long and thin. Keep this ring and fill in the rest.

    The obvious way there is to train for it, conditioning the network on
    whatever you want to ask for. Both tasks below take the other route and
    leave the trained model **exactly as it is**: the instruction enters in the
    *sampler*. They share one recipe — every iteration, force the state onto
    the constraint, then let the model repair whatever that broke. Because a
    model call always follows the nudge, the repair is chemistry rather than
    interpolation: what comes out satisfies the constraint *and* survives the
    denoiser. The alternation is the whole trick; either half alone does not
    work.

    Two kinds of constraint worth knowing, one each:

    | | the constraint | how it enters |
    |---|---|---|
    | **a** | a **global, continuous** property — the structure's shape | a linear map on the state, before every model call |
    | **b** | **exact positions** for some atoms — a scaffold | a custom prior, plus rows the loop never updates |

    **How to work them.** Each task ships two cells: one holding a class for
    you to fill in — it *runs* as shipped, it just doesn't steer anything yet —
    and, under it, a cell that samples with whatever you wrote and plays the
    result as a movie. So the loop is short: edit, re-run, watch the movie
    change. Both classes subclass `DirectDenoisingSampler`, whose loop is four
    lines and carries no schedule to stay consistent with, which makes it the
    one to interfere with.

    One thing is switched for you. §7's `USE_BIG_MODEL` decides what §7 samples
    with; **these two tasks always use the research-scale model**, whatever
    that switch says. You are steering a sampler here, and the effect of
    steering is only legible when the model underneath is not the bottleneck —
    the teaching-sized network of §6 would swamp both signals with its own
    error.
    """)
    return


@app.cell
def _(
    DEVICE,
    big_model,
    fully_connected_batch,
    make_model_fn,
    numbers,
    properties,
    to_device,
):
    # §7's layout again, ten molecules wide this time, and bound to the
    # research-scale model whatever USE_BIG_MODEL says
    task_batch = to_device(fully_connected_batch(numbers, n_mol=10), DEVICE)
    task_model_fn = make_model_fn(big_model, task_batch, "pseudo_force_pred")
    n_task = int(task_batch[properties.n_atoms].sum())
    return n_task, task_batch, task_model_fn


@app.cell
def _(mo):
    mo.md(r"""
    ### a) Shape-guided direct denoising

    **Shape** — rod, disc, ball — is a property of the atom cloud rather than
    of its chemistry, and three numbers hold it. Center a structure and form
    the 3×3 covariance of its positions,

    $$C = \frac{1}{n}\sum_i x_i x_i^\top
        = V \operatorname{diag}(\lambda_1 \ge \lambda_2 \ge \lambda_3) V^\top .$$

    The eigenvectors are the **principal axes** and $\lambda_i$ the variance
    along each. Their sum $\operatorname{tr} C$ is the mean squared distance
    from the center — the structure's *size* — and the normalized triple
    $r_i = \lambda_i / \operatorname{tr} C$ is its *shape*: unchanged by
    rotation, and independent of size. Our 181 isomers run from
    $r = (0.41, 0.33, 0.26)$, the roundest of them, to $(0.89, 0.09, 0.02)$,
    the open chain §5 kept noising — and $r_3$ averages $0.03$ over the set,
    which is these molecules saying they are flat.

    **Make the sampler generate at a *prescribed* $r^\ast$.** Once per
    iteration, before the model is called, stretch and squeeze the current
    structure onto the target: eigendecompose its covariance, then scale along
    each principal axis by

    $$s_i = \sqrt{\frac{r_i^\ast \operatorname{tr} C}{\lambda_i}},
      \qquad x \leftarrow V \operatorname{diag}(s)\, V^\top x,$$

    and hand *that* to the model. Since $\sum_i r_i^\ast = 1$ the new variances
    again sum to $\operatorname{tr} C$: the map **preserves the total** and
    only moves variance between axes. That restriction is what keeps the task
    well-posed. Total variance is fixed by bond lengths and atom count, neither
    of which you are free to choose — a guidance that inflated it too would be
    demanding a molecule whose bonds are 20% too long, and the next model call
    would spend itself undoing the demand instead of following it.

    Why before *every* call rather than once on the finished sample? Applied
    once, this is not guidance but damage: every bond along the long axis
    stretched by $s_1$, and the validity check will say so. Applied every
    iteration, each squeeze is small and the denoising step right after repairs
    it — toward a structure that is already leaning the way you asked. That
    alternation is the general recipe: the model supplies *is a molecule*, the
    map supplies *has this shape*, and the loop looks for something that is
    both.

    Two edits, both marked `TODO` in the cell below: write `reshape`, and put
    it in the loop. As shipped it reshapes nothing, so the movie you get is
    §7's unguided sampler — your job is to make that movie change. The panel
    captions measure you: each reads out the $r_1$ its molecule actually
    reached, next to the molecule it turned out to be.

    **Check your implementation against these questions:** how close does the
    achieved $r_1$ get to the target — and which target does the model fight
    hardest? (Change `SHAPE_TARGET` to the disc and to the ball; hold all three
    against the dataset's range above.) On the ball run, read the molecule
    names: a planar ring cannot fill three dimensions, so what does the model
    reach for instead? What happens if you drop the trace preservation and
    scale the total variance by 1.5 as well — and at which point in the loop
    does that show? And what do the names do if you reshape *only once*, on the
    finished sample?
    """)
    return


@app.cell
def _(DirectDenoisingSampler, torch):
    class ShapeGuidedDenoising(DirectDenoisingSampler):
        """Direct denoising that re-shapes its state before every model call."""

        def __init__(
            self, process, parametrization, idx_m, n_atoms, target, **kwargs
        ):
            super().__init__(process, parametrization, **kwargs)
            self.idx_m = idx_m  # which molecule each row belongs to
            self.n_mol = int(n_atoms.shape[0])
            target = torch.as_tensor(target, dtype=torch.float32)
            # normalized and descending, to line up with the sorted eigenvalues
            self.target = (target / target.sum()).sort(descending=True).values

        def reshape(self, x):
            """Scale each molecule along its own principal axes onto `target`."""
            out = x.clone()
            for m in range(self.n_mol):
                rows = self.idx_m == m
                pos = x[rows]
                pos = pos - pos.mean(0)  # each molecule on its own center
                # TODO ------------------------------------------------------
                # Eigendecompose this molecule's 3x3 covariance
                # (`torch.linalg.eigh` returns ascending eigenvalues and the
                # matching axes as *columns*), build the per-axis factor s_i of
                # the formula above, and write the rescaled positions back.
                out[rows] = pos  # as shipped: no reshaping at all
                # -----------------------------------------------------------
            return out

        def denoise(self, model, x_t, n_steps, cond=None):
            x = x_t
            t = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
            for k in range(1, n_steps + 1):
                # the base class's decaying noise injection, unchanged
                noise_scale = self.stochastic_lambda * (1.0 - k / n_steps)
                if noise_scale > 0.0:
                    x = x + noise_scale * torch.randn_like(x)
                # TODO: one line — the state that goes into the model should be
                # the reshaped one
                x = self.parametrization.to_x0(
                    self.process, model(x, t, cond), x, t
                )
            return x
    return (ShapeGuidedDenoising,)


@app.cell
def _(
    DEVICE,
    ShapeGuidedDenoising,
    force_param,
    n_task,
    process,
    properties,
    rdkit_verdict,
    recording_model_fn,
    task_batch,
    task_model_fn,
    torch,
    viz,
):
    # rod. Also try the disc, (0.50, 0.45, 0.05), and the ball, (1/3, 1/3, 1/3)
    SHAPE_TARGET = (0.85, 0.13, 0.02)

    def principal_ratios(x, layout):
        """Per molecule: the variance along each principal axis, normalized."""
        rows = []
        for m in range(int(layout[properties.n_atoms].shape[0])):
            pos = x[layout[properties.idx_m] == m]
            pos = pos - pos.mean(0)
            lam = torch.linalg.eigvalsh(pos.T @ pos / len(pos))  # ascending
            rows.append(lam.flip(0) / lam.sum())
        return torch.stack(rows).cpu()

    torch.manual_seed(7)
    shaped = ShapeGuidedDenoising(
        process,
        force_param,
        task_batch[properties.idx_m],
        task_batch[properties.n_atoms],
        SHAPE_TARGET,
    )
    shape_start = shaped.prior.sample(
        (n_task, 3), device=DEVICE, context=task_batch
    )
    watched_shape, shape_frames = recording_model_fn(task_model_fn)
    with torch.no_grad():
        x_shaped = shaped.denoise(watched_shape, shape_start, n_steps=60)
    shape_frames.append(x_shaped)

    # every panel says what its molecule reached and what it became
    shape_r = principal_ratios(x_shaped, task_batch)
    shape_idx_m = task_batch[properties.idx_m]
    shape_titles = [
        f"r₁={float(shape_r[m][0]):.2f} · "
        + (
            rdkit_verdict(
                task_batch[properties.Z][shape_idx_m == m],
                x_shaped[shape_idx_m == m],
            )[1]
            or "—"
        )
        for m in range(len(shape_r))
    ]
    viz.show_trajectory(
        shape_frames,
        task_batch,
        titles=shape_titles,
        zoom=0.35,
        cell_px=190,
        frame_ms=120,
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### b) Scaffold-conditioned generation

    Fix part of a molecule, generate the rest. This is the question generative
    chemistry actually gets asked: the scaffold is the part that already works
    — a ring that binds, a core a synthesis route exists for — and what is
    wanted is everything around it.

    Ours: **uracil**. C₄H₄N₂O₂ is uracil's formula, so the RNA base is one of
    our 181 isomers (index 7) and its six-membered ring is a scaffold worth
    keeping. Keep the ring — atoms 1, 2, 3, 4, 5, 7 (C, N, C, C, C, N) — and
    generate the rest: the two carbonyl oxygens (0, 6) and the four hydrogens
    (8–11). Half the molecule given, half to place. The next cell builds
    exactly that, with the atom indices drawn on so you can check them, and
    leaves you four things to work with: `scaffold_batch` (the layout, in
    uracil's atom order), `scaffold_model_fn` (the model bound to it), `x_kept`
    (the ring coordinates, repeated per copy) and `free_atoms` (the mask of
    rows a sampler may touch).

    Two axes carry the task, one each:

    **The prior.** §5 named it as an axis it does not vary — vary it here.
    `GaussianPrior` draws every atom from
    $\mathcal{N}(0, \sigma_\text{max}^2)$; a **scaffold prior** draws only the
    *free* rows that way and puts the kept atoms at their given coordinates.
    That is a legal prior, and the library's rule says why: an endpoint may
    depend on anything known before generation — composition, atom count,
    which atoms are fixed and where — but never on the data values of the batch
    it is generating. So `ScaffoldPrior` subclasses `Prior` and is handed to
    the sampler as its `prior=` argument, exactly like the process and the
    parametrization.

    **The sampler.** A prior only decides where the run *starts*; the first
    model call would move the ring like anything else. So freeze it: after
    every update write the scaffold coordinates back, and keep the noise
    injection off those rows too. The free atoms see a ring that never moves,
    and get pulled into place around it.

    Three `TODO`s, all in the cell after next. As shipped, the prior is an
    ordinary Gaussian and nothing is frozen — so what you get is an
    unconditional sample that ignores the ring, and the panels say **⚠ ring
    moved**. Get all three right and that warning disappears: the ring stands
    still through the whole movie while the oxygens and hydrogens fly in around
    it.

    **Check your implementation against these questions:** did it rebuild
    uracil (`O=c1cc[nH]c(=O)[nH]1`)? Several completions will be *other*
    molecules on the same ring — enols, tautomers — which is worth deciding
    about: failure, or the point? Then the honest worry: this start is
    **out of distribution**. The model was trained on structures where every
    atom carries the *same* noise level, and here six atoms are exact while six
    sit 30 Å out. Does that show — and does starting the free atoms at a
    smaller `std` (3 Å, say) buy better completions, or only less diverse ones?
    Finally, freeze less: keep only the two ring nitrogens. How much of a
    molecule does this model need before it can finish it?
    """)
    return


@app.cell
def _(
    AtomsLoader,
    DEVICE,
    big_model,
    dataset,
    fully_connected_batch,
    make_model_fn,
    properties,
    to_device,
    torch,
    viz,
):
    URACIL_IDX = 7  # C4H4N2O2 is uracil's formula — the RNA base is in the set
    RING = [1, 2, 3, 4, 5, 7]  # its six-membered ring, in this molecule's order
    N_SCAFFOLD = 10  # completions to generate

    uracil = next(iter(AtomsLoader(dataset, sampler=[URACIL_IDX])))
    # the layout has to follow *this* molecule's atom order — the isomers share
    # a composition, not an ordering
    scaffold_batch = to_device(
        fully_connected_batch(uracil[properties.Z].tolist(), n_mol=N_SCAFFOLD), DEVICE
    )
    scaffold_model_fn = make_model_fn(big_model, scaffold_batch, "pseudo_force_pred")

    # the same ring in every copy, at the dataset's (centered) coordinates...
    x_kept = uracil[properties.R].repeat(N_SCAFFOLD, 1).to(DEVICE)
    # ...and the mask of rows a sampler is allowed to touch
    kept = torch.zeros(int(uracil[properties.n_atoms]), dtype=torch.bool)
    kept[RING] = True
    free_atoms = (~kept).repeat(N_SCAFFOLD).to(DEVICE)

    viz.show_batch(
        uracil,
        titles=["uracil — keep 1,2,3,4,5,7; generate 0,6 and 8–11"],
        atom_index=True,
        cell_px=300,
        zoom=1.5,
    )
    return N_SCAFFOLD, free_atoms, scaffold_batch, scaffold_model_fn, x_kept


@app.cell
def _(DirectDenoisingSampler, torch):
    from schnetpack.generative import GaussianPrior, Prior

    class ScaffoldPrior(Prior):
        """Noise on the free atoms, the given coordinates on the rest.

        `gaussian` stays False, the base class default: half these rows are not
        random at all, and that flag is what gates the library's Gaussian-only
        closed forms. Nothing here needs them — direct denoising only ever asks
        a prior for a starting state.
        """

        def __init__(self, x_scaffold, free, idx_m, std):
            self.x_scaffold = x_scaffold
            self.free = free[:, None]  # (n_atoms, 1), to broadcast over x, y, z
            self.idx_m = idx_m
            self.std = std

        def sample(self, shape, dtype=None, device=None, context=None):
            x = self.std * torch.randn(
                *shape,
                dtype=dtype or self.x_scaffold.dtype,
                device=device or self.x_scaffold.device,
            )
            # the same zero-COM frame everything else lives in, per molecule
            x = GaussianPrior.center(x, self.idx_m)
            # TODO: the free rows start as noise, the scaffold rows start at
            # the coordinates they are supposed to keep (`torch.where`)
            return x

    class ScaffoldDenoising(DirectDenoisingSampler):
        """Direct denoising in which the scaffold rows never move."""

        def __init__(self, process, parametrization, x_scaffold, free, **kwargs):
            super().__init__(process, parametrization, **kwargs)
            self.x_scaffold, self.free = x_scaffold, free[:, None]

        def denoise(self, model, x_t, n_steps, cond=None):
            x = x_t
            t = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
            for k in range(1, n_steps + 1):
                noise_scale = self.stochastic_lambda * (1.0 - k / n_steps)
                if noise_scale > 0.0:
                    # TODO: the scaffold does not move, not even by the injection
                    x = x + noise_scale * torch.randn_like(x)
                x0_hat = self.parametrization.to_x0(
                    self.process, model(x, t, cond), x, t
                )
                # TODO: only the free rows take the update
                x = x0_hat
            return x
    return ScaffoldDenoising, ScaffoldPrior


@app.cell
def _(
    DEVICE,
    N_SCAFFOLD,
    SIGMA_MAX,
    ScaffoldDenoising,
    ScaffoldPrior,
    force_param,
    free_atoms,
    process,
    properties,
    rdkit_verdict,
    recording_model_fn,
    scaffold_batch,
    scaffold_model_fn,
    torch,
    viz,
    x_kept,
):
    torch.manual_seed(11)
    scaffolded = ScaffoldDenoising(
        process,
        force_param,
        x_kept,
        free_atoms,
        prior=ScaffoldPrior(
            x_kept, free_atoms, scaffold_batch[properties.idx_m], SIGMA_MAX
        ),
        stochastic_lambda=1.0,
    )
    watched_scaffold, scaffold_frames = recording_model_fn(scaffold_model_fn)
    with torch.no_grad():
        x_scaffold = scaffolded.sample(
            watched_scaffold, shape=x_kept.shape, n_steps=60, device=DEVICE
        )
    scaffold_frames.append(x_scaffold)

    # the ring has to come back bit-identical — that is what "frozen" means
    ring_frozen = torch.equal(x_scaffold[~free_atoms], x_kept[~free_atoms])
    scaffold_idx_m = scaffold_batch[properties.idx_m]
    scaffold_titles = [
        ("" if ring_frozen else "⚠ ring moved · ")
        + (
            rdkit_verdict(
                scaffold_batch[properties.Z][scaffold_idx_m == m],
                x_scaffold[scaffold_idx_m == m],
            )[1]
            or "—"
        )
        for m in range(N_SCAFFOLD)
    ]
    viz.show_trajectory(
        scaffold_frames,
        scaffold_batch,
        titles=scaffold_titles,
        zoom=0.35,
        cell_px=190,
        frame_ms=120,
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Where to go from here

    Three things this tutorial skipped, all in the library. Of the five axes
    only the *coupling* is left untouched — how the $(x_0, x_1)$ pairs are
    matched up, and the slot flow matching with optimal transport plugs into;
    it is a constructor argument exactly like the prior §8b replaced.
    §7 assembled `Sampler` with one integrator and the default grid, and both
    slots hold more: `Euler` and `Heun` integrate the reverse SDE or the
    probability-flow ODE (`churn=0`) instead of stepping the exact posterior,
    and a warped `grid` spends steps where the structure actually appears
    rather than spreading them evenly — the usual first thing to tune when a
    sampler needs to get cheaper. And the force-field side this tutorial rode
    in on — property prediction, ML-driven molecular dynamics, and the
    Lightning/CLI training stack — is covered by the SchNetPack documentation
    and tutorials.
    """)
    return


if __name__ == "__main__":
    app.run()
