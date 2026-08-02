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
    # Generative models in SchNetPack

    **ML4Chem hands-on tutorial** — from force fields to generative models
    """)
    return


@app.cell
def _():
    import os as _os

    import viz as _viz

    # one run of §7's direct-denoising sampler, rendered ahead of time by the
    # same viewer every section below uses
    _here = (
        _os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else "."
    )
    with open(_os.path.join(_here, "assets", "denoising.html"), encoding="utf-8") as _fh:
        _page = _fh.read()

    _viz.show_page(_page, height=210)
    return


@app.cell
def _(mo):
    mo.md(r"""
    - **Context.** Niklas Gebauer's talk covered the theory of generative
      models for molecules. This session is its practical counterpart: the
      code that turns that theory into a model you can train and sample from.
    - **What we build.** GPFF — the generative pseudo-force field — trained
      twice: on a **181-molecule slice of QM9**, small enough to train live in
      this notebook, and, shipped as a checkpoint, on **all of QM9** at
      research scale.
    - **Scope.** We generate **equilibrium structures only** — the 3D geometry.
      The composition is given: positions diffuse, **atom types do not**.
    - **The code.** This is the **work-in-progress `schnetpack.generative`
      module of SchNetPack 3**, on its way to a general toolbox for generative
      models. The structure you see here is meant to stay; what grows around it
      is more of the same kind — flow matching, further processes and samplers.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## What this notebook covers

    1. **Setup** — the repo, and the runtime it needs
    2. **Introduction** — SchNetPack today, and where GPFF plugs in
    3. **Your own data** — databases, transforms, batches
    4. **Roadmap** — the three parts of a diffusion model
    5. **Forward process** — noising structures, defining labels
    6. **Model and training** — a force field on noised structures
    7. **Sampling** — ancestral, direct denoising, validation
    8. **Your tasks** — steering the sampler
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Setup

    Everything for this tutorial lives in one repository:

    **https://github.com/stefaanhessmann/ml4chem-tutorial**

    Open it and click the **"Open in Colab"** badge — that is the quickest way
    in, and the first code cell below then pulls everything into the runtime.

    Everything ships in one folder:

    ```
    ML4Chem-tutorial/
    ├── notebook.py            ← this tutorial
    ├── data/qm9_c4h4n2o2.xyz  ← the dataset: 181 QM9 isomers of C₄H₄N₂O₂
    ├── checkpoints/
    │   ├── gpff.pt            ← the §6 model; loaded unless RETRAIN = True
    │   └── gpff_big.pt        ← the same model at research scale (all of QM9)
    ├── helpers.py             ← glue: sampling batches, model adapters
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

    In a [marimo](https://marimo.io) notebook, cells form a dependency graph
    and re-run when their inputs change. Every cell is plain Python —
    everything here works the same in a script.

    **Hardware.** The first code cell points `DEVICE` at a GPU if there is one
    and the CPU otherwise; nothing below is device-specific. The one
    GPU-hungry step is opt-in — training §6's model with `RETRAIN = True`,
    ~15 min on a GPU — and that cell loads a checkpoint by default.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Introduction

    **SchNetPack** is an open-source toolbox for atomistic machine learning.

    **SchNetPack 2 — what the released package covers:**

    - **Machine-learned force fields (MLFFs)** — energies and forces at a
      fraction of the cost of the electronic-structure method they learn from.
    - **Architectures** — SchNet, PaiNN, SO3net: equivariant message passing on
      atomic neighborhoods, interchangeable inside one model interface.
    - **Property prediction** — any per-atom or per-molecule quantity a dataset
      carries, through the same training stack.
    - **Data and MD tooling** — ASE-backed databases, transforms, loaders, a
      configurable training setup, and interfaces that run molecular dynamics
      with a trained model.

    **SchNetPack 3 — what we are adding.** A force field *evaluates* structures
    it is given; a **generative model** *produces* them, learning the
    distribution behind a dataset's geometries so new, plausible ones can be
    drawn from it. `schnetpack.generative` is meant to make that a first-class
    part of the toolbox:

    - **the forward process** — noise schedules, parametrizations, priors and
      couplings, assembled from swappable pieces;
    - **samplers** — the trained model run backwards, from noise to structure;
    - and, from the same interfaces, **flow matching** and further processes and
      samplers as the module grows.

    **GPFF plugs straight into the existing stack.** Its denoising network reads
    a noised structure and predicts a 3-vector per atom — architecturally a
    force field, of the non-energy-conserving kind. So it reuses the *same
    architectures* (here PaiNN), the *same datasets and transform pipeline*, and
    the *same training loop*. What is genuinely new is the forward process that
    manufactures the training data and the sampler that runs the model
    backwards.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Using your own data in SchNetPack

    SchNetPack reads training data from an **ASE-backed SQLite database**;
    whatever format your structures start in, step one is to put them in a db.
    Ours is an xyz file with all **181 isomers of C₄H₄N₂O₂** in QM9 — one
    fixed composition, so the generative model only has to learn *where the
    atoms go*, not which atoms to place.

    Two calls build it: `ASEAtomsData.create` declares the stored properties
    with their units, `add_systems` fills it with `ase.Atoms`. We store the U0
    energies from the xyz — unused here, but a database declares its properties
    up front and real datasets carry them.
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
    return (
        ASEAtomsData,
        AtomsLoader,
        DB_PATH,
        DEVICE,
        HERE,
        np,
        numbers,
        os,
        torch,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ### Transforms and batches

    - A dataset item is a **dict of tensors** — positions `R`, atomic numbers
      `Z`, … — keyed by `schnetpack.properties`. That dict is the universal
      interface: every SchNetPack model consumes it, and everything we build
      below writes into it.
    - **Transforms** are per-structure preprocessing owned by the dataset,
      re-run on every load. Here: center (`SubtractCenterOfGeometry`), build
      the neighbor list (`MatScipyNeighborList`, 10 Å — which fully connects a
      molecule), cast to float32 (`CastTo32`).
    - **Batches are not padded.** `AtomsLoader` concatenates all atoms along
      one axis and records in `idx_m` which molecule each belongs to: 8
      twelve-atom molecules make one `(96, 3)` position tensor.
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

    In code, a diffusion-based generative model is **three parts**, and each
    gets one section:

    | | part | what it is | where |
    |---|---|---|---|
    | a | **forward process** | noising structures and computing labels, inside the dataloader | §5 |
    | b | **model architecture** | an MLFF-shaped network applied to noised structures | §6 |
    | c | **sampling** | iterating the trained model from noise to structures | §7 |

    **Training** (§6) is where a and b meet; after sampling we **validate**
    what came out (§7). The goal, plainly: train GPFF on our 181-molecule QM9
    slice and generate new C₄H₄N₂O₂ geometries.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. The forward process: making training data from noise

    A force field trains on labels the dataset ships — energies, forces. A
    diffusion model **manufactures its own**: noise a clean structure, then ask
    the network for the way back.

    ### Noising

    `schnetpack.generative` writes the forward process as an **interpolation**
    between the data $x_0$ and an endpoint $x_1$ drawn from a prior:

    $$x_t = a(t)\,x_0 + b(t)\,x_1, \qquad t \in [0, 1],$$

    with $a(0) = 1,\, b(0) \approx 0$ (the data) and $b(1) = 1$ (pure noise). A
    `Process` owns $a$, $b$, the prior, and the noise level
    $\sigma(t) = b(t)\,\sigma_\text{prior}$. The two standard schedules:

    - **`VP`** (variance preserving — DDPM): the data is scaled away as noise
      of fixed scale blends in; the total variance stays constant.
    - **`VE`** (variance exploding — score matching): the data is never scaled
      ($a \equiv 1$), and noise is simply *added* until it drowns the
      structure, with $\sigma(t)$ growing geometrically.

    We take **VE**, from $\sigma_\text{min} = 0.05$ to
    $\sigma_\text{max} = 30$ Å. $\sigma_\text{max}$ must at least match the
    data scale — rule of thumb: the largest pairwise distance, ~7.7 Å here.
    Too small and the endpoint still remembers the data, too large and training
    wastes capacity, and *neither failure is loud*. 30 Å is what §7's
    research-scale model was trained at, which keeps every model here
    interchangeable.

    Below — and in every illustration that follows — one elongated open-chain
    isomer, easy to track through the noise, under both processes with the same
    noise draw (slider = $t$). VP shrinks it into a small fixed-size cloud; VE
    leaves it in place and buries it under a 30 Å one.
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
    ### Labels: what should the network predict?

    Noised structures are half the training data; the **label** is the other
    half, and choosing it is the second axis — the `Parametrization`. All three
    below are one map away from each other. What separates them is how the
    target's magnitude scales with $\sigma$, which is exactly what a plain L2
    loss sees.

    | parametrization | target | scales with $\sigma$ as | recover $x_0$ |
    |---|---|---|---|
    | `EpsParametrization` | $\varepsilon$ | constant | $x_t - \sigma\varepsilon$ |
    | `ScoreParametrization` | $s = \nabla_x \log p_t(x_t) = -\varepsilon/\sigma$ | $1/\sigma$ | $x_t + \sigma^2 s$ |
    | `PseudoForceParametrization` | $F = 2\,(x_0 - x_t)$ | $\sigma$ | $x_t + F/2$ |

    GPFF takes the **pseudo force**, whose magnitude grows in proportion to
    $\sigma$. That buys two things:

    - getting home is one addition, $\hat x_0 = x_t + F/2$ — no division, so
      nothing degenerates as $\sigma \to 0$;
    - the magnitude of $F$ *carries* the noise level, so a GPFF network needs
      **no time input at all**. Remember that for §6 and §7. Read backwards,
      the same identity says a trained GPFF can be *asked* how noisy a
      structure is.

    Below, the same path three times, with the **target drawn as an arrow on
    every atom**: eps arrows keep their size everywhere, score arrows explode
    as $\sigma \to 0$, and pseudo-force arrows shrink to nothing as the
    structure comes home. That last row is drawn at **half length**,
    $F/2 = x_0 - x_t$, so each arrow tip lands on the clean structure.
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
    Two further axes this section does not vary: the **coupling** (how the
    drawn $(x_0, x_1)$ pairs are matched up) and the **prior** (what $x_1$ is
    drawn from) — equally swappable constructor arguments. The prior returns in
    §8b, where replacing it is half the task.

    ### Wrapping it into a transform

    Building diffusion training data is preprocessing, so it is a **transform**
    like §3's. `Diffuse(process, parametrization)` runs the forward process
    inside the dataloader: per structure it draws a time, noises the positions,
    and writes the label into the item dict.

    **Where those draws land** is its own choice — the `t_sampler`. Uniform $t$
    on a geometric schedule is *log-uniform* in $\sigma$: equal weight to every
    decade from 0.05 to 30 Å, so a third of the budget lands above 3 Å, where
    the target is nearly the endpoint itself and there is little to learn.
    `LogNormalSigmaTimes` states the density where it means something, in
    $\sigma$: log-normal, median ~0.5 Å, most of the mass between 0.15 and
    1.7 Å — the band where bonds live and denoising is genuinely hard. This is
    GPFF's own training density (and EDM's, for images); the process converts
    to $t$ through `t_of_sigma`. `truncate=True` redraws the few draws falling
    outside the schedule instead of piling them onto its ends.

    Only the transform order needs thought:

    1. `SubtractCenterOfGeometry` — diffusion lives in the centered frame, and
       the prior draws its endpoints there too. A translation-invariant network
       could never predict a displacement of a whole structure, so an
       off-center endpoint would be unlearnable noise in every label.
    2. `Diffuse` — overwrites `R` with $x_t$, writes `"pseudo_force"` and `"t"`.
    3. `AllToAllNeighborList` — **after** noising. A *distance*-based list
       built at one noise level is wrong at another, and a cutoff wide enough
       for a fully noised cloud (~90 Å across) returns every pair anyway, at
       the cost of searching for them. Pairs the model's cutoff function
       downweights to zero cost nothing.
    4. `CastTo32`.

    An ordinary MSE against `"pseudo_force"` is then the whole objective.
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
    Those batches *are* the training set — so look at one. Ten structures from
    the same loader, each at its own drawn time, captioned with its noise
    level, the **label drawn as an arrow on every atom** (again at half length,
    so each arrow ends where its atom belongs).

    Read it as a difficulty gradient: at $\sigma \lesssim 0.5$ Å the molecule
    is intact and the arrows are tiny corrections; at several Ångström there is
    no molecule left and the arrows span the whole cloud. The label's scale
    runs with $\sigma$ — exactly what §6's loss has to compensate. And note
    what the time sampler did: most draws sit below ~2 Å, where denoising is
    hard but learnable.
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
    ## 6. Model and training

    A diffusion denoiser uses the **same architecture as an MLFF**, of the
    *non-energy-conserving* kind: a 3-vector read out per atom, rather than one
    energy per molecule and differentiated. Diffusion models generally need
    time conditioning on top, because the same noised geometry means a
    different target at a different $t$. **GPFF does not** — the magnitude of
    the pseudo force carries the noise level — so this network is *exactly* an
    ordinary force field.

    `NeuralNetworkPotential` stacks three stages, each an `nn.Module` acting on
    the batch dict:

    1. **input** — `PairwiseDistances`: positions + neighbor list to distance
       vectors.
    2. **representation** — `PaiNN`, message passing to per-atom features. It
       is *equivariant*: besides scalars it carries vector features that rotate
       with the molecule, which is what lets a head output a well-behaved
       vector per atom. (SchNet and SO3net are drop-in.)
    3. **output** — `AtomwiseVector`, a 3-vector per atom. (An energy model
       would end in `Atomwise`: a scalar per atom, summed per molecule.)

    Two settings are concessions to *noised* inputs:

    - **cutoff 150 Å**, since a fully noised cloud is ~90 Å across — with 600
      `GaussianRBF` functions across it, one every 0.25 Å. Too few, and a 1.0 Å
      contact and a 1.4 Å bond get near-identical embeddings; a denoiser that
      cannot tell a clash from a bond will happily generate both.
    - **`norm_epsilon=1`**: PaiNN normalizes each pair direction as
      $r_{ij}/(d_{ij} + 1)$ rather than $r_{ij}/d_{ij}$, which stays finite
      when two atoms of a noise cloud land on top of each other.
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

    Data augmentation and model meet in an ordinary PyTorch loop: pull a batch
    from an `AtomsLoader` over §5's diffused dataset, compare against the
    `"pseudo_force"` label, step the optimizer. Nothing in the loop knows it is
    training a generative model — the transforms did that part.

    **The objective** is an MSE against the pseudo-force label, weighted per
    draw by $w(t) = \min(\sigma(t)^{-2}, w_\text{max})$.
    The $1/\sigma^2$ undoes the label's $\sigma$-scaling, so what is minimized
    is the *relative* error at every noise level rather than the absolute one;
    without it the deep-noise samples, whose labels are tens of Ångström long,
    drown out everything else. The ceiling decides how much of the
    small-$\sigma$ end survives, and it is easy to set too low: at
    $w_\text{max} = 1$ it binds for 71% of the draws, flattening the weight
    across the whole band where bond lengths are decided. At **100** it is the
    honest $1/\sigma^2$ almost everywhere.

    Three things about the loop:

    - **Every step sees fresh $(t, \varepsilon)$ draws** — the transforms
      re-run on every item the loader hands out, so the dataset is effectively
      infinite. A model fed a fixed set of noised structures would memorize
      them instead of learning the denoising field.
    - The weights used downstream are an **exponential moving average** of the
      ones the optimizer visited: a few thousand steps is a noisy place to
      stop, and the average samples visibly better.
    - The loader draws **with replacement**, `num_samples = BATCH * STEPS`, so
      the run is *one* epoch of 12000 batches. Otherwise 181 structures at
      batch 64 is under three batches per epoch, and a dataloader that throws
      away its prefetch queue that often never gets ahead of the GPU.

    The cell **loads** `checkpoints/gpff.pt` by default; `RETRAIN = True` runs
    the loop instead and overwrites it. Either way the curve below is real —
    the checkpoint stores its loss history alongside its weights. It plateaus
    well above zero because every noise level keeps an irreducible error, and
    that is healthy.
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
    return gpff_model, to_device


@app.cell
def _(mo):
    mo.md(r"""
    ## 7. Sampling

    Sampling runs the process backwards: start from a draw of the prior — a
    30 Å cloud — and call the model over and over until a molecule is left.
    Both samplers below do that with the same trained model; what they disagree
    about is *how*.

    ### The model that generates

    §6's network is **teaching-sized** and saw 181 structures. Denoising has to
    be accurate exactly where geometry is decided, at $\sigma \lesssim 0.3$ Å
    where bond lengths live — precisely where the time sampler concentrated its
    training. Expect roughly half the draws to be chemically valid molecules
    and nearly all of them to be sane geometry; the cells below measure it.

    The bundle also ships `checkpoints/gpff_big.pt`: same pseudo-force target,
    same VE process, same $\sigma$-focused time sampling, at **research scale —
    5.1M parameters, trained on all ~130k molecules of QM9**. The cell below
    assembles it exactly like §6's model, only wider, and `USE_BIG_MODEL` swaps
    it into every sampling and validation cell that follows. Flip it after one
    pass: how far the numbers move is the most honest measure of what scale
    buys.
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

    The **classical** route — the reverse process of the lecture, and what
    every time-conditioned diffusion model uses. Walk a *prescribed* ladder of
    noise levels from $t = 1$ down to $0$ and take one exact step down each
    rung: one model call estimates the clean structure,
    $\hat x_0 = x_t + F_\theta(x_t)/2$, then the iterate moves to the next
    rung by the exact Gaussian posterior $p(x_{t_{k-1}} \mid x_{t_k}, \hat x_0)$
    — an interpolation between where it is and $\hat x_0$, plus a matched
    noise injection. No approximation beyond the model's own error; the closer
    two rungs sit, the less the estimate is trusted in one go.

    SchNetPack assembles it from four parts — §5's two, plus two only sampling
    needs:

    | part | what it decides | here |
    |---|---|---|
    | `process` | the noise schedule $\sigma(t)$ | the `VE` of §5 |
    | `parametrization` | what the network's output means | pseudo force |
    | `integrator` | how one step down the ladder is taken | `Ancestral` |
    | `grid` | where the rungs sit | uniform in $t$ (the default) |

    Because VE's $\sigma$ grows *geometrically* in $t$, a uniform grid already
    gives the geometric ladder score matching wants — rungs that bunch up where
    $\sigma$ is small — so no schedule code is needed. (Warping the grid stays
    an option, and `grid` is where it would go.)

    Three pieces of glue from `helpers.py`:

    - `fully_connected_batch` — 8 copies of our composition as one flat batch:
      the topology (`Z`, `idx_m`, a neighbor list) for molecules that do not
      exist yet. It stays static, since the cutoff function handles the
      changing distances.
    - `make_model_fn` — adapts our batch-dict model to the sampler's
      plain-tensor `model(x, t, cond)` contract.
    - `recording_model_fn` — what makes the viewer a **movie**: a sampler
      returns only the structure it ended on, but every step passes its state
      through the model, so wrapping the model captures the whole run.

    Scrub the slider. The frames carry the grid's own times, so each caption
    reads out the rung it sits on, $t = 1$ down to $0$: the ladder comes down
    *gradually*, and a structure appears only over the last handful of rungs.
    Frame 0 is a ~80 Å cloud against a ~3 Å molecule — no camera holds a 25×
    range, so the view is framed on the finished structure and pulled back
    (`zoom=0.35`), and the atoms fly in from outside. That gap *is* the scale
    the model closes.
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

    GPFF's own sampler has no schedule, no time grid, and never asks how noisy
    its iterate is. It can afford that because the pseudo force is the way home
    in *one* step, $\hat x_0 = x + F_\theta(x)/2$. Taken from pure noise, that
    single jump lands on the model's *conditional mean* over every structure
    that could hide under it — a blob, not a molecule — so the sampler iterates
    instead: re-noise a little, jump home ($x + F_\theta(x)/2$), repeat, with
    the injection scaled by $\lambda$ and decaying linearly to zero over the
    run. No $t$ appears anywhere, which is only possible because the model
    does not need one either. That loop is `DirectDenoisingSampler`.

    $\lambda = 0$ — the plain repeated jump, no injection at all — looks like
    the natural choice and is the wrong one. With nothing put back, the loop is
    a deterministic fixed-point iteration that converges to whatever the
    model's map happens to attract: atoms stranded off the structure, and
    occasionally a run that leaves the finite numbers behind altogether. The
    injection keeps the iterate inside the band the model was trained on, and
    the decay walks it down. Here $\lambda = 1$, and on the counts below it is
    the difference between a third of the samples passing and nearly all.

    **Cost.** The ladder above took **64** model calls; this loop takes **60**
    — close, because this model is small and wants the iterations. The budget
    is a dial rather than a schedule, so a stronger model gets away with far
    fewer (flip `USE_BIG_MODEL` and 15 is plenty).

    Watch the difference in the movie: where the ladder descended gradually and
    revealed a structure only near the bottom, direct denoising is at molecular
    size after two or three calls and spends the rest tidying up. Same model,
    same starting noise — only the route differs.
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
    structure's nearest-neighbor distances:

    - two atoms closer than **0.7 Å** are fused — a *clash*;
    - an atom whose nearest neighbor is beyond **2.5 Å** is bonded to
      nothing — a *stray*.

    The dataset row calibrates both: real bond lengths here are ~1.0–1.5 Å, and
    a clean batch scores zero on each by construction. Being able to say *how
    many* is the point — that number is what tells you whether a change to the
    model, the process or the sampler actually helped.
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

    Distance checks ask whether the geometry is *sane* — never whether it is a
    *molecule*. RDKit can: `rdDetermineBonds` infers a bond graph from nothing
    but the coordinates, and a structure counts as **chemically valid** only if
    that graph works out — every valence satisfied as a neutral molecule, no
    unpaired electrons left over, everything in one connected piece. What
    passes earns a **SMILES** string: the molecule's identity, independent of
    coordinates.

    - the **valid fraction** is the standard headline metric for molecular
      generative models;
    - **SMILES** says *which* molecule each sample is — compare against the
      dataset's to see whether the model reproduced a training isomer or found
      a new one.

    It is a *strict* judge: a single fused pair already sinks a sample, which
    is what makes it honest. Flip `USE_BIG_MODEL` and see what a network
    trained on all of QM9 does to it.
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

    def smiles_per_molecule(x, layout):
        """One SMILES per structure, or an em dash where there is no molecule."""
        idx_m = layout[properties.idx_m]
        return [
            rdkit_verdict(layout[properties.Z][idx_m == m], x[idx_m == m])[1] or "—"
            for m in range(int(layout[properties.n_atoms].shape[0]))
        ]

    def chemistry_summary(x, layout):
        found = smiles_per_molecule(x, layout)
        valid = sorted(s for s in found if s != "—")
        return {"valid": f"{len(valid)}/{len(found)}", "SMILES": valid}

    {
        "dataset (reference)": chemistry_summary(batch[properties.R], batch),
        "ancestral (8)": chemistry_summary(x_ancestral, sampling_batch),
        "direct denoising (8)": chemistry_summary(x_direct, sampling_batch),
    }
    return (smiles_per_molecule,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 8. Your tasks: steering the sampler

    Both samplers of §7 draw from the *whole* distribution the model learned:
    hand them noise and they hand back some molecule. Neither takes an
    instruction — and nearly every real use of a generative model is one. *Make
    it long and thin. Keep this ring and fill in the rest.*

    The obvious way there is to train for it. Both tasks below take the other
    route and leave the trained model **exactly as it is**: the instruction
    enters in the *sampler*, through one shared recipe.

    > Every iteration, force the state onto the constraint, then let the model
    > repair whatever that broke.

    Because a model call always follows the nudge, the repair is chemistry
    rather than interpolation: what comes out satisfies the constraint *and*
    survives the denoiser. The alternation is the whole trick — either half
    alone does not work.

    | | the constraint | how it enters |
    |---|---|---|
    | **a** | a **global, continuous** property — the structure's shape | a linear map on the state, before every model call |
    | **b** | **exact positions** for some atoms — a scaffold | a custom prior, plus rows the loop never updates |

    **How to work them.** Each task is three cells: a class with `# TODO`
    markers to fill in, a **folded reference solution** under it, and a runner
    that samples and plays the result as a movie. The class **runs as
    shipped** — it just steers nothing yet, so the first movie you get is §7's
    unguided sampler. Your job is to make it change: edit, re-run, watch.

    The runner picks its sampler on its first line, so nobody is stuck. Leave
    it on your own class; point it at the reference to see the intended
    behaviour, and switch back to compare.

    Both subclass `DirectDenoisingSampler`, whose loop is four lines and
    carries no schedule to stay consistent with — which makes it the one to
    interfere with. And both always use the **research-scale model**, whatever
    §7's `USE_BIG_MODEL` says: steering is only legible when the model
    underneath is not the bottleneck.
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
    the 3×3 covariance of its positions:

    $$C = \frac{1}{n}\sum_i x_i x_i^\top
        = V \operatorname{diag}(\lambda_1 \ge \lambda_2 \ge \lambda_3) V^\top .$$

    - the columns of $V$ are the **principal axes**, $\lambda_i$ the variance
      along each;
    - $\operatorname{tr} C = \sum_i \lambda_i$ is the mean squared distance
      from the center — the structure's **size**;
    - $r_i = \lambda_i / \operatorname{tr} C$ is its **shape**: unchanged by
      rotation, and independent of size.

    Our 181 isomers run from $r = (0.41, 0.33, 0.26)$, the roundest of them, to
    $(0.89, 0.09, 0.02)$, the open chain §5 kept noising — and $r_3$ averages
    $0.03$, which is these molecules saying they are flat.

    **Your task: make the sampler generate at a prescribed $r^\ast$.** Once per
    iteration, *before* the model is called, stretch and squeeze the current
    structure onto the target — eigendecompose its covariance, then scale along
    principal axis $i$ by

    $$s_i = \sqrt{\frac{r_i^\ast \operatorname{tr} C}{\lambda_i}},
      \qquad x \;\leftarrow\; V \operatorname{diag}(s)\, V^\top x,$$

    and hand *that* to the model.

    Since $\sum_i r_i^\ast = 1$, the new variances again sum to
    $\operatorname{tr} C$: the map **preserves the total** and only moves
    variance between axes. That restriction keeps the task well-posed — total
    variance is fixed by bond lengths and atom count, neither of which you are
    free to choose, and a guidance that inflated it too would be demanding a
    molecule whose bonds are 20% too long.

    Why before *every* call rather than once on the finished sample? Applied
    once, this is not guidance but damage: every bond along the long axis
    stretched by $s_1$, and the validity check will say so. Applied every
    iteration, each squeeze is small and the denoising step right after repairs
    it — toward a structure already leaning the way you asked.

    **Two `# TODO`s in the next cell:** write `reshape`, then call it in the
    loop. The movie is the measurement — a rod and a ball are not subtle — and
    each panel is captioned with the molecule its structure turned out to be.

    **Then ask:**

    - Does the batch come out the shape you asked for, and which target does
      the model fight hardest? Try `SHAPE_TARGET` as the disc and the ball, and
      hold all three against the dataset's range above.
    - On the ball run, read the molecule names: a planar ring cannot fill three
      dimensions, so what does the model reach for instead?
    - Drop the trace preservation and scale the total variance by 1.5 as well.
      What breaks — and at which point in the loop does it show?
    - Reshape *only once*, on the finished sample. What happens to the names?
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


@app.cell(hide_code=True)
def _(DirectDenoisingSampler, torch):
    # @title 🔑 Reference solution — task a (click to reveal the code)
    class ShapeGuidedSolution(DirectDenoisingSampler):
        """The same class with both TODOs filled in."""

        def __init__(
            self, process, parametrization, idx_m, n_atoms, target, **kwargs
        ):
            super().__init__(process, parametrization, **kwargs)
            self.idx_m = idx_m
            self.n_mol = int(n_atoms.shape[0])
            target = torch.as_tensor(target, dtype=torch.float32)
            self.target = (target / target.sum()).sort(descending=True).values

        def reshape(self, x):
            target = self.target.to(x.device, x.dtype)
            out = x.clone()
            for m in range(self.n_mol):
                rows = self.idx_m == m
                pos = x[rows]
                pos = pos - pos.mean(0)
                lam, axes = torch.linalg.eigh(pos.T @ pos / len(pos))
                lam, axes = lam.flip(0), axes.flip(1)  # ascending -> descending
                # sum(target) == 1, so these variances still add up to tr C
                scale = (target * lam.sum() / lam.clamp(min=1e-8)).sqrt()
                # rotate into the principal frame, scale there, rotate back
                out[rows] = ((pos @ axes) * scale) @ axes.T
            return out

        def denoise(self, model, x_t, n_steps, cond=None):
            x = x_t
            t = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
            for k in range(1, n_steps + 1):
                noise_scale = self.stochastic_lambda * (1.0 - k / n_steps)
                if noise_scale > 0.0:
                    x = x + noise_scale * torch.randn_like(x)
                x = self.reshape(x)  # the model only ever sees the target shape
                x = self.parametrization.to_x0(
                    self.process, model(x, t, cond), x, t
                )
            return x

    return


@app.cell
def _(
    DEVICE,
    ShapeGuidedDenoising,
    force_param,
    n_task,
    process,
    properties,
    recording_model_fn,
    smiles_per_molecule,
    task_batch,
    task_model_fn,
    torch,
    viz,
):
    SAMPLER = ShapeGuidedDenoising  # yours; ShapeGuidedSolution is the reference
    SHAPE_TARGET = (0.85, 0.13, 0.02)  # rod · disc (0.50, 0.45, 0.05) · ball (1/3, 1/3, 1/3)

    torch.manual_seed(7)
    shaped = SAMPLER(
        process,
        force_param,
        task_batch[properties.idx_m],
        task_batch[properties.n_atoms],
        SHAPE_TARGET,
    )
    x_start = shaped.prior.sample((n_task, 3), device=DEVICE, context=task_batch)
    watched, shape_frames = recording_model_fn(task_model_fn)
    with torch.no_grad():
        x_shaped = shaped.denoise(watched, x_start, n_steps=60)
    shape_frames.append(x_shaped)

    viz.show_trajectory(
        shape_frames,
        task_batch,
        titles=smiles_per_molecule(x_shaped, task_batch),
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
    — a group that binds, a core a synthesis route exists for — and what is
    wanted is everything around it.

    Ours is the **amide**, `N-C(=O)`: the peptide bond, and the single most
    common linkage in drug molecules. It is taken from
    3-(hydroxyimino)pyrrolidin-2-one, a small QM9 lactam, and it is *all* that
    is kept — four atoms of fourteen, three of them heavy. Everything else is
    generated: **5 of the 8 heavy atoms are free**, so the model is not
    decorating a fixed core here, it is building a molecule around an anchor.

    That is deliberate. Freeze more and the model mostly re-derives the
    molecule you took the scaffold from; freeze this little and each run is a
    different molecule that happens to contain your amide — which is what
    fragment-based design actually looks like.

    The molecule comes from QM9 rather than from our 181 isomers, so the next
    cell simply *states* it: three arrays — positions in Å (already centered),
    atomic numbers, and the indices to keep. That is all a scaffold is. It
    works because §8 samples with the model trained on all of QM9, which knows
    this composition even though §3's dataset never mentions it.

    That cell also draws the molecule with its atom indices on, and leaves you
    four things: `scaffold_batch` (the layout, in this molecule's atom order),
    `scaffold_model_fn` (the model bound to it), `x_kept` (the coordinates,
    repeated per copy) and `free_atoms` (the mask of rows a sampler may touch).

    Two axes carry the task, one each.

    **The prior.** §5 named it as an axis it does not vary — vary it here.
    `GaussianPrior` draws every atom from $\mathcal N(0, \sigma_\text{max}^2)$;
    a **scaffold prior** draws only the *free* rows that way and puts the kept
    atoms at their given coordinates. That is a legal prior, and the library's
    rule says why: an endpoint may depend on anything known before generation —
    composition, atom count, which atoms are fixed and where — but never on the
    data values of the batch it is generating. So `ScaffoldPrior` subclasses
    `Prior` and is handed to the sampler as `prior=`, exactly like the process
    and the parametrization.

    **The sampler.** A prior only decides where the run *starts*; the first
    model call would move the amide like anything else. So freeze it: after
    every update write the scaffold coordinates back, and keep the noise
    injection off those rows too. The free atoms then see an anchor that never
    moves, and assemble around it.

    **Three `# TODO`s in the next cell.** As shipped, the prior is an ordinary
    Gaussian and nothing is frozen, so the amide drifts off with everything
    else and each run is an unconditional sample. Get all three right and the
    anchor stands still through the whole movie while the other ten atoms fly
    in around it.

    **Then ask:**

    - Read the panel captions: how many completions still contain the amide,
      and what did the model build onto it — rings, chains? None of them has to
      be the molecule the fragment came from, and most will not be.
    - This start is **out of distribution**: the model was trained where every
      atom carries the *same* noise level, and here four atoms are exact while
      ten sit 30 Å out. Does that show — and does starting the free atoms at a
      smaller `std` (3 Å, say) buy better completions, or only less diverse
      ones?
    - Freeze more: add the ring carbon next to the amide, or the whole ring.
      How much structure does the model need before it stops inventing and
      starts reconstructing?
    """)
    return


@app.cell
def _(
    DEVICE,
    big_model,
    fully_connected_batch,
    make_model_fn,
    np,
    properties,
    to_device,
    torch,
    viz,
):
    # 3-(hydroxyimino)pyrrolidin-2-one, one QM9 structure written out in full:
    # nothing here needs the dataset, and a scaffold is only ever these arrays.
    SCAFFOLD_Z = np.array([8, 7, 6, 6, 6, 7, 6, 8, 1, 1, 1, 1, 1, 1])
    SCAFFOLD_R = np.array(  # positions in Angstrom, center of geometry at 0
        [
            [1.73577, 2.33192, -0.19861],  # 0   O, oxime
            [1.79019, 0.95051, -0.14187],  # 1   N, oxime
            [0.66031, 0.39076, 0.04573],  # 2   C, ring
            [0.53740, -1.11094, 0.13951],  # 3   C, ring
            [-0.96261, -1.36931, -0.13537],  # 4   C, ring
            [-1.57289, -0.08586, 0.16992],  # 5   N, amide  <- kept
            [-0.72108, 0.99488, 0.17524],  # 6   C, amide  <- kept
            [-1.04806, 2.15558, 0.26989],  # 7   O, amide  <- kept
            [2.65830, 2.56393, -0.35520],  # 8   H, on the oxime O
            [1.19530, -1.63266, -0.55682],  # 9   H, on C3
            [0.79316, -1.44270, 1.15216],  # 10  H, on C3
            [-1.13046, -1.65728, -1.18229],  # 11  H, on C4
            [-1.36859, -2.16296, 0.49881],  # 12  H, on C4
            [-2.56675, 0.07413, 0.11890],  # 13  H, on the amide N  <- kept
        ]
    )
    SCAFFOLD = np.array([5, 6, 7, 13])  # the amide N-C(=O) and its hydrogen
    N_SCAFFOLD = 10  # completions to generate

    scaffold_batch = to_device(
        fully_connected_batch(SCAFFOLD_Z.tolist(), n_mol=N_SCAFFOLD), DEVICE
    )
    scaffold_model_fn = make_model_fn(big_model, scaffold_batch, "pseudo_force_pred")

    # the same anchor in every copy...
    x_kept = (
        torch.tensor(SCAFFOLD_R, dtype=torch.float32).repeat(N_SCAFFOLD, 1).to(DEVICE)
    )
    # ...and the mask of rows a sampler is allowed to touch
    kept = torch.zeros(len(SCAFFOLD_Z), dtype=torch.bool)
    kept[torch.as_tensor(SCAFFOLD)] = True
    free_atoms = (~kept).repeat(N_SCAFFOLD).to(DEVICE)

    # the same arrays as one molecule, only to look at
    scaffold_view = fully_connected_batch(SCAFFOLD_Z.tolist(), n_mol=1)
    scaffold_view[properties.R] = torch.tensor(SCAFFOLD_R, dtype=torch.float32)
    viz.show_batch(
        scaffold_view,
        titles=["keep the amide 5, 6, 7 and its hydrogen 13 — generate the rest"],
        atom_index=True,
        cell_px=300,
        zoom=1.5,
    )
    return free_atoms, scaffold_batch, scaffold_model_fn, x_kept


@app.cell
def _(DirectDenoisingSampler, torch):
    from schnetpack.generative import GaussianPrior, Prior

    class ScaffoldPrior(Prior):
        """Noise on the free atoms, the given coordinates on the rest.

        `gaussian` stays False, the base class default: some of these rows are
        not random at all, and that flag is what gates the library's
        Gaussian-only closed forms. Nothing here needs them — direct denoising
        only ever asks a prior for a starting state.
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


@app.cell(hide_code=True)
def _(DirectDenoisingSampler, torch):
    # @title 🔑 Reference solution — task b (click to reveal the code)
    from schnetpack.generative import GaussianPrior as _GaussianPrior
    from schnetpack.generative import Prior as _Prior

    class ScaffoldPriorSolution(_Prior):
        """The same prior with its TODO filled in."""

        def __init__(self, x_scaffold, free, idx_m, std):
            self.x_scaffold = x_scaffold
            self.free = free[:, None]
            self.idx_m = idx_m
            self.std = std

        def sample(self, shape, dtype=None, device=None, context=None):
            x = self.std * torch.randn(
                *shape,
                dtype=dtype or self.x_scaffold.dtype,
                device=device or self.x_scaffold.device,
            )
            x = _GaussianPrior.center(x, self.idx_m)
            return torch.where(self.free, x, self.x_scaffold)

    class ScaffoldDenoisingSolution(DirectDenoisingSampler):
        """The same sampler with both TODOs filled in."""

        def __init__(self, process, parametrization, x_scaffold, free, **kwargs):
            super().__init__(process, parametrization, **kwargs)
            self.x_scaffold, self.free = x_scaffold, free[:, None]

        def denoise(self, model, x_t, n_steps, cond=None):
            x = torch.where(self.free, x_t, self.x_scaffold)  # pin, then start
            t = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
            for k in range(1, n_steps + 1):
                noise_scale = self.stochastic_lambda * (1.0 - k / n_steps)
                if noise_scale > 0.0:
                    # the scaffold does not move, not even by the injection
                    x = x + noise_scale * torch.randn_like(x) * self.free
                x0_hat = self.parametrization.to_x0(
                    self.process, model(x, t, cond), x, t
                )
                x = torch.where(self.free, x0_hat, self.x_scaffold)
            return x

    return


@app.cell
def _(
    DEVICE,
    SIGMA_MAX,
    ScaffoldDenoising,
    ScaffoldPrior,
    force_param,
    free_atoms,
    process,
    properties,
    recording_model_fn,
    scaffold_batch,
    scaffold_model_fn,
    smiles_per_molecule,
    torch,
    viz,
    x_kept,
):
    # yours; the references are ScaffoldPriorSolution and ScaffoldDenoisingSolution
    PRIOR, SAMPLER_B = ScaffoldPrior, ScaffoldDenoising

    torch.manual_seed(11)
    scaffolded = SAMPLER_B(
        process,
        force_param,
        x_kept,
        free_atoms,
        prior=PRIOR(x_kept, free_atoms, scaffold_batch[properties.idx_m], SIGMA_MAX),
        stochastic_lambda=1.0,
    )
    watched_scaffold, scaffold_frames = recording_model_fn(scaffold_model_fn)
    with torch.no_grad():
        x_scaffold = scaffolded.sample(
            watched_scaffold, shape=x_kept.shape, n_steps=60, device=DEVICE
        )
    scaffold_frames.append(x_scaffold)

    viz.show_trajectory(
        scaffold_frames,
        scaffold_batch,
        titles=smiles_per_molecule(x_scaffold, scaffold_batch),
        zoom=0.35,
        cell_px=190,
        frame_ms=120,
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Where to go from here

    Three things this tutorial skipped, all in the library:

    - Of the five axes only the **coupling** is left untouched — how the
      $(x_0, x_1)$ pairs are matched up, and the slot flow matching with
      optimal transport plugs into. A constructor argument exactly like the
      prior §8b replaced.
    - §7 assembled `Sampler` with one integrator and the default grid; both
      slots hold more. `Euler` and `Heun` integrate the reverse SDE or the
      probability-flow ODE (`churn=0`) instead of stepping the exact posterior,
      and a warped `grid` spends steps where the structure actually appears —
      the usual first thing to tune when a sampler needs to get cheaper.
    - The force-field side this tutorial rode in on — property prediction,
      ML-driven molecular dynamics, the Lightning/CLI training stack — is
      covered by the SchNetPack documentation and tutorials.
    """)
    return


if __name__ == "__main__":
    app.run()
