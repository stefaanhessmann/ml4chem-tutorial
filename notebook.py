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
def _(mo):
    # A still of §7's direct-denoising sampler, drawn by the same viewer every
    # section below uses. Carried inline, base64: this cell renders before §1
    # has installed anything, so it can import nothing and read no file.
    mo.md(
        r"""
        <img width="760" alt="noise denoised into a molecule over successive model calls"
             src="data:image/png;base64,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">
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    - **Context.** Niklas Gebauer's talk covered the theory of generative
      models for molecules. This session is its practical counterpart: the
      code that turns that theory into a model you can train and sample from.
    - **What we build.** A **Generative Pseudo-Force Field (GPFF)**, trained on
      **QM9** — a 181-molecule slice of it here, small enough to train live in
      this notebook, with a research-scale checkpoint on all of QM9 to sample
      from.
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
