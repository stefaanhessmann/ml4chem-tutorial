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
    7. **Sampling** — direct denoising, and checking what we made
    8. **Your task** — a variance-conditioned sampler

    ## 1. Setup

    Everything you need ships in one folder:

    ```
    ML4Chem-tutorial/
    ├── notebook.py            ← this tutorial
    ├── data/
    │   └── qm9_c4h4n2o2.xyz   ← the dataset: 181 small molecules from QM9
    ├── checkpoints/
    │   └── gpff.pt            ← a trained model — nothing has to train today
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
    pip install marimo matplotlib scipy
    marimo edit notebook.py
    ```

    This is a [marimo](https://marimo.io) notebook: cells form a dependency
    graph and re-run automatically when something they use changes. Every cell
    is plain Python — everything here works the same in a script.

    **Hardware.** The first code cell sets `DEVICE` to a GPU when one is
    available and to the CPU otherwise; the model and every batch follow it,
    and nothing below is device-specific. The CPU is fast enough for the whole
    tutorial — the trained model ships as a checkpoint — so the GPU only
    matters if you set `RETRAIN = True` in §6.
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
      the structure. $\sigma_\text{max}$ must match the data scale — rule of
      thumb: the largest pairwise distance, ~7.7 Å here. Too small and the
      endpoint still remembers the data; too large and training wastes
      capacity — and *neither failure is loud*.

    For all illustrations we use a single molecule: an elongated, open-chain
    isomer whose shape is easy to track through the noise. Below, that chain
    under both processes with the same underlying noise draw (slider = $t$):
    VP shrinks it into a small fixed-size cloud, VE leaves it in place and
    buries it under a 10 Å one.
    """)
    return


@app.cell
def _(AtomsLoader, batch, dataset, properties, torch, viz):
    from schnetpack.generative import VE, VP

    SIGMA_MIN, SIGMA_MAX = 0.05, 10.0  # ≈ largest pairwise distance + margin
    ve = VE(sigma_min=SIGMA_MIN, sigma_max=SIGMA_MAX)
    vp = VP(scale=float(batch[properties.R].std()))  # VP wants the data scale

    CHAIN_IDX = 90  # the most elongated open-chain isomer of the dataset
    chain = next(iter(AtomsLoader(dataset, sampler=[CHAIN_IDX])))
    x0 = chain[properties.R]  # the structure every illustration below noises

    # a trajectory is just interpolate() evaluated along a grid of times
    torch.manual_seed(3)
    t_noise = torch.linspace(0.0, 1.0, 13)
    z_noise = torch.randn_like(x0)  # shared draw — only its scale differs
    frames_vp = [vp.interpolate(x0, vp.prior.std * z_noise, t) for t in t_noise]
    frames_ve = [ve.interpolate(x0, ve.prior.std * z_noise, t) for t in t_noise]

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
    return SIGMA_MAX, SIGMA_MIN, VE, chain, ve, x0


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
    remember this for §6, and the second identity for §8, where we ask a
    trained model *how noisy is this structure?*

    The lecture introduced the alternatives, and the library has them too:
    `EpsParametrization` predicts the unit noise that was mixed in (same
    typical arrow length at every noise level — the DDPM convention), and
    `ScoreParametrization` the score, whose magnitude runs like $1/\sigma$.

    Below, the same noising path three times, with the **training target drawn
    as an arrow on every atom**. The pseudo-force arrows are long in deep
    noise and shrink to nothing as the structure comes home — that is
    $\mathrm{RMS}(F) = 2\sigma$ made visible. The eps arrows keep their size
    everywhere; the score arrows explode as $\sigma \to 0$.
    """)
    return


@app.cell
def _(chain, torch, ve, viz, x0):
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
    x1_param = ve.prior.sample_like(x0)
    ts_param = [torch.full((len(x0),), float(ti)) for ti in t_param]
    frames_param = [ve.interpolate(x0, x1_param, t) for t in ts_param]
    targets = {
        name: [p.target(ve, x0, x1_param, t) for t in ts_param]
        for name, p in (
            ("pseudo-force target", force_param),
            ("eps target", eps_param),
            ("score target", score_param),
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
    Two further axes exist that this tutorial does not vary: the **coupling**
    (how the drawn $(x_0, x_1)$ pairs are matched up) and the **prior** (what
    $x_1$ is drawn from) — equally swappable constructor arguments; see the
    `schnetpack.generative` API docs.

    ### Wrapping it into a transform

    Building diffusion training data is preprocessing — so it is a
    **transform** like the ones in §3. `Diffuse(process, parametrization)`
    runs the forward process inside the dataloader: per structure it draws a
    time, noises the positions, and writes the label into the item dict. Only
    the order needs thought:

    1. `SubtractCenterOfGeometry` — diffusion lives in the centered frame;
    2. `Diffuse` — overwrites `R` with $x_t$, writes the label
       `"pseudo_force"` and the time `"t"`;
    3. `MatScipyNeighborList` — **after** noising, with a cutoff sized for
       *noised* structures (30 Å, not 10);
    4. `CastTo32`.

    An ordinary MSE against `"pseudo_force"` is then the whole training
    objective — no special training loop anywhere.
    """)
    return


@app.cell
def _(
    ASEAtomsData,
    AtomsLoader,
    DB_PATH,
    SIGMA_MAX,
    SIGMA_MIN,
    VE,
    force_param,
    trn,
):
    from schnetpack.generative import Diffuse, PermutationCoupling

    CUTOFF = 30.0  # must cover *noised* structures, not just clean ones

    # the same VE schedule as above, plus a coupling: it re-pairs atoms and
    # noise draws (nearest gets nearest), which shortens the paths the model
    # must learn. One of the two axes we skip — passed along silently because
    # the shipped checkpoint was trained with it.
    gpff_process = VE(
        sigma_min=SIGMA_MIN, sigma_max=SIGMA_MAX, coupling=PermutationCoupling()
    )

    diffused = ASEAtomsData(
        DB_PATH,
        load_properties=[],
        transforms=[
            trn.SubtractCenterOfGeometry(),
            Diffuse(gpff_process, force_param, label_key="pseudo_force", time_key="t"),
            trn.MatScipyNeighborList(cutoff=CUTOFF),
            trn.CastTo32(),
        ],
    )
    train_loader = AtomsLoader(diffused, batch_size=100, shuffle=True)
    # a second, smaller draw from the same pipeline — only so the picture below
    # stays a picture; a hundred viewers on one page is not one
    peek = next(iter(AtomsLoader(diffused, batch_size=10, shuffle=True)))
    {key: tuple(peek[key].shape) for key in ("_positions", "pseudo_force", "t")}
    return CUTOFF, gpff_process, peek, train_loader


@app.cell
def _(mo):
    mo.md(r"""
    Those batches *are* the training set of our generative model — so look at
    one. Below, ten structures drawn from the same loader: each at its own
    randomly drawn time, captioned with its noise level, with the
    **pseudo-force label drawn as an arrow on every atom**.

    Read it as a difficulty gradient. At $\sigma \lesssim 0.5$ Å the molecule
    is intact and the arrows are tiny corrections; at $\sigma \sim 10$ Å there
    is no molecule left and the arrows span the whole cloud — the label's
    scale runs with $\sigma$ over two orders of magnitude, which is exactly
    what the loss in §6 will have to compensate.
    """)
    return


@app.cell
def _(gpff_process, peek, properties, viz):
    # one box per structure of the batch, captioned with its own noise level
    sigma_peek = gpff_process.sigma(peek["t_structure"])
    viz.show_trajectory(
        [peek[properties.R]],
        peek,
        vectors=[peek["pseudo_force"]],
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

    The cutoff is 30 Å because this force field sees *noised* structures,
    whose atoms are tens of Ångström apart.
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
            n_atom_basis=32,
            n_interactions=3,
            radial_basis=snn.GaussianRBF(n_rbf=30, cutoff=CUTOFF),
            cutoff_fn=snn.CosineCutoff(CUTOFF),
        ),
        input_modules=[PairwiseDistances()],
        output_modules=[
            AtomwiseVector(n_in=32, n_layers=1, output_key="pseudo_force_pred")
        ],
    ).to(DEVICE)
    f"GPFF model: {sum(p.numel() for p in gpff_net.parameters()):,} parameters on {DEVICE}"
    return gpff_net, snn


@app.cell
def _(mo):
    mo.md(r"""
    ### Training

    Model and data augmentation meet in an ordinary PyTorch loop: fetch a
    batch from the §5 loader, compare the model's prediction against the
    `"pseudo_force"` label, step the optimizer. The only diffusion-specific
    line is the **loss weight** $\min(1/\sigma(t)^2, 1)$: it undoes the
    label's $\sigma$-scaling so the heavily-noised samples don't drown out the
    nearly-clean ones.

    One thing to appreciate about the loop's shape: every pass over
    `train_loader` re-runs the transforms, so **every epoch sees fresh
    $(t, \text{noise})$ draws** — the dataset is effectively infinite. If you
    cache the batches instead (e.g. `itertools.cycle(train_loader)`), the
    noise freezes and the model memorizes a fixed set of noised structures
    rather than learning the denoising field.

    A trained checkpoint ships with the tutorial and is loaded by default; set
    `RETRAIN = True` to train from scratch — roughly 4 minutes on a GPU against
    40 on a CPU, which is the one place in this notebook the hardware
    genuinely matters. Don't over-read
    the loss curve: it plateaus at a high-looking value because the deepest
    noise levels keep a large irreducible error — that is healthy.
    """)
    return


@app.cell
def _(DEVICE, HERE, gpff_net, gpff_process, os, torch, train_loader):
    import matplotlib.pyplot as plt
    from tqdm.auto import tqdm

    from helpers import to_device

    CKPT = os.path.join(HERE, "checkpoints", "gpff.pt")
    RETRAIN = False  # flip to train from scratch instead of loading the checkpoint

    def gpff_loss(pred, inputs):
        # 1/sigma^2 undoes the label's sigma-scaling; the clamp keeps
        # nearly-clean samples from dominating
        weight = (1.0 / gpff_process.sigma(inputs["t"]) ** 2).clamp(max=1.0)
        diff = pred["pseudo_force_pred"] - inputs["pseudo_force"]
        return (weight[:, None] * diff**2).mean()

    if os.path.exists(CKPT) and not RETRAIN:
        ckpt_state = torch.load(CKPT, weights_only=True, map_location=DEVICE)
        gpff_net.load_state_dict(ckpt_state["state_dict"])
        history = [tuple(h) for h in ckpt_state["history"]]
    else:
        EPOCHS, LR_START, LR_END = 500, 5e-4, 1e-5
        optimizer = torch.optim.Adam(gpff_net.parameters(), lr=LR_START)
        # decay the step size geometrically from LR_START to LR_END across the
        # run — one factor per epoch, so the last epochs only polish
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=(LR_END / LR_START) ** (1 / EPOCHS)
        )
        history, step = [], 0
        # the bar counts epochs; its postfix is that epoch's mean loss, the same
        # number the curve below plots
        epochs = tqdm(range(EPOCHS), desc="epoch", unit="ep")
        for epoch in epochs:
            running = 0.0
            for train_batch in train_loader:  # fresh (t, noise) draws every epoch
                step += 1
                train_batch = to_device(train_batch, DEVICE)  # transforms ran on CPU
                loss = gpff_loss(gpff_net(train_batch), train_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running += loss.item()
                history.append((step, loss.item()))
            scheduler.step()
            epochs.set_postfix(
                loss=f"{running / len(train_loader):.4f}",
                lr=f"{scheduler.get_last_lr()[0]:.1e}",
            )
        torch.save({"state_dict": gpff_net.state_dict(), "history": history}, CKPT)

    gpff_model = gpff_net.eval()  # downstream cells use the *trained* model

    loss_fig, loss_ax = plt.subplots(figsize=(6, 3))
    loss_ax.plot(*zip(*history), lw=1.2)
    loss_ax.set_xlabel("step")
    loss_ax.set_ylabel("weighted pseudo-force MSE")
    loss_ax.grid(alpha=0.3)
    loss_fig
    return gpff_model, plt, to_device


@app.cell
def _(mo):
    mo.md(r"""
    ## 7. Sampling

    ### Direct denoising

    The pseudo force is the way home in one step: $\hat x_0 = x + F/2$. Taken
    from pure noise that single jump lands on the model's *conditional mean*
    over every structure that could hide under it — a blob, not a molecule —
    so sampling iterates instead: denoise, put a little noise back, repeat.

    That loop is `DirectDenoisingSampler`. At iteration $k$ of $N$ it injects
    $\lambda\,(1 - k/N)$ of noise before the model call, decaying to zero.
    There is no time grid and no schedule in it at all: it never asks what
    noise level its iterate sits at, which is only possible because the model
    doesn't either. The injected noise is what buys sample diversity;
    $\lambda = 0$ would be the plain repeated jump.

    Two pieces of glue from `helpers.py`. `fully_connected_batch` lays out 8
    copies of our composition as one flat batch — the topology the model needs
    (`Z`, `idx_m`, and a neighbor list) for molecules that don't exist yet.
    `make_model_fn` adapts our batch-dict model to the plain-tensor contract
    `model(x, t, cond)` the sampler expects, binding everything except the
    positions. The neighbor list is static: atoms move at every step, but the
    cutoff function handles the changing distances, so nothing is rebuilt.
    """)
    return


@app.cell
def _(
    DEVICE,
    force_param,
    gpff_model,
    gpff_process,
    numbers,
    properties,
    to_device,
    torch,
    viz,
):
    from helpers import fully_connected_batch, make_model_fn
    from schnetpack.generative import DirectDenoisingSampler

    torch.manual_seed(2)
    # 8 molecules-to-be, laid out where the model lives
    sampling_batch = to_device(fully_connected_batch(numbers, n_mol=8), DEVICE)
    model_fn = make_model_fn(gpff_model, sampling_batch, "pseudo_force_pred")

    direct = DirectDenoisingSampler(gpff_process, force_param, stochastic_lambda=1.0)

    # draw the start with the batch layout as context — each molecule's cloud
    # centered on its own — then run the denoising loop
    n_total = int(sampling_batch[properties.n_atoms].sum())
    x_init = direct.prior.sample((n_total, 3), device=DEVICE, context=sampling_batch)
    with torch.no_grad():
        x_sampled = direct.denoise(model_fn, x_init, n_steps=15)

    viz.show_batch({**sampling_batch, properties.R: x_sampled}, cell_px=170)
    return model_fn, sampling_batch, x_sampled


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

    Our model is small (~48k parameters) and saw 181 structures, so don't
    expect perfection — some samples come out with atoms fused together. Being
    able to say *how many* is the point: that number is what tells you whether
    a change to the model, the process or the sampler actually helped.
    """)
    return


@app.cell
def _(batch, properties, sampling_batch, torch, x_sampled):
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
        "generated (8 samples)": geometry_summary(x_sampled, sampling_batch),
    }
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 8. Your task: a variance-conditioned direct-denoising sampler

    `DirectDenoisingSampler` still marches down a *prescribed* schedule: its
    injected noise is $\lambda(1 - k/N)$, decided before the run and identical
    for every molecule in the batch. But a GPFF model reveals the noise level
    in its own prediction:

    $$\hat\sigma_m = \tfrac{1}{2}\sqrt{\langle F^2 \rangle_m}
      \qquad \text{(per-molecule } \mathrm{RMS}(F)/2\text{)}.$$

    **Write a `DirectDenoisingSampler` subclass that conditions the injection
    on that estimate instead.** Override `denoise`; per iteration:

    1. predict the pseudo force $F$ for the current structures;
    2. estimate each molecule's noise level $\hat\sigma_m$ from $F$;
    3. direct-denoise: $\hat x_0 = x + F/2$;
    4. if $\max_m \hat\sigma_m \le \sigma_\text{min}$: stop, return $\hat x_0$;
    5. otherwise re-noise a *shrunken* amount onto the estimate,
       $x \leftarrow \hat x_0 + \kappa\,\hat\sigma_m\,\varepsilon$ with
       $\varepsilon \sim \mathcal N(0, I)$, $\kappa \approx 0.7$ — and go to 1.

    No grid and no schedule: the model paces its own descent, and every
    molecule descends at its own rate. Note what the class needs that the base
    class does not: $\hat\sigma_m$ is *per molecule*, so the batch layout
    (`idx_m`, `n_atoms`) has to be handed to the constructor — the
    tensor-level core knows only an anonymous sample axis.

    Useful pieces: `model_fn(x, None)`, `snn.scatter_add` for per-molecule
    sums, `sampling_batch[properties.idx_m]` to broadcast per-molecule values
    back to atoms.

    ```python
    class VarianceConditionedDenoising(DirectDenoisingSampler):
        def __init__(self, process, parametrization, idx_m, n_atoms,
                     shrink=0.7, sigma_stop=SIGMA_MIN, **kwargs):
            super().__init__(process, parametrization, **kwargs)
            ...

        def denoise(self, model, x_t, n_steps, cond=None):
            x = x_t
            for _ in range(n_steps):
                F = model(x, None, cond)   # 1. predict — no time input
                sigma_hat = ...            # 2. per-molecule RMS(F) / 2
                x0_hat = ...               # 3. direct denoising
                if ...:                    # 4. converged?
                    return x0_hat
                x = ...                    # 5. shrink & re-noise
            return x
    ```

    **Check your implementation against these questions:** how many model
    calls does it take before it stops (the fixed loop above used 15)? What
    happens for $\kappa \to 1$ and for $\kappa \to 0$? Why is it fine that
    molecules reach $\sigma_\text{min}$ at different iterations?
    """)
    return


@app.cell
def _(
    DEVICE,
    SIGMA_MIN,
    force_param,
    gpff_process,
    model_fn,
    plt,
    properties,
    sampling_batch,
    snn,
    torch,
    viz,
):
    from schnetpack.generative import DirectDenoisingSampler as _DDS

    class VarianceConditionedDenoising(_DDS):
        """Direct denoising paced by the model's own per-molecule sigma estimate."""

        def __init__(
            self,
            process,
            parametrization,
            idx_m,
            n_atoms,
            shrink=0.7,
            sigma_stop=SIGMA_MIN,
            **kwargs,
        ):
            super().__init__(process, parametrization, **kwargs)
            self.idx_m = idx_m  # which molecule each row belongs to
            self.n_atoms = n_atoms
            self.shrink = shrink
            self.sigma_stop = sigma_stop
            self.frames, self.sigma_track = [], []  # for the movie below

        def sigma_hat(self, F):
            """Per-molecule noise estimate RMS(F) / 2 — the variance conditioning."""
            msq = snn.scatter_add(
                (F**2).mean(-1), self.idx_m, dim_size=self.n_atoms.shape[0]
            ) / self.n_atoms.to(F.dtype)
            return msq.sqrt() / 2

        def denoise(self, model, x_t, n_steps, cond=None):
            x = x_t
            self.frames, self.sigma_track = [x.clone()], []
            for _ in range(n_steps):
                F = model(x, None, cond)  # 1. predict — no time input
                sigma_hat = self.sigma_hat(F)  # 2. how noisy does the model think it is?
                self.sigma_track.append(sigma_hat.clone())
                # 3. direct denoising — the pseudo force is the way home
                x0_hat = self.parametrization.to_x0(self.process, F, x, None)
                # 4. stop when even the noisiest molecule is (estimated) clean
                if float(sigma_hat.max()) <= self.sigma_stop:
                    self.frames.append(x0_hat.clone())
                    return x0_hat
                # 5. re-noise a shrunken amount onto the estimate
                x = x0_hat + (self.shrink * sigma_hat)[self.idx_m].unsqueeze(
                    -1
                ) * torch.randn_like(x)
                self.frames.append(x.clone())
            return x

    def solution_view():
        torch.manual_seed(6)
        vcd = VarianceConditionedDenoising(
            gpff_process,
            force_param,
            sampling_batch[properties.idx_m],
            sampling_batch[properties.n_atoms],
        )
        x_start = vcd.prior.sample(
            (int(sampling_batch[properties.n_atoms].sum()), 3),
            device=DEVICE,
            context=sampling_batch,
        )
        with torch.no_grad():
            vcd.denoise(model_fn, x_start, n_steps=80)

        fig, ax = plt.subplots(figsize=(5.5, 3.2))
        ax.semilogy(torch.stack(vcd.sigma_track).cpu().numpy(), alpha=0.8)
        ax.axhline(SIGMA_MIN, color="k", ls=":", lw=1)
        ax.set_xlabel("iteration (= model calls)")
        ax.set_ylabel(r"estimated $\hat\sigma_m$ [Å]")
        ax.set_title(
            f"self-paced descent: {len(vcd.sigma_track)} model calls, one line per molecule"
        )
        ax.grid(alpha=0.3, which="both")
        # one animated box per molecule; the σ̂ story is the plot above
        viewer = viz.show_trajectory(vcd.frames, sampling_batch, cell_px=170)
        return fig, viewer

    viz.details("🔑 Reference solution — try it yourself first!", *solution_view())
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Where to go from here

    Three things this tutorial skipped, all in the library. The two remaining
    axes — the *coupling* and the *prior* — are constructor arguments exactly
    like the process and the parametrization, and swapping them is how methods
    like flow matching with optimal transport or shaped priors are built.
    Schedule-based sampling (`Sampler` with the `Ancestral` integrator and
    friends) is the classical reverse-process route, and the one to use for
    time-conditioned diffusion models. And the force-field side this tutorial
    rode in on — property prediction, ML-driven molecular dynamics, and the
    Lightning/CLI training stack — is covered by the SchNetPack documentation
    and tutorials.
    """)
    return


if __name__ == "__main__":
    app.run()
