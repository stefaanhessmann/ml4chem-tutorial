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

    **ML4Chem hands-on tutorial**: from force fields to generative models
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    - **Context.** Niklas Gebauer's talk covered the theory of generative
      models for molecules. This session is its practical counterpart: the
      code that turns that theory into a model you can train and sample from.
    - **What we build.** A **Generative Pseudo-Force Field (GPFF)**, a
      **diffusion-based generative model**, trained on **QM9**. A 181-molecule
      slice of it here, small enough to train live in this notebook, plus a
      research-scale checkpoint on all of QM9 to sample from.
    - **Scope.** We generate **equilibrium structures only**, meaning the 3D
      geometry. The composition is given: positions diffuse, **atom types do
      not**.
    - **The code.** This is the **work-in-progress `schnetpack.generative`
      module of SchNetPack 3**, on its way to a general toolbox for
      diffusion-based generative models. The structure you see here is meant to
      stay; what grows around it is more of the same kind, such as flow
      matching and further processes and samplers.
    - **Not covered.** Diffusion is one family among several. Autoregressive
      models such as **G-SchNet**, which place atoms one after another, solve
      the same problem a different way. SchNetPack 3 implements the
      diffusion-based family.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## What this notebook covers

    1. **Setup**: the repo, and the runtime it needs
    2. **Introduction**: SchNetPack today, and where GPFF plugs in
    3. **Your own data**: databases, transforms, batches
    4. **Roadmap**: the three parts of a diffusion model
    5. **Forward process**: noising as data augmentation, and the labels it
       writes
    6. **Model and training**: a force field on noised structures
    7. **Sampling**: ancestral, direct denoising, validation
    8. **Your tasks**: steering the sampler
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Setup

    - **Everything lives in one repository:**
      **https://github.com/stefaanhessmann/ml4chem-tutorial**
    - **Open it and click the "Open in Colab" badge.** That is the quickest way
      in, and the first code cell below then pulls everything into the runtime.
    - **Everything ships in one folder**, so nothing has to be collected by
      hand:

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

    - **To run it locally instead**, create an environment, install SchNetPack
      from the tutorial branch, and start the notebook:

    ```bash
    conda create -n ml4chem python=3.12
    conda activate ml4chem
    pip install "git+https://github.com/atomistic-machine-learning/schnetpack.git@sh/v3"
    pip install marimo matplotlib scipy rdkit
    marimo edit notebook.py
    ```

    - **That local copy is a [marimo](https://marimo.io) notebook**, where
      cells form a dependency graph and re-run when their inputs change. Every
      cell is plain Python, so everything here works the same in a script.
    - **Hardware.** The first code cell points `DEVICE` at a GPU if there is
      one and the CPU otherwise, and nothing below is device-specific.
    - **Only one step is GPU-hungry**, training §6's model with
      `RETRAIN = True` (~15 min on a GPU), and it is opt-in: that cell loads a
      checkpoint by default.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Introduction to SchNetPack

    **SchNetPack** is an open-source toolbox for atomistic machine learning.

    **SchNetPack 2, what the released package covers:**

    - **Machine-learned force fields (MLFFs)**: energies and forces at a
      fraction of the cost of the electronic-structure method they learn from,
      plus the interfaces that run molecular dynamics with them.
    - **Property prediction**: any per-atom or per-molecule quantity a dataset
      carries, through the same training stack.
    - **Architectures**: SchNet, PaiNN, SO3net, invariant and equivariant
      message passing on atomic neighborhoods, interchangeable inside one model
      interface.
    - **Datasets**: ASE-backed databases, transforms and loaders, with the
      standard benchmarks (QM9, MD17, Materials Project) ready to download.
    - **A command-line interface, Lightning and Hydra configs**: `spktrain`
      runs a training from composable YAML rather than from a script, so a
      model, a dataset or an optimizer is swapped by overriding a config group
      on the command line. This notebook builds its loop by hand instead, to
      keep every part visible.

    **SchNetPack 3, what we are adding:**

    - **Generative models.** A force field *evaluates* structures it is given;
      a generative model *produces* them, learning the distribution behind a
      dataset's geometries so new, plausible ones can be drawn from it.
    - **`schnetpack.generative`** makes **diffusion-based** generative models a
      first-class part of the toolbox: the **forward process** (noise
      schedules, parametrizations, priors and couplings, assembled from
      swappable pieces) and the **samplers** that run a trained model backwards
      from noise to structure.
    - **More of the same to come**, built from those interfaces: **flow
      matching**, further processes and further samplers.

    ### How generative models fit in

    - **GPFF states diffusion in the language of force fields.** Its target is
      a **pseudo force**, and that force defines a **pseudo potential energy
      surface**: noised structures sit uphill, and the field points every atom
      back down toward the clean structure. Where an MLFF learns the forces of
      a real PES, GPFF learns the forces of this one.
    - **So the rest carries over.** Same architectures, same data pipeline and
      transforms, same training loop. Only the forward process that makes the
      training data and the sampler that runs the model backwards are new.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Using your own data in SchNetPack

    - **Atomistic data comes in many formats.** **xyz** and **extended xyz**
      for molecules, **PDB** and **SDF/MOL** in the chemistry and biology
      tools, **CIF** and **POSCAR** for periodic structures, and whatever your
      electronic-structure code writes out (Gaussian, ORCA, VASP, CP2K).
    - **SchNetPack reads one of them**: an **ASE-backed SQLite database**. So
      whatever your structures start in, step one is a conversion.
    - **`schnetpack.data` provides the tools for it.** `ASEAtomsData.create`
      declares the stored properties with their units, and `add_systems` fills
      the database with `ase.Atoms`, which ASE will have parsed from any of the
      formats above.
    - **Our data** is an xyz file with all **181 isomers of C₄H₄N₂O₂** in QM9.
      One fixed composition, so the generative model only has to learn *where
      the atoms go*, not which atoms to place.
    - **We also store the U0 energies** from the xyz. They are unused here, but
      a database declares its properties up front and real datasets carry them.
    """)
    return


@app.cell
def _():
    import os
    import numpy as np
    import torch
    from ase.io import read
    from schnetpack.data import ASEAtomsData, AtomsLoader

    HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else "."
    XYZ_FILE = os.path.join(HERE, "data", "qm9_c4h4n2o2.xyz")
    DB_PATH = os.path.join(HERE, "data", "qm9_c4h4n2o2.db")
    # Everything downstream follows this one line: the GPU when this machine
    # (or this Colab runtime) has one, the CPU otherwise.
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # the tutorial's data: one xyz of isomers, read once and put into a db
    molecules = read(XYZ_FILE, index=":")  # a list of ase.Atoms
    numbers = molecules[0].get_atomic_numbers().tolist()  # every isomer: same composition

    if not os.path.exists(DB_PATH):
        db = ASEAtomsData.create(
            datapath=DB_PATH,
            distance_unit="Ang",
            property_unit_dict={"energy": "eV"},
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

    - A dataset item is a **dict of tensors** (positions `R`, atomic numbers
      `Z`, …) keyed by `schnetpack.properties`. That dict is the universal
      interface: every SchNetPack model consumes it, and everything we build
      below writes into it.
    - **Transforms** are per-structure preprocessing owned by the dataset,
      re-run on every load. Here: center (`SubtractCenterOfGeometry`), build
      the neighbor list (`MatScipyNeighborList`, 10 Å, which fully connects a
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
        datapath=DB_PATH,
        load_properties=[],  # skip the stored energies, not needed here
        transforms=[
            trn.SubtractCenterOfGeometry(),
            trn.MatScipyNeighborList(cutoff=10.0),  # fully connects a molecule
            trn.CastTo32(),
        ],
    )
    loader = AtomsLoader(dataset=dataset, batch_size=8, shuffle=False)
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

    # the loaded batch in 3D: drag to rotate
    viz.show_batch(batch, cell_px=170)
    return (viz,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Generative models in SchNetPack: the roadmap

    In code, a diffusion-based generative model is **three parts**, and each
    gets one section:

    - **I. Forward process** (§5): noising structures and computing labels,
      inside the dataloader.
    - **II. Model architecture** (§6): an MLFF-shaped network applied to noised
      structures.
    - **III. Sampling** (§7): iterating the trained model from noise to
      structures.

    - **Training** (§6) is where I and II meet.
    - **Validation** (§7) follows sampling: turn "it looks like molecules" into
      numbers.
    - **The goal, plainly:** train GPFF on our 181-molecule QM9 slice and
      generate new C₄H₄N₂O₂ geometries.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. The forward process: noising as data augmentation

    A force field trains on labels the dataset ships, energies and forces. A
    diffusion model **manufactures its own**, and the forward process is where
    that happens: it is **data augmentation**. Noise a clean structure, write
    down the way back, and one dataset becomes an endless supply of labelled
    training pairs.

    ### I. Process: noising the geometry

    `schnetpack.generative` writes the forward process as an **interpolation**
    between the data $x_0$ and an endpoint $x_1$ drawn from a prior:

    $$x_t = a(t)\,x_0 + b(t)\,x_1, \qquad t \in [0, 1],$$

    with $a(0) = 1,\, b(0) \approx 0$ (the data) and $b(1) = 1$ (pure noise). A
    `Process` owns $a$, $b$, the prior, and the noise level
    $\sigma(t) = b(t)\,\sigma_\text{prior}$. The two standard schedules:

    - **`VP`** (variance preserving, as in DDPM): the data is scaled away as
      noise of fixed scale blends in; the total variance stays constant.
    - **`VE`** (variance exploding, as in score matching): the data is never scaled
      ($a \equiv 1$), and noise is simply *added* until it drowns the
      structure, with $\sigma(t)$ growing geometrically.

    We take **VE**, from $\sigma_\text{min} = 0.05$ to
    $\sigma_\text{max} = 30$ Å. $\sigma_\text{max}$ has to be large enough that
    nothing of the original structure is left to see at the end of the process.

    Below, and in every illustration that follows, one elongated open-chain
    isomer, easy to track through the noise, under both processes with the same
    noise draw (slider = $t$). VP shrinks it into a small fixed-size cloud; VE
    leaves it in place and buries it under a 30 Å one.
    """)
    return


@app.cell
def _(AtomsLoader, batch, dataset, properties, torch, viz):
    from schnetpack.generative import VE, VP

    SIGMA_MIN, SIGMA_MAX = 0.05, 30.0  # as the §7 generation model was trained
    CHAIN_IDX = 90  # the most elongated open-chain isomer of the dataset
    chain = next(iter(AtomsLoader(dataset=dataset, sampler=[CHAIN_IDX])))
    x0 = chain[properties.R]  # the structure every illustration below noises

    # the VE process every later section shares
    ve = VE(sigma_min=SIGMA_MIN, sigma_max=SIGMA_MAX)
    vp = VP(scale=float(batch[properties.R].std()))  # VP wants the data scale

    SEED = 3  # the one seed this notebook uses; every draw below reuses it

    # a trajectory is just interpolate() evaluated along a grid of times
    torch.manual_seed(SEED)
    t_noise = torch.linspace(0.0, 1.0, 13)
    z_noise = torch.randn_like(x0)  # shared draw: only its scale differs
    frames_vp = [vp.interpolate(x0=x0, x1=vp.prior.std * z_noise, t=t) for t in t_noise]
    frames_ve = [
        ve.interpolate(x0=x0, x1=ve.prior.std * z_noise, t=t) for t in t_noise
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
    return SEED, SIGMA_MAX, chain, ve, x0


@app.cell
def _(mo):
    mo.md(r"""
    ### II. Parametrization: defining the training targets

    Noised structures are half the training data; the **label** is the other
    half, and choosing it is the second axis, the `Parametrization`. All three
    below are one map away from each other. What separates them is how the
    target's magnitude scales with $\sigma$, which is exactly what a plain L2
    loss sees.

    - **`EpsParametrization`**: $\varepsilon$
    - **`ScoreParametrization`**:
      $s = \nabla_x \log p_t(x_t) = -\varepsilon/\sigma$
    - **`PseudoForceParametrization`**: $F = 2\,(x_0 - x_t)$

    GPFF takes the **pseudo force**, whose magnitude grows in proportion to
    $\sigma$. That buys two things:

    - getting home is one addition, $\hat x_0 = x_t + F/2$, with no division,
      so nothing degenerates as $\sigma \to 0$;
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
def _(SEED, chain, torch, ve, viz, x0):
    from schnetpack.generative import (
        EpsParametrization,
        PseudoForceParametrization,
        ScoreParametrization,
    )

    force_param = PseudoForceParametrization()  # F = 2 (x0 - x_t), GPFF's target
    eps_param = EpsParametrization()
    score_param = ScoreParametrization()

    # one path, three targets: `target` turns the same (x0, x1, t) into
    # whichever field the network is asked to predict
    torch.manual_seed(SEED)
    t_param = torch.linspace(1.0, 0.2, 13)
    x1_param = ve.prior.sample_like(x0)
    ts_param = [torch.full((len(x0),), float(ti)) for ti in t_param]
    xt_frames = [ve.interpolate(x0=x0, x1=x1_param, t=t) for t in ts_param]
    targets = {
        # the pseudo force is drawn at half length: F/2 = x0 - x_t is the
        # offset itself, so each arrow lands exactly on the clean structure
        name: [scale * p.target(process=ve, x0=x0, x1=x1_param, t=t) for t in ts_param]
        for name, p, scale in (
            ("eps target", eps_param, 1.0),
            ("score target", score_param, 1.0),
            ("pseudo-force target (F/2)", force_param, 0.5),
        )
    }

    viz.show_frames(
        {name: xt_frames for name in targets},
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
    drawn from), equally swappable constructor arguments. The prior returns in
    §8b, where replacing it is half the task.

    ### III. Diffuse: wrapping both into a transform

    - **It is preprocessing, so it is a transform** like §3's.
      `Diffuse(process, parametrization)` runs the forward process inside the
      dataloader: per structure it draws a time, noises the positions, and
      writes the label into the item dict.
    - **Where those draws land is its own choice**, the `t_sampler`. Timesteps
      can be sampled in different ways, and we follow Karras et al.:
      `LogNormalSigmaTimes` concentrates them in the range of noise levels
      where the model can actually learn something. Its values were measured
      empirically.

    Only the transform order needs thought:

    1. `SubtractCenterOfGeometry`: diffusion lives in the centered frame, and
       the prior draws its endpoints there too. A translation-invariant network
       could never predict a displacement of a whole structure, so an
       off-center endpoint would be unlearnable noise in every label.
    2. `Diffuse`: overwrites `R` with $x_t$, writes `"pseudo_force"` and `"t"`.
    3. `AllToAllNeighborList`, **after** noising. A *distance*-based list
       built at one noise level is wrong at another, and a cutoff wide enough
       for a fully noised cloud (~90 Å across) returns every pair anyway, at
       the cost of searching for them. Pairs the model's cutoff function
       downweights to zero cost nothing.
    4. `CastTo32`.

    An ordinary MSE against `"pseudo_force"` is then the whole objective.
    """)
    return


@app.cell
def _(ASEAtomsData, AtomsLoader, DB_PATH, force_param, ve, trn):
    from schnetpack.generative import Diffuse, LogNormalSigmaTimes

    CUTOFF = 150.0  # must cover *noised* structures: clouds ~90 Å across

    # the forward process, wrapped into the dataset's transform pipeline:
    # train mostly around half an Ångström of displacement, GPFF's density
    t_sampler = LogNormalSigmaTimes(process=ve, mean=-0.7, std=1.2, truncate=True)
    diffused = ASEAtomsData(
        datapath=DB_PATH,
        load_properties=[],
        transforms=[
            trn.SubtractCenterOfGeometry(),
            # the same schedule the frames above walked, sampled where it helps
            Diffuse(
                process=ve,
                parametrization=force_param,
                t_sampler=t_sampler,
                label_key="pseudo_force",
                time_key="t",
            ),
            trn.AllToAllNeighborList(),
            trn.CastTo32(),
        ],
    )
    # a small batch from the pipeline, only so the picture below stays a
    # picture; a hundred viewers on one page is not one
    diffused_batch = next(
        iter(AtomsLoader(dataset=diffused, batch_size=10, shuffle=True))
    )
    {key: tuple(diffused_batch[key].shape) for key in ("_positions", "pseudo_force", "t")}
    return CUTOFF, diffused, diffused_batch


@app.cell
def _(mo):
    mo.md(r"""
    Those batches *are* the training set, so look at one. Ten structures from
    the same loader, each at its own drawn time, captioned with its noise
    level, the **label drawn as an arrow on every atom** (again at half length,
    so each arrow ends where its atom belongs).

    Read it as a difficulty gradient: at $\sigma \lesssim 0.5$ Å the molecule
    is intact and the arrows are tiny corrections; at several Ångström there is
    no molecule left and the arrows span the whole cloud. The label's scale
    runs with $\sigma$, exactly what §6's loss has to compensate. And note
    what the time sampler did: most draws sit below ~2 Å, where denoising is
    hard but learnable.
    """)
    return


@app.cell
def _(diffused_batch, ve, properties, viz):
    # one box per structure of the batch, captioned with its own noise level
    sigma_batch = ve.sigma(diffused_batch["t_structure"])
    viz.show_trajectory(
        [diffused_batch[properties.R]],
        diffused_batch,
        # F/2 = x0 - x_t, so each arrow ends on the clean structure
        vectors=[diffused_batch["pseudo_force"] / 2],
        titles=[f"σ = {float(s):.2f} Å" for s in sigma_batch],
        cell_px=170,
        zoom=1.0,
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. Model and training

    - **Same architecture as an MLFF**, of the *non-energy-conserving* kind: a
      3-vector read out per atom, rather than one energy per molecule and
      differentiated.
    - **No time conditioning.** Diffusion models generally need it, because the
      same noised geometry means a different target at a different $t$. GPFF
      does not, because the magnitude of the pseudo force carries the noise
      level, so this network is *exactly* an ordinary force field.

    `NeuralNetworkPotential` stacks three stages, each an `nn.Module` acting on
    the batch dict:

    1. **input**: `PairwiseDistances`, positions + neighbor list to distance
       vectors.
    2. **representation**: `PaiNN`, message passing to per-atom features. This
       one has to be *equivariant* rather than merely *invariant*: besides
       scalar features it carries vector features that rotate with the
       molecule, which is what lets a head output a well-behaved vector per
       atom. An invariant representation like `SchNet` cannot, since features
       that do not turn with the molecule give a head nothing to build a
       direction from. (`SO3net` is equivariant too, and drop-in.)
    3. **output**: `AtomwiseVector`, a 3-vector per atom. (An energy model
       would end in `Atomwise`: a scalar per atom, summed per molecule.)

    Two settings are concessions to *noised* inputs:

    - **cutoff 150 Å**, since a fully noised cloud is ~90 Å across, with 600
      `GaussianRBF` functions across it, one every 0.25 Å. Too few, and a 1.0 Å
      contact and a 1.4 Å bond get near-identical embeddings; a denoiser that
      cannot tell a clash from a bond will happily generate both.
    - **`norm_epsilon=1`**: PaiNN normalizes each pair direction as
      $r_{ij}/(d_{ij} + 1)$ rather than $r_{ij}/d_{ij}$, which stays finite
      when two atoms of a noise cloud land on top of each other.
    """)
    return


@app.cell
def _(CUTOFF, DEVICE, SEED, torch):
    import schnetpack.nn as snn
    from schnetpack.model import (
        AtomwiseVector,
        NeuralNetworkPotential,
        PaiNN,
        PairwiseDistances,
    )

    torch.manual_seed(SEED)

    # the denoiser: an ordinary MLFF, read out as a vector per atom
    gpff_net = NeuralNetworkPotential(
        representation=PaiNN(
            n_atom_basis=128,
            n_interactions=4,
            radial_basis=snn.GaussianRBF(n_rbf=600, cutoff=CUTOFF),
            cutoff_fn=snn.CosineCutoff(cutoff=CUTOFF),
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
    training a generative model; the transforms did that part.

    **The objective** is an MSE against the pseudo-force label, weighted per
    draw by $w(t) = \min(\sigma(t)^{-2}, w_\text{max})$.

    - **The $1/\sigma^2$ undoes the label's $\sigma$-scaling**, so what is
      minimized is the *relative* error at every noise level rather than the
      absolute one. Without it the deep-noise samples, whose labels are tens of
      Ångström long, drown out everything else.
    - **The ceiling decides how much of the small-$\sigma$ end survives**, and
      it is easy to set too low: at $w_\text{max} = 1$ it binds for 71% of the
      draws, flattening the weight across the whole band where bond lengths are
      decided. At **100** it is the honest $1/\sigma^2$ almost everywhere.

    Three things about the loop:

    - **Every step sees fresh $(t, \varepsilon)$ draws**: the transforms
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
    the loop instead and overwrites it. Either way the curve below is real,
    since the checkpoint stores its loss history alongside its weights. It plateaus
    well above zero because every noise level keeps an irreducible error, and
    that is healthy.
    """)
    return


@app.cell
def _(AtomsLoader, DEVICE, HERE, diffused, gpff_net, os, ve, torch):
    import matplotlib.pyplot as plt
    from tqdm.auto import tqdm
    from helpers import to_device

    CKPT = os.path.join(HERE, "checkpoints", "gpff.pt")
    RETRAIN = False  # True: run the loop below instead of loading the checkpoint

    # the training run itself, from its hyperparameters down
    STEPS, BATCH, LR_START, LR_END, EMA_DECAY = 12000, 64, 1e-3, 1e-5, 0.999

    def gpff_loss(pred, inputs):
        # 1/sigma^2 undoes the label's sigma-scaling; the ceiling keeps the
        # small-sigma end (where bonds are decided) from being flattened away
        weight = (1.0 / ve.sigma(inputs["t"]) ** 2).clamp(max=100.0)
        diff = pred["pseudo_force_pred"] - inputs["pseudo_force"]
        return (weight[:, None] * diff**2).mean()

    if os.path.exists(CKPT) and not RETRAIN:
        # the checkpoint carries its loss curve as well as its weights, so the
        # plot below is the real one from the run that produced them
        ckpt_state = torch.load(CKPT, weights_only=True, map_location=DEVICE)
        gpff_net.load_state_dict(ckpt_state["state_dict"])
        history = [tuple(h) for h in ckpt_state["history"]]
    else:
        # the whole run as one epoch of STEPS batches, drawn with replacement,
        # which is what lets the workers stay ahead of the GPU (see above)
        train_loader = AtomsLoader(
            dataset=diffused,
            batch_size=BATCH,
            sampler=torch.utils.data.RandomSampler(
                diffused, replacement=True, num_samples=BATCH * STEPS
            ),
            num_workers=4,
            persistent_workers=True,
        )

        optimizer = torch.optim.Adam(gpff_net.parameters(), lr=LR_START)
        # decay the step size geometrically from LR_START to LR_END across the
        # run, so the last steps only polish
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=(LR_END / LR_START) ** (1 / STEPS)
        )
        # the running average of the weights, which is what samples at the end
        ema = {k: v.detach().clone().float() for k, v in gpff_net.state_dict().items()}

        history = []
        steps = tqdm(train_loader, desc="step", unit="it", total=STEPS)
        for step, train_batch in enumerate(steps):
            train_batch = to_device(batch=train_batch, device=DEVICE)  # CPU-side
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

    Sampling runs the process backwards: start from a draw of the prior, a
    30 Å cloud, and call the model over and over until a molecule is left.
    Both samplers below do that with the same trained model; what they disagree
    about is *how*.

    ### The model that generates

    - **§6's network is teaching-sized** and saw 181 structures. Denoising has
      to be accurate exactly where geometry is decided, at
      $\sigma \lesssim 0.3$ Å where bond lengths live, precisely where the time
      sampler concentrated its training. Expect a good share of the draws to
      come out as chemically valid molecules, direct denoising ahead of the
      ladder; the cell below measures it.
    - **The bundle also ships `checkpoints/gpff_big.pt`**: same pseudo-force
      target, same VE process, same $\sigma$-focused time sampling, at
      **research scale, 5.1M parameters trained on all ~130k molecules of
      QM9**. The cell below assembles it exactly like §6's model, only wider.
    - **`USE_BIG_MODEL` swaps it in** for every sampling and validation cell
      that follows. Flip it after one pass: how far the numbers move is the
      most honest measure of what scale buys.
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
            cutoff_fn=snn.CosineCutoff(cutoff=CUTOFF),
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

    The **classical** route, the reverse process of the lecture, and what
    every time-conditioned diffusion model uses. Walk a *prescribed* ladder of
    noise levels from $t = 1$ down to $0$ and take one exact step down each
    rung: one model call estimates the clean structure,
    $\hat x_0 = x_t + F_\theta(x_t)/2$, then the iterate moves to the next
    rung by the exact Gaussian posterior
    $p(x_{t_{k-1}} \mid x_{t_k}, \hat x_0)$, an interpolation between where it
    is and $\hat x_0$ plus a matched noise injection. No approximation beyond
    the model's own error; the closer two rungs sit, the less the estimate is
    trusted in one go.

    SchNetPack assembles it from four parts, §5's two plus two that only
    sampling needs:

    | part | what it decides | here |
    |---|---|---|
    | `process` | the noise schedule $\sigma(t)$ | the `VE` of §5 |
    | `parametrization` | what the network's output means | pseudo force |
    | `integrator` | how one step down the ladder is taken | `Ancestral` |
    | `grid` | where the rungs sit | uniform in $t$ (the default) |

    Because VE's $\sigma$ grows *geometrically* in $t$, a uniform grid already
    gives the geometric ladder score matching wants, rungs that bunch up where
    $\sigma$ is small, so no schedule code is needed. (Warping the grid stays
    an option, and `grid` is where it would go.)

    Three pieces of glue from `helpers.py`:

    - `fully_connected_batch`: 8 copies of our composition as one flat batch,
      the topology (`Z`, `idx_m`, a neighbor list) for molecules that do not
      exist yet. It stays static, since the cutoff function handles the
      changing distances.
    - `make_model_fn`: adapts our batch-dict model to the sampler's
      plain-tensor `model(x, t, cond)` contract.
    - `recording_model_fn`: what makes the viewer a **movie**. A sampler
      returns only the structure it ended on, but every step passes its state
      through the model, so wrapping the model captures the whole run.

    Scrub the slider. The frames carry the grid's own times, so each caption
    reads out the rung it sits on, $t = 1$ down to $0$: the ladder comes down
    *gradually*, and a structure appears only over the last handful of rungs.
    Frame 0 is a ~80 Å cloud against a ~3 Å molecule, and no camera holds a 25×
    range, so the view is framed on the finished structure and pulled back
    (`zoom=0.35`), and the atoms fly in from outside. That gap *is* the scale
    the model closes.
    """)
    return


@app.cell
def _(
    DEVICE,
    SEED,
    force_param,
    gen_model,
    numbers,
    properties,
    to_device,
    torch,
    ve,
    viz,
):
    from helpers import fully_connected_batch, make_model_fn, recording_model_fn
    from schnetpack.generative import Sampler
    from schnetpack.generative.integrators import Ancestral

    N_LADDER = 64
    torch.manual_seed(SEED)

    # 8 molecules-to-be, laid out where the model lives
    sampling_batch = to_device(
        batch=fully_connected_batch(numbers=numbers, n_mol=8), device=DEVICE
    )
    model_fn = make_model_fn(
        model=gen_model, static_batch=sampling_batch, output_key="pseudo_force_pred"
    )
    n_total = int(sampling_batch[properties.n_atoms].sum())

    # process + parametrization as before, plus the two sampling-only parts;
    # `grid` is left at its default (uniform in t = geometric in sigma on VE)
    ancestral = Sampler(
        process=ve, parametrization=force_param, integrator=Ancestral()
    )

    # the sampler returns the final structure only; wrapping the model keeps
    # every state it was asked about, which is the trajectory
    watched_anc, ancestral_frames = recording_model_fn(model_fn=model_fn)
    with torch.no_grad():
        x_ancestral = ancestral.sample(
            model=watched_anc,
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
        frame_ms=120,  # 65 frames: play them faster than the default
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
    that could hide under it, a blob rather than a molecule, so it iterates
    instead: re-noise a little, jump home ($x + F_\theta(x)/2$), repeat, with
    the injection scaled by $\lambda$ and decaying linearly to zero over the
    run. No $t$ appears anywhere, which is only possible because the model
    does not need one either. That loop is `DirectDenoisingSampler`.

    $\lambda = 0$, the plain repeated jump with no injection at all, looks like
    the natural choice and is the wrong one. With nothing put back, the loop is
    a deterministic fixed-point iteration that converges to whatever the
    model's map happens to attract: atoms stranded off the structure, and
    occasionally a run that leaves the finite numbers behind altogether. The
    injection keeps the iterate inside the band the model was trained on, and
    the decay walks it down. Here $\lambda = 1$, and on the counts below it is
    the difference between a third of the samples passing and nearly all.

    **Cost.** The ladder above took **64** model calls; this loop takes **60**,
    close, because this model is small and wants the iterations. The budget
    is a dial rather than a schedule, so a stronger model gets away with far
    fewer (flip `USE_BIG_MODEL` and 15 is plenty).

    Watch the difference in the movie: where the ladder descended gradually and
    revealed a structure only near the bottom, direct denoising is at molecular
    size after two or three calls and spends the rest tidying up. Same model,
    same starting noise, only the route differs.
    """)
    return


@app.cell
def _(
    DEVICE,
    force_param,
    model_fn,
    n_total,
    recording_model_fn,
    sampling_batch,
    SEED,
    torch,
    ve,
    viz,
):
    from schnetpack.generative import DirectDenoisingSampler

    torch.manual_seed(SEED)
    direct = DirectDenoisingSampler(
        process=ve, parametrization=force_param, stochastic_lambda=1.0
    )

    # draw the start from the prior (a 30 Å cloud per molecule), then run
    # the denoising loop
    x_init = direct.prior.sample(
        shape=(n_total, 3), device=DEVICE, context=sampling_batch
    )

    watched_fn, direct_frames = recording_model_fn(model_fn=model_fn)
    with torch.no_grad():
        x_direct = direct.denoise(model=watched_fn, x_t=x_init, n_steps=60)
    direct_frames.append(x_direct)  # ...plus the structure it ended on

    viz.show_trajectory(
        direct_frames, sampling_batch, zoom=0.35, cell_px=170, frame_ms=120
    )
    return DirectDenoisingSampler, x_direct


@app.cell
def _(mo):
    mo.md(r"""
    ### Did we make molecules? Validity

    A rendered batch can look right and still be no molecule at all, so turn
    "it looks like molecules" into a number.

    - **RDKit reads chemistry out of coordinates.** `rdDetermineBonds` infers a
      bond graph from nothing but the positions, and a structure counts as
      **chemically valid** only if that graph works out: every valence
      satisfied as a neutral molecule, no unpaired electrons left over,
      everything in one connected piece.
    - **The valid fraction** is the standard headline metric for molecular
      generative models, and the dataset row calibrates it: a real batch scores
      full marks by construction.
    - **What passes earns a SMILES string**, the molecule's identity
      independent of coordinates. It says *which* molecule each sample is, so
      compare against the dataset's to see whether the model reproduced a
      training isomer or found a new one.
    - **It is a strict judge.** A single fused pair already sinks a sample,
      which is what makes it honest. Flip `USE_BIG_MODEL` and see what a
      network trained on all of QM9 does to it.
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
    instruction, and nearly every real use of a generative model is one. *Make
    it long and thin. Keep this ring and fill in the rest.*

    The obvious way there is to train for it. Both tasks below take the other
    route and leave the trained model **exactly as it is**: the instruction
    enters in the *sampler*, through one shared recipe.

    > Every iteration, force the state onto the constraint, then let the model
    > repair whatever that broke.

    Because a model call always follows the nudge, the repair is chemistry
    rather than interpolation: what comes out satisfies the constraint *and*
    survives the denoiser. The alternation is the whole trick; either half
    alone does not work.

    | | the constraint | how it enters |
    |---|---|---|
    | **a** | a **global, continuous** property, the structure's shape | a linear map on the state, before every model call |
    | **b** | **exact positions** for some atoms, a scaffold | a custom prior, plus rows the loop never updates |

    **How to work them.**

    - **Each task is three cells:** a class with `# TODO` markers to fill in, a
      **folded reference solution** under it, and a runner that samples and
      plays the result as a movie.
    - **The class runs as shipped**, it just steers nothing yet, so the first
      movie you get is §7's unguided sampler. Your job is to make it change:
      edit, re-run, watch.
    - **Nobody is stuck.** The runner picks its sampler on its first line.
      Leave it on your own class; point it at the reference to see the intended
      behaviour, and switch back to compare.
    - **Both subclass `DirectDenoisingSampler`**, whose loop is four lines and
      carries no schedule to stay consistent with, which makes it the one to
      interfere with.
    - **Both always use the research-scale model**, whatever §7's
      `USE_BIG_MODEL` says: steering is only legible when the model underneath
      is not the bottleneck.
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
    task_batch = to_device(
        batch=fully_connected_batch(numbers=numbers, n_mol=10), device=DEVICE
    )
    task_model_fn = make_model_fn(
        model=big_model, static_batch=task_batch, output_key="pseudo_force_pred"
    )
    n_task = int(task_batch[properties.n_atoms].sum())
    return n_task, task_batch, task_model_fn


@app.cell
def _(mo):
    mo.md(r"""
    ### a) Shape-guided direct denoising

    **Generate molecules of a prescribed shape**: a rod, a disc, a ball.

    A structure's shape lives in the spread of its atoms along its three
    principal axes, and `SHAPE_TARGET` says how that spread should be divided
    between them. Stretching the cloud along one axis and squeezing it along
    another is a rotation and a rescaling away, and `torch.linalg.eigh` gives
    you the axes.

    **What to do**, two `# TODO`s in the next cell:

    - **Write `reshape`.** Per molecule, centered: find the principal axes,
      rescale along each so the spread matches the target, keeping the overall
      size unchanged.
    - **Call it inside the loop**, so the model always sees the reshaped state.

    Nudging then denoising, every iteration, is what keeps the result a
    molecule: each squeeze is small and the model call right after repairs it.
    The movie is the measurement, and each panel is captioned with whatever
    molecule came out.

    **Then ask:**

    - Does the batch come out the shape you asked for, and which target does
      the model fight hardest? Try `SHAPE_TARGET` as the disc and the ball.
    - On the ball run, read the molecule names: a planar ring cannot fill three
      dimensions, so what does the model reach for instead?
    - Let the total size grow as well, not just its division between axes.
      What breaks?
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
            super().__init__(process=process, parametrization=parametrization, **kwargs)
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
                # Find this molecule's principal axes and how far it spreads
                # along each (`torch.linalg.eigh` of the covariance: ascending
                # spreads, axes as *columns*), rescale so the spread matches
                # `self.target` without changing the total, and write it back.
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
                # TODO: one line. The state that goes into the model should be
                # the reshaped one
                x = self.parametrization.to_x0(
                    process=self.process, output=model(x, t, cond), x_t=x, t=t
                )
            return x

    return (ShapeGuidedDenoising,)


@app.cell(hide_code=True)
def _(DirectDenoisingSampler, torch):
    # @title 🔑 Reference solution, task a (click to reveal the code)
    class ShapeGuidedSolution(DirectDenoisingSampler):
        """The same class with both TODOs filled in."""

        def __init__(
            self, process, parametrization, idx_m, n_atoms, target, **kwargs
        ):
            super().__init__(process=process, parametrization=parametrization, **kwargs)
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
                    process=self.process, output=model(x, t, cond), x_t=x, t=t
                )
            return x

    return


@app.cell
def _(
    DEVICE,
    force_param,
    n_task,
    properties,
    recording_model_fn,
    SEED,
    ShapeGuidedDenoising,
    smiles_per_molecule,
    task_batch,
    task_model_fn,
    torch,
    ve,
    viz,
):
    SAMPLER = ShapeGuidedDenoising  # yours; ShapeGuidedSolution is the reference
    SHAPE_TARGET = (0.85, 0.13, 0.02)  # rod · disc (0.50, 0.45, 0.05) · ball (1/3, 1/3, 1/3)

    torch.manual_seed(SEED)
    shaped = SAMPLER(
        ve,
        force_param,
        task_batch[properties.idx_m],
        task_batch[properties.n_atoms],
        SHAPE_TARGET,
    )
    x_start = shaped.prior.sample(
        shape=(n_task, 3), device=DEVICE, context=task_batch
    )
    watched, shape_frames = recording_model_fn(model_fn=task_model_fn)
    with torch.no_grad():
        x_shaped = shaped.denoise(model=watched, x_t=x_start, n_steps=60)
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
    chemistry actually gets asked. The scaffold is the part that already works,
    a group that binds or a core a synthesis route exists for, and what is
    wanted is everything around it.

    Ours is an **alcohol**, `C-C-OH`, taken from 1-octanol — a plain
    eight-carbon chain with a hydroxyl on the end. Two carbons, the oxygen and
    its hydrogen are kept; the other 23 atoms, **6 of the 9 heavy ones**, are
    generated. So the model is not decorating a fixed core here, it is building
    a molecule around an anchor, and what comes back are different C₈H₁₈O
    skeletons wearing the same alcohol.

    The next cell states the scaffold as three arrays, positions, atomic
    numbers and the indices to keep, since that is all a scaffold is. It leaves
    you `scaffold_batch` (the layout), `scaffold_model_fn` (the model bound to
    it), `x_kept` (the coordinates) and `free_atoms` (the rows a sampler may
    touch).

    **Generate molecules that contain the given scaffold, in its given
    geometry.** Two things have to change, and the sampler as shipped does
    neither: it starts every atom in noise and moves every atom every step.

    **What to do**, three `# TODO`s in the next cell:

    - **Start right.** The prior decides where a run begins. Draw the free
      atoms from noise as usual, but place the kept ones at their coordinates.
    - **Keep them there.** Inside the loop, exclude the scaffold rows from the
      noise injection and from the update, so only the free atoms move.

    Get it right and the anchor stands still through the whole movie while the
    other 23 atoms assemble around it.

    **Then ask:**

    - Read the panel captions: they are alcohols, but which ones? Straight
      chains, branched ones, 1-octanol itself? None of them has to be the
      molecule the fragment came from, and most will not be. Watch for the odd
      one where the oxygen ends up *inside* the chain as an ether — the anchor
      pins three atoms, not what they are bonded to.
    - Seventeen of the free atoms are hydrogens, and RDKit only calls a
      structure a molecule if every one lands within bonding distance. That is
      why this task runs longer and quieter than §7: 150 iterations at half the
      noise injection. Put both back to §7's values (60 and 1.0) and count what
      survives.
    - This start is **out of distribution**: the model was trained where every
      atom carries the *same* noise level, and here four atoms are exact while
      23 sit 30 Å out. Does that show, and does starting the free atoms at a
      smaller `std` (3 Å, say) buy better completions, or only less diverse
      ones?
    - Freeze more: keep the next carbon along the chain too, or half of it. How
      much structure does the model need before it stops inventing and starts
      reconstructing?
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
    SCAFFOLD_Z = np.array(
        # fmt: off
        [6, 6, 6, 6, 6, 6, 6, 6, 8,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        # fmt: on
    )
    SCAFFOLD_R = np.array(  # positions in Angstrom, center of geometry at 0
        [
            [-1.21787, 4.08052, 2.08149],  # 0   C, the far end of the chain
            [-1.17900, 2.55043, 2.04083],  # 1   C
            [-0.47184, 1.99695, 0.79893],  # 2   C
            [-0.42804, 0.46613, 0.74944],  # 3   C
            [0.27930, -0.08542, -0.49288],  # 4   C
            [0.32289, -1.61667, -0.54188],  # 5   C
            [1.03114, -2.16083, -1.78589],  # 6   C   <- kept
            [1.07236, -3.68214, -1.83112],  # 7   C   <- kept, carries the OH
            [1.74801, -4.07032, -3.01888],  # 8   O   <- kept
            [-0.20643, 4.50216, 2.07927],  # 9   H, on C0
            [-1.74468, 4.48618, 1.21045],  # 10  H, on C0
            [-1.72828, 4.44535, 2.97846],  # 11  H, on C0
            [-0.67681, 2.17219, 2.94129],  # 12  H, on C1
            [-2.20337, 2.15633, 2.07912],  # 13  H, on C1
            [-0.97369, 2.37580, -0.10268],  # 14  H, on C2
            [0.55333, 2.39161, 0.75979],  # 15  H, on C2
            [0.07390, 0.08792, 1.65122],  # 16  H, on C3
            [-1.45342, 0.07210, 0.78862],  # 17  H, on C3
            [-0.22238, 0.29168, -1.39487],  # 18  H, on C4
            [1.30472, 0.30755, -0.53245],  # 19  H, on C4
            [0.82466, -1.99299, 0.36072],  # 20  H, on C5
            [-0.70313, -2.00885, -0.50216],  # 21  H, on C5
            [0.53088, -1.80043, -2.69335],  # 22  H, on C6
            [2.06088, -1.78477, -1.82911],  # 23  H, on C6
            [1.58879, -4.06678, -0.93582],  # 24  H, on C7
            [0.04512, -4.08270, -1.80869],  # 25  H, on C7
            [1.77295, -5.03099, -3.04987],  # 26  H, on the O  <- kept
        ]
    )
    SCAFFOLD = np.array([6, 7, 8, 26])  # C-C-OH: two carbons, the oxygen, its H
    N_SCAFFOLD = 10  # completions to generate

    scaffold_batch = to_device(
        batch=fully_connected_batch(numbers=SCAFFOLD_Z.tolist(), n_mol=N_SCAFFOLD),
        device=DEVICE,
    )
    scaffold_model_fn = make_model_fn(
        model=big_model, static_batch=scaffold_batch, output_key="pseudo_force_pred"
    )

    # the same anchor in every copy...
    x_kept = (
        torch.tensor(SCAFFOLD_R, dtype=torch.float32).repeat(N_SCAFFOLD, 1).to(DEVICE)
    )
    # ...and the mask of rows a sampler is allowed to touch
    kept = torch.zeros(len(SCAFFOLD_Z), dtype=torch.bool)
    kept[torch.as_tensor(SCAFFOLD)] = True
    free_atoms = (~kept).repeat(N_SCAFFOLD).to(DEVICE)

    # the same arrays as one molecule, only to look at
    scaffold_view = fully_connected_batch(numbers=SCAFFOLD_Z.tolist(), n_mol=1)
    scaffold_view[properties.R] = torch.tensor(SCAFFOLD_R, dtype=torch.float32)
    viz.show_batch(
        scaffold_view,
        titles=["keep C-C-OH — atoms 6, 7, 8 and the hydroxyl H 26"],
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
        Gaussian-only closed forms. Nothing here needs them, since direct
        denoising only ever asks a prior for a starting state.
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
            x = GaussianPrior.center(x=x, segments=self.idx_m)
            # TODO: the free rows start as noise, the scaffold rows start at
            # the coordinates they are supposed to keep (`torch.where`)
            return x

    class ScaffoldDenoising(DirectDenoisingSampler):
        """Direct denoising in which the scaffold rows never move."""

        def __init__(self, process, parametrization, x_scaffold, free, **kwargs):
            super().__init__(process=process, parametrization=parametrization, **kwargs)
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
                    process=self.process, output=model(x, t, cond), x_t=x, t=t
                )
                # TODO: only the free rows take the update
                x = x0_hat
            return x

    return ScaffoldDenoising, ScaffoldPrior


@app.cell(hide_code=True)
def _(DirectDenoisingSampler, torch):
    # @title 🔑 Reference solution, task b (click to reveal the code)
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
            x = _GaussianPrior.center(x=x, segments=self.idx_m)
            return torch.where(self.free, x, self.x_scaffold)

    class ScaffoldDenoisingSolution(DirectDenoisingSampler):
        """The same sampler with both TODOs filled in."""

        def __init__(self, process, parametrization, x_scaffold, free, **kwargs):
            super().__init__(process=process, parametrization=parametrization, **kwargs)
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
                    process=self.process, output=model(x, t, cond), x_t=x, t=t
                )
                x = torch.where(self.free, x0_hat, self.x_scaffold)
            return x

    return


@app.cell
def _(
    DEVICE,
    force_param,
    free_atoms,
    properties,
    recording_model_fn,
    scaffold_batch,
    scaffold_model_fn,
    ScaffoldDenoising,
    ScaffoldPrior,
    SEED,
    SIGMA_MAX,
    smiles_per_molecule,
    torch,
    ve,
    viz,
    x_kept,
):
    # yours; the references are ScaffoldPriorSolution and ScaffoldDenoisingSolution
    PRIOR, SAMPLER_B = ScaffoldPrior, ScaffoldDenoising

    torch.manual_seed(SEED)
    scaffolded = SAMPLER_B(
        ve,
        force_param,
        x_kept,
        free_atoms,
        prior=PRIOR(x_kept, free_atoms, scaffold_batch[properties.idx_m], SIGMA_MAX),
        # 23 free atoms is a lot to place: half the injection of §7, over more
        # iterations, roughly doubles how many completions come out as molecules
        stochastic_lambda=0.5,
    )
    watched_scaffold, scaffold_frames = recording_model_fn(
        model_fn=scaffold_model_fn
    )
    with torch.no_grad():
        x_scaffold = scaffolded.sample(
            model=watched_scaffold, shape=x_kept.shape, n_steps=150, device=DEVICE
        )
    scaffold_frames.append(x_scaffold)

    viz.show_trajectory(
        scaffold_frames,
        scaffold_batch,
        titles=smiles_per_molecule(x_scaffold, scaffold_batch),
        stride=3,  # 150 iterations is more frames than a scrubber needs
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

    - Of the five axes only the **coupling** is left untouched: how the
      $(x_0, x_1)$ pairs are matched up, and the slot flow matching with
      optimal transport plugs into. A constructor argument exactly like the
      prior §8b replaced.
    - §7 assembled `Sampler` with one integrator and the default grid; both
      slots hold more. `Euler` and `Heun` integrate the reverse SDE or the
      probability-flow ODE (`churn=0`) instead of stepping the exact posterior,
      and a warped `grid` spends steps where the structure actually appears,
      the usual first thing to tune when a sampler needs to get cheaper.
    - The force-field side this tutorial rode in on (property prediction,
      ML-driven molecular dynamics, the Lightning/CLI training stack) is
      covered by the SchNetPack documentation and tutorials.
    """)
    return


if __name__ == "__main__":
    app.run()
