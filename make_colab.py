"""
Build the Colab notebook from ``notebook.py``.

``notebook.py`` stays the single source of truth: this derives ``notebook.ipynb``
from it and is the only thing allowed to edit the result. Never hand-edit the
generated file — the next build overwrites it.

    python make_colab.py --repo https://github.com/<owner>/<bundle> \\
                         --schnetpack ~/projects/schnetpack

Three edits are applied to marimo's export:

1. drop the ``import marimo as mo`` cell — after §8 moved to ``viz.details``,
   no code cell touches the marimo API, so nothing else needs it;
2. rewrite the local-install instructions, which describe conda and
   ``marimo edit`` and are wrong in a hosted runtime;
3. insert the setup cell that installs SchNetPack and fetches the bundle,
   directly under the prose that introduces it.

With ``--schnetpack`` it also builds a wheel into this folder and has the setup
cell install *that*. It is worth the extra step for a class: ``pip install
git+https://...`` makes every student clone ~66 MB of history, where the wheel
is ~330 KB and needs no build. Being immutable, it also guarantees the whole
room runs identical code, which a branch — or even a retagged tag — does not.

The dependency only ever points bundle -> library: the wheel is *built from* a
SchNetPack checkout and committed *here*. Nothing from this tutorial belongs in
the SchNetPack repository.

Requires ``nbformat`` (marimo's exporter does).
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# The paragraph in §1 that only makes sense on a local machine: from the
# "create an environment" lead-in through the marimo explainer. Anchored on
# both ends so a reworded middle still matches, and asserted to hit exactly
# once so a rewrite that breaks the anchors fails loudly instead of silently
# shipping conda instructions to Colab.
LOCAL_SETUP = re.compile(
    r"Create an environment.*?works the same in a script\.",
    re.DOTALL,
)

COLAB_SETUP = """\
Nothing to install by hand: the cell below fetches SchNetPack and the tutorial
files into this runtime. Run it first — it takes a couple of minutes, and only
has to happen once per session.

This notebook asks Colab for a **GPU** runtime. If you got one it is used
automatically; if Colab handed you a CPU instead — free GPUs are rationed and
not guaranteed — everything still runs, just see *Hardware* below.\
"""


def build_wheel(checkout: Path) -> str:
    """Build a wheel from ``checkout`` into this folder. Returns its filename."""
    for stale in HERE.glob("schnetpack-*.whl"):
        stale.unlink()
    r = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "-w",
            str(HERE),
            str(checkout),
        ],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        sys.exit(f"wheel build failed:\n{r.stdout[-2000:]}{r.stderr[-2000:]}")
    built = sorted(HERE.glob("schnetpack-*.whl"))
    if len(built) != 1:
        sys.exit(f"expected exactly one wheel, got {[p.name for p in built]}")
    print(f"built {built[0].name} ({built[0].stat().st_size // 1024} KB)")
    return built[0].name


def setup_cell(repo: str, pin: str, wheel: str = None) -> dict:
    # Absolute paths throughout, so re-running the cell — which students do,
    # usually after the runtime has recycled — never nests a checkout inside a
    # checkout. An existing directory is *refreshed*, not skipped: the notebook
    # is fetched from GitHub every time it is opened, the checkout beside it is
    # not, so a runtime warm from before a push would otherwise run new cells
    # against old helpers.py — an ImportError deep in §7, far from its cause.
    # rdkit rides along: Colab images don't ship it, and §7's validity check needs it
    install = (
        f"!pip install -q rdkit /content/tutorial/{wheel}"
        if wheel
        else '!pip install -q rdkit "git+https://github.com/'
        f'atomistic-machine-learning/schnetpack.git@{pin}"'
    )
    # Every build carries the same version string, so pip counts an updated
    # wheel as already satisfied and keeps the stale one. Force this one file.
    refresh = (
        f"\n!pip install -q --force-reinstall --no-deps /content/tutorial/{wheel}"
        if wheel
        else ""
    )
    src = f"""\
# Fetch SchNetPack and the tutorial files into this runtime. Run me first.
# Safe to re-run: it refreshes what is already here rather than duplicating it.
import os

if os.path.isdir("/content/tutorial"):
    !git -C /content/tutorial fetch -q --depth 1 origin HEAD
    !git -C /content/tutorial reset -q --hard FETCH_HEAD
else:
    !git clone -q --depth 1 {repo} /content/tutorial
%cd /content/tutorial
{install}{refresh}

import schnetpack
print("schnetpack", schnetpack.__version__, "ready")\
"""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src.splitlines(True),
    }


def build(repo: str, pin: str, out: Path, schnetpack: Path = None) -> None:
    wheel = build_wheel(schnetpack) if schnetpack else None
    export = subprocess.run(
        [
            sys.executable,
            "-m",
            "marimo",
            "export",
            "ipynb",
            str(HERE / "notebook.py"),
            "--sort",
            "top-down",
        ],
        capture_output=True,
        text=True,
    )
    if export.returncode != 0:
        sys.exit(f"marimo export failed:\n{export.stderr}")
    nb = json.loads(export.stdout)

    # 1. the marimo import is dead weight once no code cell uses `mo`
    cells = nb["cells"]
    for cell in cells:
        if cell["cell_type"] == "code" and "mo." in "".join(cell["source"]):
            sys.exit(
                "a code cell still calls the marimo API — it would fail on "
                "Colab; route it through viz.py's frontend seam first"
            )
    cells = [c for c in cells if "".join(c["source"]).strip() != "import marimo as mo"]

    # 2. swap the local install instructions for the hosted ones
    hits = []
    for i, cell in enumerate(cells):
        if cell["cell_type"] != "markdown":
            continue
        text = "".join(cell["source"])
        text, n = LOCAL_SETUP.subn(COLAB_SETUP, text)
        if n:
            hits += [i] * n
            cell["source"] = text.replace(
                "notebook.py            ← this tutorial",
                "notebook.ipynb          ← this tutorial",
            ).splitlines(True)
    if len(hits) != 1:
        sys.exit(
            f"expected exactly one local-setup block to rewrite, found {len(hits)}"
        )

    # 3. the setup cell goes directly under the prose that introduces it, so the
    #    notebook still opens on its title and "the cell below" stays true
    at = hits[0] + 1
    nb["cells"] = cells[:at] + [setup_cell(repo, pin, wheel)] + cells[at:]
    # Colab reads these when opening the notebook and starts a GPU runtime.
    # It is only a request: if none is granted the notebook's own
    # cuda.is_available() check falls back to CPU, which runs the tutorial fine.
    nb["metadata"]["colab"] = {"provenance": [], "toc_visible": True, "gpuType": "T4"}
    nb["metadata"]["accelerator"] = "GPU"

    out.write_text(json.dumps(nb, indent=1))
    n_code = sum(c["cell_type"] == "code" for c in nb["cells"])
    source = wheel if wheel else f"git @ {pin}"
    print(
        f"{out.name}: {len(nb['cells'])} cells ({n_code} code), schnetpack from {source}"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--repo", required=True, help="public git URL of the tutorial bundle"
    )
    p.add_argument(
        "--schnetpack",
        type=Path,
        help="path to a SchNetPack checkout; builds a wheel here and installs that",
    )
    p.add_argument(
        "--pin",
        default="sh/v3",
        help="schnetpack git ref, used only without --schnetpack (prefer a tag)",
    )
    p.add_argument("--out", type=Path, default=HERE / "notebook.ipynb")
    a = p.parse_args()
    build(a.repo, a.pin, a.out, a.schnetpack)
