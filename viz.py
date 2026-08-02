"""
Live 3D visualization for the tutorial — the one piece that is *not* library code.

Three viewers, plus one way to tuck them away:

``show_batch(batch)``
    A grid of interactive 3Dmol.js viewers, one box per molecule of a batch.
    Drag to rotate. This is "here is what the dataloader handed us".

``show_frames(trajectory, batch, n_frames=5, ...)``
    The same trajectory as a *filmstrip*: ``n_frames`` stills, evenly spaced
    from the first frame to the last, side by side and going nowhere. Takes the
    same overlays as ``show_trajectory``; each still carries them for its own
    frame, so the trace grows across the strip. For print, and for a figure that
    does not need a reader to drag anything.

``show_trajectory(trajectory, batch, ...)``
    One row per trajectory, scrubbed by a single shared slider so a whole batch
    moves as one motion. A row is up to three panels — the first frame, the
    animated frame, the last frame — and *which* panels exist follows the
    arguments: ``start=True`` and ``end=True`` add the statics, both omitted
    leaves a single wide box. The panels' cameras are linked, so dragging or
    zooming any one of them moves the whole row.

    On top of the animated panel, each overlay is switched on by handing over
    the data (or index) it needs, never by a mode flag:

    ``ghost_id=0``     a translucent copy of that frame, to see how far x_t has
                       drifted from where it started (or ``-1``: from where it
                       ended up).
    ``trace=True``     each atom's path so far as a dashed, colour-per-atom
                       polyline from frame 0 to the current frame.
    ``vectors=[...]``  one arrow per atom per frame — a training target, a
                       predicted force. In a scrubbable view they are suppressed
                       on exactly those frames that are *also* shown as a bare
                       static panel, so the statics and the animation agree at
                       the endpoints; a filmstrip draws them on every still.

Both take batch-level data: positions are ``(n_atoms_total, 3)`` over the whole
batch and are split per molecule with ``idx_m`` internally.

``details(summary, *parts)``
    A collapsed block holding any of the above plus matplotlib figures — the
    reference solution a reader should attempt before opening.

The viewer library (``assets/3Dmol-min.js``) is inlined into the page, so the
result is a single self-contained HTML string with no network access: it works
offline, and it is the *same* string whichever notebook is running. Only the
last step differs, and only in one place — see "the frontend seam" below, which
is what lets this notebook run under both marimo and Jupyter/Colab.
"""

import base64
import io
import json
import os
import re
from html import escape as _escape
from typing import List, Optional, Sequence, Union

import numpy as np

from schnetpack import properties

_ASSET = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "3Dmol-min.js")


def _load_3dmol() -> str:
    with open(_ASSET, "r", encoding="utf-8", errors="ignore") as fh:
        return fh.read()


def _numpy(x) -> np.ndarray:
    return np.asarray(x.detach().cpu() if hasattr(x, "detach") else x, dtype=float)


def _frame_to_xyz(symbols: Sequence[str], coords: np.ndarray, comment: str = "") -> str:
    lines = [str(len(symbols)), comment]
    for s, (x, y, z) in zip(symbols, coords):
        lines.append(f"{s} {x:.4f} {y:.4f} {z:.4f}")
    return "\n".join(lines)


def _hill_formula(symbols: Sequence[str]) -> str:
    """Chemical formula in Hill order (C, H, then others alphabetically)."""
    from collections import Counter

    counts = Counter(symbols)

    def term(el):
        n = counts.pop(el, 0)
        if n == 0:
            return ""
        return el if n == 1 else f"{el}{n}"

    out = term("C") + term("H")
    for el in sorted(counts):
        out += term(el)
    return out or "".join(symbols)


def _molecules(batch: dict) -> List[dict]:
    """Per-molecule ``{"mask", "symbols"}`` of a batch, in batch order."""
    from ase import Atoms

    idx_m = _numpy(batch[properties.idx_m]).astype(int)
    numbers = _numpy(batch[properties.Z]).astype(int)
    n_mol = int(_numpy(batch[properties.n_atoms]).shape[0])
    out = []
    for m in range(n_mol):
        mask = idx_m == m
        out.append(
            {"mask": mask, "symbols": Atoms(numbers=numbers[mask]).get_chemical_symbols()}
        )
    return out


def _unpack(trajectory, times):
    """``(frames, times)`` from a frame list, a tensor, or a recorder object."""
    if hasattr(trajectory, "frames"):  # anything that recorded its own frames
        if times is None:
            times = list(trajectory.times)
        trajectory = trajectory.frames
    frames = [_numpy(f) for f in trajectory]
    if times is None:
        times = list(range(len(frames)))
    return frames, [float(t) for t in times]


def _keep_indices(n: int, stride: int, pin: Sequence[int] = ()) -> List[int]:
    """Strided frame indices, always including the last frame and ``pin``."""
    keep = set(range(0, n, max(stride, 1))) | {n - 1} | {i for i in pin}
    return sorted(keep)


def _payload(
    trajectory,
    batch: dict,
    times=None,
    vectors=None,
    stride: int = 1,
    titles: Optional[Sequence[str]] = None,
    ghost_id: Optional[int] = None,
) -> List[dict]:
    """One entry per (run × molecule) — the JSON the page is built from.

    ``trajectory`` is a list of batch-level frames, or a ``dict`` of them, in
    which case each key becomes a row of its own (``vectors`` may be a dict with
    the same keys). Rows are ordered run-major.
    """
    runs = trajectory if isinstance(trajectory, dict) else {None: trajectory}
    vecs = vectors if isinstance(vectors, dict) else {None: vectors}
    mols = _molecules(batch)

    entries = []
    for name, traj in runs.items():
        run_times = times.get(name) if isinstance(times, dict) else times
        frames, ts = _unpack(traj, run_times)
        run_vectors = vecs.get(name) if isinstance(vectors, dict) else vectors
        gid = None if ghost_id is None else range(len(frames))[ghost_id]
        keep = _keep_indices(len(frames), stride, pin=() if gid is None else (gid,))
        for m, mol in enumerate(mols):
            label = name if name is not None else _hill_formula(mol["symbols"])
            if name is not None and len(mols) > 1:
                label = f"{name} #{m}"
            entry = {
                "label": label,
                "frames": [
                    _frame_to_xyz(mol["symbols"], frames[i][mol["mask"]], f"t={ts[i]:.3f}")
                    for i in keep
                ],
                "times": [ts[i] for i in keep],
                "ghost": None if gid is None else keep.index(gid),
            }
            if run_vectors is not None:
                entry["vectors"] = [
                    np.round(_numpy(run_vectors[i])[mol["mask"]], 4).tolist() for i in keep
                ]
            entries.append(entry)

    if titles is not None:
        for entry, title in zip(entries, titles):
            entry["label"] = title
    return entries


_PAGE_TEMPLATE = """\
<div id="wrap" style="font-family:-apple-system,Segoe UI,Roboto,sans-serif;color:#222;">
  <div style="display:{controls_display};align-items:center;gap:14px;margin:6px 4px 12px;flex-wrap:wrap;">
    <button id="play" style="padding:5px 14px;border:1px solid #bbb;border-radius:6px;background:#f5f5f5;cursor:pointer;font-size:13px;">&#9654; play</button>
    <input id="slider" type="range" min="0" max="{max_frame}" value="{init_frame}" style="flex:1;min-width:220px;">
    <span id="caption" style="font-variant-numeric:tabular-nums;font-size:13px;color:#444;min-width:150px;">step 0</span>
  </div>
  <div id="grid" style="display:flex;flex-wrap:wrap;gap:10px;"></div>
  <div id="axhint" style="font-size:12px;color:#888;margin:8px 4px 2px;">
    view along axis: press <b>x</b>/<b>1</b> · <b>y</b>/<b>2</b> · <b>z</b>/<b>3</b>
    &nbsp;(click a panel first to give it focus)
  </div>
</div>
<script>{threedmol}</script>
<script>
const $3Dmol = window.$3Dmol || window["3Dmol"];
const DATA = {data_json};
const N_FRAMES = {n_frames};
const CELL = {cell_px};
const SHOW_INDEX = {show_index};
// one entry per column: {{kind:"start"|"end"|"moving"}} or {{kind:"frame", i}}.
// "start"/"end" are the bare endpoint statics of a scrubbable view; "frame" is
// a still of one frame *with* the overlays, which is what a filmstrip is made of
const PANELS = {panels_json};
const PANEL_LABELS = {panel_labels_json};
const TRACE = {trace};
const GHOST_ALPHA = {ghost_alpha};
const VSCALE = {arrow_scale};
const ZOOM = {zoom};
const FRAME_MS = {frame_ms};
const SKIP_ARROWS_FIRST = {skip_arrows_first};
const SKIP_ARROWS_LAST = {skip_arrows_last};
const moving = [];                   // the panels the slider drives
const allViewers = [];               // every panel, statics included — axis keys

function xyzOf(mol, i) {{ return mol.frames[Math.min(i, mol.frames.length - 1)]; }}
function lastOf(mol) {{ return mol.frames.length - 1; }}
function panelIndex(panel, mol) {{
  if (panel.kind === "start") return 0;
  if (panel.kind === "end") return lastOf(mol);
  return Math.min(panel.i, lastOf(mol));
}}

// pull (x,y,z) rows out of one xyz frame string
function parseXYZ(s) {{
  const lines = s.split("\\n");
  const n = parseInt(lines[0]);
  const out = [];
  for (let i = 0; i < n; i++) {{
    const p = lines[2 + i].trim().split(/\\s+/);
    out.push({{x:parseFloat(p[1]), y:parseFloat(p[2]), z:parseFloat(p[3])}});
  }}
  return out;
}}

// a distinct, saturated colour per atom so individual paths stay followable
function atomColor(k, n) {{
  const h = (360 * k / n) / 360, s = 0.65, l = 0.5;
  const q = l < 0.5 ? l * (1 + s) : l + s - l * s, p = 2 * l - q;
  const f = (t) => {{
    if (t < 0) t += 1; if (t > 1) t -= 1;
    if (t < 1/6) return p + (q - p) * 6 * t;
    if (t < 1/2) return q;
    if (t < 2/3) return p + (q - p) * (2/3 - t) * 6;
    return p;
  }};
  const to = (v) => Math.round(v * 255).toString(16).padStart(2, "0");
  return "#" + to(f(h + 1/3)) + to(f(h)) + to(f(h - 1/3));
}}

// 3Dmol's addLine draws a plain segment — it has no dash option — so a dotted
// path is drawn as short segments with gaps: DASH long, DASH/2 apart, at most
// MAX_DASH per frame-to-frame step (long steps early in a trajectory would
// otherwise flood the scene with geometry).
const DASH = 0.18, MAX_DASH = 6;
function dashedLine(v, a, b, color) {{
  const dx = b.x - a.x, dy = b.y - a.y, dz = b.z - a.z;
  const len = Math.sqrt(dx*dx + dy*dy + dz*dz);
  if (len < 1e-6) return;
  const n = Math.max(1, Math.min(MAX_DASH, Math.round(len / (1.5 * DASH))));
  const step = 1 / n, duty = 2 / 3;   // dash occupies 2/3 of each step
  for (let k = 0; k < n; k++) {{
    const s = k * step, e = s + duty * step;
    v.addLine({{
      start: {{x:a.x + s*dx, y:a.y + s*dy, z:a.z + s*dz}},
      end:   {{x:a.x + e*dx, y:a.y + e*dy, z:a.z + e*dz}},
      color: color,
    }});
  }}
}}

// One panel, one frame. `opts.overlays` is false for the static endpoint
// panels: they show the bare structure, so the animation lines up with them.
function drawFrame(entry, i, opts) {{
  const v = entry.viewer, mol = entry.mol;
  const overlays = !(opts && opts.overlays === false);
  v.removeAllModels();
  v.removeAllShapes();
  v.removeAllLabels();

  if (overlays && mol.ghost !== null && mol.ghost !== i) {{
    // translucent reference structure: how far has x_t drifted from it?
    const g = v.addModel(xyzOf(mol, mol.ghost), "xyz");
    g.setStyle({{}}, {{
      stick:{{radius:0.10, opacity:GHOST_ALPHA}},
      sphere:{{scale:0.24, opacity:GHOST_ALPHA}},
    }});
  }}

  if (overlays && TRACE && i > 0) {{
    // each atom's path so far, dotted and coloured per atom
    const coords = [];
    for (let f = 0; f <= i; f++) coords.push(parseXYZ(xyzOf(mol, f)));
    const na = coords[0].length;
    for (let a = 0; a < na; a++) {{
      const col = atomColor(a, na);
      for (let f = 0; f < i; f++) {{
        dashedLine(v, coords[f][a], coords[f + 1][a], col);
      }}
    }}
  }}

  const model = v.addModel(xyzOf(mol, i), "xyz");
  model.setStyle({{}}, {{stick:{{radius:0.12}}, sphere:{{scale:0.28}}}});

  // arrows are suppressed at an endpoint only when that endpoint is also shown
  // bare somewhere in the row (a static panel), so the panels cannot disagree
  // about the same frame. A filmstrip has no such panel: every still keeps its
  // arrows.
  const atEndpoint =
    (i === 0 && SKIP_ARROWS_FIRST) || (i === lastOf(mol) && SKIP_ARROWS_LAST);
  if (overlays && mol.vectors && !atEndpoint) {{
    const pos = parseXYZ(xyzOf(mol, i));
    const vecs = mol.vectors[Math.min(i, mol.vectors.length - 1)];
    vecs.forEach((w, k) => {{
      const p = pos[k];
      v.addArrow({{
        start: {{x:p.x, y:p.y, z:p.z}},
        end: {{x:p.x + VSCALE*w[0], y:p.y + VSCALE*w[1], z:p.z + VSCALE*w[2]}},
        radius: 0.06, radiusRatio: 2.5, mid: 0.8, color: "#d62828",
      }});
    }});
  }}

  if (SHOW_INDEX) {{
    // label each atom with its row index, so atom ordering is visible
    model.selectedAtoms({{}}).forEach((a, k) => {{
      v.addLabel(String(k), {{
        position: {{x:a.x, y:a.y, z:a.z}},
        fontSize: 10, fontColor: "black", inFront: true,
        showBackground: true, backgroundColor: "white", backgroundOpacity: 0.55,
        borderThickness: 0.0,
      }});
    }});
  }}
  v.render();
}}

// size the camera on the last frame, then never zoomTo again — so nothing
// jumps while scrubbing. Pass a small `zoom` when the far end of a trajectory
// (a large-sigma noise cloud) has to stay in frame.
function initCamera(v, mol) {{
  v.addModel(xyzOf(mol, lastOf(mol)), "xyz");
  v.setStyle({{}}, {{sphere:{{scale:0.28}}, stick:{{radius:0.12}}}});
  v.zoomTo();
  v.zoom(ZOOM);
}}

// A row of linked panels only earns its width when there is more than one of
// them. One panel per molecule — a batch of structures, or a batch being
// denoised — reads better as a wrapping grid of independent boxes.
const SINGLE = PANELS.length <= 1;
const grid = document.getElementById("grid");
if (!SINGLE) {{ grid.style.flexDirection = "column"; grid.style.alignItems = "flex-start"; }}

DATA.forEach((mol) => {{
  const cell = document.createElement("div");
  cell.style.cssText = "border:1px solid #e2e2e2;border-radius:8px;padding:4px;background:#fbfbfb;";
  const title = document.createElement("div");
  title.style.cssText = "font-size:12px;color:#555;text-align:center;padding:2px 0 4px;";
  title.textContent = mol.label;
  cell.appendChild(title);
  grid.appendChild(cell);

  if (SINGLE) {{
    // one box per molecule, in a wrapping grid. Animated if the panel is the
    // moving one (the slider then drives every box at once), otherwise a still
    const holder = document.createElement("div");
    holder.style.cssText = `width:${{CELL}}px;height:${{CELL}}px;position:relative;`;
    cell.appendChild(holder);
    const v = $3Dmol.createViewer(holder, {{backgroundColor:"white"}});
    const entry = {{viewer:v, mol:mol}};
    allViewers.push(entry);
    initCamera(v, mol);
    const panel = PANELS[0];
    if (panel && panel.kind === "moving") moving.push(entry);   /* show() draws it */
    else if (panel) drawFrame(entry, panelIndex(panel, mol), {{overlays: panel.kind === "frame"}});
    else drawFrame(entry, 0);
    return;
  }}

  // one row per trajectory: the requested panels side by side in ONE canvas
  // (one WebGL context per row) with control_all, so the cameras stay linked
  const holder = document.createElement("div");
  holder.style.cssText = `width:${{PANELS.length * CELL}}px;height:${{CELL}}px;position:relative;`;
  cell.appendChild(holder);
  const caps = document.createElement("div");
  caps.style.cssText = "display:flex;";
  PANELS.forEach((panel, j) => {{
    const cap = document.createElement("div");
    cap.style.cssText = `width:${{CELL}}px;font-size:11px;color:#888;text-align:center;padding-top:2px;`;
    cap.textContent = PANEL_LABELS[j];
    caps.appendChild(cap);
  }});
  cell.appendChild(caps);

  const row = $3Dmol.createViewerGrid(
    holder, {{rows: 1, cols: PANELS.length, control_all: true}}, {{backgroundColor: "white"}}
  )[0];
  row.forEach(v => {{ initCamera(v, mol); allViewers.push({{viewer: v, mol: mol}}); }});

  PANELS.forEach((panel, j) => {{
    const entry = {{viewer: row[j], mol: mol}};
    if (panel.kind === "moving") {{ moving.push(entry); return; }}
    // a still: bare for the endpoint statics of a scrubbable view, with the
    // overlays for a filmstrip's frames
    drawFrame(entry, panelIndex(panel, mol), {{overlays: panel.kind === "frame"}});
  }});
}});

const slider = document.getElementById("slider");
const caption = document.getElementById("caption");
function show(i) {{
  moving.forEach(e => drawFrame(e, i));
  const t = DATA[0].times[Math.min(i, DATA[0].times.length - 1)];
  caption.textContent = `step ${{i}} / ${{N_FRAMES - 1}}   —   t = ${{t.toFixed(3)}}`;
}}
slider.addEventListener("input", e => show(parseInt(e.target.value)));

let timer = null;
const play = document.getElementById("play");
play.addEventListener("click", () => {{
  if (timer) {{ clearInterval(timer); timer = null; play.innerHTML = "&#9654; play"; return; }}
  play.innerHTML = "&#10073;&#10073; pause";
  timer = setInterval(() => {{
    let i = (parseInt(slider.value) + 1) % N_FRAMES;
    slider.value = i; show(i);
  }}, FRAME_MS);
}});

// ASE-style "look along axis": align the camera down x / y / z, keeping the
// current centre and zoom, for every panel at once. Quaternions rotate the
// chosen world axis onto the view axis (z = the default front view).
const AXIS_QUAT = {{
  x: [0, -0.7071067811865476, 0, 0.7071067811865476],
  y: [0.7071067811865476, 0, 0, 0.7071067811865476],
  z: [0, 0, 0, 1],
}};
function lookAlong(axis) {{
  const q = AXIS_QUAT[axis];
  allViewers.forEach(e => {{
    const view = e.viewer.getView();  // [cx, cy, cz, zoom, qx, qy, qz, qw]
    e.viewer.setView([view[0], view[1], view[2], view[3], q[0], q[1], q[2], q[3]]);
    e.viewer.render();
  }});
}}
const AXIS_KEY = {{x:"x", "1":"x", y:"y", "2":"y", z:"z", "3":"z"}};
document.addEventListener("keydown", (ev) => {{
  const axis = AXIS_KEY[ev.key.toLowerCase()];
  if (axis) {{ lookAlong(axis); ev.preventDefault(); }}
}});

if (moving.length) show({init_frame});
</script>
"""


def _block_comments(template: str) -> str:
    """Rewrite the page's ``//`` comments as ``/* */``.

    ``mo.iframe`` collapses its argument onto a single line before escaping it
    into the ``srcdoc`` attribute. A ``//`` comment then runs to the end of *the
    whole script* and silently swallows everything after it — the page renders
    blank. Block comments survive the collapse, so the source keeps ordinary
    line comments and they are converted here, once, at import time.
    """
    return re.sub(
        r"//(.*)$",
        lambda m: "/*" + m.group(1).replace("*/", "* /") + "*/",
        template,
        flags=re.M,
    )


_PAGE_TEMPLATE = _block_comments(_PAGE_TEMPLATE)


def _page(
    entries: List[dict],
    panels: Sequence[dict],
    panel_labels: Sequence[str],
    init_frame: int = 0,
    trace: bool = False,
    ghost_alpha: float = 0.35,
    arrow_scale: float = 1.0,
    zoom: float = 1.7,
    cell_px: int = 230,
    frame_ms: int = 350,
    atom_index: bool = False,
    skip_arrows_first: bool = False,
    skip_arrows_last: bool = False,
) -> str:
    n_frames = max((len(e["frames"]) for e in entries), default=1)
    init = range(n_frames)[init_frame] if n_frames else 0
    has_moving = any(p["kind"] == "moving" for p in panels)
    return _PAGE_TEMPLATE.format(
        threedmol=_load_3dmol(),
        data_json=json.dumps(entries),
        n_frames=n_frames,
        max_frame=max(n_frames - 1, 0),
        init_frame=init,
        cell_px=cell_px,
        controls_display="flex" if has_moving and n_frames > 1 else "none",
        show_index="true" if atom_index else "false",
        panels_json=json.dumps(list(panels)),
        panel_labels_json=json.dumps(list(panel_labels)),
        trace="true" if trace else "false",
        ghost_alpha=ghost_alpha,
        arrow_scale=arrow_scale,
        zoom=zoom,
        frame_ms=frame_ms,
        skip_arrows_first="true" if skip_arrows_first else "false",
        skip_arrows_last="true" if skip_arrows_last else "false",
    )


def batch_html(
    batch: dict,
    titles: Optional[Sequence[str]] = None,
    atom_index: bool = False,
    cell_px: int = 230,
    zoom: float = 1.7,
) -> str:
    """The self-contained page for a batch of structures (one box per molecule).

    Args:
        batch: batch dict — ``R``, ``Z``, ``idx_m``, ``n_atoms``.
        titles: caption per box; defaults to each molecule's Hill formula.
        atom_index: write each atom's row index next to it, so the ordering of
            atoms (and what a permutation coupling does to it) is visible.
        cell_px: pixel size of each molecule's viewer box.
        zoom: camera zoom factor applied after framing the structure.
    """
    entries = _payload([batch[properties.R]], batch, times=[0.0], titles=titles)
    # no panels at all: a wrapping grid of independent boxes, not rows
    return _page(entries, panels=(), panel_labels=(), zoom=zoom, cell_px=cell_px,
                 atom_index=atom_index)


def trajectory_html(
    trajectory,
    batch: dict,
    times=None,
    start: bool = False,
    end: bool = False,
    ghost_id: Optional[int] = None,
    ghost_alpha: float = 0.35,
    trace: bool = False,
    vectors=None,
    arrow_scale: float = 1.0,
    stride: int = 1,
    init_frame: int = 0,
    panel_labels: Sequence[str] = ("x₀", "xₜ", "x₁"),
    titles: Optional[Sequence[str]] = None,
    zoom: float = 1.7,
    cell_px: int = 230,
    frame_ms: int = 350,
    atom_index: bool = False,
) -> str:
    """The self-contained page for one or more batch-level trajectories.

    Args:
        trajectory: the frames, each ``(n_atoms_total, 3)`` over the whole
            batch — a list, a stacked tensor, or an object carrying ``frames``
            (and optionally ``times``). A ``dict`` of any of these gives one row
            per entry, titled by its key.
        batch: batch dict supplying the topology (``Z``, ``idx_m``,
            ``n_atoms``) the frames are laid out in.
        times: one time per frame, for the caption; a ``dict`` keyed like
            ``trajectory`` when the runs live on different grids. Defaults to
            frame indices.
        start: show ``frames[0]`` as a static panel left of the animation.
        end: show ``frames[-1]`` as a static panel right of the animation.
        ghost_id: index of the frame drawn as a translucent overlay on the
            animated panel (``0``: where the trajectory started, ``-1``: where
            it ended). ``None`` draws no ghost.
        ghost_alpha: opacity of that overlay.
        trace: draw each atom's path from frame 0 up to the current frame as a
            dashed, colour-per-atom polyline. Tangled paths reveal a costly
            coupling; radial ones an aligned pairing.
        vectors: one ``(n_atoms_total, 3)`` field per frame, drawn as an arrow
            on every atom (a training target, a predicted force). A ``dict``
            when ``trajectory`` is one, keyed the same. Arrows are suppressed on
            frames that are also shown as a static panel.
        arrow_scale: length multiplier for those arrows.
        stride: keep only every n-th frame (the last one is always kept).
        init_frame: slider position on load; ``-1`` for the last frame.
        panel_labels: captions for the start / animated / end panels; only the
            panels that exist are captioned.
        titles: row captions, overriding the dict keys or Hill formulas.
        zoom: camera zoom factor, applied after framing ``frames[-1]``. Values
            below 1 pull back — needed when the far end of the trajectory is a
            wide noise cloud that should stay in frame.
        cell_px: pixel size of one panel.
        frame_ms: milliseconds per frame when playing (bigger = slower).
        atom_index: write each atom's row index next to it.
    """
    entries = _payload(
        trajectory,
        batch,
        times=times,
        vectors=vectors,
        stride=stride,
        titles=titles,
        ghost_id=ghost_id,
    )
    panels, labels = [], []
    if start:
        panels.append({"kind": "start"})
        labels.append(panel_labels[0])
    panels.append({"kind": "moving"})
    labels.append(panel_labels[1])
    if end:
        panels.append({"kind": "end"})
        labels.append(panel_labels[2])
    return _page(
        entries,
        panels=panels,
        panel_labels=labels,
        init_frame=init_frame,
        trace=trace,
        ghost_alpha=ghost_alpha,
        arrow_scale=arrow_scale,
        zoom=zoom,
        cell_px=cell_px,
        frame_ms=frame_ms,
        atom_index=atom_index,
        skip_arrows_first=start,
        skip_arrows_last=end,
    )


_CELL_WIDTH_PX = 720
"""Usable width of a marimo cell at ``width="medium"``.

Both the filmstrip's panel size and the wrapping grid's height estimate are
derived from it, so a page never grows an inner scrollbar it was not sized for.
"""


def _film_cell_px(n_frames: int) -> int:
    """Panel size that keeps a strip of ``n_frames`` stills inside a notebook cell.

    A filmstrip is one fixed-width row — too wide and it grows an inner
    scrollbar instead of being readable at a glance.
    """
    return min(180, max(90, _CELL_WIDTH_PX // max(int(n_frames), 1)))


def frames_html(
    trajectory,
    batch: dict,
    n_frames: int = 5,
    times=None,
    ghost_id: Optional[int] = None,
    ghost_alpha: float = 0.35,
    trace: bool = False,
    vectors=None,
    arrow_scale: float = 1.0,
    panel_labels: Union[str, Sequence[str], None] = None,
    titles: Optional[Sequence[str]] = None,
    zoom: float = 1.7,
    cell_px: Optional[int] = None,
    atom_index: bool = False,
) -> str:
    """The self-contained page for a trajectory as a row of *stills*.

    Same trajectory and same overlays as :func:`trajectory_html`, but instead of
    one animated panel there are ``n_frames`` static ones, evenly spaced from the
    first frame to the last (both always included) — a filmstrip. Each still
    carries the overlays for *its* frame: the trace grows from panel to panel and
    the ghost sits behind every one of them. Cameras are linked across the row,
    so rotating one still rotates the strip.

    Args:
        trajectory: as in :func:`trajectory_html` (frames, an object carrying
            ``frames``, or a ``dict`` of either for one row each).
        batch: batch dict supplying the topology.
        n_frames: number of stills, including the first and last frame. ``5``
            gives first + 3 intermediates + last.
        times: as in :func:`trajectory_html`; used for the column captions.
        ghost_id: frame drawn as a translucent overlay on every still.
        ghost_alpha: opacity of that overlay.
        trace: draw each atom's path from frame 0 up to that still's frame.
        vectors: per-frame arrow field, drawn on every still including the first
            and the last. (In a scrubbable view the endpoints *are* suppressed,
            but only because a static panel shows the same frame bare there;
            a filmstrip has no such panel to disagree with.)
        arrow_scale: length multiplier for those arrows.
        panel_labels: column captions. A format string over the still's frame
            index and time (``"t = {t:.2f}"``, ``"step {i}"``), a callable
            ``(i, t) -> str``, or one caption per still. Defaults to the time
            when ``times`` are known, else the frame index.
        titles: row captions, overriding the dict keys or Hill formulas.
        zoom: camera zoom factor, applied after framing the last frame.
        cell_px: pixel size of one still. Defaults to whatever keeps the whole
            strip inside a notebook cell (see :func:`_film_cell_px`) — a strip
            of 5 is narrower per panel than a strip of 3.
        atom_index: write each atom's row index next to it.

    Note:
        No frames are dropped and no ``stride`` is taken — the stills are picked
        from the full trajectory, so which frames you see does not depend on how
        densely it was recorded.
    """
    if cell_px is None:
        cell_px = _film_cell_px(n_frames)
    entries = _payload(
        trajectory, batch, times=times, vectors=vectors, titles=titles, ghost_id=ghost_id
    )
    n_total = max((len(e["frames"]) for e in entries), default=1)
    n_show = max(2, min(int(n_frames), n_total)) if n_total > 1 else 1
    if n_show == 1:
        picks = [0]
    else:
        picks = [round(k * (n_total - 1) / (n_show - 1)) for k in range(n_show)]
    panels = [{"kind": "frame", "i": i} for i in picks]

    ts = entries[0]["times"] if entries else []
    if panel_labels is None:
        auto = times is not None or hasattr(trajectory, "frames")
        if isinstance(trajectory, dict):
            auto = auto or any(hasattr(t, "frames") for t in trajectory.values())
        panel_labels = "t = {t:.2f}" if auto else "frame {i}"
    if isinstance(panel_labels, str):
        # a template over the still's frame index and time, e.g. "step {i}"
        panel_labels = [
            panel_labels.format(i=i, t=ts[i] if i < len(ts) else float("nan"))
            for i in picks
        ]
    elif callable(panel_labels):
        panel_labels = [panel_labels(i, ts[i] if i < len(ts) else None) for i in picks]

    return _page(
        entries,
        panels=panels,
        panel_labels=list(panel_labels),
        trace=trace,
        ghost_alpha=ghost_alpha,
        arrow_scale=arrow_scale,
        zoom=zoom,
        cell_px=cell_px,
        atom_index=atom_index,
    )


# --- the frontend seam --------------------------------------------------- #
# Everything above builds a self-contained page; everything below puts one on
# screen. Two frontends matter — marimo, and IPython/Jupyter/Colab for the
# exported notebook — and both get the page sealed into an *iframe*.
#
# That is not decoration. A page names its controls with fixed ids (``slider``,
# ``grid``, ``play``), and its script declares them at top level, so two pages
# sharing one document collide: the second script redeclares the first's
# ``const`` and throws before drawing anything, leaving a live slider attached
# to an empty grid. marimo isolates cells for us; a Jupyter document does not,
# and every output lands in the same DOM. The iframe is what keeps them apart.


def _in_marimo() -> bool:
    """Are we rendering inside a marimo notebook (not merely importable)?"""
    try:
        import marimo as mo
    except ImportError:
        return False
    return bool(mo.running_in_notebook())


def _iframe(html: str, height: int) -> str:
    """A page sealed into a document of its own.

    Wrapped in a div because ``IPython.display.HTML`` warns about content that
    *starts* with an iframe, and the advice it gives (use ``IFrame``) does not
    apply — that class takes a URL, and there is no URL here.
    """
    return (
        f'<div><iframe srcdoc="{_escape(html)}" width="100%" height="{height}" '
        'style="border:none;"></iframe></div>'
    )


def _guess_height(n_rows: int, n_panels: int, cell_px: int) -> int:
    if n_panels > 1:  # one full-width row of linked panels per trajectory
        return 100 + n_rows * (cell_px + 52)
    per_row = max(1, _CELL_WIDTH_PX // (cell_px + 12))  # wrapping grid of boxes
    rows = (n_rows + per_row - 1) // per_row
    return 90 + rows * (cell_px + 32)


def _embed(html: str, n_rows: int, n_panels: int, cell_px: int, height: Optional[int]):
    """Put a page on screen, guessing a height if none was given."""
    if height is None:
        height = _guess_height(n_rows, n_panels, cell_px)
    if _in_marimo():
        import marimo as mo

        return mo.iframe(html, height=f"{height}px")
    try:
        from IPython.display import HTML
    except ImportError:
        return html  # a plain script: hand back the page itself
    return HTML(_iframe(html, height))


def figure_html(fig, dpi: int = 110) -> str:
    """A matplotlib figure as a self-contained ``<img>``.

    Composable with the pages above, which is what lets :func:`details` hold a
    plot and a viewer in one block. Closes the figure, so an inline backend
    does not also draw it where it was made.
    """
    import matplotlib.pyplot as plt

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    data = base64.b64encode(buf.getvalue()).decode()
    return f'<img src="data:image/png;base64,{data}" style="max-width:100%;">'


def image_html(path: str, width: Optional[int] = None) -> str:
    """A PNG on disk as a self-contained ``<img>``, like :func:`figure_html`.

    Inlined rather than linked, so a notebook exported with its outputs carries
    the picture with it.
    """
    with open(path, "rb") as fh:
        data = base64.b64encode(fh.read()).decode()
    size = f"width:{int(width)}px;" if width else ""
    return f'<img src="data:image/png;base64,{data}" style="max-width:100%;{size}">'


def show_image(path: str, width: Optional[int] = None):
    """A PNG on disk, on screen — the still counterpart to the viewers above."""
    html = image_html(path, width)
    if _in_marimo():
        import marimo as mo

        return mo.Html(html)
    try:
        from IPython.display import HTML
    except ImportError:
        return html
    return HTML(html)


def _as_html(part, height: int) -> str:
    if hasattr(part, "savefig"):  # a matplotlib figure
        return figure_html(part)
    if hasattr(part, "data"):  # what _embed returns here: IPython's HTML
        return part.data
    return _iframe(str(part), height)  # a bare page, sealed like any other


def details(summary: str, *parts, height: int = 460):
    """A collapsed block — for a reference solution the reader should try first.

    Each part is whatever this frontend already renders: a matplotlib figure,
    or a viewer from :func:`show_batch` / :func:`show_trajectory` /
    :func:`show_frames`. ``height`` sizes bare page strings only; viewers
    arrive already sized.
    """
    if _in_marimo():
        import marimo as mo

        return mo.accordion({summary: mo.vstack(list(parts))})

    body = "".join(_as_html(p, height) for p in parts)
    html = (
        '<details style="border:1px solid #e2e2e2;border-radius:8px;padding:8px 12px;">'
        f'<summary style="cursor:pointer;font-weight:600;">{_escape(summary)}</summary>'
        f'<div style="padding-top:8px;">{body}</div></details>'
    )
    try:
        from IPython.display import HTML
    except ImportError:
        return html
    return HTML(html)


def show_batch(batch: dict, height: Optional[int] = None, **kwargs):
    """A grid of viewers, one per molecule of ``batch`` (see :func:`batch_html`)."""
    html = batch_html(batch, **kwargs)
    n_mol = int(_numpy(batch[properties.n_atoms]).shape[0])
    return _embed(html, n_mol, 0, kwargs.get("cell_px", 230), height)


def show_trajectory(trajectory, batch: dict, height: Optional[int] = None, **kwargs):
    """One scrubbable row per trajectory (see :func:`trajectory_html`)."""
    html = trajectory_html(trajectory, batch, **kwargs)
    n_panels = 1 + int(kwargs.get("start", False)) + int(kwargs.get("end", False))
    return _embed(html, _n_rows(trajectory, batch), n_panels, kwargs.get("cell_px", 230), height)


def show_frames(trajectory, batch: dict, height: Optional[int] = None, **kwargs):
    """One row of stills per trajectory (see :func:`frames_html`)."""
    html = frames_html(trajectory, batch, **kwargs)
    n_frames = kwargs.get("n_frames", 5)
    cell_px = kwargs.get("cell_px") or _film_cell_px(n_frames)
    return _embed(html, _n_rows(trajectory, batch), n_frames, cell_px, height)


def _n_rows(trajectory, batch: dict) -> int:
    n_runs = len(trajectory) if isinstance(trajectory, dict) else 1
    return n_runs * int(_numpy(batch[properties.n_atoms]).shape[0])


def export_html(html: str, path: str) -> str:
    """Write a page built above to a standalone ``.html`` file. Returns ``path``."""
    page = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>molecular trajectories</title></head><body>{html}</body></html>"
    )
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(page)
    return path
