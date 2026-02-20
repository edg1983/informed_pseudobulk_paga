# Trajectory Inference — Split-Lineage DPT (v2)

> `trajectory_inference_iterative_v2.py`

An enhanced single-cell trajectory inference pipeline built on **Scanpy**, with
optional GPU acceleration via **RAPIDS / rapids_singlecell**.  Version 2
introduces [Split-Lineage Diffusion Pseudotime](#split-lineage-approach) to
address the variance-dominance problem inherent in a single global DPT
computation.

---

## Table of Contents

1. [Motivation](#motivation)
2. [Split-Lineage Approach — Algorithm Overview](#split-lineage-approach)
   1. [Step 0 — Global backbone](#step-0--global-backbone)
   2. [Step 1 — Root group identification](#step-1--root-group-identification)
   3. [Step 2 — Per-lineage iteration](#step-2--per-lineage-iteration)
   4. [Step 3 — Root group stitched pseudotime](#step-3--root-group-stitched-pseudotime)
   5. [Step 4 — Reference global DPT](#step-4--reference-global-dpt)
3. [AnnData Storage Layout](#anndata-storage-layout)
   1. [`obs` columns](#obs-columns)
   2. [`uns` entries](#uns-entries)
   3. [`obsm` / `obsp` entries](#obsm--obsp-entries)
4. [Hierarchical PAGA Masking](#hierarchical-paga-masking)
5. [Plotting & Pseudotime Modes](#plotting--pseudotime-modes)
6. [CLI Reference](#cli-reference)
7. [Examples](#examples)
8. [Differences from v1](#differences-from-v1)

---

## Motivation

When running a **single global DPT** on a dataset that contains multiple
differentiation lineages (e.g. myeloid, lymphoid, erythroid), the resulting
pseudotime axis is dominated by whichever lineage has the highest variance in
diffusion-component space.  Cells in lower-variance lineages receive compressed,
poorly resolved pseudotime values that do not faithfully represent their
differentiation progression.

The **split-lineage** approach solves this by computing DPT **independently
per lineage**, ensuring that each branch receives the full `[0, 1]` pseudotime
range, then stitching the results back together.

---

## Split-Lineage Approach

Implemented in `LargeScaleTrajectory.run_split_lineage_trajectory()`.

### Step 0 — Global backbone

Before any per-lineage work:

1. **Neighbour graph** — computed on the full dataset using `X_pca`
   (k = `--n-neighbors`, default 30).
2. **Diffusion map** — computed globally (stored in `obsm['X_diffmap']`).
3. **Global hierarchical PAGA** — run through all hierarchy levels so the
   global PAGA graph is available for visualisation.

These global structures remain in the AnnData and are used for the overview
plots.

### Step 1 — Root group identification

The pipeline identifies the **root (progenitor) group** — the unique value in
the `split_by` column that contains the cells matching `--root-label`.

```
root_label = "HSC_MPP"
split_by   = "lineage_1"

→ The split_by group that contains HSC_MPP cells is the root group.
```

The root group is **not** processed as its own standalone lineage.  Instead,
its cells are **prepended to every lineage subset** so that each per-lineage
DPT can anchor at the biological origin.

### Step 2 — Per-lineage iteration

For every non-root value in the `split_by` column (the "iteration lineages"):

#### 2a. Subset construction

```
subset = cells where (split_by == lineage) | (split_by == root_group)
```

Both the lineage-specific cells **and** the root/progenitor cells are present
in each subset.  This guarantees the DPT root cell is always available.

Categorical columns are re-levelled to drop unused categories so that PAGA
only considers cell types actually present in the subset.

#### 2b. Recompute neighbours + diffusion map

A **fresh** k-NN graph and diffusion map are computed on the combined subset
using `X_pca`.  This is critical — re-using the global graph would not give
lineage-appropriate diffusion components.

#### 2c. Hierarchical PAGA on the subset

The same top-down hierarchical PAGA masking algorithm (see
[below](#hierarchical-paga-masking)) is run on the subset AnnData, using only
the hierarchy levels that have more than one unique value in this subset.

The resulting PAGA connectivities are saved to the master AnnData as
`uns['paga_{lineage}']`.

#### 2d. Diffusion Pseudotime (DPT)

`sc.tl.dpt` is called on the subset with `n_dcs` diffusion components
(default 10).  The root cell (`iroot`) is set to the first cell matching
`root_label` in the subset.

#### 2e. Min-max scaling to [0, 1]

Raw DPT values are scaled:

$$
\text{pt\_scaled}_{i} = \frac{\text{pt}_{i} - \min(\text{pt})}{\max(\text{pt}) - \min(\text{pt})}
$$

Only finite values participate in the min/max calculation.

#### 2f. Write-back into master AnnData

| Target column | Which cells receive a value | Value written |
|---|---|---|
| `dpt_pseudotime_{lineage}` | **All** cells in the subset (lineage + root) | Raw DPT |
| `dpt_pseudotime_{lineage}_scaled` | **All** cells in the subset (lineage + root) | Scaled DPT [0, 1] |
| `dpt_pseudotime_stitched` | **Only lineage-specific** cells (not root group) | Scaled DPT [0, 1] |

Root-group cells get per-lineage column values (they were in the subset), but
their stitched value is handled separately in Step 3.

#### 2g. Optional AnnData storage

If `--store-lineage-adata` is set, the per-lineage subset AnnData (with its
own neighbours, diffusion map, PAGA) is kept in `self.lineage_adatas[lineage]`.
Otherwise it is deleted and garbage-collected to save memory.

### Step 3 — Root group stitched pseudotime

After all lineages are processed:

1. **Stitched pseudotime** for root-group cells is set to **0.0** — they are
   the biological origin shared by all lineages.

2. A synthetic per-lineage column for the root group itself is computed as the
   **mean** of all lineage-specific scaled values:

   ```
   dpt_pseudotime_{root_group}        = mean across lineage columns
   dpt_pseudotime_{root_group}_scaled = mean across lineage columns
   ```

   This gives root cells a reasonable value in their own column while still
   correctly anchoring them at 0 in the stitched column.

### Step 4 — Reference global DPT

A single global DPT is also computed on the full dataset and stored in the
standard `obs['dpt_pseudotime']` column.  This serves as a comparison baseline
and is **not** used for the stitched pseudotime.

---

## AnnData Storage Layout

After `run_split_lineage_trajectory` completes, the AnnData object contains the
following additions.

### `obs` columns

| Column | Dtype | Description |
|---|---|---|
| `dpt_pseudotime` | `float64` | Global DPT (single pass on all cells). Reference only. |
| `dpt_pseudotime_stitched` | `float64` | **Primary output.** Union of all per-lineage scaled pseudotime values. Root group cells = 0.0. Lineage cells = their lineage-specific scaled value. Cells not assigned to any processed lineage = `NaN`. |
| `dpt_pseudotime_{L}` | `float64` | Raw (unscaled) DPT for lineage *L*, computed on the combined subset of lineage *L* + root group cells. `NaN` for cells not in that subset. |
| `dpt_pseudotime_{L}_scaled` | `float64` | Min-max scaled [0, 1] DPT for lineage *L*. Same cell coverage as the raw column. |

Where `{L}` is each unique value in the `split_by` column (including a
synthetic column for the root group).

**Example** with `split_by="lineage_1"` having values `Progenitor`, `Myeloid`,
`Lymphoid`:

| Column | Cells covered |
|---|---|
| `dpt_pseudotime_Myeloid` | Myeloid + Progenitor cells |
| `dpt_pseudotime_Myeloid_scaled` | Myeloid + Progenitor cells |
| `dpt_pseudotime_Lymphoid` | Lymphoid + Progenitor cells |
| `dpt_pseudotime_Lymphoid_scaled` | Lymphoid + Progenitor cells |
| `dpt_pseudotime_Progenitor` | Progenitor cells only (mean of Myeloid + Lymphoid) |
| `dpt_pseudotime_Progenitor_scaled` | Progenitor cells only (mean) |
| `dpt_pseudotime_stitched` | All cells: Progenitor = 0.0, Myeloid = Myeloid_scaled, Lymphoid = Lymphoid_scaled |

### `uns` entries

| Key | Type | Description |
|---|---|---|
| `paga` | dict | Global PAGA (connectivities, connectivities_tree, etc.) computed on all cells at the finest hierarchy level. |
| `paga_{L}` | dict | Per-lineage PAGA for lineage *L*. Contains `connectivities` (sparse matrix), `connectivities_tree`, and `groups` (list of category names present in that subset). |
| `iroot` | int | Index of the root cell (last set during the global DPT step). |
| `neighbors` | dict | Global neighbour graph parameters. |
| `diffmap_evals` | array | Eigenvalues from the global diffusion map. |

### `obsm` / `obsp` entries

| Key | Description |
|---|---|
| `obsm['X_pca']` | Input PCA embeddings (or pre-computed embeddings loaded via `--embedding-key`). |
| `obsm['X_diffmap']` | Global diffusion map coordinates. |
| `obsm['X_umap']` | UMAP coordinates (computed during `plot_results` if not already present). |
| `obsm['X_draw_graph_fa']` | ForceAtlas2 layout (PAGA-initialised, computed during plotting). |
| `obsp['distances']` | Global k-NN distance matrix. |
| `obsp['connectivities']` | Global k-NN connectivity matrix. |

> **Note:** Per-lineage neighbour graphs and diffusion maps are **not** stored
> back into the master AnnData (they live only in the temporary subset objects).
> If you need them, pass `--store-lineage-adata` and access
> `analyzer.lineage_adatas[lineage]`.

---

## Hierarchical PAGA Masking

Both global and per-lineage PAGA use the same top-down masking algorithm
(`compute_hierarchical_paga`):

1. **Coarsest level** → enforce a **star topology** at the root.  Only edges
   touching the root's coarse-level group are kept; all cross-lineage edges are
   severed.
2. **Each finer level** → PAGA is recomputed. Edges between fine-grained
   clusters are **removed** if their parent groups at the previous level were
   not connected above the threshold.
3. Allowed edges propagate down level by level, progressively constraining the
   graph.

Thresholds can be set globally (`--paga-threshold 0.05`) or per-level
(programmatically via the `thresholds` dict argument).

When running on a per-lineage subset, `compute_hierarchical_paga` automatically
detects which hierarchy levels are present in the subset (via
`active_levels`) and only iterates over those.

---

## Plotting & Pseudotime Modes

`plot_results()` generates two figure types:

| Figure | Panels |
|---|---|
| **UMAP** (2 panels) | Left: cells coloured by finest cell type. Right: cells coloured by pseudotime + directed PAGA arrows. |
| **Branching** (2 panels) | Left: X = pseudotime, Y = ForceAtlas2 Y, coloured by cell type. Right: same axes, coloured by pseudotime. |

### `--pseudotime-mode`

| Value | Global plot | Per-group plot |
|---|---|---|
| `global` (default) | Uses `dpt_pseudotime_stitched` (or `dpt_pseudotime`). | Uses the same global/stitched pseudotime. |
| `lineage` | Uses `dpt_pseudotime_stitched` (or `dpt_pseudotime`). | Uses `dpt_pseudotime_{group}_scaled` and the per-lineage PAGA graph (`uns['paga_{group}']`). Arrows are drawn using the lineage-specific PAGA connectivities and node positions. |

Per-group plots highlight relevant cells in the foreground and grey out the
rest.

---

## CLI Reference

```
usage: trajectory_inference_iterative_v2.py [-h] -i INPUT
    [--hierarchy LEVEL [LEVEL ...]] [--root-label LABEL]
    [-o OUTPUT] [--h5ad-type {full,processed}]
    [--mode {split,global}] [--split-by COLUMN]
    [--embedding-key KEY | --embedding-file PATH]
    [--n-components N] [--n-neighbors K] [--n-dcs N]
    [--color-by COLUMN] [--save-prefix PREFIX]
    [--group-by COLUMN] [--edge-color-by COLUMN]
    [--pseudotime-mode {global,lineage}]
    [--skip-plots] [--plots-only]
    [--paga-threshold FLOAT]
    [--store-lineage-adata]
```

### Required arguments

| Flag | Description |
|---|---|
| `-i, --input` | Path to input `.h5ad` file. |
| `--hierarchy` | Hierarchy-level column names in coarse-to-fine order (e.g. `lineage_1 lineage_2 celltype_1 celltype_2 celltype_3`). Required unless `--plots-only`. |
| `--root-label` | Label identifying the root/progenitor cells (searched across all hierarchy levels). Required unless `--plots-only`. |

### Mode selection

| Flag | Default | Description |
|---|---|---|
| `--mode {split,global}` | `split` | `split` = per-lineage DPT (recommended). `global` = single DPT pass (v1 behaviour). |
| `--split-by` | First hierarchy level | obs column defining major lineage branches for split mode. |

### Computation parameters

| Flag | Default | Description |
|---|---|---|
| `--n-neighbors` | 30 | k for the k-NN graph. |
| `--n-dcs` | 10 | Number of diffusion components passed to `sc.tl.dpt`. |
| `--n-components` | 50 | PCA components (only used with incremental PCA). |
| `--paga-threshold` | 0.05 | Default PAGA connectivity threshold. |

### Embedding options (mutually exclusive)

| Flag | Description |
|---|---|
| `--embedding-key` | Load pre-computed embeddings from `.obsm[KEY]` (e.g. `X_scVI`). |
| `--embedding-file` | Load embeddings from an external TSV/CSV file. |

### Visualisation

| Flag | Default | Description |
|---|---|---|
| `--color-by` | `dpt_pseudotime` | Column for the colour axis. Automatically resolves to `dpt_pseudotime_stitched` in split mode. |
| `--save-prefix` | `trajectory` | Filename prefix for saved PNGs. |
| `--group-by` | — | obs column for per-group UMAP/branching plots. |
| `--edge-color-by` | — | obs column for colouring PAGA arrows by lineage. |
| `--pseudotime-mode` | `global` | `global` or `lineage` (see [above](#plotting--pseudotime-modes)). |
| `--skip-plots` | — | Skip all plot generation. |
| `--plots-only` | — | Load a pre-processed h5ad and only generate plots. |

### Memory / storage

| Flag | Default | Description |
|---|---|---|
| `--store-lineage-adata` | off | Keep per-lineage AnnData objects in `self.lineage_adatas`. When off, subsets are deleted after write-back to conserve memory. |

### Output

| Flag | Default | Description |
|---|---|---|
| `-o, --output` | `trajectory_results.h5ad` | Path for the output `.h5ad` file. |
| `--h5ad-type` | `full` | Input type: `full` (raw) or `processed` (already lightweight). |

---

## Examples

### Split-lineage mode (recommended)

```bash
python trajectory_inference_iterative_v2.py \
    -i data.h5ad \
    -o results.h5ad \
    --hierarchy lineage_1 lineage_2 celltype_1 celltype_2 celltype_3 \
    --root-label HSC_MPP \
    --split-by lineage_1 \
    --n-neighbors 30 \
    --n-dcs 15 \
    --group-by lineage_1 \
    --edge-color-by lineage_2 \
    --pseudotime-mode lineage \
    --save-prefix split_results
```

### Global mode (single DPT, v1 behaviour)

```bash
python trajectory_inference_iterative_v2.py \
    -i data.h5ad \
    -o results.h5ad \
    --hierarchy lineage_1 lineage_2 celltype_1 \
    --root-label HSC_MPP \
    --mode global
```

### Pre-computed embeddings (e.g. scVI latent space)

```bash
python trajectory_inference_iterative_v2.py \
    -i data.h5ad \
    -o results.h5ad \
    --hierarchy lineage_1 celltype_1 \
    --root-label HSC_MPP \
    --embedding-key X_scVI \
    --split-by lineage_1
```

### Regenerate plots from a previously processed file

```bash
python trajectory_inference_iterative_v2.py \
    -i results.h5ad \
    --plots-only \
    --hierarchy lineage_1 lineage_2 celltype_1 celltype_2 celltype_3 \
    --group-by lineage_2 \
    --edge-color-by lineage_2 \
    --pseudotime-mode lineage \
    --save-prefix replotted
```

### Keep per-lineage AnnData objects for downstream inspection

```bash
python trajectory_inference_iterative_v2.py \
    -i data.h5ad \
    -o results.h5ad \
    --hierarchy lineage_1 celltype_1 \
    --root-label HSC_MPP \
    --split-by lineage_1 \
    --store-lineage-adata
```

---

## Differences from v1

| Feature | v1 (`trajectory_inference_iterative.py`) | v2 (`trajectory_inference_iterative_v2.py`) |
|---|---|---|
| DPT strategy | Single global pass | Per-lineage with progenitor prepending + global reference |
| `n_dcs` control | Hardcoded (Scanpy default) | CLI argument `--n-dcs` |
| Per-lineage pseudotime | Not available | `dpt_pseudotime_{L}`, `dpt_pseudotime_{L}_scaled` |
| Stitched pseudotime | Not available | `dpt_pseudotime_stitched` |
| Per-lineage PAGA | Not stored | `uns['paga_{L}']` |
| Progenitor handling | N/A | Root group cells prepended to every lineage subset; stitched PT = 0.0 |
| Min-max scaling | Not applied | Each lineage scaled to [0, 1] independently |
| Pseudotime plot modes | Global only | `--pseudotime-mode global|lineage` |
| Per-lineage AnnData storage | N/A | `--store-lineage-adata` flag |
| `compute_hierarchical_paga` | Operates on `self.adata` only | Accepts `target_adata` for subsets; auto-detects `active_levels` |
| `_ensure_cpu_adata` | Instance method on `self.adata` | Static method accepting any AnnData |
| Plotting helpers | Fixed pseudotime source | `pt_values` / `pt_override` parameters for lineage-specific rendering |
