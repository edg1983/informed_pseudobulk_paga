# Large Scale Single-Cell Trajectory Inference (10M Cells)

This solution uses a hybrid CPU/GPU approach to handle 10 million cells efficiently.PrerequisitesFor a dataset of this size (10M cells), a GPU is highly recommended.

## Prerequisites

For a dataset of this size (10M cells), a GPU is highly recommended.

The script uses rapids-singlecell to accelerate graph construction and diffusion map calculations, reducing runtime from days (CPU) to minutes (GPU).

## Environment Setup

You need a Python environment with Scanpy and RAPIDS.

### Option 1: Conda (Recommended)

```bash
conda create -n sc_gpu -c rapidsai -c conda-forge -c nvidia \
    rapids=23.12 python=3.10 cudatoolkit=11.8 \
    scanpy python-igraph leidenalg
conda activate sc_gpu
pip install rapids-singlecell
```

### Option 2: Pip (if CUDA is already installed)

```bash
pip install scanpy rapids-singlecell cupy-cuda11x
```

(Note: Replace cupy-cuda11x with the version matching your CUDA installation)

## How it Works

### Memory Management (Incremental PCA)

It is impossible to load 10M raw expression profiles into GPU memory (requires >200GB VRAM).
The script uses scanpy in backed='r' mode to read the .h5ad from disk without loading it into RAM.
It uses sklearn.IncrementalPCA to process the data in batches (e.g., 50k cells at a time) on the CPU.
Result: A lightweight PCA matrix (10M x 50) that fits easily into ~2GB of memory.

### GPU Acceleration

The lightweight PCA matrix is loaded into rapids_singlecell.
Nearest Neighbors, UMAP (optional), and PAGA are computed on the GPU.

### Guided Trajectory (PAGA + DPT)

- **PAGA Backbone:** We run PAGA using your finest label (celltype_3). This abstracts the 10M cells into a simpler graph of cell types, enforcing the biological topology you expect.
- **Root:** We identify "HSPC" cells to set the root iroot.
- **DPT:** We run Diffusion Pseudotime. By running PAGA first, DPT uses the PAGA connectivity to guide the random walk, ensuring the pseudotime respects the global structure of your hierarchy.

## Usage

The script is fully driven by command-line arguments — no source edits are required.

```
python trajectory_inference.py -i INPUT.h5ad
                                --hierarchy COL1 [COL2 ...]
                                --root-label-column COL
                                --root-label LABEL
                                [-o OUTPUT.h5ad]
                                [--h5ad-type {full,processed}]
                                [--embedding-key KEY | --embedding-file FILE]
                                [--n-components N]
                                [--n-neighbors N]
                                [--color-by KEY]
                                [--save-prefix PREFIX]
                                [--skip-plots]
```

### Required Arguments

| Argument | Description |
|---|---|
| `-i` / `--input` | Path to the input `.h5ad` file. |
| `--hierarchy` | Space-separated list of `obs` column names defining the cell-type hierarchy in order (coarsest → finest). |
| `--root-label-column` | The `obs` column that contains the root cell label. |
| `--root-label` | Label value identifying the root cell population (e.g. `HSC_MPP`). |

### Optional Arguments

| Argument | Default | Description |
|---|---|---|
| `-o` / `--output` | `trajectory_results.h5ad` | Path to the output `.h5ad` file. |
| `--h5ad-type` | `full` | `full` — raw counts (Incremental PCA is run). `processed` — loads the object as-is (embeddings expected). |
| `--embedding-key` | — | Read pre-computed embeddings from `.obsm[KEY]` (e.g. `X_scVI`). Skips PCA. Mutually exclusive with `--embedding-file`. |
| `--embedding-file` | — | Path to a TSV/CSV file with pre-computed embeddings. Skips PCA. Mutually exclusive with `--embedding-key`. |
| `--n-components` | `50` | Number of PCA components (Incremental PCA only). |
| `--n-neighbors` | `30` | Number of neighbors for the nearest-neighbor graph. |
| `--color-by` | `dpt_pseudotime` | `obs` key / column used to colour the output plots. |
| `--save-prefix` | `trajectory` | Filename prefix for saved plot images. |
| `--skip-plots` | `false` | Pass this flag to disable plot generation entirely. |

### Examples

**Minimal — raw counts, Incremental PCA:**
```bash
python trajectory_inference.py \
  -i large_dataset.h5ad \
  --hierarchy lineage_1 lineage_2 celltype_1 celltype_2 celltype_3 \
  --root-label-column lineage_2 \
  --root-label HSC_MPP
```

**Pre-computed scVI embeddings stored in the h5ad:**
```bash
python trajectory_inference.py \
  -i large_dataset.h5ad \
  -o results_scvi.h5ad \
  --hierarchy lineage_1 lineage_2 celltype_1 celltype_2 celltype_3 \
  --root-label-column lineage_2 \
  --root-label HSC_MPP \
  --embedding-key X_scVI
```

**Pre-computed embeddings from an external TSV file:**
```bash
python trajectory_inference.py \
  -i large_dataset.h5ad \
  --hierarchy lineage_1 lineage_2 celltype_1 \
  --root-label-column lineage_2 \
  --root-label HSC_MPP \
  --embedding-file embeddings.tsv \
  --n-neighbors 15
```

**Already-processed h5ad, skip plots:**
```bash
python trajectory_inference.py \
  -i processed.h5ad \
  --h5ad-type processed \
  --hierarchy lineage_1 celltype_1 \
  --root-label-column celltype_1 \
  --root-label Progenitor \
  --skip-plots
```

## Output

The script saves a new `.h5ad` file (default: `trajectory_results.h5ad`) containing:

- `obsm['X_pca']`: The PCA coordinates (or the provided pre-computed embeddings).
- `obs['dpt_pseudotime']`: The calculated pseudotime (0.0 to 1.0).
- `uns['paga']`: The graph abstraction data.
- `obsp['connectivities']`: The sparse nearest-neighbor graph.

Plot files are saved to the working directory with names matching `<save-prefix>_*.png` (e.g. `trajectory_umap_directed.png`, `trajectory_branching.png`).