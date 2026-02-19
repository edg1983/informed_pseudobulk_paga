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

1. Open `trajectory_inference.py`.
2. Update the `H5AD_FILE` variable with the path to your dataset.
3. Ensure your .h5ad obs dataframe contains the hierarchy columns: ['lineage_1', 'lineage_2', 'celltype_1', 'celltype_2', 'celltype_3'].
4. Run the script:

```bash
python trajectory_inference.py
```

## Output

The script saves a new file `trajectory_results.h5ad` containing:

- `obsm['X_pca']`: The PCA coordinates.
- `obs['dpt_pseudotime']`: The calculated pseudotime (0.0 to 1.0).
- `uns['paga']`: The graph abstraction data.
- `obsp['connectivities']`: The sparse neighbor graph.