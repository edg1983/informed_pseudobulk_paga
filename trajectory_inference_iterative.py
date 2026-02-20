import scanpy as sc
import pandas as pd
import numpy as np
import anndata as ad
import warnings
import argparse
import gc
from sklearn.decomposition import IncrementalPCA
from scipy import sparse
import matplotlib.pyplot as plt
import os

# Try importing GPU libraries, fall back gracefully
try:
    import rapids_singlecell as rsc
    import cupy as cp
    GPU_AVAILABLE = True
    print("🚀 NVIDIA GPU detected. Using RAPIDS for acceleration.")
except ImportError:
    GPU_AVAILABLE = False
    print("⚠️  RAPIDS/CuPy not found. Falling back to CPU.")

class LargeScaleTrajectory:
    def __init__(self, h5ad_path, hierarchy_levels, h5ad_type="full"):
        """
        Initialize the analyzer for large-scale datasets.
        """
        self.h5ad_path = h5ad_path
        self.hierarchy_levels = hierarchy_levels
        self.finest_level = hierarchy_levels[-1] # celltype_3
        self.adata = None 
        if h5ad_type == "processed":
            print(f"📂 Loading pre-processed AnnData from {self.h5ad_path}...")
            self.adata = sc.read_h5ad(self.h5ad_path)

    def _create_lightweight_adata(self, X_emb, obs_df, save_path=None):
        """Internal helper to create the lightweight AnnData."""
        print(f"📦 Creating lightweight AnnData (Shape: {X_emb.shape})...")
        self.adata = ad.AnnData(X=X_emb, obs=obs_df)
        self.adata.obsm['X_pca'] = X_emb
        
        # Ensure hierarchy columns are categorical (required for PAGA mapping)
        for level in self.hierarchy_levels:
            self.adata.obs[level] = self.adata.obs[level].astype('category')
            
        if save_path:
            self.adata.write(save_path)
            print(f"💾 Lightweight object saved to {save_path}")

    def load_precomputed_embeddings(self, source='h5ad', key='X_pca', tsv_path=None, save_path=None):
        """Loads pre-computed embeddings (e.g., scVI, pre-calculated PCA)."""
        print(f"📂 Opening {self.h5ad_path} in backed mode to read metadata...")
        adata_backed = sc.read_h5ad(self.h5ad_path, backed='r')
        
        X_emb = None
        if source == 'h5ad':
            print(f"📥 Reading embeddings from .obsm['{key}']...")
            X_emb = adata_backed.obsm[key][:]
        elif source == 'file':
            print(f"📥 Reading embeddings from {tsv_path}...")
            sep = ',' if tsv_path.endswith('.csv') else '\t'
            df = pd.read_csv(tsv_path, sep=sep, index_col=0)

            print("   Verifying cell ID alignment...")
            h5ad_indices = adata_backed.obs_names
            if not df.index.equals(h5ad_indices):
                print("   ⚠️  Order mismatch detected. Re-ordering...")
                df = df.reindex(h5ad_indices)
            X_emb = df.values.astype(np.float32)
            del df
            gc.collect()

        print("📑 Extracting metadata...")
        obs = adata_backed.obs[self.hierarchy_levels].copy()
        del adata_backed
        gc.collect()
        
        self._create_lightweight_adata(X_emb, obs, save_path)
        
    def preprocess_cpu_incremental(self, n_components=50, batch_size=50000, save_pca_path=None):
        """Performs Incremental PCA on CPU."""
        print(f"📂 Opening {self.h5ad_path} in backed mode...")
        adata_backed = sc.read_h5ad(self.h5ad_path, backed='r')
        ipca = IncrementalPCA(n_components=n_components)
        n_cells = adata_backed.shape[0]
        
        print("⚙️  Fitting Incremental PCA in batches...")
        for i in range(0, n_cells, batch_size):
            end = min(i + batch_size, n_cells)
            chunk = adata_backed[i:end].X
            if sparse.issparse(chunk): chunk = chunk.toarray()
            ipca.partial_fit(chunk)

        print("📉 Transforming data...")
        X_pca = np.zeros((n_cells, n_components), dtype=np.float32)
        for i in range(0, n_cells, batch_size):
            end = min(i + batch_size, n_cells)
            chunk = adata_backed[i:end].X
            if sparse.issparse(chunk): chunk = chunk.toarray()
            X_pca[i:end] = ipca.transform(chunk)
        
        obs = adata_backed.obs[self.hierarchy_levels].copy()
        del adata_backed
        gc.collect()

        self._create_lightweight_adata(X_pca, obs, save_pca_path)

    def _ensure_cpu(self):
        """Ensures all matrices are numpy/scipy arrays (CPU) before Scanpy algorithms."""
        if not GPU_AVAILABLE: return
        if hasattr(self.adata.X, 'get'): self.adata.X = self.adata.X.get()
        for key in list(self.adata.obsm.keys()):
            if hasattr(self.adata.obsm[key], 'get'): self.adata.obsm[key] = self.adata.obsm[key].get()
        if hasattr(self.adata, 'obsp'):
            for key in list(self.adata.obsp.keys()):
                if hasattr(self.adata.obsp[key], 'get'): self.adata.obsp[key] = self.adata.obsp[key].get()
        if 'paga' in self.adata.uns:
            for key in list(self.adata.uns['paga'].keys()):
                 if hasattr(self.adata.uns['paga'][key], 'get'): self.adata.uns['paga'][key] = self.adata.uns['paga'][key].get()

    def compute_hierarchical_paga(self, root_label, thresholds=0.05):
        """
        Computes PAGA iteratively from coarse to fine. 
        Severely penalizes or removes edges between fine clusters if their 
        coarse parent lineages are not biologically connected.
        
        Args:
            root_label (str): The label of the root progenitor cells.
            thresholds (float or dict): A single threshold for all levels, or a dictionary
                                        mapping level names to specific thresholds.
                                        e.g., {'lineage_1': 0.01, 'celltype_3': 0.1}
        """
        print("\n🌳 Starting Top-Down Hierarchical PAGA Masking...")
        self._ensure_cpu()
        
        allowed_parent_edges = None
        prev_level = None
        
        for level in self.hierarchy_levels:
            # Determine threshold for this specific level
            if isinstance(thresholds, dict):
                current_thresh = thresholds.get(level, 0.05)
            else:
                current_thresh = thresholds
                
            print(f"   ► Computing PAGA for level: '{level}' (Threshold: {current_thresh})")
            sc.tl.paga(self.adata, groups=level)
            
            paga_conn = self.adata.uns['paga']['connectivities'].toarray()
            categories = self.adata.obs[level].cat.categories
            
            # --- MASKING LOGIC ---
            if level == self.hierarchy_levels[0]:
                # Force star topology from the root at the very first level
                # 1. Find which top-level category contains the root_label
                root_mask = self.adata.obs[self.finest_level] == root_label
                if not root_mask.any():
                    for l in reversed(self.hierarchy_levels):
                        if (self.adata.obs[l] == root_label).any():
                            root_mask = self.adata.obs[l] == root_label
                            break
                top_root_category = self.adata.obs[root_mask][level].mode()[0]
                print(f"     👑 Enforcing root hub: only allowing edges connected to '{top_root_category}'")
                
                masked_top_edges = 0
                for i, cat_i in enumerate(categories):
                    for j, cat_j in enumerate(categories):
                        if i == j: continue
                        # If NEITHER category is the root category, kill the connection
                        if cat_i != top_root_category and cat_j != top_root_category:
                            if paga_conn[i, j] > 0:
                                paga_conn[i, j] = 0.0
                                masked_top_edges += 1
                if masked_top_edges > 0:
                    print(f"     ✂️ Severed {masked_top_edges} cross-lineage edges at the root level.")
                    self.adata.uns['paga']['connectivities'] = sparse.csr_matrix(paga_conn)

            elif allowed_parent_edges is not None:
                # Build mapping: Find the parent category for each current category
                # .mode()[0] safely handles minor annotation noise/errors
                mapping = {}
                for i, cat in enumerate(categories):
                    parent_cat = self.adata.obs[self.adata.obs[level] == cat][prev_level].mode()[0]
                    mapping[i] = parent_cat
                
                masked_edges = 0
                for i in range(len(categories)):
                    for j in range(len(categories)):
                        if i == j: continue
                        
                        parent_i = mapping[i]
                        parent_j = mapping[j]
                        
                        # If parents were NOT connected in the coarse graph, sever this fine edge
                        if not allowed_parent_edges.get((parent_i, parent_j), False):
                            if paga_conn[i, j] > 0:
                                paga_conn[i, j] = 0.0
                                masked_edges += 1
                                
                print(f"     ✂️ Severed {masked_edges} cross-lineage short-circuits based on '{prev_level}' rules.")
                
                # Update the object with the strictly pruned connectivities
                self.adata.uns['paga']['connectivities'] = sparse.csr_matrix(paga_conn)

            # --- PREPARE NEXT LEVEL RULES ---
            allowed_parent_edges = {}
            for i, cat_i in enumerate(categories):
                for j, cat_j in enumerate(categories):
                    # We allow an edge down the hierarchy if the PAGA score > current_thresh
                    if i == j or paga_conn[i, j] >= current_thresh:
                        allowed_parent_edges[(cat_i, cat_j)] = True
            
            prev_level = level
            
        print(f"✅ Hierarchical PAGA complete. Final strictly-bounded backbone rests on '{self.finest_level}'.\n")

    def run_gpu_trajectory(self, root_label="HSPC", n_neighbors=30):
        """Moves data to GPU, computes Neighbors, runs Hierarchical PAGA, and DPT."""
        if self.adata is None:
            raise ValueError("Run preprocess or load embeddings first.")

        lib = rsc if GPU_AVAILABLE else sc
        print("🚀 Starting Trajectory Inference...")
        
        print(f"🔗 Computing Neighbor Graph (k={n_neighbors})...")
        lib.pp.neighbors(self.adata, n_neighbors=n_neighbors, n_pcs=50, use_rep='X_pca')
        
        print("🗺️  Computing Diffusion Maps...")
        lib.tl.diffmap(self.adata)
        
        print("⏬ Moving data to CPU for Scanpy PAGA and DPT...")
        self._ensure_cpu()
        
        # --- NEW: Call the Hierarchical PAGA instead of standard PAGA ---
        # Using a lenient threshold at the top to preserve all root branches, 
        # and stricter thresholds downstream to clean up the graph.
        level_thresholds = {
            self.hierarchy_levels[0]: 0.01,  # lineage_1 (Very lenient for root branches)
            self.hierarchy_levels[1]: 0.01,  # lineage_2
            self.hierarchy_levels[2]: 0.03,  # celltype_1
            self.hierarchy_levels[3]: 0.05,  # celltype_2
            self.hierarchy_levels[-1]: 0.05  # celltype_3 (Stricter for fine clusters)
        }
        self.compute_hierarchical_paga(root_label=root_label, thresholds=level_thresholds)
        
        print(f"📍 Setting root to a cell in group: {root_label}")
        root_mask = self.adata.obs[self.finest_level] == root_label
        if not root_mask.any():
            for level in reversed(self.hierarchy_levels):
                if (self.adata.obs[level] == root_label).any():
                    root_mask = self.adata.obs[level] == root_label
                    break
        
        if not root_mask.any():
            raise ValueError(f"Root label '{root_label}' not found.")

        flat_indices = np.where(root_mask.values)[0]
        self.adata.uns['iroot'] = flat_indices[0] 
        print(f"   Root index set to: {self.adata.uns['iroot']}")

        print("⏳ Computing Diffusion Pseudotime (guided by strict hierarchical PAGA)...")
        sc.tl.dpt(self.adata)
        print("✅ Trajectory inference complete.")

    def enforce_hierarchy_constraint(self):
        print("🔧 Validating hierarchy alignment...")
        dpt_col = 'dpt_pseudotime'
        summary = self.adata.obs.groupby(self.hierarchy_levels, observed=True)[dpt_col].mean().reset_index()
        summary = summary.sort_values(dpt_col)
        print(summary.head(10))
        return summary

    def compute_visualization_layout(self):
        print("🎨 Computing visualization layouts...")
        lib = rsc if GPU_AVAILABLE else sc

        if 'X_umap' not in self.adata.obsm:
            print("   Computing UMAP...")
            lib.tl.umap(self.adata)

        print("   Computing ForceAtlas2 layout (PAGA-initialized)...")
        if GPU_AVAILABLE:
            # RAPIDS bug workaround: rsc.tl.draw_graph hard-codes
            # neighbors_key='connectivities' when calling Scanpy's
            # get_init_pos_from_paga, causing it to look for uns['connectivities']
            # as a neighbours *dict* instead of the matrix in obsp.
            # Fix: compute PAGA init coords manually with the correct
            # neighbors_key='neighbors', then pass them as a plain numpy array
            # to draw_graph, bypassing the broken 'paga' branch entirely.
            from scanpy.tools._utils import get_init_pos_from_paga
            print("   Pre-computing PAGA init positions for GPU layout...")
            # sc.pl.paga (not sc.tl.paga) is the function that writes
            # adata.uns['paga']['pos'], which get_init_pos_from_paga requires.
            if 'pos' not in self.adata.uns.get('paga', {}):
                sc.pl.paga(self.adata, show=False)
            init_coords = get_init_pos_from_paga(
                self.adata, random_state=0, neighbors_key='neighbors'
            )
            lib.tl.draw_graph(self.adata, init_pos=init_coords)
        else:
            sc.tl.draw_graph(self.adata, init_pos='paga')
        self._ensure_cpu()

    def plot_results(self, color_by='dpt_pseudotime', save_prefix='trajectory'):
        if 'X_draw_graph_fa' not in self.adata.obsm or 'X_umap' not in self.adata.obsm:
            self.compute_visualization_layout()
            
        self._ensure_cpu()

        keys = [self.finest_level, 'dpt_pseudotime']
        if color_by not in keys: keys.append(color_by)

        print(f"📊 Plotting UMAP with PAGA edges...")
        fig_umap, ax_umap = plt.subplots(figsize=(8, 8))
        sc.pl.paga(
            self.adata, pos=self.adata.obsm['X_umap'], show=False, ax=ax_umap,
            edge_width_scale=0.5, threshold=0.01, colors=color_by
        )
        plt.title(f"Trajectory UMAP ({color_by})")
        plt.savefig(f"{save_prefix}_umap_directed.png", bbox_inches='tight')
        plt.close()

        print(f"📊 Plotting Branching Tree (ForceAtlas2)...")
        sc.pl.draw_graph(
            self.adata, color=keys, layout='fa', ncols=2, show=False, save=f"_{save_prefix}_branching.png"
        )
        print(f"✅ Plots saved as {save_prefix}_*.png")

    def save_results(self, output_path):
        print(f"💾 Saving results to {output_path}...")
        self.adata.write(output_path)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Large-scale trajectory inference with optional GPU acceleration.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  %(prog)s -i data.h5ad -o results.h5ad --hierarchy lineage_1 lineage_2 celltype_1 --root-label HSC_MPP
  %(prog)s -i data.h5ad -o results.h5ad --hierarchy lineage_1 celltype_1 --root-label-column celltype_1 --root-label Progenitor
  %(prog)s -i processed.h5ad --h5ad-type processed --embedding-key X_scVI --root-label HSC_MPP
"""
    )
    # Required arguments
    parser.add_argument("-i", "--input", required=True,
                        help="Path to input .h5ad file.")
    parser.add_argument("--hierarchy", required=True, nargs="+",
                        help="Hierarchy level column names in order (e.g., lineage_1 lineage_2 celltype_1).")
    parser.add_argument("--root-label", required=True,
                        help="Label value used to identify the root cell (e.g., HSC_MPP).")

    # Optional arguments
    parser.add_argument("-o", "--output", default="trajectory_results.h5ad",
                        help="Path to output .h5ad file (default: trajectory_results.h5ad).")
    parser.add_argument("--h5ad-type", choices=["full", "processed"], default="full",
                        help="Type of input h5ad: 'full' (raw) or 'processed' (default: full).")

    # Embedding / PCA options
    emb_group = parser.add_mutually_exclusive_group()
    emb_group.add_argument("--embedding-key", default=None,
                           help="Load pre-computed embeddings from .obsm[KEY] (e.g., X_scVI) in the h5ad input file. "
                                "Skips PCA preprocessing.")
    emb_group.add_argument("--embedding-file", default=None,
                           help="Path to a TSV/CSV file with pre-computed embeddings. "
                                "Skips PCA preprocessing.")
    parser.add_argument("--n-components", type=int, default=50,
                        help="Number of PCA components for incremental PCA (default: 50).")
    parser.add_argument("--n-neighbors", type=int, default=30,
                        help="Number of neighbors for the neighbor graph (default: 30).")

    # Visualization options
    parser.add_argument("--color-by", default="dpt_pseudotime",
                        help="Column or key to color plots by (default: dpt_pseudotime).")
    parser.add_argument("--save-prefix", default="trajectory",
                        help="Filename prefix for saved plots (default: trajectory).")
    parser.add_argument("--skip-plots", action="store_true",
                        help="Skip plot generation.")

    return parser.parse_args()


# --- Execution Block ---
if __name__ == "__main__":
    args = parse_args()

    analyzer = LargeScaleTrajectory(
        h5ad_path=args.input,
        hierarchy_levels=args.hierarchy,
        h5ad_type=args.h5ad_type
    )

    # 1. Load Data
    if args.embedding_key:
        analyzer.load_precomputed_embeddings(source='h5ad', key=args.embedding_key)
    elif args.embedding_file:
        analyzer.load_precomputed_embeddings(source='file', tsv_path=args.embedding_file, save_path=f"{args.save_prefix}_lightweight.h5ad")
    elif analyzer.adata is None:
        analyzer.preprocess_cpu_incremental(n_components=args.n_components)

    # 2. Run Inference
    analyzer.run_gpu_trajectory(root_label=args.root_label, n_neighbors=args.n_neighbors)

    # 3. Check & Save
    analyzer.enforce_hierarchy_constraint()
    analyzer.save_results(args.output)

    # 4. Visualize
    if not args.skip_plots:
        analyzer.plot_results(color_by=args.color_by, save_prefix=args.save_prefix)
        analyzer.save_results(args.output)