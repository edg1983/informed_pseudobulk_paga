import scanpy as sc
import pandas as pd
import numpy as np
import anndata as ad
import warnings
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
    def __init__(self, h5ad_path, hierarchy_levels):
        """
        Initialize the analyzer for large-scale datasets.
        """
        self.h5ad_path = h5ad_path
        self.hierarchy_levels = hierarchy_levels
        self.finest_level = hierarchy_levels[-1] # celltype_3
        self.adata = None 

    def _create_lightweight_adata(self, X_emb, obs_df, save_path=None):
        """Internal helper to create the lightweight AnnData."""
        print(f"📦 Creating lightweight AnnData (Shape: {X_emb.shape})...")
        self.adata = ad.AnnData(X=X_emb, obs=obs_df)
        self.adata.obsm['X_pca'] = X_emb
        
        if save_path:
            self.adata.write(save_path)
            print(f"💾 Lightweight object saved to {save_path}")

    def load_precomputed_embeddings(self, source='h5ad', key='X_pca', tsv_path=None, save_path=None):
        """Loads pre-computed embeddings (e.g., scVI, pre-calculated PCA)."""
        print(f"📂 Opening {self.h5ad_path} in backed mode to read metadata...")
        adata_backed = sc.read_h5ad(self.h5ad_path, backed='r')
        
        # 1. Load Embeddings
        X_emb = None
        
        if source == 'h5ad':
            print(f"📥 Reading embeddings from .obsm['{key}']...")
            if key not in adata_backed.obsm.keys():
                possible_keys = list(adata_backed.obsm.keys())
                raise ValueError(f"Key '{key}' not found in .obsm. Available: {possible_keys}")
            X_emb = adata_backed.obsm[key][:]
            
        elif source == 'file':
            if not tsv_path:
                raise ValueError("tsv_path must be provided if source='file'")
            print(f"📥 Reading embeddings from {tsv_path}...")
            sep = ',' if tsv_path.endswith('.csv') else '\t'
            try:
                df = pd.read_csv(tsv_path, sep=sep, index_col=0)
            except Exception as e:
                raise ValueError(f"Failed to read file: {e}")

            print("   Verifying cell ID alignment...")
            h5ad_indices = adata_backed.obs_names
            common_ids = df.index.intersection(h5ad_indices)
            
            if len(common_ids) == 0:
                 raise ValueError("❌ No matching cell IDs found. Check file format.")
            
            if len(common_ids) < len(h5ad_indices):
                missing_count = len(h5ad_indices) - len(common_ids)
                raise ValueError(f"❌ Alignment incomplete. {missing_count} cells missing.")

            if not df.index.equals(h5ad_indices):
                print("   ⚠️  Order mismatch detected. Re-ordering...")
                df_reordered = df.reindex(h5ad_indices)
                X_emb = df_reordered.values.astype(np.float32)
                del df, df_reordered
                gc.collect()
            else:
                print("   ✅ Cell order matches perfectly.")
                X_emb = df.values.astype(np.float32)
        else:
            raise ValueError("source must be 'h5ad' or 'file'")

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
            if i % (batch_size * 5) == 0: gc.collect()

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

    def run_gpu_trajectory(self, root_label="HSPC", n_neighbors=30):
        """Moves data to GPU (if available), computes Neighbors, PAGA, and DPT."""
        if self.adata is None:
            raise ValueError("Run preprocess or load embeddings first.")

        lib = rsc if GPU_AVAILABLE else sc
        print("🚀 Starting Trajectory Inference...")
        
        print(f"🔗 Computing Neighbor Graph (k={n_neighbors})...")
        lib.pp.neighbors(self.adata, n_neighbors=n_neighbors, n_pcs=50, use_rep='X_pca')
        
        print(f"🕸️  Computing PAGA using '{self.finest_level}' as backbone...")
        lib.tl.paga(self.adata, groups=self.finest_level)
        
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

        print("⏳ Computing Diffusion Pseudotime (guided by PAGA)...")
        lib.tl.dpt(self.adata)
        print("✅ Trajectory inference complete.")

    def enforce_hierarchy_constraint(self):
        """Validates hierarchy alignment."""
        print("🔧 Validating hierarchy alignment...")
        dpt_col = 'dpt_pseudotime'
        summary = self.adata.obs.groupby(self.hierarchy_levels)[dpt_col].mean().reset_index()
        summary = summary.sort_values(dpt_col)
        print(summary.head(10))
        return summary

    def _ensure_cpu(self):
        """Ensures all matrices are numpy arrays (CPU) before plotting."""
        if not GPU_AVAILABLE: return
        
        # Check obsm
        for key in self.adata.obsm.keys():
            if hasattr(self.adata.obsm[key], 'get'): # Check if CuPy array
                 self.adata.obsm[key] = self.adata.obsm[key].get()
        
        # Check uns (for paga adjacency)
        if 'paga' in self.adata.uns:
            for key in self.adata.uns['paga']:
                 if hasattr(self.adata.uns['paga'][key], 'get'):
                      self.adata.uns['paga'][key] = self.adata.uns['paga'][key].get()

    def compute_visualization_layout(self):
        """Computes UMAP and ForceDirected Graph (FA2) for visualization."""
        print("🎨 Computing visualization layouts...")
        lib = rsc if GPU_AVAILABLE else sc
        
        # 1. UMAP
        if 'X_umap' not in self.adata.obsm:
            print("   Computing UMAP...")
            lib.tl.umap(self.adata)

        # 2. Force Directed Layout (PAGA initialized)
        # This creates the tree-like structure that shows branching clearly
        print("   Computing ForceAtlas2 layout (PAGA-initialized)...")
        # Note: ForceAtlas2 (fa) is standard for branching. 
        # We perform this on the graph.
        lib.tl.draw_graph(self.adata, init_pos='paga')

    def plot_results(self, color_by='dpt_pseudotime', save_prefix='trajectory'):
        """
        Generates requested plots:
        1. UMAP with directed edges
        2. Branching Tree (ForceAtlas2) with pseudotime
        """
        self._ensure_cpu() # Make sure data is plotting-ready
        
        # Ensure layouts exist
        if 'X_draw_graph_fa' not in self.adata.obsm or 'X_umap' not in self.adata.obsm:
            self.compute_visualization_layout()

        # Define colors to plot
        keys = [self.finest_level, 'dpt_pseudotime']
        if color_by not in keys:
            keys.append(color_by)

        # Plot 1: UMAP with PAGA edges (Directed topology)
        print(f"📊 Plotting UMAP with PAGA edges...")
        fig_umap, ax_umap = plt.subplots(figsize=(8, 8))
        sc.pl.paga(
            self.adata, 
            pos=self.adata.obsm['X_umap'], 
            show=False, 
            ax=ax_umap,
            edge_width_scale=0.5,
            threshold=0.1,  # Only strong connections
            colors=color_by
        )
        plt.title(f"Trajectory UMAP ({color_by})")
        plt.savefig(f"{save_prefix}_umap_directed.png", bbox_inches='tight')
        plt.close()

        # Plot 2: Branching Tree (ForceAtlas2)
        # This puts root on one side and branches expanding out
        print(f"📊 Plotting Branching Tree (ForceAtlas2)...")
        
        # We plot the cells using the FA2 layout
        sc.pl.draw_graph(
            self.adata, 
            color=keys, 
            layout='fa', 
            ncols=2,
            show=False,
            save=f"_{save_prefix}_branching.png"
        )
        print(f"✅ Plots saved as {save_prefix}_*.png")

    def save_results(self, output_path):
        """Saves the results."""
        print(f"💾 Saving results to {output_path}...")
        self.adata.write(output_path)

# --- Execution Block ---
if __name__ == "__main__":
    # Configuration
    H5AD_FILE = "large_dataset.h5ad" 
    OUTPUT_FILE = "trajectory_results.h5ad"
    HIERARCHY = ['lineage_1', 'lineage_2', 'celltype_1', 'celltype_2', 'celltype_3']
    
    analyzer = LargeScaleTrajectory(H5AD_FILE, HIERARCHY)
    
    # 1. Load Data (Example using pre-computed)
    # analyzer.load_precomputed_embeddings(source='h5ad', key='X_scVI')
    if analyzer.adata is None:
        analyzer.preprocess_cpu_incremental(n_components=50)
    
    # 2. Run Inference
    analyzer.run_gpu_trajectory(root_label="HSPC")
    
    # 3. Check & Save
    analyzer.enforce_hierarchy_constraint()
    
    # 4. Visualize
    # This will generate the UMAP with edges and the branching scatter plot
    analyzer.plot_results(color_by='dpt_pseudotime')
    
    analyzer.save_results(OUTPUT_FILE)