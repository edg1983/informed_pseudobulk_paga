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

    def _is_numeric_col(self, col):
        """Returns True if obs column `col` holds numeric values."""
        return col in self.adata.obs.columns and \
               np.issubdtype(self.adata.obs[col].dtype, np.number)

    def _draw_umap_pair(self, ax_ct, ax_pt,
                        umap_coords, cell_colors, color_by,
                        palette, groups, handles,
                        paga_pos, paga_conn,
                        mask=None, title_suffix='',
                        mean_pt_by_group=None,
                        edge_node_lineage=None,
                        edge_palette=None,
                        edge_legend_handles=None,
                        relevant_node_indices=None):
        """
        Render one UMAP pair (celltype left, color_by + directed PAGA right).

        Directed arrows run from lower-pseudotime node to higher-pseudotime node.
        If `edge_node_lineage` and `edge_palette` are provided, each arrow is
        coloured by the *source* node's lineage group.

        If `mask` is provided, non-masked cells are drawn in light grey.
        If `relevant_node_indices` is provided (set of ints), only edges where
        at least one endpoint is a relevant node are drawn — used to hide
        unrelated edges in per-group plots.
        """
        bg_kw  = dict(s=0.2, linewidths=0, rasterized=True)
        fg_kw  = dict(s=0.5, linewidths=0, rasterized=True)

        for ax in (ax_ct, ax_pt):
            ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
            ax.set_aspect('equal')

        def _scatter_with_mask(ax, colors, cmap_vals=None, cmap='viridis'):
            if mask is None:
                if cmap_vals is not None:
                    sc = ax.scatter(umap_coords[:, 0], umap_coords[:, 1],
                                    c=cmap_vals, cmap=cmap, **bg_kw)
                    return sc
                ax.scatter(umap_coords[:, 0], umap_coords[:, 1],
                           c=colors, **bg_kw)
                return None
            ax.scatter(umap_coords[~mask, 0], umap_coords[~mask, 1],
                       c='#d0d0d0', **bg_kw)
            if cmap_vals is not None:
                sc = ax.scatter(umap_coords[mask, 0], umap_coords[mask, 1],
                                c=cmap_vals[mask], cmap=cmap, **fg_kw)
                return sc
            ax.scatter(umap_coords[mask, 0], umap_coords[mask, 1],
                       c=colors[mask], **fg_kw)
            return None

        # ── Left panel: celltype ─────────────────────────────────────────────
        _scatter_with_mask(ax_ct, cell_colors)
        ax_ct.legend(handles=handles, fontsize=5, markerscale=3,
                     loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0)
        ax_ct.set_title(f"UMAP — {self.finest_level}{title_suffix}", fontsize=10)

        # ── Right panel: color_by + directed PAGA arrows ─────────────────────
        cmap_vals = self.adata.obs[color_by].values if self._is_numeric_col(color_by) else None
        sc = _scatter_with_mask(ax_pt, cell_colors, cmap_vals=cmap_vals)
        if sc is not None:
            plt.colorbar(sc, ax=ax_pt, label=color_by, shrink=0.6)

        # Directed PAGA arrows (source = lower pseudotime)
        n_g   = len(groups)
        max_w = paga_conn.max() if paga_conn.max() > 0 else 1.0
        group_list = list(groups)  # index ↔ name

        for i in range(n_g):
            for j in range(i + 1, n_g):
                w = paga_conn[i, j]
                if w < 0.01:
                    continue

                # Skip edges not touching any relevant node (per-group filter)
                if relevant_node_indices is not None and \
                        i not in relevant_node_indices and \
                        j not in relevant_node_indices:
                    continue

                lw = (w / max_w) * 3

                # Determine direction from pseudotime
                pt_i = mean_pt_by_group.get(group_list[i], 0.0) if mean_pt_by_group else 0.0
                pt_j = mean_pt_by_group.get(group_list[j], 0.0) if mean_pt_by_group else 1.0
                src, dst = (i, j) if pt_i <= pt_j else (j, i)

                # Determine edge colour from source node's lineage
                if edge_node_lineage and edge_palette:
                    src_lineage = edge_node_lineage.get(group_list[src])
                    edge_color  = edge_palette.get(src_lineage, '#333333')
                else:
                    edge_color = '#333333'

                ax_pt.annotate(
                    "",
                    xy=paga_pos[dst], xytext=paga_pos[src],
                    xycoords='data', textcoords='data',
                    arrowprops=dict(
                        arrowstyle=f"->, head_width={lw*0.25:.2f}, head_length={lw*0.18:.2f}",
                        color=edge_color,
                        lw=lw,
                        connectionstyle='arc3,rad=0.08',
                        alpha=0.85,
                    ),
                    zorder=6,
                )

        # Edge-colour legend (one entry per lineage)
        if edge_legend_handles:
            legend_cells = ax_pt.get_legend()
            ax_pt.legend(handles=edge_legend_handles, fontsize=6,
                         title='Edge lineage', title_fontsize=7,
                         loc='lower left', bbox_to_anchor=(1.01, 0), borderaxespad=0)

        ax_pt.set_title(f"UMAP — {color_by} + directed PAGA{title_suffix}", fontsize=10)

    def _draw_branching_pair(self, ax_b1, ax_b2,
                             pseudotime, fa2_coords, cell_colors, color_by,
                             handles, mask=None, title_suffix=''):
        """
        Render one branching pair (celltype left, color_by right).
        X = DPT pseudotime, Y = FA2 Y (branch-divergence axis).
        If `mask` is provided, non-masked cells are grey context.
        """
        bg_kw = dict(s=0.2, linewidths=0, rasterized=True)
        fg_kw = dict(s=0.5, linewidths=0, rasterized=True)

        for ax in (ax_b1, ax_b2):
            ax.set_xlabel("Pseudotime (DPT)")
            ax.set_ylabel("FA2 Y (branch axis)")

        fa2_y = fa2_coords[:, 1]
        order = np.argsort(pseudotime)

        def _scatter_branch(ax, colors, cmap_vals=None, cmap='viridis'):
            if mask is None:
                if cmap_vals is not None:
                    sc = ax.scatter(pseudotime[order], fa2_y[order],
                                    c=cmap_vals[order], cmap=cmap, **bg_kw)
                    return sc
                ax.scatter(pseudotime[order], fa2_y[order],
                           c=colors[order], **bg_kw)
                return None
            # background
            bg = ~mask
            ax.scatter(pseudotime[bg], fa2_y[bg],
                       c='#d0d0d0', **bg_kw)
            # foreground
            fg_order = np.argsort(pseudotime[mask])
            fg_pt  = pseudotime[mask][fg_order]
            fg_fa2 = fa2_y[mask][fg_order]
            if cmap_vals is not None:
                sc = ax.scatter(fg_pt, fg_fa2,
                                c=cmap_vals[mask][fg_order], cmap=cmap, **fg_kw)
                return sc
            ax.scatter(fg_pt, fg_fa2, c=colors[mask][fg_order], **fg_kw)
            return None

        # Left — celltype
        _scatter_branch(ax_b1, cell_colors)
        ax_b1.legend(handles=handles, fontsize=5, markerscale=3,
                     loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0)
        ax_b1.set_title(f"Branching — {self.finest_level}{title_suffix}", fontsize=10)

        # Right — color_by
        cmap_vals = self.adata.obs[color_by].values if self._is_numeric_col(color_by) else None
        sc = _scatter_branch(ax_b2, cell_colors, cmap_vals=cmap_vals)
        if sc is not None:
            plt.colorbar(sc, ax=ax_b2, label=color_by, shrink=0.6)
        ax_b2.set_title(f"Branching — {color_by}{title_suffix}", fontsize=10)

    def plot_results(self, color_by='dpt_pseudotime', save_prefix='trajectory',
                     group_by=None, edge_color_by=None):
        """
        Generates output figures.

        Global (all cells):
          <save_prefix>_umap.png      — 2-panel UMAP:
            Left : cells coloured by finest_level.
            Right: cells coloured by `color_by` with directed PAGA arrows.
                   Arrows point from progenitor to differentiated (by DPT mean).
                   When `edge_color_by` is set each arrow is coloured by its
                   source node's parent group (e.g. B/T/NK lineage).
          <save_prefix>_branching.png — 2-panel pseudotime branching:
            X = dpt_pseudotime, Y = FA2 Y (branch axis).

        Per-group (when `group_by` column is set):
          For each unique value G in obs[group_by]:
            <save_prefix>_<G>_umap.png + _branching.png  (G cells highlighted).
        """
        if 'X_draw_graph_fa' not in self.adata.obsm or 'X_umap' not in self.adata.obsm:
            self.compute_visualization_layout()

        self._ensure_cpu()
        from matplotlib.patches import Patch, FancyArrow

        umap_coords  = self.adata.obsm['X_umap']
        fa2_coords   = self.adata.obsm['X_draw_graph_fa']
        pseudotime   = self.adata.obs['dpt_pseudotime'].values
        celltype_col = self.adata.obs[self.finest_level]

        groups    = celltype_col.cat.categories
        n_groups  = len(groups)
        palette   = dict(zip(groups, plt.cm.tab20.colors[:n_groups] if n_groups <= 20
                             else [plt.cm.hsv(i / n_groups) for i in range(n_groups)]))
        cell_colors = np.array([palette[g] for g in celltype_col])
        handles     = [Patch(color=palette[g], label=g) for g in groups]

        paga_pos  = np.array([umap_coords[celltype_col == g].mean(axis=0) for g in groups])
        paga_conn = self.adata.uns['paga']['connectivities'].toarray()

        # ── Mean pseudotime per PAGA node (drives arrow direction) ──────────
        mean_pt_by_group = {
            g: float(pseudotime[celltype_col == g].mean())
            for g in groups
        }

        # ── Edge colouring: map each finest-level node → parent lineage ──────
        edge_node_lineage   = None
        edge_palette        = None
        edge_legend_handles = None

        if edge_color_by is not None and edge_color_by in self.adata.obs.columns:
            lineage_col = self.adata.obs[edge_color_by]
            lineages = lineage_col.cat.categories if hasattr(lineage_col, 'cat') \
                       else lineage_col.unique()
            n_lin = len(lineages)
            # Use a visually distinct colormap for lineages
            lin_colors = plt.cm.Set1.colors[:n_lin] if n_lin <= 9 \
                         else [plt.cm.tab10(i / n_lin) for i in range(n_lin)]
            edge_palette = dict(zip(lineages, lin_colors))

            # For each finest-level cluster, find its dominant parent lineage
            edge_node_lineage = {}
            for g in groups:
                parent = self.adata.obs[celltype_col == g][edge_color_by].mode()
                edge_node_lineage[g] = parent.iloc[0] if len(parent) > 0 else lineages[0]

            edge_legend_handles = [
                Patch(color=edge_palette[lin], label=lin) for lin in lineages
            ]
            print(f"   Edge colours mapped from '{edge_color_by}' "
                  f"({n_lin} lineages: {list(lineages)})")
        elif edge_color_by is not None:
            print(f"⚠️  --edge-color-by column '{edge_color_by}' not found in obs "
                  f"— edges will be drawn in dark grey.")

        saved = []

        def _save_pair(umap_path, branch_path, mask=None, title_suffix=''):
            # Compute which PAGA nodes have cells in this group (for edge filtering)
            relevant_node_indices = None
            if mask is not None:
                relevant_node_indices = {
                    i for i, g in enumerate(groups)
                    if np.any(mask & (celltype_col == g).values)
                }

            # Filter legend to only cell types present in the masked subset
            active_handles = handles
            if mask is not None:
                present_groups = set(celltype_col[mask].unique())
                active_handles = [h for h in handles if h.get_label() in present_groups]

            fig, (ax_ct, ax_pt) = plt.subplots(1, 2, figsize=(18, 7))
            fig.subplots_adjust(wspace=0.35)
            self._draw_umap_pair(
                ax_ct, ax_pt, umap_coords, cell_colors, color_by,
                palette, groups, active_handles, paga_pos, paga_conn,
                mask=mask, title_suffix=title_suffix,
                mean_pt_by_group=mean_pt_by_group,
                edge_node_lineage=edge_node_lineage,
                edge_palette=edge_palette,
                edge_legend_handles=edge_legend_handles,
                relevant_node_indices=relevant_node_indices,
            )
            fig.savefig(umap_path, dpi=150, bbox_inches='tight')
            plt.close(fig)

            fig2, (ax_b1, ax_b2) = plt.subplots(1, 2, figsize=(18, 6))
            fig2.subplots_adjust(wspace=0.3)
            self._draw_branching_pair(ax_b1, ax_b2, pseudotime, fa2_coords,
                                      cell_colors, color_by, active_handles,
                                      mask=mask, title_suffix=title_suffix)
            fig2.savefig(branch_path, dpi=150, bbox_inches='tight')
            plt.close(fig2)

            saved.extend([umap_path, branch_path])
            print(f"   Saved → {umap_path}, {branch_path}")

        # ── Global plots ─────────────────────────────────────────────────────
        print("📊 Plotting global UMAP and branching (all cells)...")
        _save_pair(f"{save_prefix}_umap.png", f"{save_prefix}_branching.png")

        # ── Per-group plots ───────────────────────────────────────────────────
        if group_by is not None:
            if group_by not in self.adata.obs.columns:
                print(f"⚠️  --group-by column '{group_by}' not found in obs — skipping per-group plots.")
            else:
                group_col = self.adata.obs[group_by]
                subgroups = group_col.cat.categories if hasattr(group_col, 'cat') \
                            else group_col.unique()
                print(f"📊 Plotting per-group views for {len(subgroups)} groups in '{group_by}'...")
                for grp in subgroups:
                    mask = (group_col == grp).values
                    safe_name = str(grp).replace('/', '_').replace(' ', '_')
                    suffix = f" [{grp}]"
                    _save_pair(
                        f"{save_prefix}_{safe_name}_umap.png",
                        f"{save_prefix}_{safe_name}_branching.png",
                        mask=mask, title_suffix=suffix
                    )

        print(f"✅ {len(saved)} plot files saved with prefix '{save_prefix}'.")

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
    parser.add_argument("--hierarchy", required=False, nargs="+", default=None,
                        help="Hierarchy level column names in order (e.g., lineage_1 lineage_2 celltype_1). "
                             "Required unless --plots-only is set.")
    parser.add_argument("--root-label", required=False, default=None,
                        help="Label value used to identify the root cell (e.g., HSC_MPP). "
                             "Required unless --plots-only is set.")

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
    parser.add_argument("--group-by", default=None,
                        help="obs column used to split per-group plots (e.g., lineage_2). "
                             "For each unique value a highlighted UMAP and branching plot "
                             "is saved alongside the global plots.")
    parser.add_argument("--edge-color-by", default=None,
                        help="obs column used to colour PAGA arrows by lineage group "
                             "(e.g., lineage_2). Each arrow is coloured by its source "
                             "node's dominant group, letting you trace B/T/NK paths "
                             "as distinct colour trajectories.")
    parser.add_argument("--skip-plots", action="store_true",
                        help="Skip plot generation.")
    parser.add_argument("--plots-only", action="store_true",
                        help="Load a fully pre-processed h5ad (must include dpt_pseudotime, "
                             "X_umap, X_draw_graph_fa, paga, neighbors) and generate plots "
                             "only — skips all preprocessing and trajectory inference steps. "
                             "Requires --hierarchy so the finest-level column is known; "
                             "--root-label is not needed.")

    return parser.parse_args()


# --- Execution Block ---
if __name__ == "__main__":
    args = parse_args()

    # ── Plots-only mode ──────────────────────────────────────────────────────
    if args.plots_only:
        if args.hierarchy is None:
            raise SystemExit("❌ --hierarchy is required with --plots-only "
                             "(needed to identify the finest-level obs column).")
        print(f"🖼️  Plots-only mode: loading '{args.input}'...")
        analyzer = LargeScaleTrajectory(
            h5ad_path=args.input,
            hierarchy_levels=args.hierarchy,
            h5ad_type="processed",   # always load as processed in this mode
        )
        # Validate that the required results are present
        missing = []
        for key in ('dpt_pseudotime',):
            if key not in analyzer.adata.obs.columns:
                missing.append(f"obs['{key}']")
        for key in ('paga', 'neighbors'):
            if key not in analyzer.adata.uns:
                missing.append(f"uns['{key}']")
        if missing:
            raise SystemExit(
                "❌ The h5ad is missing results needed for plotting:\n  " +
                "\n  ".join(missing) +
                "\nRun the full pipeline first (without --plots-only) to generate them."
            )
        analyzer.plot_results(
            color_by=args.color_by,
            save_prefix=args.save_prefix,
            group_by=args.group_by,
            edge_color_by=args.edge_color_by,
        )
        analyzer.save_results(args.output)
        raise SystemExit(0)

    # ── Full pipeline mode ───────────────────────────────────────────────────
    if args.hierarchy is None:
        raise SystemExit("❌ --hierarchy is required for the full pipeline.")
    if args.root_label is None:
        raise SystemExit("❌ --root-label is required for the full pipeline.")

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
        analyzer.plot_results(color_by=args.color_by, save_prefix=args.save_prefix,
                              group_by=args.group_by, edge_color_by=args.edge_color_by)
        analyzer.save_results(args.output)