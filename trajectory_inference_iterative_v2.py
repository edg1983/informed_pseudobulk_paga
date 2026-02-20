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
from copy import deepcopy

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
        self.finest_level = hierarchy_levels[-1]  # celltype_3
        self.adata = None
        # Per-lineage AnnData subsets stored after split-lineage trajectory
        # Keys are lineage names, values are the subset AnnData objects
        self.lineage_adatas = {}
        if h5ad_type == "processed":
            print(f"📂 Loading pre-processed AnnData from {self.h5ad_path}...")
            self.adata = sc.read_h5ad(self.h5ad_path)

    # ──────────────────────────────────────────────────────────────────────────
    # Data loading helpers (unchanged from v1)
    # ──────────────────────────────────────────────────────────────────────────

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
            if sparse.issparse(chunk):
                chunk = chunk.toarray()
            ipca.partial_fit(chunk)

        print("📉 Transforming data...")
        X_pca = np.zeros((n_cells, n_components), dtype=np.float32)
        for i in range(0, n_cells, batch_size):
            end = min(i + batch_size, n_cells)
            chunk = adata_backed[i:end].X
            if sparse.issparse(chunk):
                chunk = chunk.toarray()
            X_pca[i:end] = ipca.transform(chunk)

        obs = adata_backed.obs[self.hierarchy_levels].copy()
        del adata_backed
        gc.collect()

        self._create_lightweight_adata(X_pca, obs, save_pca_path)

    # ──────────────────────────────────────────────────────────────────────────
    # GPU / CPU helpers
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _ensure_cpu_adata(adata):
        """Ensures all matrices in the given AnnData are numpy/scipy (CPU)."""
        if not GPU_AVAILABLE:
            return
        if hasattr(adata.X, 'get'):
            adata.X = adata.X.get()
        for key in list(adata.obsm.keys()):
            if hasattr(adata.obsm[key], 'get'):
                adata.obsm[key] = adata.obsm[key].get()
        if hasattr(adata, 'obsp'):
            for key in list(adata.obsp.keys()):
                if hasattr(adata.obsp[key], 'get'):
                    adata.obsp[key] = adata.obsp[key].get()
        if 'paga' in adata.uns:
            for key in list(adata.uns['paga'].keys()):
                if hasattr(adata.uns['paga'][key], 'get'):
                    adata.uns['paga'][key] = adata.uns['paga'][key].get()

    def _ensure_cpu(self):
        """Ensures all matrices on self.adata are numpy/scipy arrays (CPU)."""
        self._ensure_cpu_adata(self.adata)

    # ──────────────────────────────────────────────────────────────────────────
    # Hierarchical PAGA (unchanged from v1)
    # ──────────────────────────────────────────────────────────────────────────

    def compute_hierarchical_paga(self, root_label, thresholds=0.05, target_adata=None):
        """
        Computes PAGA iteratively from coarse to fine.
        Severely penalizes or removes edges between fine clusters if their
        coarse parent lineages are not biologically connected.

        Args:
            root_label (str): The label of the root progenitor cells.
            thresholds (float or dict): A single threshold for all levels, or a dictionary
                                        mapping level names to specific thresholds.
            target_adata (AnnData, optional): If provided, compute PAGA on this
                                              object instead of self.adata. Useful
                                              for per-lineage subsets.
        """
        adata = target_adata if target_adata is not None else self.adata

        print("\n🌳 Starting Top-Down Hierarchical PAGA Masking...")
        self._ensure_cpu_adata(adata)

        allowed_parent_edges = None
        prev_level = None

        # Determine which hierarchy levels are actually present in this subset
        active_levels = [
            lvl for lvl in self.hierarchy_levels
            if lvl in adata.obs.columns and adata.obs[lvl].nunique() > 0
        ]

        for level in active_levels:
            # Determine threshold for this specific level
            if isinstance(thresholds, dict):
                current_thresh = thresholds.get(level, 0.05)
            else:
                current_thresh = thresholds

            print(f"   ► Computing PAGA for level: '{level}' (Threshold: {current_thresh})")
            sc.tl.paga(adata, groups=level)

            paga_conn = adata.uns['paga']['connectivities'].toarray()
            categories = adata.obs[level].cat.categories

            # --- MASKING LOGIC ---
            finest = active_levels[-1]
            if level == active_levels[0]:
                # Force star topology from the root at the very first level
                root_mask = adata.obs[finest] == root_label
                if not root_mask.any():
                    for l in reversed(active_levels):
                        if (adata.obs[l] == root_label).any():
                            root_mask = adata.obs[l] == root_label
                            break
                if root_mask.any():
                    top_root_category = adata.obs[root_mask][level].mode()[0]
                    print(f"     👑 Enforcing root hub: only allowing edges connected to '{top_root_category}'")

                    masked_top_edges = 0
                    for i, cat_i in enumerate(categories):
                        for j, cat_j in enumerate(categories):
                            if i == j:
                                continue
                            if cat_i != top_root_category and cat_j != top_root_category:
                                if paga_conn[i, j] > 0:
                                    paga_conn[i, j] = 0.0
                                    masked_top_edges += 1
                    if masked_top_edges > 0:
                        print(f"     ✂️ Severed {masked_top_edges} cross-lineage edges at the root level.")
                        adata.uns['paga']['connectivities'] = sparse.csr_matrix(paga_conn)
                else:
                    print(f"     ⚠️  Root label '{root_label}' not found at first level — skipping star enforcement.")

            elif allowed_parent_edges is not None:
                mapping = {}
                for i, cat in enumerate(categories):
                    parent_cat = adata.obs[adata.obs[level] == cat][prev_level].mode()[0]
                    mapping[i] = parent_cat

                masked_edges = 0
                for i in range(len(categories)):
                    for j in range(len(categories)):
                        if i == j:
                            continue
                        parent_i = mapping[i]
                        parent_j = mapping[j]
                        if not allowed_parent_edges.get((parent_i, parent_j), False):
                            if paga_conn[i, j] > 0:
                                paga_conn[i, j] = 0.0
                                masked_edges += 1

                print(f"     ✂️ Severed {masked_edges} cross-lineage short-circuits based on '{prev_level}' rules.")
                adata.uns['paga']['connectivities'] = sparse.csr_matrix(paga_conn)

            # --- PREPARE NEXT LEVEL RULES ---
            allowed_parent_edges = {}
            for i, cat_i in enumerate(categories):
                for j, cat_j in enumerate(categories):
                    if i == j or paga_conn[i, j] >= current_thresh:
                        allowed_parent_edges[(cat_i, cat_j)] = True

            prev_level = level

        print(f"✅ Hierarchical PAGA complete. Final backbone rests on '{active_levels[-1]}'.\n")

    # ──────────────────────────────────────────────────────────────────────────
    # Global trajectory (preserved from v1, now with n_dcs)
    # ──────────────────────────────────────────────────────────────────────────

    def run_gpu_trajectory(self, root_label="HSPC", n_neighbors=30, n_dcs=10):
        """Moves data to GPU, computes Neighbors, runs Hierarchical PAGA, and DPT.

        Args:
            root_label: Label of the root progenitor cluster.
            n_neighbors: k for the nearest-neighbour graph.
            n_dcs: Number of diffusion components used by ``sc.tl.dpt``.
        """
        if self.adata is None:
            raise ValueError("Run preprocess or load embeddings first.")

        lib = rsc if GPU_AVAILABLE else sc
        print("🚀 Starting Trajectory Inference (global)...")

        print(f"🔗 Computing Neighbor Graph (k={n_neighbors})...")
        lib.pp.neighbors(self.adata, n_neighbors=n_neighbors, n_pcs=50, use_rep='X_pca')

        print("🗺️  Computing Diffusion Maps...")
        lib.tl.diffmap(self.adata)

        print("⏬ Moving data to CPU for Scanpy PAGA and DPT...")
        self._ensure_cpu()

        # Hierarchical PAGA
        level_thresholds = {
            self.hierarchy_levels[0]: 0.01,
            self.hierarchy_levels[-1]: 0.05,
        }
        if len(self.hierarchy_levels) > 1:
            level_thresholds[self.hierarchy_levels[1]] = 0.01
        if len(self.hierarchy_levels) > 2:
            level_thresholds[self.hierarchy_levels[2]] = 0.03
        if len(self.hierarchy_levels) > 3:
            level_thresholds[self.hierarchy_levels[3]] = 0.05
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

        print(f"⏳ Computing Diffusion Pseudotime (n_dcs={n_dcs})...")
        sc.tl.dpt(self.adata, n_dcs=n_dcs)
        print("✅ Global trajectory inference complete.")

    # ──────────────────────────────────────────────────────────────────────────
    # *** NEW: Split-Lineage Trajectory ***
    # ──────────────────────────────────────────────────────────────────────────

    def run_split_lineage_trajectory(self, root_label="HSPC",
                                     split_by=None,
                                     n_neighbors=30, n_dcs=10,
                                     paga_thresholds=0.05,
                                     store_lineage_adata=False):
        """
        Compute trajectory inference independently for each major lineage.

        Instead of running a single global DPT that is dominated by the
        highest-variance lineage, we:
          1. Subset *self.adata* by each unique value of ``split_by``.
          2. **Recompute** neighbours **and** diffusion maps per subset.
          3. Run PAGA + DPT per lineage.
          4. Min-max scale each lineage's pseudotime to [0, 1].
          5. Stitch results back into *self.adata.obs* as::
               - ``dpt_pseudotime_{lineage}`` — raw per-lineage pseudotime
               - ``dpt_pseudotime_{lineage}_scaled`` — min-max [0, 1]
               - ``dpt_pseudotime_stitched`` — union of all scaled values

        Per-lineage PAGA connectivities are saved in::
            self.adata.uns[f'paga_{lineage}']

        The per-lineage AnnData objects (with their own neighbours,
        diffusion maps, and PAGA) are also saved in ``self.lineage_adatas``
        for downstream inspection.

        Args:
            root_label: Label of the root progenitor cluster (searched
                        across all hierarchy levels).
            split_by: obs column that defines the major lineage branches
                      (e.g. ``lineage_1``). If ``None``, defaults to
                      ``self.hierarchy_levels[0]``.
            n_neighbors: k for the nearest-neighbor graph in each subset.
            n_dcs: Number of diffusion components for ``sc.tl.dpt``.
            paga_thresholds: Passed to ``compute_hierarchical_paga``.
            store_lineage_adata: If ``True``, keep the full per-lineage
                                 AnnData objects in ``self.lineage_adatas``.
                                 Set to ``False`` (default) to save memory.
        """
        if self.adata is None:
            raise ValueError("Run preprocess or load embeddings first.")

        if split_by is None:
            split_by = self.hierarchy_levels[0]
        if split_by not in self.adata.obs.columns:
            raise ValueError(f"split_by column '{split_by}' not found in adata.obs.")

        lib = rsc if GPU_AVAILABLE else sc

        # ── 0. Global neighbours + diffmap (needed for global UMAP later) ────
        print("🚀 Starting Split-Lineage Trajectory Inference...")
        print(f"   Split column: '{split_by}'")
        print(f"🔗 Computing global Neighbor Graph (k={n_neighbors})...")
        lib.pp.neighbors(self.adata, n_neighbors=n_neighbors, n_pcs=50, use_rep='X_pca')
        print("🗺️  Computing global Diffusion Maps...")
        lib.tl.diffmap(self.adata)
        self._ensure_cpu()

        # Also run global PAGA so the global graph is available for plotting
        print("   Computing global hierarchical PAGA (for visualisation)...")
        level_thresholds = {
            self.hierarchy_levels[0]: 0.01,
            self.hierarchy_levels[-1]: 0.05,
        }
        if len(self.hierarchy_levels) > 1:
            level_thresholds[self.hierarchy_levels[1]] = 0.01
        if len(self.hierarchy_levels) > 2:
            level_thresholds[self.hierarchy_levels[2]] = 0.03
        if len(self.hierarchy_levels) > 3:
            level_thresholds[self.hierarchy_levels[3]] = 0.05
        self.compute_hierarchical_paga(root_label=root_label,
                                       thresholds=level_thresholds)

        # ── 1. Identify the root (progenitor) group ─────────────────────────
        lineage_col = self.adata.obs[split_by]
        lineages = lineage_col.cat.categories if hasattr(lineage_col, 'cat') \
            else sorted(lineage_col.unique())

        # Find which split_by group contains the root_label cells
        root_cell_mask = self.adata.obs[self.finest_level] == root_label
        if not root_cell_mask.any():
            for lvl in reversed(self.hierarchy_levels):
                if (self.adata.obs[lvl] == root_label).any():
                    root_cell_mask = self.adata.obs[lvl] == root_label
                    break
        if not root_cell_mask.any():
            raise ValueError(
                f"Root label '{root_label}' not found in any hierarchy level. "
                f"Cannot determine the progenitor group for split-lineage mode."
            )

        root_group = self.adata.obs.loc[root_cell_mask, split_by].mode()[0]
        root_group_mask = (lineage_col == root_group).values
        root_group_idx = self.adata.obs_names[root_group_mask]
        n_root = int(root_group_mask.sum())
        print(f"   👑 Root group identified: '{root_group}' ({n_root} cells)")
        print(f"      These progenitor cells will be prepended to every "
              f"lineage subset so DPT can anchor at the origin.")

        # Non-root lineages to iterate over
        iter_lineages = [lin for lin in lineages if lin != root_group]
        print(f"   Lineages to process independently: {iter_lineages}")
        print(f"   (Root group '{root_group}' is shared — not processed alone)")

        # Prepare columns in the master adata
        self.adata.obs['dpt_pseudotime_stitched'] = np.nan
        for lin in lineages:
            self.adata.obs[f'dpt_pseudotime_{lin}'] = np.nan
            self.adata.obs[f'dpt_pseudotime_{lin}_scaled'] = np.nan

        self.lineage_adatas = {}

        # ── 2. Iterate over non-root lineages ────────────────────────────────
        for lin in iter_lineages:
            print(f"\n{'='*60}")
            print(f"🔬 Processing lineage: {lin}  (+ {n_root} root cells from '{root_group}')")
            print(f"{'='*60}")

            lin_mask = (lineage_col == lin).values
            n_lin_cells = int(lin_mask.sum())
            print(f"   {n_lin_cells} lineage-specific cells + {n_root} root cells "
                  f"= {n_lin_cells + n_root} total in subset.")

            if n_lin_cells < 10:
                print(f"   ⚠️  Too few lineage cells ({n_lin_cells}), skipping.")
                continue

            # ── 2a. Subset: lineage cells + root group cells ──────────────────
            combined_mask = lin_mask | root_group_mask
            adata_sub = self.adata[combined_mask].copy()

            # Track which cells in the subset are lineage-specific vs root
            is_root_in_sub = adata_sub.obs_names.isin(root_group_idx)
            is_lineage_in_sub = ~is_root_in_sub
            lineage_sub_idx = adata_sub.obs_names[is_lineage_in_sub]

            # Re-categorise so categories only contain values present in subset
            for lvl in self.hierarchy_levels:
                if lvl in adata_sub.obs.columns:
                    adata_sub.obs[lvl] = adata_sub.obs[lvl].cat.remove_unused_categories()

            # ── 2b. Recompute neighbours + diffmap on combined subset ─────────
            print(f"   🔗 Computing Neighbor Graph for '{lin}' (k={n_neighbors})...")
            lib.pp.neighbors(adata_sub, n_neighbors=n_neighbors, n_pcs=50, use_rep='X_pca')

            print(f"   🗺️  Computing Diffusion Maps for '{lin}'...")
            lib.tl.diffmap(adata_sub)
            self._ensure_cpu_adata(adata_sub)

            # ── 2c. Hierarchical PAGA on combined subset ──────────────────────
            print(f"   Running hierarchical PAGA for '{lin}'...")
            self.compute_hierarchical_paga(
                root_label=root_label,
                thresholds=paga_thresholds,
                target_adata=adata_sub,
            )

            # ── 2d. Set root and run DPT ──────────────────────────────────────
            # Root cells are guaranteed to be in the subset
            root_mask_sub = adata_sub.obs[self.finest_level] == root_label
            if not root_mask_sub.any():
                for lvl in reversed(self.hierarchy_levels):
                    if lvl in adata_sub.obs.columns and (adata_sub.obs[lvl] == root_label).any():
                        root_mask_sub = adata_sub.obs[lvl] == root_label
                        break

            if not root_mask_sub.any():
                # Extremely unlikely given we prepended root cells, but be safe
                print(f"   ⚠️  Root label '{root_label}' not found in combined "
                      f"subset — using cell closest to PCA centroid as root.")
                pca = adata_sub.obsm['X_pca']
                centroid = pca.mean(axis=0)
                dists = np.linalg.norm(pca - centroid, axis=1)
                adata_sub.uns['iroot'] = int(np.argmin(dists))
            else:
                flat_idx = np.where(root_mask_sub.values)[0]
                adata_sub.uns['iroot'] = int(flat_idx[0])

            print(f"   📍 Root index for '{lin}': {adata_sub.uns['iroot']}")
            print(f"   ⏳ Computing DPT for '{lin}' (n_dcs={n_dcs})...")
            sc.tl.dpt(adata_sub, n_dcs=n_dcs)
            print(f"   ✅ DPT complete for '{lin}'.")

            # ── 2e. Min-max scale pseudotime [0, 1] ──────────────────────────
            pt = adata_sub.obs['dpt_pseudotime'].values.copy()
            valid = np.isfinite(pt)
            if valid.any():
                pt_min = pt[valid].min()
                pt_max = pt[valid].max()
                denom = pt_max - pt_min if pt_max > pt_min else 1.0
                pt_scaled = np.where(valid, (pt - pt_min) / denom, np.nan)
            else:
                pt_scaled = pt

            adata_sub.obs['dpt_pseudotime_scaled'] = pt_scaled

            # ── 2f. Write back into master adata ──────────────────────────────
            # Write per-lineage raw & scaled PT for ALL cells in the subset
            # (including root cells — they get a value in every lineage column)
            all_sub_idx = adata_sub.obs_names
            self.adata.obs.loc[all_sub_idx, f'dpt_pseudotime_{lin}'] = pt
            self.adata.obs.loc[all_sub_idx, f'dpt_pseudotime_{lin}_scaled'] = pt_scaled

            # Stitched column: only write for lineage-specific cells
            # (root cells are handled separately below)
            self.adata.obs.loc[lineage_sub_idx, 'dpt_pseudotime_stitched'] = \
                pt_scaled[is_lineage_in_sub.values]

            # ── 2g. Store per-lineage PAGA in master uns ──────────────────────
            if 'paga' in adata_sub.uns:
                self.adata.uns[f'paga_{lin}'] = deepcopy(adata_sub.uns['paga'])
                self.adata.uns[f'paga_{lin}']['groups'] = \
                    list(adata_sub.obs[self.finest_level].cat.categories)

            # Optionally save the full per-lineage AnnData for downstream use
            if store_lineage_adata:
                self.lineage_adatas[lin] = adata_sub
            else:
                del adata_sub
                gc.collect()

        # ── 3. Assign stitched pseudotime for root group cells ────────────────
        # Root cells participated in every lineage's DPT.  Their stitched
        # value is set to 0 (they are the origin for all trajectories).
        # We also create a per-lineage column for the root group itself
        # using the mean of all lineage-specific scaled values.
        print(f"\n   📌 Setting stitched pseudotime for root group "
              f"'{root_group}' ({n_root} cells) to 0.0")
        self.adata.obs.loc[root_group_idx, 'dpt_pseudotime_stitched'] = 0.0

        # Compute the root-group's own column as the mean across lineages
        lin_scaled_cols = [f'dpt_pseudotime_{lin}_scaled' for lin in iter_lineages
                          if f'dpt_pseudotime_{lin}_scaled' in self.adata.obs.columns]
        if lin_scaled_cols:
            root_mean_pt = self.adata.obs.loc[root_group_idx, lin_scaled_cols].mean(axis=1)
            self.adata.obs.loc[root_group_idx, f'dpt_pseudotime_{root_group}'] = \
                root_mean_pt.values
            self.adata.obs.loc[root_group_idx, f'dpt_pseudotime_{root_group}_scaled'] = \
                root_mean_pt.values

        # ── 4. Also compute a "global" DPT for reference ─────────────────────
        print(f"\n{'='*60}")
        print("📊 Computing global DPT (for reference comparison)...")
        print(f"{'='*60}")
        root_mask = self.adata.obs[self.finest_level] == root_label
        if not root_mask.any():
            for level in reversed(self.hierarchy_levels):
                if (self.adata.obs[level] == root_label).any():
                    root_mask = self.adata.obs[level] == root_label
                    break

        if root_mask.any():
            flat_indices = np.where(root_mask.values)[0]
            self.adata.uns['iroot'] = int(flat_indices[0])
            sc.tl.dpt(self.adata, n_dcs=n_dcs)
            print("   ✅ Global DPT stored in obs['dpt_pseudotime'].")
        else:
            print(f"   ⚠️  Root label '{root_label}' not found globally — "
                  f"skipping global DPT.")

        processed_lineages = list(iter_lineages)
        print(f"\n✅ Split-lineage trajectory complete.")
        print(f"   Root group: '{root_group}' (prepended to every subset)")
        print(f"   Lineages processed: {processed_lineages}")
        print(f"   Per-lineage columns: dpt_pseudotime_<lineage>, "
              f"dpt_pseudotime_<lineage>_scaled")
        print(f"   Stitched column:     dpt_pseudotime_stitched")
        print(f"   Global column:       dpt_pseudotime")
        if store_lineage_adata:
            print(f"   Per-lineage AnnData objects stored in self.lineage_adatas "
                  f"({len(self.lineage_adatas)} objects).")
        else:
            print(f"   Per-lineage AnnData objects discarded (--store-lineage-adata not set).")

    # ──────────────────────────────────────────────────────────────────────────
    # Validation
    # ──────────────────────────────────────────────────────────────────────────

    def enforce_hierarchy_constraint(self):
        print("🔧 Validating hierarchy alignment...")
        # Use stitched pseudotime if available, else fall back to global
        dpt_col = 'dpt_pseudotime_stitched' if 'dpt_pseudotime_stitched' in self.adata.obs.columns \
            else 'dpt_pseudotime'
        summary = self.adata.obs.groupby(self.hierarchy_levels, observed=True)[dpt_col].mean().reset_index()
        summary = summary.sort_values(dpt_col)
        print(summary.head(10))
        return summary

    # ──────────────────────────────────────────────────────────────────────────
    # Visualization layout
    # ──────────────────────────────────────────────────────────────────────────

    def compute_visualization_layout(self):
        print("🎨 Computing visualization layouts...")
        lib = rsc if GPU_AVAILABLE else sc

        if 'X_umap' not in self.adata.obsm:
            print("   Computing UMAP...")
            lib.tl.umap(self.adata)

        print("   Computing ForceAtlas2 layout (PAGA-initialized)...")
        if GPU_AVAILABLE:
            from scanpy.tools._utils import get_init_pos_from_paga
            print("   Pre-computing PAGA init positions for GPU layout...")
            if 'pos' not in self.adata.uns.get('paga', {}):
                sc.pl.paga(self.adata, show=False)
            init_coords = get_init_pos_from_paga(
                self.adata, random_state=0, neighbors_key='neighbors'
            )
            lib.tl.draw_graph(self.adata, init_pos=init_coords)
        else:
            sc.tl.draw_graph(self.adata, init_pos='paga')
        self._ensure_cpu()

    # ──────────────────────────────────────────────────────────────────────────
    # Plotting helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _is_numeric_col(self, col):
        """Returns True if obs column ``col`` holds numeric values."""
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
                        relevant_node_indices=None,
                        pt_values=None):
        """
        Render one UMAP pair (celltype left, color_by + directed PAGA right).

        ``pt_values`` overrides the default pseudotime data for colouring
        (used to pass lineage-specific pseudotime for per-group plots).
        """
        bg_kw = dict(s=0.2, linewidths=0, rasterized=True)
        fg_kw = dict(s=0.5, linewidths=0, rasterized=True)

        for ax in (ax_ct, ax_pt):
            ax.set_xlabel("UMAP 1")
            ax.set_ylabel("UMAP 2")
            ax.set_aspect('equal')

        def _scatter_with_mask(ax, colors, cmap_vals=None, cmap='viridis'):
            if mask is None:
                if cmap_vals is not None:
                    sc_obj = ax.scatter(umap_coords[:, 0], umap_coords[:, 1],
                                        c=cmap_vals, cmap=cmap, **bg_kw)
                    return sc_obj
                ax.scatter(umap_coords[:, 0], umap_coords[:, 1],
                           c=colors, **bg_kw)
                return None
            ax.scatter(umap_coords[~mask, 0], umap_coords[~mask, 1],
                       c='#d0d0d0', **bg_kw)
            if cmap_vals is not None:
                sc_obj = ax.scatter(umap_coords[mask, 0], umap_coords[mask, 1],
                                    c=cmap_vals[mask], cmap=cmap, **fg_kw)
                return sc_obj
            ax.scatter(umap_coords[mask, 0], umap_coords[mask, 1],
                       c=colors[mask], **fg_kw)
            return None

        # ── Left panel: celltype ─────────────────────────────────────────────
        _scatter_with_mask(ax_ct, cell_colors)
        ax_ct.legend(handles=handles, fontsize=5, markerscale=3,
                     loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0)
        ax_ct.set_title(f"UMAP — {self.finest_level}{title_suffix}", fontsize=10)

        # ── Right panel: color_by + directed PAGA arrows ─────────────────────
        # Allow caller to override pseudotime values for colouring
        if pt_values is not None and self._is_numeric_col(color_by):
            cmap_vals = pt_values
        elif pt_values is not None and color_by.startswith('dpt_pseudotime'):
            cmap_vals = pt_values
        else:
            cmap_vals = self.adata.obs[color_by].values if self._is_numeric_col(color_by) else None

        sc_obj = _scatter_with_mask(ax_pt, cell_colors, cmap_vals=cmap_vals)
        if sc_obj is not None:
            plt.colorbar(sc_obj, ax=ax_pt, label=color_by, shrink=0.6)

        # Directed PAGA arrows
        n_g = len(groups)
        max_w = paga_conn.max() if paga_conn.max() > 0 else 1.0
        group_list = list(groups)

        for i in range(n_g):
            for j in range(i + 1, n_g):
                w = paga_conn[i, j]
                if w < 0.01:
                    continue

                if relevant_node_indices is not None and \
                        i not in relevant_node_indices and \
                        j not in relevant_node_indices:
                    continue

                lw = (w / max_w) * 3

                pt_i = mean_pt_by_group.get(group_list[i], 0.0) if mean_pt_by_group else 0.0
                pt_j = mean_pt_by_group.get(group_list[j], 0.0) if mean_pt_by_group else 1.0
                src, dst = (i, j) if pt_i <= pt_j else (j, i)

                if edge_node_lineage and edge_palette:
                    src_lineage = edge_node_lineage.get(group_list[src])
                    edge_color = edge_palette.get(src_lineage, '#333333')
                else:
                    edge_color = '#333333'

                ax_pt.annotate(
                    "",
                    xy=paga_pos[dst], xytext=paga_pos[src],
                    xycoords='data', textcoords='data',
                    arrowprops=dict(
                        arrowstyle=f"->, head_width={lw * 0.25:.2f}, head_length={lw * 0.18:.2f}",
                        color=edge_color,
                        lw=lw,
                        connectionstyle='arc3,rad=0.08',
                        alpha=0.85,
                    ),
                    zorder=6,
                )

        if edge_legend_handles:
            ax_pt.legend(handles=edge_legend_handles, fontsize=6,
                         title='Edge lineage', title_fontsize=7,
                         loc='lower left', bbox_to_anchor=(1.01, 0), borderaxespad=0)

        ax_pt.set_title(f"UMAP — {color_by} + directed PAGA{title_suffix}", fontsize=10)

    def _draw_branching_pair(self, ax_b1, ax_b2,
                             pseudotime, fa2_coords, cell_colors, color_by,
                             handles, mask=None, title_suffix='',
                             pt_values=None):
        """
        Render one branching pair (celltype left, color_by right).
        X = DPT pseudotime, Y = FA2 Y (branch-divergence axis).

        ``pt_values`` overrides pseudotime for both the x-axis and
        colour mapping, allowing lineage-specific pseudotime in per-group plots.
        """
        bg_kw = dict(s=0.2, linewidths=0, rasterized=True)
        fg_kw = dict(s=0.5, linewidths=0, rasterized=True)

        # Use overridden pseudotime if provided
        pt = pt_values if pt_values is not None else pseudotime

        for ax in (ax_b1, ax_b2):
            ax.set_xlabel("Pseudotime (DPT)")
            ax.set_ylabel("FA2 Y (branch axis)")

        fa2_y = fa2_coords[:, 1]
        order = np.argsort(pt)

        def _scatter_branch(ax, colors, cmap_vals=None, cmap='viridis'):
            if mask is None:
                if cmap_vals is not None:
                    sc_obj = ax.scatter(pt[order], fa2_y[order],
                                        c=cmap_vals[order], cmap=cmap, **bg_kw)
                    return sc_obj
                ax.scatter(pt[order], fa2_y[order],
                           c=colors[order], **bg_kw)
                return None
            # background
            bg = ~mask
            ax.scatter(pt[bg], fa2_y[bg], c='#d0d0d0', **bg_kw)
            # foreground
            fg_order = np.argsort(pt[mask])
            fg_pt = pt[mask][fg_order]
            fg_fa2 = fa2_y[mask][fg_order]
            if cmap_vals is not None:
                sc_obj = ax.scatter(fg_pt, fg_fa2,
                                    c=cmap_vals[mask][fg_order], cmap=cmap, **fg_kw)
                return sc_obj
            ax.scatter(fg_pt, fg_fa2, c=colors[mask][fg_order], **fg_kw)
            return None

        # Left — celltype
        _scatter_branch(ax_b1, cell_colors)
        ax_b1.legend(handles=handles, fontsize=5, markerscale=3,
                     loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0)
        ax_b1.set_title(f"Branching — {self.finest_level}{title_suffix}", fontsize=10)

        # Right — color_by
        if pt_values is not None and color_by.startswith('dpt_pseudotime'):
            cmap_vals = pt_values
        else:
            cmap_vals = self.adata.obs[color_by].values if self._is_numeric_col(color_by) else None
        sc_obj = _scatter_branch(ax_b2, cell_colors, cmap_vals=cmap_vals)
        if sc_obj is not None:
            plt.colorbar(sc_obj, ax=ax_b2, label=color_by, shrink=0.6)
        ax_b2.set_title(f"Branching — {color_by}{title_suffix}", fontsize=10)

    # ──────────────────────────────────────────────────────────────────────────
    # Main plotting entry point
    # ──────────────────────────────────────────────────────────────────────────

    def plot_results(self, color_by='dpt_pseudotime', save_prefix='trajectory',
                     group_by=None, edge_color_by=None,
                     pseudotime_mode='global'):
        """
        Generate output figures.

        Args:
            color_by: obs column to use for the colour axis on the right
                      panels. When ``pseudotime_mode='lineage'`` and
                      ``color_by`` starts with ``dpt_pseudotime``, the
                      per-lineage pseudotime column is used automatically
                      in per-group plots.
            save_prefix: Filename prefix.
            group_by: obs column for per-group plots.
            edge_color_by: obs column for arrow colouring.
            pseudotime_mode: ``'global'`` — use the stitched/global
                             pseudotime for all plots (default).
                             ``'lineage'`` — for per-group plots use the
                             lineage-specific pseudotime columns
                             (``dpt_pseudotime_{grp}_scaled``).

        Global (all cells):
          <save_prefix>_umap.png      — 2-panel UMAP
          <save_prefix>_branching.png — 2-panel pseudotime branching

        Per-group:
          <save_prefix>_<G>_umap.png + _branching.png
        """
        if 'X_draw_graph_fa' not in self.adata.obsm or 'X_umap' not in self.adata.obsm:
            self.compute_visualization_layout()

        self._ensure_cpu()
        from matplotlib.patches import Patch

        umap_coords = self.adata.obsm['X_umap']
        fa2_coords = self.adata.obsm['X_draw_graph_fa']

        # Determine the primary pseudotime column for global plots
        if 'dpt_pseudotime_stitched' in self.adata.obs.columns and \
                self.adata.obs['dpt_pseudotime_stitched'].notna().any():
            global_pt_col = 'dpt_pseudotime_stitched'
        else:
            global_pt_col = 'dpt_pseudotime'

        pseudotime = self.adata.obs[global_pt_col].values
        celltype_col = self.adata.obs[self.finest_level]

        groups = celltype_col.cat.categories
        n_groups = len(groups)
        palette = dict(zip(groups, plt.cm.tab20.colors[:n_groups] if n_groups <= 20
                           else [plt.cm.hsv(i / n_groups) for i in range(n_groups)]))
        cell_colors = np.array([palette[g] for g in celltype_col])
        handles = [Patch(color=palette[g], label=g) for g in groups]

        paga_pos = np.array([umap_coords[celltype_col == g].mean(axis=0) for g in groups])
        paga_conn = self.adata.uns['paga']['connectivities'].toarray()

        mean_pt_by_group = {
            g: float(np.nanmean(pseudotime[celltype_col == g]))
            for g in groups
        }

        # ── Edge colouring ────────────────────────────────────────────────────
        edge_node_lineage = None
        edge_palette = None
        edge_legend_handles = None

        if edge_color_by is not None and edge_color_by in self.adata.obs.columns:
            lineage_col_ec = self.adata.obs[edge_color_by]
            lineages = lineage_col_ec.cat.categories if hasattr(lineage_col_ec, 'cat') \
                else lineage_col_ec.unique()
            n_lin = len(lineages)
            lin_colors = plt.cm.Set1.colors[:n_lin] if n_lin <= 9 \
                else [plt.cm.tab10(i / n_lin) for i in range(n_lin)]
            edge_palette = dict(zip(lineages, lin_colors))

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

        def _save_pair(umap_path, branch_path, mask=None, title_suffix='',
                       pt_override=None, color_by_override=None,
                       paga_conn_override=None, paga_pos_override=None,
                       groups_override=None, mean_pt_override=None):
            """Save a UMAP + branching figure pair.

            ``pt_override``: per-cell array of pseudotime values override.
            ``color_by_override``: label for the colour axis.
            ``paga_conn_override``: per-lineage PAGA connectivities.
            ``paga_pos_override`` / ``groups_override`` / ``mean_pt_override``:
                allow fully replacing the PAGA graph for per-lineage views.
            """
            _groups = groups_override if groups_override is not None else groups
            _paga_conn = paga_conn_override if paga_conn_override is not None else paga_conn
            _paga_pos = paga_pos_override if paga_pos_override is not None else paga_pos
            _color_by = color_by_override if color_by_override is not None else color_by
            _mean_pt = mean_pt_override if mean_pt_override is not None else mean_pt_by_group
            _pt = pt_override if pt_override is not None else pseudotime

            relevant_node_indices = None
            if mask is not None:
                relevant_node_indices = {
                    i for i, g in enumerate(_groups)
                    if np.any(mask & (celltype_col == g).values)
                }

            active_handles = handles
            if mask is not None:
                present_groups = set(celltype_col[mask].unique())
                active_handles = [h for h in handles if h.get_label() in present_groups]

            fig, (ax_ct, ax_pt) = plt.subplots(1, 2, figsize=(18, 7))
            fig.subplots_adjust(wspace=0.35)
            self._draw_umap_pair(
                ax_ct, ax_pt, umap_coords, cell_colors, _color_by,
                palette, _groups, active_handles, _paga_pos, _paga_conn,
                mask=mask, title_suffix=title_suffix,
                mean_pt_by_group=_mean_pt,
                edge_node_lineage=edge_node_lineage,
                edge_palette=edge_palette,
                edge_legend_handles=edge_legend_handles,
                relevant_node_indices=relevant_node_indices,
                pt_values=pt_override,
            )
            fig.savefig(umap_path, dpi=150, bbox_inches='tight')
            plt.close(fig)

            fig2, (ax_b1, ax_b2) = plt.subplots(1, 2, figsize=(18, 6))
            fig2.subplots_adjust(wspace=0.3)
            self._draw_branching_pair(ax_b1, ax_b2, _pt, fa2_coords,
                                      cell_colors, _color_by, active_handles,
                                      mask=mask, title_suffix=title_suffix,
                                      pt_values=pt_override)
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
                print(f"⚠️  --group-by column '{group_by}' not found in obs — "
                      f"skipping per-group plots.")
            else:
                group_col = self.adata.obs[group_by]
                subgroups = group_col.cat.categories if hasattr(group_col, 'cat') \
                    else group_col.unique()
                print(f"📊 Plotting per-group views for {len(subgroups)} groups "
                      f"in '{group_by}' (pseudotime_mode={pseudotime_mode})...")

                for grp in subgroups:
                    mask = (group_col == grp).values
                    safe_name = str(grp).replace('/', '_').replace(' ', '_')
                    suffix = f" [{grp}]"

                    pt_override = None
                    color_by_override = None
                    paga_conn_lin = None
                    paga_pos_lin = None
                    groups_lin = None
                    mean_pt_lin = None

                    if pseudotime_mode == 'lineage':
                        # Try to use lineage-specific pseudotime
                        lin_pt_col = f'dpt_pseudotime_{grp}_scaled'
                        if lin_pt_col in self.adata.obs.columns:
                            pt_override = self.adata.obs[lin_pt_col].values
                            color_by_override = lin_pt_col
                            suffix = f" [{grp}] (lineage PT)"

                            # Use per-lineage PAGA if available
                            paga_key = f'paga_{grp}'
                            if paga_key in self.adata.uns:
                                paga_info = self.adata.uns[paga_key]
                                paga_conn_lin = paga_info['connectivities'].toarray()
                                lin_groups_list = paga_info.get('groups', None)
                                if lin_groups_list is not None:
                                    groups_lin = pd.Index(lin_groups_list)
                                    paga_pos_lin = np.array([
                                        umap_coords[celltype_col == g].mean(axis=0)
                                        if (celltype_col == g).any()
                                        else np.array([0.0, 0.0])
                                        for g in groups_lin
                                    ])
                                    # Compute mean pseudotime per node using lineage PT
                                    mean_pt_lin = {}
                                    for g in groups_lin:
                                        g_mask = (celltype_col == g).values
                                        vals = pt_override[g_mask]
                                        valid = np.isfinite(vals)
                                        mean_pt_lin[g] = float(np.nanmean(vals)) if valid.any() else 0.0
                        else:
                            print(f"   ⚠️  Column '{lin_pt_col}' not found — "
                                  f"falling back to global PT for '{grp}'.")

                    _save_pair(
                        f"{save_prefix}_{safe_name}_umap.png",
                        f"{save_prefix}_{safe_name}_branching.png",
                        mask=mask, title_suffix=suffix,
                        pt_override=pt_override,
                        color_by_override=color_by_override,
                        paga_conn_override=paga_conn_lin,
                        paga_pos_override=paga_pos_lin,
                        groups_override=groups_lin,
                        mean_pt_override=mean_pt_lin,
                    )

        print(f"✅ {len(saved)} plot files saved with prefix '{save_prefix}'.")

    # ──────────────────────────────────────────────────────────────────────────
    # Save
    # ──────────────────────────────────────────────────────────────────────────

    def save_results(self, output_path):
        print(f"💾 Saving results to {output_path}...")
        self.adata.write(output_path)


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Large-scale trajectory inference with split-lineage DPT and optional GPU acceleration.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  # Split-lineage mode (recommended)
  %(prog)s -i data.h5ad -o results.h5ad \\
           --hierarchy lineage_1 lineage_2 celltype_1 \\
           --root-label HSC_MPP --split-by lineage_1

  # Global mode (single DPT, like v1)
  %(prog)s -i data.h5ad -o results.h5ad \\
           --hierarchy lineage_1 lineage_2 celltype_1 \\
           --root-label HSC_MPP --mode global

  # Pre-computed embeddings
  %(prog)s -i data.h5ad -o results.h5ad \\
           --hierarchy lineage_1 celltype_1 \\
           --root-label HSC_MPP --embedding-key X_scVI --split-by lineage_1

  # Plots only with lineage-specific pseudotime
  %(prog)s -i processed.h5ad --plots-only \\
           --hierarchy lineage_1 celltype_1 \\
           --group-by lineage_1 --pseudotime-mode lineage
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

    # Mode selection
    parser.add_argument("--mode", choices=["split", "global"], default="split",
                        help="Trajectory inference mode. "
                             "'split': per-lineage DPT (recommended, fixes variance issues). "
                             "'global': single global DPT (like v1). (default: split)")
    parser.add_argument("--split-by", default=None,
                        help="obs column to split lineages by (e.g., lineage_1). "
                             "Defaults to the first hierarchy level when --mode=split.")

    # Embedding / PCA options
    emb_group = parser.add_mutually_exclusive_group()
    emb_group.add_argument("--embedding-key", default=None,
                           help="Load pre-computed embeddings from .obsm[KEY] (e.g., X_scVI). "
                                "Skips PCA preprocessing.")
    emb_group.add_argument("--embedding-file", default=None,
                           help="Path to a TSV/CSV file with pre-computed embeddings. "
                                "Skips PCA preprocessing.")
    parser.add_argument("--n-components", type=int, default=50,
                        help="Number of PCA components for incremental PCA (default: 50).")
    parser.add_argument("--n-neighbors", type=int, default=30,
                        help="Number of neighbors for the neighbor graph (default: 30).")
    parser.add_argument("--n-dcs", type=int, default=10,
                        help="Number of diffusion components for sc.tl.dpt (default: 10).")

    # Visualization options
    parser.add_argument("--color-by", default="dpt_pseudotime",
                        help="Column or key to color plots by (default: dpt_pseudotime). "
                             "For split-lineage mode this defaults to the stitched "
                             "pseudotime in global plots.")
    parser.add_argument("--save-prefix", default="trajectory",
                        help="Filename prefix for saved plots (default: trajectory).")
    parser.add_argument("--group-by", default=None,
                        help="obs column used to split per-group plots (e.g., lineage_1).")
    parser.add_argument("--edge-color-by", default=None,
                        help="obs column used to colour PAGA arrows by lineage group.")
    parser.add_argument("--pseudotime-mode", choices=["global", "lineage"], default="global",
                        help="For per-group plots: 'global' uses the stitched (or global) "
                             "pseudotime; 'lineage' uses the per-lineage-specific "
                             "pseudotime columns and per-lineage PAGA. (default: global)")
    parser.add_argument("--skip-plots", action="store_true",
                        help="Skip plot generation.")
    parser.add_argument("--plots-only", action="store_true",
                        help="Load a pre-processed h5ad and generate plots only — "
                             "skips all preprocessing and trajectory inference.")

    # PAGA thresholds
    parser.add_argument("--paga-threshold", type=float, default=0.05,
                        help="Default PAGA connectivity threshold (default: 0.05).")

    # Memory / storage
    parser.add_argument("--store-lineage-adata", action="store_true",
                        help="Keep per-lineage AnnData objects in memory (and save "
                             "them alongside the main output). Disabled by default "
                             "to conserve memory and storage.")

    return parser.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# Main execution
# ══════════════════════════════════════════════════════════════════════════════

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
            h5ad_type="processed",
        )
        # Validate required results
        missing = []
        # Check for at least one pseudotime column
        has_pt = any(
            c in analyzer.adata.obs.columns
            for c in ('dpt_pseudotime', 'dpt_pseudotime_stitched')
        )
        if not has_pt:
            missing.append("obs['dpt_pseudotime'] or obs['dpt_pseudotime_stitched']")
        for key in ('paga', 'neighbors'):
            if key not in analyzer.adata.uns:
                missing.append(f"uns['{key}']")
        if missing:
            raise SystemExit(
                "❌ The h5ad is missing results needed for plotting:\n  " +
                "\n  ".join(missing) +
                "\nRun the full pipeline first (without --plots-only) to generate them."
            )
        # Resolve color_by for split-lineage results
        effective_color_by = args.color_by
        if effective_color_by == 'dpt_pseudotime' and \
                'dpt_pseudotime_stitched' in analyzer.adata.obs.columns:
            effective_color_by = 'dpt_pseudotime_stitched'

        analyzer.plot_results(
            color_by=effective_color_by,
            save_prefix=args.save_prefix,
            group_by=args.group_by,
            edge_color_by=args.edge_color_by,
            pseudotime_mode=args.pseudotime_mode,
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
        h5ad_type=args.h5ad_type,
    )

    # 1. Load Data
    if args.embedding_key:
        analyzer.load_precomputed_embeddings(source='h5ad', key=args.embedding_key)
    elif args.embedding_file:
        analyzer.load_precomputed_embeddings(
            source='file', tsv_path=args.embedding_file,
            save_path=f"{args.save_prefix}_lightweight.h5ad",
        )
    elif analyzer.adata is None:
        analyzer.preprocess_cpu_incremental(n_components=args.n_components)

    # 2. Run Inference
    if args.mode == 'split':
        analyzer.run_split_lineage_trajectory(
            root_label=args.root_label,
            split_by=args.split_by,
            n_neighbors=args.n_neighbors,
            n_dcs=args.n_dcs,
            paga_thresholds=args.paga_threshold,
            store_lineage_adata=args.store_lineage_adata,
        )
    else:
        analyzer.run_gpu_trajectory(
            root_label=args.root_label,
            n_neighbors=args.n_neighbors,
            n_dcs=args.n_dcs,
        )

    # 3. Check & Save
    analyzer.enforce_hierarchy_constraint()
    analyzer.save_results(args.output)

    # 4. Visualize
    if not args.skip_plots:
        # Resolve color_by
        effective_color_by = args.color_by
        if args.mode == 'split' and effective_color_by == 'dpt_pseudotime':
            effective_color_by = 'dpt_pseudotime_stitched'

        analyzer.plot_results(
            color_by=effective_color_by,
            save_prefix=args.save_prefix,
            group_by=args.group_by,
            edge_color_by=args.edge_color_by,
            pseudotime_mode=args.pseudotime_mode,
        )
        analyzer.save_results(args.output)
