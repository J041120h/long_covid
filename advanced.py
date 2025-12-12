"""
Advanced CCA Analysis and Visualization
========================================
This module provides three main functionalities:
1. Plot sample labels on embedding visualizations
2. Perform CCA analysis within each subset with arrow visualization
3. Test alignment of CCA directions across subsets
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cross_decomposition import CCA
from scipy.stats import circmean, circstd
from scipy.spatial.distance import cosine
from matplotlib.patches import FancyArrowPatch
from anndata import AnnData
from typing import Optional, Tuple, Dict, List
import warnings


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def detect_embeddings(
    adata: AnnData,
    patterns: List[str] = None,
    verbose: bool = True
) -> List[str]:
    """
    Automatically detect all embedding keys in AnnData object.
    
    Parameters:
    -----------
    adata : AnnData
        Annotated data object
    patterns : list of str, optional
        List of patterns to match. Default: ['X_DR', 'X_pca', 'X_umap', 'X_tsne']
    verbose : bool
        Whether to print information
    
    Returns:
    --------
    list : List of embedding keys found
    """
    if patterns is None:
        patterns = ['X_DR', 'X_pca', 'X_umap', 'X_tsne', 'X_']
    
    embeddings = []
    
    # Check adata.obsm
    for key in adata.obsm.keys():
        if any(pattern in key for pattern in patterns):
            # Verify it's 2D+
            if adata.obsm[key].shape[1] >= 2:
                embeddings.append(key)
                if verbose:
                    print(f"  Found in obsm: {key} (shape: {adata.obsm[key].shape})")
    
    # Check adata.uns
    for key in adata.uns.keys():
        if any(pattern in key for pattern in patterns):
            data = adata.uns[key]
            if isinstance(data, pd.DataFrame):
                data = data.values
            if isinstance(data, np.ndarray) and len(data.shape) == 2 and data.shape[1] >= 2:
                embeddings.append(key)
                if verbose:
                    print(f"  Found in uns: {key} (shape: {data.shape})")
    
    embeddings = list(dict.fromkeys(embeddings))
    
    if verbose:
        print(f"\nTotal embeddings detected: {len(embeddings)}")
    
    return embeddings


def get_embedding_name(embedding_key: str) -> str:
    """
    Extract a clean name from embedding key for file naming.
    """
    name = embedding_key
    for prefix in ['X_DR_', 'X_']:
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    name = name.replace(' ', '_').replace('/', '_').replace('-', '_')
    return name if name else embedding_key


# ============================================================================
# PART 1: Sample Label Visualization
# ============================================================================

def plot_embedding_with_sample_labels(
    adata: AnnData,
    color_col: str,
    embedding_key: str = 'X_DR_proportion',
    sample_col: str = None,
    output_dir: str = None,
    figsize: Tuple[int, int] = (12, 10),
    point_size: int = 100,
    alpha: float = 0.7,
    colormap: str = 'viridis',
    font_size: int = 8,
    label_offset: Tuple[float, float] = (0.02, 0.02),
    show_legend: bool = True,
    dpi: int = 300,
    verbose: bool = True
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot embedding visualization with sample names labeled next to each point.
    """
    
    # Get embedding coordinates
    if embedding_key in adata.obsm:
        embedding = adata.obsm[embedding_key]
    elif embedding_key in adata.uns:
        embedding = adata.uns[embedding_key]
        if isinstance(embedding, pd.DataFrame):
            embedding = embedding.values
    else:
        raise KeyError(f"'{embedding_key}' not found in adata.obsm or adata.uns")
    
    x_coords = embedding[:, 0]
    y_coords = embedding[:, 1]
    
    # Get sample labels
    if sample_col is not None:
        sample_labels = adata.obs[sample_col].values
    else:
        sample_labels = adata.obs.index.values
    
    # Get color values
    color_values = adata.obs[color_col].values
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Detect data type for coloring
    try:
        numeric_values = pd.to_numeric(color_values, errors='coerce')
        is_numeric = ~pd.isna(numeric_values).all()
    except Exception:
        is_numeric = False
    
    # Plot points
    if is_numeric:
        valid_mask = pd.notna(numeric_values)
        scatter = ax.scatter(
            x_coords[valid_mask],
            y_coords[valid_mask],
            c=numeric_values[valid_mask],
            s=point_size,
            alpha=alpha,
            cmap=colormap,
            edgecolors='white',
            linewidths=1.5
        )
        
        if show_legend:
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
            cbar.set_label(color_col, rotation=270, labelpad=20, fontsize=12)
    else:
        # Categorical coloring
        unique_categories = pd.unique(color_values[pd.notna(color_values)])
        n_categories = len(unique_categories)
        colors = sns.color_palette("tab10" if n_categories <= 10 else "husl", n_categories)
        
        for i, category in enumerate(unique_categories):
            mask = color_values == category
            ax.scatter(
                x_coords[mask],
                y_coords[mask],
                c=[colors[i]],
                s=point_size,
                alpha=alpha,
                edgecolors='white',
                linewidths=1.5,
                label=str(category)
            )
        
        if show_legend and n_categories <= 15:
            ax.legend(loc='best', fontsize=10, frameon=True, title=color_col)
    
    # Add sample labels
    x_range = x_coords.max() - x_coords.min()
    y_range = y_coords.max() - y_coords.min()
    
    for x, y, label in zip(x_coords, y_coords, sample_labels):
        ax.annotate(
            str(label),
            xy=(x, y),
            xytext=(x + label_offset[0] * x_range, y + label_offset[1] * y_range),
            fontsize=font_size,
            alpha=0.8,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.6, edgecolor='gray')
        )
    
    ax.set_xlabel('Dimension 1', fontsize=14, fontweight='bold')
    ax.set_ylabel('Dimension 2', fontsize=14, fontweight='bold')
    ax.set_title(f'Embedding with Sample Labels\nColored by {color_col}', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        safe_color_col = color_col.replace(' ', '_').replace('/', '_')
        filename = f"embedding_with_labels_{safe_color_col}.png"
        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        if verbose:
            print(f"Sample label plot saved to: {output_path}")
    
    return fig, ax


# ============================================================================
# PART 2: CCA Analysis Within Each Subset
# ============================================================================

def find_best_2pcs_for_visualization(
    embedding_full: np.ndarray,
    cca_direction_full: np.ndarray,
    verbose: bool = False
) -> Tuple[np.ndarray, List[int], float]:
    """
    Find the best 2 PCs that capture the most variance of the CCA direction.
    
    NOTE (per user request): we now consistently use this "best 2 PCs" strategy
    to define the 2D visualization within each subset (no Rayleigh-based PC
    selection anywhere).
    """
    n_dims = embedding_full.shape[1]
    
    if n_dims <= 2:
        return embedding_full, list(range(n_dims)), 1.0
    
    abs_contributions = np.abs(cca_direction_full)
    best_pcs = np.argsort(abs_contributions)[-2:][::-1]
    
    direction_2d = cca_direction_full[best_pcs]
    explained_var = np.sum(direction_2d**2) / np.sum(cca_direction_full**2)
    
    embedding_2d = embedding_full[:, best_pcs]
    
    if verbose:
        print(f"    Best 2 PCs: PC{best_pcs[0]+1} and PC{best_pcs[1]+1}")
        print(f"    Explained variance: {explained_var:.2%}")
        print(f"    Contributions: PC{best_pcs[0]+1}={abs_contributions[best_pcs[0]]:.3f}, "
              f"PC{best_pcs[1]+1}={abs_contributions[best_pcs[1]]:.3f}")
    
    return embedding_2d, best_pcs.tolist(), explained_var


def cca_within_subsets(
    adata: AnnData,
    subset_col: str = 'OutSMART short number',
    time_col: str = 'month',
    embedding_key: str = 'X_DR_proportion',
    output_dir: str = None,
    figsize: Tuple[int, int] = (14, 10),
    arrow_scale: float = 1.0,
    arrow_width: float = 0.003,
    point_size: int = 100,
    alpha: float = 0.7,
    dpi: int = 300,
    use_all_dimensions: bool = True,
    vis_pc_selection: str = 'best2',   # UPDATED DEFAULT: always use best2 for viz
    verbose: bool = True
) -> Dict[str, Dict]:
    """
    Perform CCA analysis within each subset and visualize with directional arrows.
    
    Key points:
    - CCA is run in high-D (all PCs) if use_all_dimensions=True.
    - Visualization in 2D uses "best 2 PCs" by default (no Rayleigh-based selection).
    - For subsets with only 2 samples, a CCA fit may be unstable; instead, we compute
      the absolute direction between earliest and latest time points and still store
      that direction so it can be used in later alignment tests.
    """
    
    # Get embedding coordinates
    if embedding_key in adata.obsm:
        embedding = adata.obsm[embedding_key]
    elif embedding_key in adata.uns:
        embedding = adata.uns[embedding_key]
        if isinstance(embedding, pd.DataFrame):
            embedding = embedding.values
    else:
        raise KeyError(f"'{embedding_key}' not found")
    
    if subset_col not in adata.obs.columns:
        raise KeyError(f"'{subset_col}' not found in adata.obs")
    if time_col not in adata.obs.columns:
        raise KeyError(f"'{time_col}' not found in adata.obs")
    
    subsets = adata.obs[subset_col].unique()
    subsets = subsets[pd.notna(subsets)]
    
    if verbose:
        print(f"Found {len(subsets)} unique subsets in '{subset_col}'")
    
    results: Dict[str, Dict] = {}
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = sns.color_palette("tab10" if len(subsets) <= 10 else "husl", len(subsets))
    
    for subset_idx, subset_id in enumerate(subsets):
        subset_mask = adata.obs[subset_col] == subset_id
        subset_data = adata.obs[subset_mask]
        subset_embedding = embedding[subset_mask]
        
        time_values_raw = subset_data[time_col]
        time_values = pd.to_numeric(time_values_raw, errors='coerce').values
        
        valid_time_mask = ~np.isnan(time_values)
        if valid_time_mask.sum() < 2:
            if verbose:
                print(f"Subset {subset_id}: Too few valid time points ({valid_time_mask.sum()}), skipping")
            continue
        
        time_values = time_values[valid_time_mask]
        subset_embedding_valid = subset_embedding[valid_time_mask]
        sample_ids = subset_data.index.values[valid_time_mask]
        
        unique_times = np.unique(time_values)
        if len(unique_times) < 2:
            if verbose:
                print(f"Subset {subset_id}: Only one unique time point, skipping CCA")
            ax.scatter(subset_embedding[:, 0], subset_embedding[:, 1],
                       c=[colors[subset_idx]], s=point_size, alpha=alpha,
                       edgecolors='white', linewidths=1.5, label=f'{subset_id} (no CCA)')
            continue
        
        n_samples = len(sample_ids)
        
        if verbose:
            print(f"\nSubset {subset_id}:")
            print(f"  Samples: {n_samples}")
            print(f"  Time points: {sorted(unique_times)}")
        
        # Determine dimensions used for CCA
        if use_all_dimensions:
            X_cca = subset_embedding_valid
        else:
            X_cca = subset_embedding_valid[:, :2]
        n_cca_dims = X_cca.shape[1]
        if verbose:
            print(f"  Available dimensions: {n_cca_dims}")
        
        # Prepare result dict; fill in whether CCA actually ran
        subset_result: Dict = {
            'samples': sample_ids,
            'time_points': time_values,
            'unique_times': unique_times,
            'n_samples': n_samples,
            'n_dimensions': n_cca_dims,
            'cca_ok': False,             # NEW: marker if CCA fit succeeded
            'X_cca': X_cca               # NEW: store data used for CCA for permutations
        }
        
        # --- Case 1: Only 2 samples → compute absolute direction, skip CCA fit ---
        if n_samples == 2:
            if verbose:
                print("  Only 2 samples: using absolute difference vector instead of CCA.")
            
            # Sort by time to get earliest → latest
            order = np.argsort(time_values)
            early_idx = order[0]
            late_idx = order[-1]
            diff_vec = X_cca[late_idx] - X_cca[early_idx]
            norm = np.linalg.norm(diff_vec)
            if norm == 0:
                direction_full = np.zeros(n_cca_dims)
            else:
                direction_full = diff_vec / norm
            
            # Choose 2D visualization PCs
            if n_cca_dims <= 2:
                embedding_2d_vis = X_cca
                vis_pcs = list(range(n_cca_dims))
                explained_var = 1.0
                direction_2d_vis = direction_full[:2]
            else:
                if vis_pc_selection == 'first2':
                    vis_pcs = [0, 1]
                    embedding_2d_vis = X_cca[:, vis_pcs]
                    direction_2d_vis = direction_full[vis_pcs]
                    explained_var = np.sum(direction_2d_vis**2) / np.sum(direction_full**2)
                else:  # 'best2' (default)
                    embedding_2d_vis, vis_pcs, explained_var = find_best_2pcs_for_visualization(
                        X_cca, direction_full, verbose=verbose
                    )
                    direction_2d_vis = direction_full[vis_pcs]
            
            mean_position_2d = np.mean(embedding_2d_vis, axis=0)
            direction_2d_vis_normalized = direction_2d_vis / (np.linalg.norm(direction_2d_vis) + 1e-10)
            
            subset_result.update({
                'cca_score': np.nan,  # not meaningful with 2 points
                'direction_full': direction_full,
                'direction_2d': direction_2d_vis_normalized,
                'vis_pcs': vis_pcs,
                'explained_variance': explained_var
            })
            results[str(subset_id)] = subset_result
            
            # Plot points & arrow
            ax.scatter(embedding_2d_vis[:, 0], embedding_2d_vis[:, 1],
                       c=[colors[subset_idx]], s=point_size, alpha=alpha,
                       edgecolors='white', linewidths=1.5, label=f'{subset_id} (2 samples)')
            
            arrow = FancyArrowPatch(
                mean_position_2d,
                mean_position_2d + arrow_scale * direction_2d_vis_normalized,
                arrowstyle='->,head_width=0.4,head_length=0.8',
                color=colors[subset_idx],
                linewidth=3,
                alpha=0.9,
                zorder=100
            )
            ax.add_patch(arrow)
            
            text_offset = mean_position_2d + 1.2 * arrow_scale * direction_2d_vis_normalized
            label_text = '2 samples'
            ax.text(text_offset[0], text_offset[1],
                    label_text,
                    fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8),
                    ha='center', va='center')
            
            continue  # move to next subset
        
        # --- Case 2: ≥ 3 samples → run CCA normally ---
        try:
            y = time_values.reshape(-1, 1)
            if verbose:
                print(f"  CCA dimensions: {n_cca_dims}")
            
            cca = CCA(n_components=1)
            cca.fit(X_cca, y)
            direction_full = cca.x_weights_[:, 0]
            
            U, V = cca.transform(X_cca, y)
            cca_score = np.corrcoef(U[:, 0], V[:, 0])[0, 1]
            
            # Align direction with increasing time
            projections = X_cca @ direction_full
            time_direction_corr = np.corrcoef(projections, time_values)[0, 1]
            if time_direction_corr < 0:
                direction_full = -direction_full
                if verbose:
                    print("  Flipped direction to align with increasing time")
            
            # Choose 2D visualization PCs
            if n_cca_dims <= 2:
                embedding_2d_vis = X_cca
                vis_pcs = list(range(n_cca_dims))
                explained_var = 1.0
                direction_2d_vis = direction_full[:2]
                if verbose:
                    print(f"  Using all {n_cca_dims}D for visualization")
            else:
                if vis_pc_selection == 'first2':
                    vis_pcs = [0, 1]
                    embedding_2d_vis = X_cca[:, vis_pcs]
                    direction_2d_vis = direction_full[vis_pcs]
                    explained_var = np.sum(direction_2d_vis**2) / np.sum(direction_full**2)
                    if verbose:
                        print("  Visualization: first 2 PCs (PC1, PC2)")
                        print(f"  Explained variance: {explained_var:.2%}")
                else:  # 'best2' (default)
                    embedding_2d_vis, vis_pcs, explained_var = find_best_2pcs_for_visualization(
                        X_cca, direction_full, verbose=verbose
                    )
                    direction_2d_vis = direction_full[vis_pcs]
            
            mean_position_2d = np.mean(embedding_2d_vis, axis=0)
            direction_2d_vis_normalized = direction_2d_vis / (np.linalg.norm(direction_2d_vis) + 1e-10)
            
            subset_result.update({
                'cca_score': abs(cca_score),
                'direction_full': direction_full,
                'direction_2d': direction_2d_vis_normalized,
                'vis_pcs': vis_pcs,
                'explained_variance': explained_var,
                'cca_ok': True
            })
            results[str(subset_id)] = subset_result
            
            if verbose:
                print(f"  CCA score: {abs(cca_score):.4f}")
                print(f"  Full direction norm: {np.linalg.norm(direction_full):.4f}")
                if n_cca_dims > 2:
                    abs_contrib = np.abs(direction_full)
                    top3 = np.argsort(abs_contrib)[-3:][::-1]
                    print("  Top 3 PC contributions: " +
                          ", ".join([f"PC{i+1}={abs_contrib[i]:.3f}" for i in top3]))
            
            ax.scatter(embedding_2d_vis[:, 0], embedding_2d_vis[:, 1],
                       c=[colors[subset_idx]], s=point_size, alpha=alpha,
                       edgecolors='white', linewidths=1.5, label=f'{subset_id}')
            
            arrow = FancyArrowPatch(
                mean_position_2d,
                mean_position_2d + arrow_scale * direction_2d_vis_normalized,
                arrowstyle='->,head_width=0.4,head_length=0.8',
                color=colors[subset_idx],
                linewidth=3,
                alpha=0.9,
                zorder=100
            )
            ax.add_patch(arrow)
            
            text_offset = mean_position_2d + 1.2 * arrow_scale * direction_2d_vis_normalized
            if subset_result['explained_variance'] < 0.95 and n_cca_dims > 2:
                label_text = f'r={abs(cca_score):.3f}\n({subset_result["explained_variance"]:.0%})'
            else:
                label_text = f'r={abs(cca_score):.3f}'
            
            ax.text(text_offset[0], text_offset[1],
                    label_text,
                    fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8),
                    ha='center', va='center')
        
        except Exception as e:
            if verbose:
                print(f"  Error in CCA: {str(e)}")
            # Still plot the points (first two PCs)
            ax.scatter(subset_embedding_valid[:, 0], subset_embedding_valid[:, 1],
                       c=[colors[subset_idx]], s=point_size, alpha=alpha,
                       edgecolors='white', linewidths=1.5, label=f'{subset_id} (error)')
            # We do NOT store a direction here because CCA truly failed beyond the 2-sample case
            continue
    
    # Styling
    if len(results) > 0:
        first_result = next(iter(results.values()))
        vis_pcs = first_result.get('vis_pcs', [0, 1])
        n_dims = first_result.get('n_dimensions', 2)
        xlabel = f'PC{vis_pcs[0]+1}' if len(vis_pcs) > 0 else 'Dimension 1'
        ylabel = f'PC{vis_pcs[1]+1}' if len(vis_pcs) > 1 else 'Dimension 2'
        cca_info = f'CCA in {n_dims}D space'
    else:
        xlabel = 'Dimension 1'
        ylabel = 'Dimension 2'
        cca_info = 'CCA Analysis'
    
    ax.set_xlabel(xlabel, fontsize=14, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
    
    title_parts = [
        f'{cca_info}: Temporal Progression Within Each Subset',
        f'Subsets by {subset_col}, Time by {time_col}'
    ]
    ax.set_title('\n'.join(title_parts), fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=10, frameon=True, title=subset_col)
    
    plt.tight_layout()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        filename = f"cca_within_subsets_{embedding_key.replace('X_DR_', '')}.png"
        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        if verbose:
            print(f"\nCCA subset plot saved to: {output_path}")
        plt.close()
    
    return results


# ============================================================================
# PART 3: Test Alignment of CCA Directions Across Subsets
# ============================================================================

def rayleigh_test(angles):
    """
    Rayleigh test for non-uniformity of circular data.
    """
    angles = np.asarray(angles)
    n = len(angles)
    
    C = np.sum(np.cos(angles))
    S = np.sum(np.sin(angles))
    R = np.sqrt(C**2 + S**2) / n
    
    Z = n * R**2
    
    if n < 10:
        p_value = np.exp(-Z)
    else:
        p_value = np.exp(-Z) * (1 + (2*Z - Z**2) / (4*n) -
                                (24*Z - 132*Z**2 + 76*Z**3 - 9*Z**4) / (288*n**2))
    p_value = np.clip(p_value, 0, 1)
    return Z, p_value


def test_cca_direction_alignment(
    cca_results: Dict[str, Dict],
    output_dir: str = None,
    figsize: Tuple[int, int] = (14, 6),
    dpi: int = 300,
    verbose: bool = True,
    pc_pair: Optional[Tuple[int, int]] = None,   # NEW: global PC pair for angular analysis
    n_permutations: int = 500,                  # NEW: permutations for cosine similarity p-value
    random_state: Optional[int] = None          # NEW: RNG seed
) -> Dict[str, float]:
    """
    Test the consistency/alignment of CCA directions across subsets.
    
    NEW BEHAVIOR (per user request):
      1. Angles for Rayleigh test are defined in a SINGLE 2D PC plane shared by all
         subsets. You can fix this plane via `pc_pair`. If None, we choose the
         best two PCs globally (by average absolute contribution across subsets).
      2. Adds a cosine-similarity-based permutation test: shuffle time labels within
         each subset, recompute CCA directions, compute mean pairwise cosine similarity,
         and use its null distribution for a p-value.
      3. Subsets with only 2 samples (where CCA was replaced by an absolute direction)
         are still included in the alignment tests.
    
    Parameters
    ----------
    cca_results : dict
        Output from cca_within_subsets.
    pc_pair : tuple of int, optional
        Zero-based indices of the two PCs used to define the 2D direction plane
        for Rayleigh and angle comparisons. If None, choose global "best 2 PCs"
        based on average |direction_full|.
    n_permutations : int
        Number of permutations for cosine similarity null distribution.
    random_state : int or None
        Random seed for reproducibility.
    """
    
    directions_full = []
    subset_ids = []
    n_dimensions = []
    X_list = []
    time_list = []
    cca_ok_list = []
    
    # Collect data
    for subset_id, result in cca_results.items():
        if 'direction_full' not in result and 'direction' not in result:
            continue
        
        # full direction
        if 'direction_full' in result:
            dir_full = np.asarray(result['direction_full'], dtype=float)
        else:
            dir_full = np.asarray(result['direction'], dtype=float)
        
        directions_full.append(dir_full)
        subset_ids.append(subset_id)
        n_dimensions.append(len(dir_full))
        
        X_list.append(result.get('X_cca', None))
        time_list.append(np.asarray(result.get('time_points', [])))
        cca_ok_list.append(bool(result.get('cca_ok', False)))
    
    if len(directions_full) < 2:
        if verbose:
            print("Need at least 2 subsets with valid directions for alignment testing")
        return {}
    
    # Normalize directions_full for cosine similarity
    max_len = max(len(d) for d in directions_full)
    normed_full = []
    for d in directions_full:
        if len(d) < max_len:
            d_pad = np.pad(d, (0, max_len - len(d)))
        else:
            d_pad = d
        norm = np.linalg.norm(d_pad)
        if norm == 0:
            normed_full.append(np.zeros_like(d_pad))
        else:
            normed_full.append(d_pad / norm)
    normed_full = np.array(normed_full)
    
    n_subsets = len(normed_full)
    avg_n_dims = np.mean(n_dimensions)
    
    if verbose:
        print(f"\nTesting alignment of {n_subsets} directions...")
        print(f"  Direction dimensions per subset: {n_dimensions}")
        print(f"  Using {avg_n_dims:.0f}D directions for cosine similarity")
    
    # --- Choose global PC pair for angular analysis ---
    if pc_pair is None:
        # Global "best 2 PCs": highest average |direction| across subsets
        global_abs = np.mean(np.abs(normed_full), axis=0)
        best2 = np.argsort(global_abs)[-2:][::-1]
        pc_pair = (int(best2[0]), int(best2[1]))
        if verbose:
            print(f"  Auto-selected global best2 PCs for angles: PC{pc_pair[0]+1}, PC{pc_pair[1]+1}")
    else:
        if verbose:
            print(f"  Using user-specified PC pair for angles: PC{pc_pair[0]+1}, PC{pc_pair[1]+1}")
    
    # --- Build 2D vectors from the chosen PC plane for Rayleigh / angles ---
    directions_2d = []
    for d in normed_full:
        # ensure length
        if len(d) <= max(pc_pair):
            d_pad = np.pad(d, (0, max(pc_pair) + 1 - len(d)))
        else:
            d_pad = d
        vec2 = d_pad[list(pc_pair)]
        norm2 = np.linalg.norm(vec2)
        if norm2 == 0:
            directions_2d.append(np.array([0.0, 0.0]))
        else:
            directions_2d.append(vec2 / norm2)
    
    directions_2d = np.array(directions_2d)
    
    if verbose:
        print("  Using the same 2D PC plane for all subsets for angles and Rayleigh test.")
    
    # 1. Angular analysis
    angles = np.arctan2(directions_2d[:, 1], directions_2d[:, 0])
    mean_angle = circmean(angles)
    angular_std = circstd(angles)
    
    # 2. Rayleigh test
    rayleigh_stat, rayleigh_p = rayleigh_test(angles)
    
    # 3. Pairwise cosine similarity (observed)
    similarity_matrix = np.zeros((n_subsets, n_subsets))
    for i in range(n_subsets):
        for j in range(n_subsets):
            if i == j:
                similarity_matrix[i, j] = 1.0
            else:
                # cos similarity of normalized vectors is just dot product
                similarity_matrix[i, j] = float(np.dot(normed_full[i], normed_full[j]))
    
    upper_idx = np.triu_indices(n_subsets, k=1)
    mean_similarity = float(np.mean(similarity_matrix[upper_idx]))
    
    if verbose:
        print("\nAlignment Statistics (Observed):")
        print(f"  Mean angle (global PC plane): {np.degrees(mean_angle):.2f}°")
        print(f"  Angular std: {np.degrees(angular_std):.2f}°")
        print(f"  Rayleigh p-value: {rayleigh_p:.4f}")
        print(f"  Mean pairwise cosine similarity: {mean_similarity:.4f}")
    
    # 4. Cosine similarity permutation test
    null_means = []
    if n_permutations > 0:
        if verbose:
            print(f"\nRunning cosine-similarity permutation test with {n_permutations} permutations...")
        rng = np.random.default_rng(random_state)
        
        # Determine which subsets we can genuinely permute (>=3 samples and X_cca present)
        can_perm = []
        for X, t, ok in zip(X_list, time_list, cca_ok_list):
            if X is not None and len(t) >= 3 and ok:
                can_perm.append(True)
            else:
                can_perm.append(False)
        
        # Precompute original directions (normalized & padded) to reuse for subsets that
        # cannot be permuted (e.g., 2-sample subsets).
        orig_normed = normed_full.copy()
        
        for perm in range(n_permutations):
            perm_dirs = []
            for idx in range(n_subsets):
                if can_perm[idx]:
                    X = X_list[idx]
                    t = time_list[idx]
                    y_perm = rng.permutation(t).reshape(-1, 1)
                    try:
                        cca = CCA(n_components=1)
                        cca.fit(X, y_perm)
                        d_perm = cca.x_weights_[:, 0]
                        # normalize to same length (max_len)
                        if len(d_perm) < max_len:
                            d_perm = np.pad(d_perm, (0, max_len - len(d_perm)))
                        norm = np.linalg.norm(d_perm)
                        if norm == 0:
                            d_perm_norm = np.zeros_like(d_perm)
                        else:
                            d_perm_norm = d_perm / norm
                    except Exception:
                        # fallback: use original direction for this subset
                        d_perm_norm = orig_normed[idx]
                else:
                    # cannot permute (e.g., only 2 samples) → keep original direction
                    d_perm_norm = orig_normed[idx]
                
                perm_dirs.append(d_perm_norm)
            
            perm_dirs = np.stack(perm_dirs, axis=0)
            
            sim_mat_perm = np.zeros((n_subsets, n_subsets))
            for i in range(n_subsets):
                for j in range(n_subsets):
                    if i == j:
                        sim_mat_perm[i, j] = 1.0
                    else:
                        sim_mat_perm[i, j] = float(np.dot(perm_dirs[i], perm_dirs[j]))
            null_means.append(np.mean(sim_mat_perm[upper_idx]))
        
        null_means = np.array(null_means, dtype=float)
        # right-tailed p-value: observed similarity higher than null
        cosine_perm_p = float((1.0 + np.sum(null_means >= mean_similarity)) /
                              (n_permutations + 1.0))
        if verbose:
            print(f"  Cosine similarity permutation p-value: {cosine_perm_p:.4f}")
    else:
        null_means = None
        cosine_perm_p = np.nan
    
    # === Visualization ===
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Polar plot
    ax1 = plt.subplot(1, 3, 1, projection='polar')
    colors = sns.color_palette("husl", n_subsets)
    
    for i, (angle, subset_id) in enumerate(zip(angles, subset_ids)):
        ax1.plot([angle, angle], [0, 1], 'o-', color=colors[i],
                 linewidth=2, markersize=8, label=f'{subset_id}')
    
    ax1.plot([mean_angle, mean_angle], [0, 1.1], 'k--', linewidth=3,
             label=f'Mean (std={np.degrees(angular_std):.1f}°)')
    ax1.set_ylim(0, 1.2)
    ax1.set_title('CCA Direction Angles\n(global PC plane)', fontsize=12, fontweight='bold', pad=20)
    ax1.legend(loc='upper left', bbox_to_anchor=(1.1, 1.0), fontsize=8)
    
    # 2D direction vectors in chosen PC plane
    ax2 = axes[1]
    for i, (direction, subset_id) in enumerate(zip(directions_2d, subset_ids)):
        ax2.arrow(0, 0, direction[0], direction[1],
                  head_width=0.05, head_length=0.08,
                  fc=colors[i], ec=colors[i],
                  linewidth=2, alpha=0.7, label=f'{subset_id}')
    
    mean_direction = np.array([np.cos(mean_angle), np.sin(mean_angle)])
    ax2.arrow(0, 0, mean_direction[0], mean_direction[1],
              head_width=0.06, head_length=0.1,
              fc='black', ec='black',
              linewidth=3, alpha=0.8, label='Mean', linestyle='--')
    
    ax2.set_xlim(-1.2, 1.2)
    ax2.set_ylim(-1.2, 1.2)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='k', linewidth=0.5)
    ax2.axvline(x=0, color='k', linewidth=0.5)
    ax2.set_xlabel(f'PC{pc_pair[0]+1}', fontsize=10, fontweight='bold')
    ax2.set_ylabel(f'PC{pc_pair[1]+1}', fontsize=10, fontweight='bold')
    title2 = 'CCA Direction Vectors\n(global PC plane)'
    ax2.set_title(title2, fontsize=12, fontweight='bold')
    ax2.legend(fontsize=8, loc='best')
    
    # Similarity heatmap
    ax3 = axes[2]
    im = ax3.imshow(similarity_matrix, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')
    ax3.set_xticks(range(n_subsets))
    ax3.set_yticks(range(n_subsets))
    ax3.set_xticklabels(subset_ids, rotation=45, ha='right', fontsize=9)
    ax3.set_yticklabels(subset_ids, fontsize=9)
    title3 = (f'Cosine Similarity Matrix\n'
              f'(mean={mean_similarity:.3f}, p_perm={cosine_perm_p:.3f})')
    ax3.set_title(title3, fontsize=12, fontweight='bold')
    
    for i in range(n_subsets):
        for j in range(n_subsets):
            ax3.text(j, i, f'{similarity_matrix[i, j]:.2f}',
                     ha="center", va="center", color="black", fontsize=8)
    
    plt.colorbar(im, ax=ax3, label='Cosine Similarity')
    plt.tight_layout()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "cca_direction_alignment_test.png")
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        if verbose:
            print(f"\nAlignment test plot saved to: {output_path}")
        plt.close()
    
    # Stats panel
    fig2, ax = plt.subplots(figsize=(10, 6))
    
    stats_text = f"""
CCA Direction Alignment Test Results
{'='*50}

Number of subsets analyzed: {n_subsets}

Angular Statistics (global PC plane PC{pc_pair[0]+1} vs PC{pc_pair[1]+1}):
  • Mean direction: {np.degrees(mean_angle):.2f}°
  • Angular standard deviation: {np.degrees(angular_std):.2f}°
  • Circular concentration: {1 - angular_std / np.pi:.3f}

Rayleigh Test for Uniformity:
  • Statistic: {rayleigh_stat:.4f}
  • P-value: {rayleigh_p:.4f}
  • Interpretation: {'Directions are significantly ALIGNED' if rayleigh_p < 0.05 else 'Directions are NOT significantly aligned'}

Cosine Similarity (full {avg_n_dims:.0f}D, normalized):
  • Mean pairwise similarity: {mean_similarity:.4f}

Cosine Similarity Permutation Test:
  • Number of permutations: {n_permutations}
  • P-value (right-tailed): {cosine_perm_p:.4f}
  • Interpretation: {'Observed alignment is STRONGER than expected by chance' if cosine_perm_p < 0.05 else 'Observed alignment is NOT significantly stronger than chance'}

Overall Assessment:
"""
    if rayleigh_p < 0.05 and mean_similarity > 0.7:
        assessment = "  ✓ Directions are HIGHLY CONSISTENT across subsets"
    elif rayleigh_p < 0.05 or mean_similarity > 0.5:
        assessment = "  ~ Directions show MODERATE consistency across subsets"
    else:
        assessment = "  ✗ Directions are INCONSISTENT across subsets"
    
    stats_text += assessment
    
    ax.text(0.1, 0.5, stats_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    ax.axis('off')
    plt.tight_layout()
    
    if output_dir:
        output_path = os.path.join(output_dir, "cca_direction_alignment_stats.png")
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        if verbose:
            print(f"Alignment statistics saved to: {output_path}")
        plt.close()
    
    return {
        'mean_angle': mean_angle,
        'angular_std': angular_std,
        'rayleigh_statistic': rayleigh_stat,
        'rayleigh_p': rayleigh_p,
        'mean_cosine_similarity': mean_similarity,
        'pairwise_similarities': similarity_matrix,
        'subset_ids': subset_ids,
        'directions_full': normed_full,
        'directions_2d': directions_2d,
        'n_dimensions': n_dimensions,
        'avg_n_dimensions': avg_n_dims,
        'pc_pair': pc_pair,
        'cosine_perm_p': cosine_perm_p,
        'cosine_null_means': null_means
    }


# ============================================================================
# PART 4: Comprehensive Summary Report
# ============================================================================

def generate_comprehensive_summary(
    adata: AnnData,
    cca_results: Dict[str, Dict],
    alignment_stats: Dict[str, float],
    output_dir: str = None,
    embedding_keys: List[str] = None,
    subset_col: str = 'OutSMART short number',
    time_col: str = 'month',
    save_to_file: bool = True,
    print_to_console: bool = True,
    verbose: bool = True
) -> str:
    """
    Generate a comprehensive summary report of all CCA analyses.
    
    UPDATED: includes cosine-similarity permutation p-value.
    """
    import datetime
    
    report_lines: List[str] = []
    
    # Header
    report_lines.append("="*80)
    report_lines.append("COMPREHENSIVE CCA ANALYSIS SUMMARY REPORT")
    report_lines.append("="*80)
    report_lines.append(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # 1. Data overview
    report_lines.append("="*80)
    report_lines.append("1. DATA OVERVIEW")
    report_lines.append("="*80)
    report_lines.append("")
    
    report_lines.append(f"Dataset Shape: {adata.shape[0]} samples × {adata.shape[1]} features")
    
    if subset_col in adata.obs.columns:
        subsets = adata.obs[subset_col].unique()
        subsets = subsets[pd.notna(subsets)]
        report_lines.append(f"Number of Subsets ({subset_col}): {len(subsets)}")
        report_lines.append(f"  Subset IDs: {', '.join([str(s) for s in sorted(subsets)])}")
    
    if time_col in adata.obs.columns:
        time_vals = adata.obs[time_col].unique()
        time_vals = time_vals[pd.notna(time_vals)]
        report_lines.append(f"Time Points ({time_col}): {', '.join([str(t) for t in sorted(time_vals)])}")
    
    if embedding_keys:
        report_lines.append(f"\nEmbeddings Analyzed: {len(embedding_keys)}")
        for emb_key in embedding_keys:
            if emb_key in adata.obsm:
                n_dims = adata.obsm[emb_key].shape[1]
            elif emb_key in adata.uns:
                emb_data = adata.uns[emb_key]
                n_dims = emb_data.shape[1] if hasattr(emb_data, 'shape') else "unknown"
            else:
                n_dims = "unknown"
            report_lines.append(f"  • {emb_key}: {n_dims}D")
    report_lines.append("")
    
    # 2. CCA within subsets
    report_lines.append("="*80)
    report_lines.append("2. CCA WITHIN SUBSETS - DETAILED RESULTS")
    report_lines.append("="*80)
    report_lines.append("")
    
    if cca_results:
        scores = [r['cca_score'] for r in cca_results.values() if 'cca_score' in r and not np.isnan(r['cca_score'])]
        if scores:
            report_lines.append("Summary Statistics:")
            report_lines.append(f"  • Mean CCA Score: {np.mean(scores):.4f}")
            report_lines.append(f"  • Median CCA Score: {np.median(scores):.4f}")
            report_lines.append(f"  • Std Dev: {np.std(scores):.4f}")
            report_lines.append(f"  • Range: [{np.min(scores):.4f}, {np.max(scores):.4f}]")
            report_lines.append(f"  • Number of Subsets with CCA: {len(scores)}")
            report_lines.append("")
        
        n_dims_list = [r.get('n_dimensions', 2) for r in cca_results.values()]
        avg_dims = np.mean(n_dims_list) if n_dims_list else 2
        is_multid = avg_dims > 2
        
        if is_multid:
            report_lines.append(f"Analysis Type: Multi-Dimensional CCA (avg {avg_dims:.0f}D)")
            explained_vars = [r.get('explained_variance', 1.0) for r in cca_results.values()
                              if 'explained_variance' in r]
            if explained_vars:
                report_lines.append(f"  • Mean 2D Explained Variance: {np.mean(explained_vars):.1%}")
                report_lines.append(f"  • Min Explained Variance: {np.min(explained_vars):.1%}")
            report_lines.append("")
        else:
            report_lines.append("Analysis Type: Standard 2D CCA")
            report_lines.append("")
        
        report_lines.append("Per-Subset Results:")
        report_lines.append("-" * 80)
        
        sorted_subsets = sorted(
            cca_results.items(),
            key=lambda x: (0.0 if np.isnan(x[1].get('cca_score', np.nan)) else x[1].get('cca_score', 0.0)),
            reverse=True
        )
        
        for subset_id, data in sorted_subsets:
            report_lines.append(f"\n{subset_id}:")
            score = data.get('cca_score', np.nan)
            if np.isnan(score):
                report_lines.append(f"  CCA Score: NA (e.g. 2 samples)")
            else:
                report_lines.append(f"  CCA Score: {score:.4f}")
            
            if np.isnan(score):
                interp = "N/A (insufficient samples for robust CCA; direction from absolute difference)"
            elif score > 0.8:
                interp = "EXCELLENT - Very strong temporal progression"
            elif score > 0.6:
                interp = "GOOD - Clear temporal progression"
            elif score > 0.4:
                interp = "MODERATE - Some temporal signal"
            else:
                interp = "WEAK - Limited temporal progression"
            report_lines.append(f"  Interpretation: {interp}")
            
            n_samples = data.get('n_samples', 0)
            report_lines.append(f"  Samples: {n_samples}")
            
            if 'unique_times' in data:
                times = sorted(data['unique_times'])
                report_lines.append(f"  Time Points: {', '.join([str(t) for t in times])}")
            
            if 'n_dimensions' in data:
                n_dims = data['n_dimensions']
                report_lines.append(f"  CCA Dimensions: {n_dims}D")
            
            if 'vis_pcs' in data:
                vis_pcs = data['vis_pcs']
                pc_labels = [f"PC{i+1}" for i in vis_pcs]
                report_lines.append(f"  Visualization PCs: {' vs '.join(pc_labels)}")
            
            if 'explained_variance' in data:
                exp_var = data['explained_variance']
                report_lines.append(f"  2D Explained Variance: {exp_var:.1%}")
                if exp_var < 0.7:
                    report_lines.append("    ⚠ Note: Low explained variance - complex multi-D progression")
            
            if 'direction_2d' in data:
                dir_2d = data['direction_2d']
                angle_deg = np.degrees(np.arctan2(dir_2d[1], dir_2d[0]))
                report_lines.append(f"  Direction Angle (viz 2D): {angle_deg:.1f}°")
            
            if 'direction_full' in data and len(data['direction_full']) > 2:
                dir_full = np.asarray(data['direction_full'])
                abs_contrib = np.abs(dir_full)
                top_3_idx = np.argsort(abs_contrib)[-3:][::-1]
                top_3_str = ', '.join([f"PC{i+1}={abs_contrib[i]:.3f}" for i in top_3_idx])
                report_lines.append(f"  Top 3 PC Contributions: {top_3_str}")
        report_lines.append("")
    else:
        report_lines.append("No CCA results available.")
        report_lines.append("")
    
    # 3. Direction alignment
    report_lines.append("="*80)
    report_lines.append("3. DIRECTION ALIGNMENT ACROSS SUBSETS")
    report_lines.append("="*80)
    report_lines.append("")
    
    if alignment_stats:
        rayleigh_p = alignment_stats.get('rayleigh_p', 1.0)
        rayleigh_stat = alignment_stats.get('rayleigh_statistic', 0.0)
        mean_angle = alignment_stats.get('mean_angle', 0.0)
        angular_std = alignment_stats.get('angular_std', 0.0)
        mean_sim = alignment_stats.get('mean_cosine_similarity', 0.0)
        avg_dims = alignment_stats.get('avg_n_dimensions', 2.0)
        pc_pair = alignment_stats.get('pc_pair', (0, 1))
        cosine_perm_p = alignment_stats.get('cosine_perm_p', np.nan)
        
        report_lines.append("Statistical Tests:")
        report_lines.append("")
        
        report_lines.append("Rayleigh Test for Uniformity:")
        report_lines.append(f"  • Statistic (Z): {rayleigh_stat:.4f}")
        report_lines.append(f"  • P-value: {rayleigh_p:.4f}")
        if rayleigh_p < 0.001:
            report_lines.append("  • Result: HIGHLY SIGNIFICANT (p < 0.001) ***")
            report_lines.append("  • Interpretation: Directions are STRONGLY aligned")
        elif rayleigh_p < 0.01:
            report_lines.append("  • Result: Very significant (p < 0.01) **")
            report_lines.append("  • Interpretation: Directions are well aligned")
        elif rayleigh_p < 0.05:
            report_lines.append("  • Result: Significant (p < 0.05) *")
            report_lines.append("  • Interpretation: Directions are aligned")
        else:
            report_lines.append("  • Result: Not significant (p ≥ 0.05)")
            report_lines.append("  • Interpretation: Directions are dispersed/inconsistent")
        report_lines.append("")
        
        report_lines.append(f"Angular Statistics (PC{pc_pair[0]+1} vs PC{pc_pair[1]+1}):")
        report_lines.append(f"  • Mean Direction: {np.degrees(mean_angle):.2f}°")
        report_lines.append(f"  • Angular Std Dev: {np.degrees(angular_std):.2f}°")
        if np.degrees(angular_std) < 30:
            report_lines.append("  • Clustering: TIGHT (< 30°)")
        elif np.degrees(angular_std) < 60:
            report_lines.append("  • Clustering: MODERATE (30-60°)")
        else:
            report_lines.append("  • Clustering: LOOSE (> 60°)")
        report_lines.append("")
        
        report_lines.append(f"Cosine Similarity (full {avg_dims:.0f}D):")
        report_lines.append(f"  • Mean Pairwise Similarity: {mean_sim:.4f}")
        report_lines.append("")
        
        report_lines.append("Cosine Similarity Permutation Test:")
        report_lines.append(f"  • P-value (right-tailed): {cosine_perm_p:.4f}")
        if cosine_perm_p < 0.001:
            report_lines.append("  • Result: HIGHLY SIGNIFICANT (p < 0.001) ***")
            report_lines.append("  • Interpretation: Observed alignment is much stronger than expected by chance")
        elif cosine_perm_p < 0.01:
            report_lines.append("  • Result: Very significant (p < 0.01) **")
            report_lines.append("  • Interpretation: Observed alignment is stronger than expected by chance")
        elif cosine_perm_p < 0.05:
            report_lines.append("  • Result: Significant (p < 0.05) *")
            report_lines.append("  • Interpretation: Observed alignment is moderately stronger than chance")
        else:
            report_lines.append("  • Result: Not significant (p ≥ 0.05)")
            report_lines.append("  • Interpretation: Observed alignment is not clearly stronger than chance")
        report_lines.append("")
        
        if 'pairwise_similarities' in alignment_stats:
            sim_matrix = alignment_stats['pairwise_similarities']
            n = len(sim_matrix)
            triu_idx = np.triu_indices(n, k=1)
            similarities = sim_matrix[triu_idx]
            
            report_lines.append("Pairwise Similarity Details:")
            report_lines.append(f"  • Number of Pairs: {len(similarities)}")
            report_lines.append(f"  • Min Similarity: {np.min(similarities):.4f}")
            report_lines.append(f"  • Max Similarity: {np.max(similarities):.4f}")
            report_lines.append(f"  • Std Dev: {np.std(similarities):.4f}")
            
            subset_ids = alignment_stats.get('subset_ids', [])
            if subset_ids:
                pair_indices = [(i, j) for i in range(n) for j in range(i+1, n)]
                max_idx = np.argmax(similarities)
                most_sim_pair = pair_indices[max_idx]
                min_idx = np.argmin(similarities)
                least_sim_pair = pair_indices[min_idx]
                report_lines.append(
                    f"  • Most Similar Pair: {subset_ids[most_sim_pair[0]]} & "
                    f"{subset_ids[most_sim_pair[1]]} (similarity = {similarities[max_idx]:.4f})"
                )
                report_lines.append(
                    f"  • Least Similar Pair: {subset_ids[least_sim_pair[0]]} & "
                    f"{subset_ids[least_sim_pair[1]]} (similarity = {similarities[min_idx]:.4f})"
                )
            report_lines.append("")
    else:
        report_lines.append("No alignment statistics available.")
        report_lines.append("")
    
    # 4. Overall assessment
    report_lines.append("="*80)
    report_lines.append("4. OVERALL ASSESSMENT")
    report_lines.append("="*80)
    report_lines.append("")
    
    if cca_results and alignment_stats:
        scores = [r['cca_score'] for r in cca_results.values()
                  if 'cca_score' in r and not np.isnan(r['cca_score'])]
        mean_score = np.mean(scores) if scores else 0.0
        rayleigh_p = alignment_stats.get('rayleigh_p', 1.0)
        mean_sim = alignment_stats.get('mean_cosine_similarity', 0.0)
        cosine_perm_p = alignment_stats.get('cosine_perm_p', 1.0)
        
        report_lines.append("Key Findings:")
        report_lines.append("")
        
        if mean_score > 0.7:
            report_lines.append("✓ STRONG temporal progression within subsets")
            report_lines.append(f"  Average CCA score of {mean_score:.3f} indicates clear time-dependent patterns")
        elif mean_score > 0.5:
            report_lines.append("~ MODERATE temporal progression within subsets")
            report_lines.append(f"  Average CCA score of {mean_score:.3f} shows some time-dependent patterns")
        else:
            report_lines.append("✗ WEAK temporal progression within subsets")
            report_lines.append(f"  Average CCA score of {mean_score:.3f} suggests limited time-dependent patterns")
        report_lines.append("")
        
        if rayleigh_p < 0.05 and mean_sim > 0.7 and cosine_perm_p < 0.05:
            report_lines.append("✓ HIGHLY CONSISTENT directions across subsets")
            report_lines.append(f"  Rayleigh p = {rayleigh_p:.4f}, similarity = {mean_sim:.3f}, perm p = {cosine_perm_p:.3f}")
            report_lines.append("  → Common biological progression pattern with strong evidence beyond chance")
        elif (rayleigh_p < 0.05 or mean_sim > 0.5) and cosine_perm_p < 0.1:
            report_lines.append("~ MODERATELY CONSISTENT directions across subsets")
            report_lines.append(f"  Rayleigh p = {rayleigh_p:.4f}, similarity = {mean_sim:.3f}, perm p = {cosine_perm_p:.3f}")
            report_lines.append("  → Some shared patterns (mild to moderate evidence above chance)")
        else:
            report_lines.append("✗ INCONSISTENT or weakly supported direction alignment")
            report_lines.append(f"  Rayleigh p = {rayleigh_p:.4f}, similarity = {mean_sim:.3f}, perm p = {cosine_perm_p:.3f}")
            report_lines.append("  → Heterogeneous progression or insufficient evidence beyond random alignments")
        report_lines.append("")
        
        report_lines.append("Biological Interpretation:")
        report_lines.append("")
        
        if mean_score > 0.7 and rayleigh_p < 0.05 and mean_sim > 0.7 and cosine_perm_p < 0.05:
            report_lines.append("  The analysis reveals a STRONG and CONSISTENT temporal trajectory")
            report_lines.append("  across subsets, with strong statistical support from both directional")
            report_lines.append("  and permutation-based tests. This suggests:")
            report_lines.append("    • A common biological progression mechanism")
            report_lines.append("    • Reproducible temporal dynamics across patients")
            report_lines.append("    • High confidence for downstream trajectory-based analyses")
        elif mean_score > 0.5 and (rayleigh_p < 0.05 or mean_sim > 0.5):
            report_lines.append("  The analysis shows MODERATE temporal progression with SOME consistency")
            report_lines.append("  across subsets. This suggests:")
            report_lines.append("    • Shared progression patterns with meaningful patient-to-patient variation")
            report_lines.append("    • Potential for stratified trajectory analyses")
        else:
            report_lines.append("  The analysis reveals WEAK or INCONSISTENT temporal patterns")
            report_lines.append("  across subsets. This suggests:")
            report_lines.append("    • Highly heterogeneous responses")
            report_lines.append("    • Potential unmodeled confounders or alternative progression axes")
        report_lines.append("")
        
        report_lines.append("Recommendations:")
        report_lines.append("")
        
        if mean_score > 0.7 and cosine_perm_p < 0.05:
            report_lines.append("  1. Proceed with trajectory-based analyses (pseudotime, DE, etc.) using the common direction.")
            report_lines.append("  2. Investigate molecular drivers of the shared trajectory.")
        elif mean_score > 0.5:
            report_lines.append("  1. Examine low-CCA-score subsets individually.")
            report_lines.append("  2. Consider subset-specific or condition-specific trajectories.")
        else:
            report_lines.append("  1. Re-evaluate preprocessing and batch correction.")
            report_lines.append("  2. Explore alternative embeddings or temporal markers.")
        report_lines.append("")
    
    # 5. Output files
    if output_dir:
        report_lines.append("="*80)
        report_lines.append("5. GENERATED OUTPUT FILES")
        report_lines.append("="*80)
        report_lines.append("")
        report_lines.append(f"Output Directory: {output_dir}")
        report_lines.append("")
        report_lines.append("Visualization Files:")
        if embedding_keys:
            for emb_key in embedding_keys:
                emb_name = emb_key.replace('X_DR_', '')
                report_lines.append(f"  • embedding_with_labels_*.png - Sample label plots")
                report_lines.append(f"  • cca_within_subsets_{emb_name}.png - CCA arrows visualization")
        report_lines.append("  • cca_direction_alignment_test.png - Alignment & similarity visualization")
        report_lines.append("  • cca_direction_alignment_stats.png - Alignment statistics panel")
        report_lines.append("")
        report_lines.append("Data Files:")
        if embedding_keys:
            for emb_key in embedding_keys:
                emb_name = emb_key.replace('X_DR_', '')
                report_lines.append(f"  • cca_results_{emb_name}.csv - Detailed CCA results (if exported by user)")
        report_lines.append("  • alignment_summary.csv - Alignment statistics (if exported by user)")
        report_lines.append("  • comprehensive_summary.txt - This report")
        report_lines.append("")
    
    report_lines.append("="*80)
    report_lines.append("END OF REPORT")
    report_lines.append("="*80)
    
    report_text = '\n'.join(report_lines)
    
    if save_to_file and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        report_path = os.path.join(output_dir, 'comprehensive_summary.txt')
        with open(report_path, 'w') as f:
            f.write(report_text)
        if verbose:
            print(f"\n[INFO] Comprehensive summary saved to: {report_path}")
    
    if print_to_console:
        print("\n" + report_text)
    
    return report_text


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    import scanpy as sc
    
    print("="*70)
    print("Advanced CCA Analysis Pipeline")
    print("="*70)
    
    # Example usage – adjust paths as needed
    adata = sc.read_h5ad(
        '/dcs07/hongkai/data/harry/result/long_covid_batch_removal/rna/pseudobulk/pseudobulk_sample.h5ad'
    )
    
    output_dir = '/dcs07/hongkai/data/harry/result/long_covid_batch_removal/proportion_results_hongkai'
    os.makedirs(output_dir, exist_ok=True)
    
    # PART 1: Sample labels
    print("\n" + "="*70)
    print("PART 1: Plotting Embeddings with Sample Labels")
    print("="*70)
    
    fig1, ax1 = plot_embedding_with_sample_labels(
        adata=adata,
        color_col='month',
        embedding_key='X_DR_proportion',
        output_dir=output_dir,
        verbose=True
    )
    
    # PART 2: CCA within subsets
    print("\n" + "="*70)
    print("PART 2: CCA Analysis Within Each Subset")
    print("="*70)
    
    cca_results = cca_within_subsets(
        adata=adata,
        subset_col='OutSMART short number',
        time_col='month',
        embedding_key='X_DR_proportion',
        output_dir=output_dir,
        use_all_dimensions=True,
        vis_pc_selection='best2',
        verbose=True
    )
    
    # PART 3: Alignment across subsets
    print("\n" + "="*70)
    print("PART 3: Testing CCA Direction Alignment Across Subsets")
    print("="*70)
    
    alignment_stats = test_cca_direction_alignment(
        cca_results=cca_results,
        output_dir=output_dir,
        verbose=True,
        pc_pair= [2,3],           # None → auto-select global best2 PCs
        n_permutations=1000,
        random_state=1
    )
    
    summary_text = generate_comprehensive_summary(
        adata=adata,
        cca_results=cca_results,
        alignment_stats=alignment_stats,
        output_dir=output_dir,
        embedding_keys=['X_DR_proportion'],
        subset_col='OutSMART short number',
        time_col='month',
        save_to_file=True,
        print_to_console=True
    )
    
    print("\n" + "="*70)
    print("Analysis Complete!")
    print("="*70)
    print(f"All results saved to: {output_dir}")