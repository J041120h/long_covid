#!/usr/bin/env python3
"""
Clustering Test Framework for Long COVID Pseudobulk Data
Evaluates how well unsupervised clustering matches patient metadata categories.

Usage:
    python clustering_test_framework.py

Paths are configured for the JHPCE server.
"""

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from sklearn.preprocessing import LabelEncoder
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.stats import chi2_contingency
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
import os

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

H5AD_PATH = "/dcs07/hongkai/data/harry/result/long_covid/rna/pseudobulk/pseudobulk_sample.h5ad"
META_PATH = "/dcl01/hongkai/data/data/hjiang/Data/long_covid/sample_meta.csv"
OUTPUT_DIR = "/dcs07/hongkai/data/harry/result/long_covid/clustering_evaluation/2"

# Number of permutations for significance testing
N_PERMUTATIONS = 1000

# Random seed for reproducibility
RANDOM_SEED = 42


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data(h5ad_path, meta_path):
    """Load h5ad file and metadata CSV."""
    print(f"Loading h5ad from: {h5ad_path}")
    adata = sc.read_h5ad(h5ad_path)
    print(f"  Shape: {adata.shape}")
    print(f"  obsm keys: {list(adata.obsm.keys())}")
    
    print(f"\nLoading metadata from: {meta_path}")
    meta = pd.read_csv(meta_path, index_col='sample')
    print(f"  Samples in meta: {len(meta)}")
    
    # Ensure alignment
    common_samples = adata.obs_names.intersection(meta.index)
    print(f"  Common samples: {len(common_samples)}")
    
    adata = adata[common_samples, :]
    meta = meta.loc[common_samples]
    
    return adata, meta


def get_embeddings(adata):
    """Extract embeddings from AnnData object."""
    embeddings = {}
    
    if 'X_DR_expression' in adata.obsm:
        embeddings['expression'] = adata.obsm['X_DR_expression']
        print(f"Expression embedding shape: {embeddings['expression'].shape}")
    
    if 'X_DR_proportion' in adata.obsm:
        embeddings['proportion'] = adata.obsm['X_DR_proportion']
        print(f"Proportion embedding shape: {embeddings['proportion'].shape}")
    
    return embeddings


# ============================================================================
# METADATA CATEGORY PREPARATION
# ============================================================================

def prepare_metadata_categories(meta):
    """Prepare metadata categories for clustering comparison."""
    categories = {}
    
    # Month: 3 categories (1, 3, 6)
    categories['month'] = {
        'labels': meta['month'].values,
        'n_clusters': meta['month'].nunique(),
        'description': 'Time point (months: 1, 3, 6)'
    }
    
    # Sex: 2 categories
    categories['Sex'] = {
        'labels': meta['Sex'].values,
        'n_clusters': meta['Sex'].nunique(),
        'description': 'Biological sex (Male/Female)'
    }
    
    # LC/Recovered: 2 categories
    categories['LC/Recovered'] = {
        'labels': meta['LC/Recovered'].values,
        'n_clusters': meta['LC/Recovered'].nunique(),
        'description': 'Long COVID status'
    }
    
    # BMI category: 2 categories
    categories['BMI category'] = {
        'labels': meta['BMI category'].values,
        'n_clusters': meta['BMI category'].nunique(),
        'description': 'BMI category (Normal/Elevated)'
    }
    
    # Age: bin into groups (tertiles)
    age = meta['Age at enrollment'].values
    try:
        age_bins = pd.qcut(age, q=3, labels=['Young', 'Middle', 'Old'], duplicates='drop')
        n_age_clusters = len(age_bins.unique())
    except ValueError:
        # If qcut fails due to duplicates, use cut instead
        age_bins = pd.cut(age, bins=3, labels=['Young', 'Middle', 'Old'])
        n_age_clusters = 3
    
    # Convert to numpy array for consistency
    age_labels = np.array(age_bins)
    
    categories['Age (binned)'] = {
        'labels': age_labels,
        'n_clusters': n_age_clusters,
        'description': 'Age tertiles'
    }
    
    # Patient ID (OutSMART short number): many categories
    categories['OutSMART ID'] = {
        'labels': meta['OutSMART short number'].values,
        'n_clusters': meta['OutSMART short number'].nunique(),
        'description': 'Patient identifier (16 patients)'
    }
    
    return categories


# ============================================================================
# CLUSTERING METHODS
# ============================================================================

def run_kmeans(X, n_clusters, n_runs=10, random_state=42):
    """Run K-means clustering multiple times and return best result."""
    best_inertia = np.inf
    best_labels = None
    
    for i in range(n_runs):
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state + i, n_init=10)
        labels = kmeans.fit_predict(X)
        if kmeans.inertia_ < best_inertia:
            best_inertia = kmeans.inertia_
            best_labels = labels
    
    return best_labels


def run_hierarchical_consensus(X, n_clusters):
    """Run hierarchical clustering with multiple linkage methods and create consensus."""
    methods_linkage = ['ward', 'average', 'complete', 'single']
    all_labels = []
    
    for method in methods_linkage:
        try:
            if method == 'ward':
                Z = linkage(X, method=method, metric='euclidean')
            else:
                Z = linkage(X, method=method, metric='cosine')
            labels = fcluster(Z, n_clusters, criterion='maxclust') - 1
            all_labels.append(labels)
        except Exception as e:
            print(f"    Warning: {method} linkage failed: {e}")
    
    if len(all_labels) == 0:
        return None
    
    return majority_vote_consensus(all_labels, n_clusters)


def majority_vote_consensus(all_labels, n_clusters):
    """Create consensus clustering via majority voting."""
    if len(all_labels) == 0:
        return None
    
    n_samples = len(all_labels[0])
    aligned_labels = [all_labels[0]]
    for labels in all_labels[1:]:
        aligned = align_labels(all_labels[0], labels, n_clusters)
        aligned_labels.append(aligned)
    
    consensus = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        votes = [labels[i] for labels in aligned_labels]
        consensus[i] = Counter(votes).most_common(1)[0][0]
    
    return consensus


def align_labels(ref_labels, labels, n_clusters):
    """Align cluster labels to reference using Hungarian algorithm."""
    max_label = max(n_clusters, max(ref_labels) + 1, max(labels) + 1)
    conf_matrix = np.zeros((max_label, max_label))
    
    for i in range(len(ref_labels)):
        conf_matrix[ref_labels[i], labels[i]] += 1
    
    row_ind, col_ind = linear_sum_assignment(-conf_matrix)
    mapping = {col_ind[i]: row_ind[i] for i in range(len(row_ind))}
    aligned = np.array([mapping.get(l, l) for l in labels])
    return aligned


# ============================================================================
# EVALUATION METRICS
# ============================================================================

def compute_metrics(true_labels, pred_labels):
    """Compute clustering agreement metrics."""
    le = LabelEncoder()
    true_encoded = le.fit_transform(np.asarray(true_labels).astype(str))
    
    # Normalized Mutual Information (0 to 1)
    nmi = normalized_mutual_info_score(true_encoded, pred_labels)
    
    # Accuracy with optimal assignment (Hungarian matching)
    n_clusters = len(np.unique(pred_labels))
    n_true_clusters = len(np.unique(true_encoded))
    max_clusters = max(n_clusters, n_true_clusters)
    
    conf_matrix = np.zeros((max_clusters, max_clusters))
    for i in range(len(true_encoded)):
        if true_encoded[i] < max_clusters and pred_labels[i] < max_clusters:
            conf_matrix[true_encoded[i], pred_labels[i]] += 1
    
    row_ind, col_ind = linear_sum_assignment(-conf_matrix)
    accuracy = conf_matrix[row_ind, col_ind].sum() / len(true_encoded)
    
    return {'NMI': nmi, 'Accuracy': accuracy, 'confusion_matrix': conf_matrix}


def permutation_test(true_labels, pred_labels, n_permutations=1000, random_state=42):
    """Perform permutation test to assess significance of NMI."""
    np.random.seed(random_state)
    
    le = LabelEncoder()
    true_encoded = le.fit_transform(np.asarray(true_labels).astype(str))
    
    observed_nmi = normalized_mutual_info_score(true_encoded, pred_labels)
    
    perm_nmis = []
    for _ in range(n_permutations):
        perm_pred = np.random.permutation(pred_labels)
        perm_nmi = normalized_mutual_info_score(true_encoded, perm_pred)
        perm_nmis.append(perm_nmi)
    
    perm_nmis = np.array(perm_nmis)
    p_value = np.mean(perm_nmis >= observed_nmi)
    
    return observed_nmi, p_value, perm_nmis


# ============================================================================
# MAIN CLUSTERING COMPARISON
# ============================================================================

def run_clustering_comparison(embeddings, categories, sample_names, n_permutations=1000):
    """Run all clustering comparisons and collect results."""
    results = []
    detailed_results = {}
    
    for emb_name, X in embeddings.items():
        print(f"\n{'='*70}")
        print(f"EMBEDDING: {emb_name.upper()}")
        print(f"{'='*70}")
        
        detailed_results[emb_name] = {}
        
        for cat_name, cat_info in categories.items():
            true_labels = cat_info['labels']
            n_clusters = cat_info['n_clusters']
            
            print(f"\n  Category: {cat_name}")
            print(f"    k={n_clusters}, description: {cat_info['description']}")
            
            if n_clusters > 20:
                print(f"    Skipping - too many clusters for reliable evaluation")
                continue
            
            detailed_results[emb_name][cat_name] = {}
            
            # ---- K-means ----
            print("    Running K-means...")
            kmeans_labels = run_kmeans(X, n_clusters, n_runs=10, random_state=RANDOM_SEED)
            kmeans_metrics = compute_metrics(true_labels, kmeans_labels)
            kmeans_nmi, kmeans_pval, kmeans_perm = permutation_test(
                true_labels, kmeans_labels, n_permutations=n_permutations, random_state=RANDOM_SEED
            )
            
            results.append({
                'Embedding': emb_name,
                'Category': cat_name,
                'Method': 'K-means',
                'n_clusters': n_clusters,
                'NMI': kmeans_metrics['NMI'],
                'Accuracy': kmeans_metrics['Accuracy'],
                'p_value': kmeans_pval
            })
            
            detailed_results[emb_name][cat_name]['kmeans'] = {
                'labels': kmeans_labels,
                'metrics': kmeans_metrics,
                'perm_nmis': kmeans_perm,
                'p_value': kmeans_pval
            }
            
            sig_str = "***" if kmeans_pval < 0.001 else "**" if kmeans_pval < 0.01 else "*" if kmeans_pval < 0.05 else ""
            print(f"      NMI={kmeans_metrics['NMI']:.3f}, Acc={kmeans_metrics['Accuracy']:.3f}, p={kmeans_pval:.4f} {sig_str}")
            
            # ---- Hierarchical Consensus ----
            print("    Running Hierarchical Consensus...")
            hier_labels = run_hierarchical_consensus(X, n_clusters)
            
            if hier_labels is not None:
                hier_metrics = compute_metrics(true_labels, hier_labels)
                hier_nmi, hier_pval, hier_perm = permutation_test(
                    true_labels, hier_labels, n_permutations=n_permutations, random_state=RANDOM_SEED
                )
                
                results.append({
                    'Embedding': emb_name,
                    'Category': cat_name,
                    'Method': 'Hierarchical Consensus',
                    'n_clusters': n_clusters,
                    'NMI': hier_metrics['NMI'],
                    'Accuracy': hier_metrics['Accuracy'],
                    'p_value': hier_pval
                })
                
                detailed_results[emb_name][cat_name]['hierarchical'] = {
                    'labels': hier_labels,
                    'metrics': hier_metrics,
                    'perm_nmis': hier_perm,
                    'p_value': hier_pval
                }
                
                sig_str = "***" if hier_pval < 0.001 else "**" if hier_pval < 0.01 else "*" if hier_pval < 0.05 else ""
                print(f"      NMI={hier_metrics['NMI']:.3f}, Acc={hier_metrics['Accuracy']:.3f}, p={hier_pval:.4f} {sig_str}")
    
    return pd.DataFrame(results), detailed_results


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_nmi_heatmap_by_embedding(results_df, output_dir):
    """Create NMI heatmap for each embedding (mean across clustering methods)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for idx, emb in enumerate(['expression', 'proportion']):
        emb_data = results_df[results_df['Embedding'] == emb]
        if len(emb_data) == 0:
            continue
        
        # Average NMI across methods for each category
        nmi_by_cat = emb_data.groupby('Category')['NMI'].mean().sort_values(ascending=False)
        pval_by_cat = emb_data.groupby('Category')['p_value'].min()
        
        # Create DataFrame for heatmap
        heatmap_data = pd.DataFrame({'NMI': nmi_by_cat})
        
        # Plot heatmap
        sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlOrRd',
                   ax=axes[idx], vmin=0, vmax=0.5, cbar_kws={'label': 'NMI'})
        
        # Add significance stars to y-axis labels
        new_labels = []
        for cat in heatmap_data.index:
            p = pval_by_cat[cat]
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            new_labels.append(f"{cat} {sig}")
        axes[idx].set_yticklabels(new_labels, rotation=0)
        
        axes[idx].set_title(f'{emb.capitalize()} Embedding\n(Mean NMI across methods)', fontsize=12)
        axes[idx].set_xlabel('')
        axes[idx].set_ylabel('Metadata Category')
    
    plt.suptitle('Normalized Mutual Information by Category\n(* p<0.05, ** p<0.01, *** p<0.001)', fontsize=14, y=1.05)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/nmi_heatmap_by_embedding.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: nmi_heatmap_by_embedding.pdf")


def plot_category_comparison(results_df, output_dir):
    """Bar plot comparing NMI across categories - one plot per embedding-method combination."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    combinations = [
        ('expression', 'K-means'),
        ('expression', 'Hierarchical Consensus'),
        ('proportion', 'K-means'),
        ('proportion', 'Hierarchical Consensus')
    ]
    
    for idx, (emb, method) in enumerate(combinations):
        ax = axes[idx]
        
        # Filter data for this embedding-method combination
        subset = results_df[(results_df['Embedding'] == emb) & (results_df['Method'] == method)]
        
        if len(subset) == 0:
            ax.set_title(f'{emb.capitalize()} - {method}\n(No data)')
            continue
        
        # Sort by NMI
        subset = subset.sort_values('NMI', ascending=False)
        
        colors = ['forestgreen' if p < 0.05 else 'steelblue' for p in subset['p_value']]
        
        bars = ax.bar(range(len(subset)), subset['NMI'].values, 
                      color=colors, alpha=0.7, edgecolor='black', linewidth=1)
        
        # Add significance stars
        for i, (_, row) in enumerate(subset.iterrows()):
            y_pos = row['NMI'] + 0.02
            if row['p_value'] < 0.001:
                ax.text(i, y_pos, '***', ha='center', fontsize=12, fontweight='bold')
            elif row['p_value'] < 0.01:
                ax.text(i, y_pos, '**', ha='center', fontsize=12, fontweight='bold')
            elif row['p_value'] < 0.05:
                ax.text(i, y_pos, '*', ha='center', fontsize=12, fontweight='bold')
        
        ax.set_xticks(range(len(subset)))
        ax.set_xticklabels(subset['Category'].values, rotation=45, ha='right', fontsize=10)
        ax.set_ylabel('NMI', fontsize=11)
        ax.set_title(f'{emb.capitalize()} - {method}', fontsize=12, fontweight='bold')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_ylim(bottom=-0.05, top=1.0)
    
    plt.suptitle('Clustering Agreement with Metadata Categories\n(Green = p<0.05; * p<0.05, ** p<0.01, *** p<0.001)', 
                fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/category_comparison.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: category_comparison.pdf")


def plot_embedding_visualization(embeddings, categories, sample_names, output_dir):
    """Visualize embeddings colored by different categories."""
    from sklearn.decomposition import PCA
    
    cat_list = ['month', 'Sex', 'LC/Recovered', 'BMI category', 'Age (binned)', 'OutSMART ID']
    
    fig, axes = plt.subplots(2, len(cat_list), figsize=(4*len(cat_list), 8))
    
    for i, (emb_name, X) in enumerate(embeddings.items()):
        if X.shape[1] > 2:
            pca = PCA(n_components=2)
            X_2d = pca.fit_transform(X)
            var_explained = pca.explained_variance_ratio_
        else:
            X_2d = X[:, :2]
            var_explained = [1, 1]
        
        for j, cat_name in enumerate(cat_list):
            ax = axes[i, j]
            
            labels = categories[cat_name]['labels']
            le = LabelEncoder()
            labels_encoded = le.fit_transform(np.asarray(labels).astype(str))
            unique_labels = le.classes_
            
            n_colors = len(unique_labels)
            colors = plt.cm.Set2(np.linspace(0, 1, max(n_colors, 3)))
            
            for k, label in enumerate(unique_labels):
                mask = labels_encoded == k
                ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=[colors[k]], 
                          s=80, alpha=0.7, edgecolors='black', linewidths=0.5,
                          label=str(label))
            
            if i == 0:
                ax.set_title(cat_name, fontsize=11, fontweight='bold')
            if j == 0:
                ax.set_ylabel(f'{emb_name.capitalize()}\nPC2 ({var_explained[1]:.1%})', fontsize=10)
            ax.set_xlabel(f'PC1 ({var_explained[0]:.1%})', fontsize=9)
            
            # Only show legend for categories with few values
            if n_colors <= 4:
                ax.legend(loc='best', fontsize=8, framealpha=0.8)
    
    plt.suptitle('Embedding Visualization by Metadata Categories', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/embedding_visualization.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: embedding_visualization.pdf")


def plot_confusion_matrices(detailed_results, categories, output_dir):
    """Plot confusion matrices for best performing category-method combinations."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    cat_list = ['month', 'Sex', 'LC/Recovered', 'BMI category', 'Age (binned)', 'OutSMART ID']
    
    for idx, cat_name in enumerate(cat_list):
        ax = axes[idx]
        
        best_nmi = -1
        best_conf = None
        best_method = None
        best_emb = None
        
        for emb_name in detailed_results:
            if cat_name not in detailed_results[emb_name]:
                continue
            for method in ['kmeans', 'hierarchical']:
                if method not in detailed_results[emb_name][cat_name]:
                    continue
                nmi = detailed_results[emb_name][cat_name][method]['metrics']['NMI']
                if nmi > best_nmi:
                    best_nmi = nmi
                    best_conf = detailed_results[emb_name][cat_name][method]['metrics']['confusion_matrix']
                    best_method = method
                    best_emb = emb_name
        
        if best_conf is not None:
            conf_norm = best_conf / (best_conf.sum(axis=1, keepdims=True) + 1e-10)
            n_display = categories[cat_name]['n_clusters']
            conf_display = conf_norm[:n_display, :n_display]
            
            # Turn off annotations for large matrices
            annot = n_display <= 5
            fmt = '.2f' if annot else ''
            
            sns.heatmap(conf_display, annot=annot, fmt=fmt, cmap='Blues', ax=ax,
                       cbar_kws={'label': 'Proportion'})
            ax.set_title(f'{cat_name}\n({best_emb}, {best_method}, NMI={best_nmi:.3f})', fontsize=10)
            ax.set_xlabel('Predicted Cluster')
            ax.set_ylabel('True Label')
    
    plt.suptitle('Confusion Matrices (Best Method for Each Category)', fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/confusion_matrices.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: confusion_matrices.pdf")


def plot_permutation_distributions(detailed_results, categories, output_dir):
    """Plot permutation test distributions with p-values."""
    cat_list = ['month', 'Sex', 'LC/Recovered', 'BMI category', 'Age (binned)', 'OutSMART ID']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, cat_name in enumerate(cat_list):
        ax = axes[idx]
        
        legend_handles = []
        legend_labels = []
        
        color_map = {
            ('expression', 'kmeans'): 'blue',
            ('expression', 'hierarchical'): 'cyan',
            ('proportion', 'kmeans'): 'red',
            ('proportion', 'hierarchical'): 'orange'
        }
        
        for emb_name in detailed_results:
            if cat_name not in detailed_results[emb_name]:
                continue
            
            for method, method_label in [('kmeans', 'K-means'), ('hierarchical', 'Hier.')]:
                if method not in detailed_results[emb_name][cat_name]:
                    continue
                
                perm_nmis = detailed_results[emb_name][cat_name][method]['perm_nmis']
                observed = detailed_results[emb_name][cat_name][method]['metrics']['NMI']
                p_value = detailed_results[emb_name][cat_name][method]['p_value']
                
                color = color_map.get((emb_name, method), 'gray')
                
                # Plot histogram
                ax.hist(perm_nmis, bins=30, alpha=0.3, color=color)
                
                # Plot observed value
                line = ax.axvline(observed, linestyle='--', linewidth=2, color=color)
                
                # Format p-value string
                if p_value < 0.001:
                    p_str = "p<0.001"
                else:
                    p_str = f"p={p_value:.3f}"
                
                label = f'{emb_name[:4]}-{method_label}: NMI={observed:.3f}, {p_str}'
                legend_handles.append(line)
                legend_labels.append(label)
        
        ax.set_xlabel('NMI')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{cat_name}', fontsize=11, fontweight='bold')
        ax.legend(legend_handles, legend_labels, fontsize=7, loc='upper right')
    
    plt.suptitle('Permutation Test Distributions (with p-values)', fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/permutation_distributions.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: permutation_distributions.pdf")


def plot_detailed_metrics(results_df, output_dir):
    """Create detailed NMI and Accuracy comparison plot."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    metrics = ['NMI', 'Accuracy']
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        pivot = results_df.pivot_table(index='Category', 
                                        columns=['Embedding', 'Method'], 
                                        values=metric, aggfunc='mean')
        
        pivot.plot(kind='bar', ax=ax, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax.set_title(f'{metric}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Category', fontsize=10)
        ax.set_ylabel(metric, fontsize=10)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
        ax.legend(title='Emb-Method', fontsize=7, loc='upper right')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    plt.suptitle('Clustering Metrics Comparison', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/detailed_metrics.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: detailed_metrics.pdf")


# ============================================================================
# SUMMARY REPORT
# ============================================================================

def generate_summary_report(results_df, output_dir):
    """Generate a comprehensive summary report."""
    
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("CLUSTERING AGREEMENT ANALYSIS SUMMARY REPORT")
    report_lines.append("=" * 80)
    report_lines.append(f"\nAnalysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Number of comparisons: {len(results_df)}")
    
    # 1. Category Ranking
    report_lines.append("\n" + "=" * 80)
    report_lines.append("1. CATEGORY RANKING (by mean NMI across all methods/embeddings)")
    report_lines.append("=" * 80)
    
    cat_summary = results_df.groupby('Category').agg({
        'NMI': ['mean', 'std', 'max'],
        'Accuracy': ['mean', 'std'],
        'p_value': 'min'
    }).round(4)
    cat_summary.columns = ['NMI_mean', 'NMI_std', 'NMI_max', 'Acc_mean', 'Acc_std', 'p_value_min']
    cat_summary = cat_summary.sort_values('NMI_mean', ascending=False)
    
    for i, (cat, row) in enumerate(cat_summary.iterrows()):
        sig = "***" if row['p_value_min'] < 0.001 else "**" if row['p_value_min'] < 0.01 else "*" if row['p_value_min'] < 0.05 else ""
        report_lines.append(f"\n  {i+1}. {cat}")
        report_lines.append(f"     NMI:      {row['NMI_mean']:.4f} ± {row['NMI_std']:.4f} (max: {row['NMI_max']:.4f}) {sig}")
        report_lines.append(f"     Accuracy: {row['Acc_mean']:.4f} ± {row['Acc_std']:.4f}")
        report_lines.append(f"     p-value:  {row['p_value_min']:.4f}")
    
    # 2. Embedding Comparison
    report_lines.append("\n" + "=" * 80)
    report_lines.append("2. EMBEDDING COMPARISON")
    report_lines.append("=" * 80)
    
    emb_summary = results_df.groupby('Embedding').agg({
        'NMI': ['mean', 'std']
    }).round(4)
    
    for emb in emb_summary.index:
        report_lines.append(f"\n  {emb.capitalize()}:")
        report_lines.append(f"     Mean NMI: {emb_summary.loc[emb, ('NMI', 'mean')]:.4f} ± {emb_summary.loc[emb, ('NMI', 'std')]:.4f}")
    
    # 3. Statistical Significance Summary
    report_lines.append("\n" + "=" * 80)
    report_lines.append("3. STATISTICAL SIGNIFICANCE SUMMARY")
    report_lines.append("=" * 80)
    
    sig_001 = results_df[results_df['p_value'] < 0.001]
    sig_01 = results_df[(results_df['p_value'] >= 0.001) & (results_df['p_value'] < 0.01)]
    sig_05 = results_df[(results_df['p_value'] >= 0.01) & (results_df['p_value'] < 0.05)]
    not_sig = results_df[results_df['p_value'] >= 0.05]
    
    report_lines.append(f"\n  Highly significant (p < 0.001): {len(sig_001)}/{len(results_df)}")
    report_lines.append(f"  Significant (p < 0.01):         {len(sig_01)}/{len(results_df)}")
    report_lines.append(f"  Marginally significant (p < 0.05): {len(sig_05)}/{len(results_df)}")
    report_lines.append(f"  Not significant (p >= 0.05):    {len(not_sig)}/{len(results_df)}")
    
    if len(sig_001) > 0:
        report_lines.append("\n  Highly significant results (p < 0.001):")
        for _, row in sig_001.sort_values('NMI', ascending=False).iterrows():
            report_lines.append(f"    - {row['Category']} | {row['Embedding']} | {row['Method']}: NMI={row['NMI']:.4f}")
    
    # 4. Best Overall Combinations
    report_lines.append("\n" + "=" * 80)
    report_lines.append("4. TOP 5 BEST PERFORMING COMBINATIONS")
    report_lines.append("=" * 80)
    
    top5 = results_df.nlargest(5, 'NMI')
    for i, (_, row) in enumerate(top5.iterrows()):
        sig = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*" if row['p_value'] < 0.05 else ""
        report_lines.append(f"\n  {i+1}. {row['Category']} | {row['Embedding']} | {row['Method']}")
        report_lines.append(f"     NMI={row['NMI']:.4f}, Acc={row['Accuracy']:.4f}, p={row['p_value']:.4f} {sig}")
    
    # 5. Interpretation
    report_lines.append("\n" + "=" * 80)
    report_lines.append("5. INTERPRETATION")
    report_lines.append("=" * 80)
    
    best_cat = cat_summary.index[0]
    best_nmi = cat_summary['NMI_mean'].iloc[0]
    best_p = cat_summary['p_value_min'].iloc[0]
    
    report_lines.append(f"\n  Best matching category: {best_cat}")
    report_lines.append(f"  Mean NMI: {best_nmi:.4f}")
    report_lines.append(f"  Significance: p = {best_p:.4f}")
    
    if best_nmi > 0.4 and best_p < 0.05:
        report_lines.append("\n  STRONG AGREEMENT: The clustering strongly captures this metadata category.")
        report_lines.append("  This suggests meaningful biological signal related to this variable.")
    elif best_nmi > 0.2 and best_p < 0.05:
        report_lines.append("\n  MODERATE AGREEMENT: Clustering shows significant but partial agreement.")
        report_lines.append("  The data structure partially reflects this metadata category.")
    elif best_nmi > 0.1 and best_p < 0.05:
        report_lines.append("\n  WEAK BUT SIGNIFICANT: Small but statistically significant agreement.")
        report_lines.append("  Some signal exists but other factors likely dominate the data structure.")
    else:
        report_lines.append("\n  NO STRONG AGREEMENT: Clustering does not clearly capture any tested category.")
        report_lines.append("  The dominant structure may reflect other biological or technical factors.")
    
    report_text = "\n".join(report_lines)
    print("\n" + report_text)
    
    with open(f'{output_dir}/summary_report.txt', 'w') as f:
        f.write(report_text)
    print(f"\n  Saved: summary_report.txt")
    
    return report_text


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main function to run the clustering test framework."""
    
    print("=" * 80)
    print("CLUSTERING TEST FRAMEWORK FOR LONG COVID PSEUDOBULK DATA")
    print("=" * 80)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    
    # Step 1: Load data
    print("\n" + "-" * 40)
    print("STEP 1: Loading Data")
    print("-" * 40)
    adata, meta = load_data(H5AD_PATH, META_PATH)
    
    # Step 2: Extract embeddings
    print("\n" + "-" * 40)
    print("STEP 2: Extracting Embeddings")
    print("-" * 40)
    embeddings = get_embeddings(adata)
    
    # Step 3: Prepare metadata categories
    print("\n" + "-" * 40)
    print("STEP 3: Preparing Metadata Categories")
    print("-" * 40)
    categories = prepare_metadata_categories(meta)
    
    print("\nCategories to test:")
    for cat_name, cat_info in categories.items():
        print(f"  - {cat_name}: k={cat_info['n_clusters']} ({cat_info['description']})")
    
    # Step 4: Run clustering comparisons
    print("\n" + "-" * 40)
    print("STEP 4: Running Clustering Comparisons")
    print("-" * 40)
    results_df, detailed_results = run_clustering_comparison(
        embeddings, categories, adata.obs_names, n_permutations=N_PERMUTATIONS
    )
    
    results_df.to_csv(f'{OUTPUT_DIR}/clustering_results.csv', index=False)
    print(f"\n  Saved: clustering_results.csv")
    
    # Step 5: Generate visualizations
    print("\n" + "-" * 40)
    print("STEP 5: Generating Visualizations")
    print("-" * 40)
    
    plot_nmi_heatmap_by_embedding(results_df, OUTPUT_DIR)
    plot_category_comparison(results_df, OUTPUT_DIR)
    plot_embedding_visualization(embeddings, categories, adata.obs_names, OUTPUT_DIR)
    plot_confusion_matrices(detailed_results, categories, OUTPUT_DIR)
    plot_permutation_distributions(detailed_results, categories, OUTPUT_DIR)
    plot_detailed_metrics(results_df, OUTPUT_DIR)
    
    # Step 6: Generate summary report
    print("\n" + "-" * 40)
    print("STEP 6: Generating Summary Report")
    print("-" * 40)
    generate_summary_report(results_df, OUTPUT_DIR)
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nAll outputs saved to: {OUTPUT_DIR}")
    print("\nFiles generated:")
    print("  - clustering_results.csv")
    print("  - summary_report.txt")
    print("  - nmi_heatmap_by_embedding.pdf")
    print("  - category_comparison.pdf")
    print("  - embedding_visualization.pdf")
    print("  - confusion_matrices.pdf")
    print("  - permutation_distributions.pdf")
    print("  - detailed_metrics.pdf")
    
    return results_df, detailed_results


if __name__ == "__main__":
    results_df, detailed_results = main()