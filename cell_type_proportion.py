import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load the data
adata = sc.read_h5ad('/dcs07/hongkai/data/harry/result/long_covid/subset/T_count_for_pseudobulk.h5ad')

# Create output directory
output_dir = '/dcs07/hongkai/data/harry/result/long_covid/subset/T_cell_proportion_heatmaps'
os.makedirs(output_dir, exist_ok=True)

# Extract relevant metadata
obs_df = adata.obs[['sample', 'month', 'manual_cell_type', 'age_cluster', 'LC/Recovered', 'BMI category', 'Sex']].copy()

# Get unique values
cell_types = sorted(obs_df['manual_cell_type'].unique().tolist())
months = ['1', '3', '6']  # Ordered months

# Define category columns for stratification
category_cols = ['age_cluster', 'LC/Recovered', 'BMI category', 'Sex']

print(f"Cell types: {len(cell_types)}")
print(f"Months: {months}")

# =============================================================================
# PART 1: General Heatmap - Cell Type (rows) x Month (columns)
# =============================================================================
print("\n" + "="*60)
print("Creating General Heatmap: Cell Type x Month")
print("="*60)

# Count cells per cell type per month
general_counts = obs_df.groupby(['manual_cell_type', 'month']).size().unstack(fill_value=0)
# Reorder columns by month
general_counts = general_counts[[m for m in months if m in general_counts.columns]]

# Calculate proportions (percentage of each cell type within each month)
general_props = general_counts.div(general_counts.sum(axis=0), axis=1) * 100

fig, axes = plt.subplots(1, 2, figsize=(14, 10))

# Heatmap with counts
ax1 = axes[0]
sns.heatmap(general_counts, annot=True, fmt='d', cmap='YlOrRd', ax=ax1, 
            cbar_kws={'label': 'Cell Count'})
ax1.set_title('Cell Counts by Cell Type and Month', fontsize=12, fontweight='bold')
ax1.set_xlabel('Month', fontsize=10)
ax1.set_ylabel('Cell Type', fontsize=10)
ax1.tick_params(axis='x', rotation=0)
ax1.tick_params(axis='y', rotation=0)

# Heatmap with proportions
ax2 = axes[1]
sns.heatmap(general_props, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax2,
            cbar_kws={'label': 'Proportion (%)'})
ax2.set_title('Cell Proportions (%) by Cell Type and Month', fontsize=12, fontweight='bold')
ax2.set_xlabel('Month', fontsize=10)
ax2.set_ylabel('Cell Type', fontsize=10)
ax2.tick_params(axis='x', rotation=0)
ax2.tick_params(axis='y', rotation=0)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'general_heatmap_celltype_by_month.png'), dpi=150, bbox_inches='tight')
plt.close()

print("Saved: general_heatmap_celltype_by_month.png")

# =============================================================================
# PART 2: Per Cell Type - Month x Category Heatmaps
# =============================================================================
print("\n" + "="*60)
print("Creating Per-Cell-Type Heatmaps (Month x Category style)")
print("="*60)

for cell_type in cell_types:
    print(f"\nProcessing: {cell_type}")
    
    # Filter for this cell type
    ct_df = obs_df[obs_df['manual_cell_type'] == cell_type].copy()
    
    if len(ct_df) == 0:
        print(f"  Skipping {cell_type}: no cells found")
        continue
    
    # Create figure with subplots for each category
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, cat_col in enumerate(category_cols):
        ax = axes[idx]
        
        # Create combined column for month + category
        ct_df['month_cat'] = ct_df['month'].astype(str) + '_' + ct_df[cat_col].astype(str)
        
        # Cross-tabulation: just count cells (since we're looking at one cell type, row will be count)
        # We want Month x Category, so let's do it differently
        cross_tab = pd.crosstab(ct_df[cat_col], ct_df['month'])
        
        # Reorder columns by month
        ordered_cols = [m for m in months if m in cross_tab.columns]
        cross_tab = cross_tab[ordered_cols]
        
        # Sort rows
        cross_tab = cross_tab.sort_index()
        
        sns.heatmap(cross_tab, annot=True, fmt='d', cmap='YlOrRd', ax=ax,
                    cbar_kws={'label': 'Cell Count'}, linewidths=0.5)
        
        ax.set_title(f'{cat_col}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Month', fontsize=10)
        ax.set_ylabel(cat_col, fontsize=10)
        ax.tick_params(axis='x', rotation=0)
        ax.tick_params(axis='y', rotation=0)
    
    plt.suptitle(f'Cell Counts for: {cell_type}', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Safe filename
    safe_celltype = cell_type.replace('/', '_').replace(' ', '_').replace(':', '_')
    plt.savefig(os.path.join(output_dir, f'heatmap_{safe_celltype}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: heatmap_{safe_celltype}.png")

# =============================================================================
# PART 3: Detailed heatmap with Month x Category combinations (all cell types)
# =============================================================================
print("\n" + "="*60)
print("Creating Month x Category Detailed Heatmaps (All Cell Types)")
print("="*60)

for cat_col in category_cols:
    # Create combined column
    obs_df['month_cat'] = obs_df['month'].astype(str) + '_' + obs_df[cat_col].astype(str)
    
    # Cross-tabulation
    cross_tab = pd.crosstab(obs_df['manual_cell_type'], obs_df['month_cat'])
    
    # Reorder columns by month then category
    ordered_cols = []
    cat_values = sorted(obs_df[cat_col].unique())
    for month in months:
        for cat_val in cat_values:
            col_name = f"{month}_{cat_val}"
            if col_name in cross_tab.columns:
                ordered_cols.append(col_name)
    
    cross_tab = cross_tab[ordered_cols]
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    sns.heatmap(cross_tab, annot=True, fmt='d', cmap='YlOrRd', ax=ax,
                cbar_kws={'label': 'Cell Count'}, linewidths=0.5)
    
    # Add month separators
    n_cats = len(cat_values)
    for i in range(1, 3):
        if i * n_cats <= len(ordered_cols):
            ax.axvline(x=i * n_cats, color='black', linewidth=2)
    
    ax.set_title(f'Cell Counts: Cell Type x (Month + {cat_col})', fontsize=12, fontweight='bold')
    ax.set_xlabel(f'Month_{cat_col}', fontsize=10)
    ax.set_ylabel('Cell Type', fontsize=10)
    ax.tick_params(axis='x', rotation=45)
    ax.tick_params(axis='y', rotation=0, labelsize=9)
    
    plt.tight_layout()
    
    safe_cat = cat_col.replace('/', '_').replace(' ', '_')
    plt.savefig(os.path.join(output_dir, f'detailed_heatmap_month_x_{safe_cat}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: detailed_heatmap_month_x_{safe_cat}.png")

# Clean up temporary column
obs_df.drop('month_cat', axis=1, inplace=True)

# =============================================================================
# PART 4: Overview Heatmap - Cell Type distribution by each category
# =============================================================================
print("\n" + "="*60)
print("Creating Overview Heatmap")
print("="*60)

fig, axes = plt.subplots(2, 2, figsize=(16, 14))

for idx, cat_col in enumerate(category_cols):
    ax = axes[idx // 2, idx % 2]
    
    # Cross-tabulation: cell type x category
    cross_tab = pd.crosstab(obs_df['manual_cell_type'], obs_df[cat_col])
    
    # Sort columns
    cross_tab = cross_tab[sorted(cross_tab.columns)]
    
    sns.heatmap(cross_tab, annot=True, fmt='d', cmap='YlOrRd', ax=ax,
                cbar_kws={'label': 'Cell Count'}, linewidths=0.5)
    ax.set_title(f'Cell Type Distribution by {cat_col}', fontsize=11, fontweight='bold')
    ax.set_xlabel(cat_col, fontsize=10)
    ax.set_ylabel('Cell Type', fontsize=10)
    ax.tick_params(axis='x', rotation=45)
    ax.tick_params(axis='y', rotation=0, labelsize=8)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'overview_celltype_by_categories.png'), dpi=150, bbox_inches='tight')
plt.close()

print("Saved: overview_celltype_by_categories.png")

# =============================================================================
# PART 5: Summary CSV
# =============================================================================
print("\n" + "="*60)
print("Creating Summary Statistics")
print("="*60)

summary_data = []
for cell_type in cell_types:
    ct_df = obs_df[obs_df['manual_cell_type'] == cell_type]
    total = len(ct_df)
    for month in months:
        month_count = len(ct_df[ct_df['month'] == month])
        summary_data.append({
            'Cell Type': cell_type,
            'Month': month,
            'Count': month_count,
            'Proportion (%)': round(month_count / total * 100, 2) if total > 0 else 0
        })

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv(os.path.join(output_dir, 'cell_type_month_summary.csv'), index=False)
print("Saved: cell_type_month_summary.csv")

print("\n" + "="*60)
print(f"All outputs saved to: {output_dir}")
print("="*60)

# List all generated files
print("\nGenerated files:")
for f in sorted(os.listdir(output_dir)):
    print(f"  - {f}")