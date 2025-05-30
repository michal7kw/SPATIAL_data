# %% [markdown]
# # Combined MERSCOPE ROI Differential Gene Expression Analysis
#
# This script performs DGE analysis between corresponding ROIs from two different samples:
# - Sample 1: p0, Region R4
# - Sample 2: p30, Region R3
#
# Comparisons:
# - ROI1 (first identified ROI in Sample 1) vs. ROI1 (first identified ROI in Sample 2)
# - ROI2 (second identified ROI in Sample 1) vs. ROI2 (second identified ROI in Sample 2)

# %%
# Import necessary libraries
import sys
import os
import scanpy as sc
import anndata as ad
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import warnings
from shapely import wkt

# Add the project root to the Python path
# Assumes this script is in the project root directory.
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(script_dir) # If script is in root, project_root is script_dir
sys.path.append(project_root)

from helpers.plotting import create_volcano_plot

# Suppress FutureWarning messages
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning) # For sjoin warnings if CRS is not set

# Global plot settings
PLOT_INTERMEDIATE_ROI_VISUALIZATIONS = False # Set to True to see ROI plots during data loading
PLOT_FINAL_DGE_VOLCANO = True

# %% [markdown]
# ## Helper Function: Load and Process Sample Data

# %%
def load_sample_data(base_path, h5ad_filename, cell_boundaries_filename, roi_csv_filename, 
                     sample_prefix, target_roi_labels=['ROI1', 'ROI2'], min_gene_expression_per_cell=25):
    """
    Loads and processes data for a single MERSCOPE sample.
    - Loads h5ad, cell boundaries, and ROI geometry.
    - Filters genes and cells.
    - Identifies cells within the first two specified ROIs.
    - Assigns sample-specific ROI labels (e.g., 'sample_prefix_ROI1').
    - Returns an AnnData object with cells from these ROIs and updated .obs.
    """
    print(f"\n--- Processing sample: {sample_prefix} ---")
    h5ad_file_path = os.path.join(base_path, h5ad_filename)
    cell_boundaries_file_path = os.path.join(base_path, cell_boundaries_filename)
    roi_geometry_file_path = os.path.join(base_path, roi_csv_filename)

    # 1. Load AnnData
    print(f"Loading AnnData: {h5ad_file_path}")
    adata = sc.read_h5ad(h5ad_file_path)
    print(f"Initial adata shape: {adata.shape}")

    # 2. Filter genes (remove Blanks)
    genes_before_blank_filter = adata.shape[1]
    keep_genes = [x for x in adata.var.index.tolist() if 'Blank' not in x]
    adata = adata[:, keep_genes].copy()
    print(f"Filtered out 'Blank' genes. Kept {len(keep_genes)} out of {genes_before_blank_filter} genes. New shape: {adata.shape}")

    # 3. Filter cells by minimum total gene expression
    cells_before_min_expr_filter = adata.shape[0]
    ser_exp = adata.to_df().sum(axis=1)
    keep_cells = ser_exp[ser_exp >= min_gene_expression_per_cell].index.tolist()
    adata = adata[keep_cells, :].copy()
    print(f"Filtered cells by min expression ({min_gene_expression_per_cell}). Kept {len(keep_cells)} out of {cells_before_min_expr_filter} cells. New shape: {adata.shape}")

    if adata.shape[0] == 0 or adata.shape[1] == 0:
        print(f"Warning: AnnData for {sample_prefix} is empty after initial filtering. Skipping further processing for this sample.")
        return None

    # 3.5 Normalize and Log-transform data
    print(f"Normalizing total counts per cell to 1e4 and log-transforming data for {sample_prefix}...")
    # Ensure that adata.X is not already log-transformed if this script is re-run with modifications
    # A simple check could be if adata.X.max() is relatively small (e.g. < 50 for log data)
    # For now, assuming raw counts are loaded initially.
    if adata.X is not None and adata.X.max() > 50: # Heuristic: if max count is high, it's likely raw
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        print("Normalization and log-transformation complete.")
    elif adata.X is not None:
        print("Data seems to have low max value, potentially already normalized/log-transformed. Skipping sc.pp.normalize_total and sc.pp.log1p.")
    else:
        print("adata.X is None. Skipping normalization and log-transformation.")


    # 4. Load cell boundaries
    print(f"Loading cell boundaries: {cell_boundaries_file_path}")
    cell_boundaries_gdf = gpd.read_parquet(cell_boundaries_file_path)
    cell_boundaries_gdf = cell_boundaries_gdf.set_index('EntityID', drop=False)
    print(f"Loaded cell boundaries. Shape: {cell_boundaries_gdf.shape}")

    # 5. Load and process ROI geometries
    print(f"Loading ROI geometry: {roi_geometry_file_path}")
    roi_polygons_df = pd.read_csv(roi_geometry_file_path)
    roi_polygons_df['geometry'] = roi_polygons_df['geometry'].apply(wkt.loads)
    
    current_crs = cell_boundaries_gdf.crs # Use CRS from cell boundaries if available
    roi_gdf = gpd.GeoDataFrame(roi_polygons_df, geometry='geometry', crs=current_crs)
    print(f"Loaded and converted ROI geometries. Shape: {roi_gdf.shape}")

    # 6. Spatially select cells within ROIs
    cell_boundaries_gdf_sjoin_ready = cell_boundaries_gdf.copy()
    cell_boundaries_gdf_sjoin_ready.index.name = 'original_cell_EntityID_idx'
    
    cells_in_rois_gdf = gpd.sjoin(
        cell_boundaries_gdf_sjoin_ready,
        roi_gdf,
        how="inner",
        predicate="within",
        lsuffix='_cell',
        rsuffix='_roi'
    )
    print(f"Found {cells_in_rois_gdf.shape[0]} cells within any defined ROIs.")
    
    if cells_in_rois_gdf.empty:
        print(f"Warning: No cells found within any ROIs for {sample_prefix}. Skipping ROI assignment.")
        return None

    # 7. Assign sample-specific ROI labels (e.g., 'p0_R4_ROI1')
    # We use the first two unique groups found in the 'group' column of the ROI CSV.
    roi_group_column_in_csv = 'group' # This is 'group' in the provided examples
    available_roi_groups_from_csv = cells_in_rois_gdf[roi_group_column_in_csv].unique()
    
    adata.obs['roi_assignment'] = 'unassigned' # This will be the common column for concatenated adata
    
    # Map original cell EntityIDs (from cells_in_rois_gdf) to new prefixed ROI labels
    # Ensure EntityIDs are strings for consistent mapping with adata.obs.index
    cells_in_rois_gdf.index = cells_in_rois_gdf.index.astype(str)
    adata.obs.index = adata.obs.index.astype(str)

    assigned_count = 0
    if len(available_roi_groups_from_csv) >= 1:
        roi1_original_name_in_csv = available_roi_groups_from_csv[0]
        new_label_roi1 = f"{sample_prefix}_{target_roi_labels[0]}"
        print(f"Mapping CSV group '{roi1_original_name_in_csv}' to '{new_label_roi1}'")
        cells_for_this_roi = cells_in_rois_gdf[cells_in_rois_gdf[roi_group_column_in_csv] == roi1_original_name_in_csv].index
        
        # Intersect with adata.obs.index to only assign to cells present in adata
        valid_cells_for_assignment = adata.obs.index.intersection(cells_for_this_roi)
        adata.obs.loc[valid_cells_for_assignment, 'roi_assignment'] = new_label_roi1
        assigned_count += len(valid_cells_for_assignment)
        print(f"Assigned {len(valid_cells_for_assignment)} cells to {new_label_roi1}")

    if len(available_roi_groups_from_csv) >= 2 and len(target_roi_labels) >=2:
        roi2_original_name_in_csv = available_roi_groups_from_csv[1]
        new_label_roi2 = f"{sample_prefix}_{target_roi_labels[1]}"
        print(f"Mapping CSV group '{roi2_original_name_in_csv}' to '{new_label_roi2}'")
        cells_for_this_roi = cells_in_rois_gdf[cells_in_rois_gdf[roi_group_column_in_csv] == roi2_original_name_in_csv].index
        
        valid_cells_for_assignment = adata.obs.index.intersection(cells_for_this_roi)
        adata.obs.loc[valid_cells_for_assignment, 'roi_assignment'] = new_label_roi2
        assigned_count += len(valid_cells_for_assignment)
        print(f"Assigned {len(valid_cells_for_assignment)} cells to {new_label_roi2}")
    else:
        print(f"Warning: Fewer than 2 ROI groups found in CSV for {sample_prefix} or fewer than 2 target labels. Only ROI1 might be processed.")

    print(f"Total cells assigned to target ROIs in adata.obs for {sample_prefix}: {assigned_count}")
    print(f"Value counts for 'roi_assignment' in {sample_prefix}:\n{adata.obs['roi_assignment'].value_counts()}")

    # 8. Filter AnnData to keep only cells assigned to the target ROIs
    target_labels_for_sample = []
    if len(target_roi_labels) >= 1: target_labels_for_sample.append(f"{sample_prefix}_{target_roi_labels[0]}")
    if len(target_roi_labels) >= 2: target_labels_for_sample.append(f"{sample_prefix}_{target_roi_labels[1]}")
    
    adata_filtered_rois = adata[adata.obs['roi_assignment'].isin(target_labels_for_sample)].copy()
    print(f"Shape of AnnData for {sample_prefix} after filtering for target ROIs: {adata_filtered_rois.shape}")

    if PLOT_INTERMEDIATE_ROI_VISUALIZATIONS and not adata_filtered_rois.empty and not cells_in_rois_gdf.empty:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        # Plot all original cell boundaries lightly
        cell_boundaries_gdf.plot(ax=ax, color='lightgray', edgecolor='silver', alpha=0.3, label='All Cells (Original)')
        
        # Plot cells within any ROI, colored by their original CSV group
        # Need to merge 'group' from cells_in_rois_gdf to cell_boundaries_gdf for plotting all selected cells
        temp_plot_gdf = cell_boundaries_gdf.loc[cells_in_rois_gdf.original_cell_EntityID_idx.unique()] # Get geometries of cells in any ROI
        temp_plot_gdf = temp_plot_gdf.merge(cells_in_rois_gdf[['original_cell_EntityID_idx', roi_group_column_in_csv]].drop_duplicates(subset=['original_cell_EntityID_idx']),
                                            left_index=True, right_on='original_cell_EntityID_idx', how='left')

        temp_plot_gdf.plot(ax=ax, column=roi_group_column_in_csv, legend=True, alpha=0.7, categorical=True,
                           legend_kwds={'title': f"{sample_prefix} CSV ROI Group", 'loc': 'upper right', 'bbox_to_anchor': (1.45, 1)})
        roi_gdf.plot(ax=ax, facecolor='none', edgecolor='blue', linewidth=2, label='ROI Polygons (from CSV)')
        ax.set_title(f"Cells within Defined ROIs for {sample_prefix}")
        plt.tight_layout(rect=[0, 0, 0.80, 1])
        plt.show()

    return adata_filtered_rois

# %% [markdown]
# ## 1. Define File Paths and Parameters

# %%
# Sample 1 (p0/R4)
base_path_s1 = './202504111150_Sessa-p0-p7_VMSC10702/R4' # Adjusted path relative to project root
h5ad_filename_s1 = '202504111150_Sessa-p0-p7_VMSC10702_region_R4.h5ad'
roi_csv_filename_s1 = 'p0_R4_ROI_28-05-25_17-01_geometry.csv'
sample_prefix_s1 = 'p0_R4'

# Sample 2 (p30/R3)
base_path_s2 = './202503071102_SESSA-p30-E165_VMSC10702/R3' # Adjusted path relative to project root
h5ad_filename_s2 = '202503071102_SESSA-p30-E165_VMSC10702_region_R3.h5ad'
roi_csv_filename_s2 = 'p30_R3_ROI_28-05-25_16-57_geometry.csv'
sample_prefix_s2 = 'p30_R3'

# Common parameters
cell_boundaries_filename_common = 'cell_boundaries.parquet'
target_roi_names = ['ROI1', 'ROI2'] # Generic names we assign to the first two ROIs found in each sample

# DGE Parameters
pval_threshold = 0.05
logfc_threshold = 0.5
top_genes_to_show = 10

# %% [markdown]
# ## 2. Load Data for Both Samples

# %%
adata_s1 = load_sample_data(
    base_path=base_path_s1,
    h5ad_filename=h5ad_filename_s1,
    cell_boundaries_filename=cell_boundaries_filename_common,
    roi_csv_filename=roi_csv_filename_s1,
    sample_prefix=sample_prefix_s1,
    target_roi_labels=target_roi_names
)

# %%
adata_s2 = load_sample_data(
    base_path=base_path_s2,
    h5ad_filename=h5ad_filename_s2,
    cell_boundaries_filename=cell_boundaries_filename_common,
    roi_csv_filename=roi_csv_filename_s2,
    sample_prefix=sample_prefix_s2,
    target_roi_labels=target_roi_names
)

# %% [markdown]
# ## 3. Combine AnnData Objects

# %%
if adata_s1 is not None and not adata_s1.obs.empty and \
   adata_s2 is not None and not adata_s2.obs.empty:
    
    print(f"adata_s1 shape before concat: {adata_s1.shape}, obs keys: {adata_s1.obs.columns.tolist()}")
    print(f"adata_s1 roi_assignment example: {adata_s1.obs['roi_assignment'].unique()[:5]}")
    print(f"adata_s2 shape before concat: {adata_s2.shape}, obs keys: {adata_s2.obs.columns.tolist()}")
    print(f"adata_s2 roi_assignment example: {adata_s2.obs['roi_assignment'].unique()[:5]}")

    # Ensure var_names (genes) are identical or handle appropriately (e.g., intersection)
    # For now, assuming 'join="inner"' handles gene differences by taking intersection.
    adata_combined = ad.concat(
        {"s1": adata_s1, "s2": adata_s2},
        label="sample_batch_id",  # Adds 's1' or 's2' to obs
        join="inner",        # Keeps only common genes
        index_unique="-"     # Appends batch key to make cell indices unique if needed
    )
    print(f"Combined AnnData shape: {adata_combined.shape}")
    print(f"Combined AnnData .obs columns: {adata_combined.obs.columns}")
    print(f"Value counts for 'roi_assignment' in combined data:\n{adata_combined.obs['roi_assignment'].value_counts()}")
    print(f"Value counts for 'sample_batch_id' in combined data:\n{adata_combined.obs['sample_batch_id'].value_counts()}")
else:
    print("One or both AnnData objects are None or empty after processing. Cannot proceed with combined analysis.")
    adata_combined = None

# %% [markdown]
# ## 4. Differential Gene Expression (DGE) Analysis

# %%
def perform_dge_and_visualize(adata_input, group_to_test, reference_group, pval_thresh, logfc_thresh, top_n_genes, plot_flag=True):
    """
    Performs DGE analysis between two groups in an AnnData object and visualizes results.
    """
    if adata_input is None or adata_input.shape[0] < 2 or adata_input.shape[1] == 0: # Min 1 cell per group
        print(f"Skipping DGE for {group_to_test} vs {reference_group} due to insufficient data.")
        return None

    print(f"\n--- DGE: {group_to_test} (test) vs {reference_group} (reference) ---")
    
    # Filter for the two specific groups for this comparison
    adata_for_dge = adata_input[adata_input.obs['roi_assignment'].isin([group_to_test, reference_group])].copy()
    
    if adata_for_dge.obs.groupby('roi_assignment').size().min() < 1: # Check if any group is empty
         print(f"Warning: One of the groups ('{group_to_test}' or '{reference_group}') is empty or has too few cells after filtering for DGE. Skipping.")
         return None
    if len(adata_for_dge.obs['roi_assignment'].unique()) < 2:
        print(f"Warning: Only one group present after filtering for DGE ('{group_to_test}', '{reference_group}'). Skipping comparison.")
        return None


    print(f"Shape of AnnData for this DGE comparison: {adata_for_dge.shape}")
    print(f"Cell counts per group for DGE:\n{adata_for_dge.obs['roi_assignment'].value_counts()}")

    # Data should already be log-transformed from load_sample_data.
    # sc.pp.log1p(adata_for_dge) # REMOVED

    # Perform DGE
    dge_key = f"rank_genes_{group_to_test}_vs_{reference_group}"
    sc.tl.rank_genes_groups(
        adata_for_dge,
        groupby='roi_assignment',
        groups=[group_to_test],      # Group to test (numerator for logFC)
        reference=reference_group,   # Reference group (denominator for logFC)
        method='wilcoxon',
        corr_method='benjamini-hochberg',
        key_added=dge_key
    )
    print("DGE calculation complete.")

    # Visualize DGE results (Volcano Plot)
    if plot_flag:
        print(f"Creating volcano plot for {group_to_test} vs {reference_group}...")
        fig, ax = create_volcano_plot(
            adata_for_dge,
            group_name=group_to_test, # This is the group for which scores, pvals etc. are reported directly
            key=dge_key,
            pval_threshold=pval_thresh,
            logfc_threshold=logfc_thresh,
            title=f"Volcano: {group_to_test} vs {reference_group}",
            show_gene_labels=True,
            top_genes=top_n_genes
        )
        plt.show()

    # Display DGE results as a DataFrame
    try:
        # The results are structured under the name of the group tested (group_to_test)
        dge_results_df = pd.DataFrame(adata_for_dge.uns[dge_key]['names'])[group_to_test]
        dge_scores_df = pd.DataFrame(adata_for_dge.uns[dge_key]['scores'])[group_to_test]
        dge_pvals_df = pd.DataFrame(adata_for_dge.uns[dge_key]['pvals_adj'])[group_to_test]
        dge_logfc_df = pd.DataFrame(adata_for_dge.uns[dge_key]['logfoldchanges'])[group_to_test]

        summary_df = pd.DataFrame({
            'gene': dge_results_df,
            'score': dge_scores_df,
            'pval_adj': dge_pvals_df,
            'log2fc': dge_logfc_df
        })
        print(f"\nTop DEGs for {group_to_test} compared to {reference_group} (log2fc > 0 means higher in {group_to_test}):")
        print(summary_df.head(top_n_genes * 2)) # Show more for inspection
        return summary_df
    except KeyError as e:
        print(f"Error extracting DGE results for {group_to_test}: {e}")
        print(f"Available keys in adata_for_dge.uns['{dge_key}']: {list(adata_for_dge.uns[dge_key].keys())}")
        if 'names' in adata_for_dge.uns[dge_key]:
             print(f"Structure of 'names': {type(adata_for_dge.uns[dge_key]['names'])}")
             if isinstance(adata_for_dge.uns[dge_key]['names'], np.recarray):
                  print(f"Field names in 'names' recarray: {adata_for_dge.uns[dge_key]['names'].dtype.names}")
        return None


# %% [markdown]
# ### 4.1 DGE: ROI1 (Sample 2 vs Sample 1)

# %%
if adata_combined is not None:
    roi1_s1_label = f"{sample_prefix_s1}_{target_roi_names[0]}" # e.g., p0_R4_ROI1
    roi1_s2_label = f"{sample_prefix_s2}_{target_roi_names[0]}" # e.g., p30_R3_ROI1

    # Check if both ROI1 labels exist in the combined data
    if roi1_s1_label in adata_combined.obs['roi_assignment'].unique() and \
       roi1_s2_label in adata_combined.obs['roi_assignment'].unique():
        
        print(f"Performing DGE for ROI1: {roi1_s2_label} (test) vs {roi1_s1_label} (reference)")
        summary_roi1 = perform_dge_and_visualize(
            adata_input=adata_combined,
            group_to_test=roi1_s2_label,    # Test group
            reference_group=roi1_s1_label,  # Reference group
            pval_thresh=pval_threshold,
            logfc_thresh=logfc_threshold,
            top_n_genes=top_genes_to_show,
            plot_flag=PLOT_FINAL_DGE_VOLCANO
        )
    else:
        print(f"Skipping DGE for ROI1 as one or both required ROI groups ({roi1_s1_label}, {roi1_s2_label}) are not present in the combined data.")
        print(f"Available ROI assignments: {adata_combined.obs['roi_assignment'].unique()}")

# %% [markdown]
# ### 4.2 DGE: ROI2 (Sample 2 vs Sample 1)

# %%
if adata_combined is not None and len(target_roi_names) >= 2:
    roi2_s1_label = f"{sample_prefix_s1}_{target_roi_names[1]}" # e.g., p0_R4_ROI2
    roi2_s2_label = f"{sample_prefix_s2}_{target_roi_names[1]}" # e.g., p30_R3_ROI2

    # Check if both ROI2 labels exist
    if roi2_s1_label in adata_combined.obs['roi_assignment'].unique() and \
       roi2_s2_label in adata_combined.obs['roi_assignment'].unique():

        print(f"Performing DGE for ROI2: {roi2_s2_label} (test) vs {roi2_s1_label} (reference)")
        summary_roi2 = perform_dge_and_visualize(
            adata_input=adata_combined,
            group_to_test=roi2_s2_label,    # Test group
            reference_group=roi2_s1_label,  # Reference group
            pval_thresh=pval_threshold,
            logfc_thresh=logfc_threshold,
            top_n_genes=top_genes_to_show,
            plot_flag=PLOT_FINAL_DGE_VOLCANO
        )
    else:
        print(f"Skipping DGE for ROI2 as one or both required ROI groups ({roi2_s1_label}, {roi2_s2_label}) are not present in the combined data.")
        print(f"Available ROI assignments: {adata_combined.obs['roi_assignment'].unique()}")
elif adata_combined is not None:
    print("Skipping DGE for ROI2 as only one target ROI name was specified.")

# %%
print("\n--- End of Combined DEG Analysis Script ---")