# %% [markdown]
# # MERSCOPE Region R4 Analysis

# %% [markdown]
# This notebook performs an analysis of MERSCOPE data for region R1, focusing on data loading, exploratory data analysis (EDA), and visualization.

# %%
# Import necessary libraries
import scanpy as sc
import anndata as ad
import pandas as pd
import geopandas as gpd # For potential .parquet file with geometries
import matplotlib.pyplot as plt
import matplotlib.image as mpimg # For displaying images
import seaborn as sns
import numpy as np # For calculations if needed
import os
import squidpy
import warnings

# Suppress FutureWarning messages
warnings.filterwarnings('ignore', category=FutureWarning)

# Set plotting style
# plt.style.use('seaborn-v0_8-whitegrid')
# sc.settings.set_figure_params(dpi=100, frameon=True, figsize=(6, 6), facecolor='white')

# %%
# Define file paths
base_path = '../202503071102_SESSA-p30-E165_VMSC10702/R4'
summary_image_file = os.path.join(base_path, 'summary.png')

# %% [markdown]
# ## 1. Data Loading
# 
# We will prioritize loading the AnnData file (`.h5ad`) as it should contain the most comprehensive data. If needed, we will supplement with other files.

# %%
h5ad_file = os.path.join(base_path, '202503071102_SESSA-p30-E165_VMSC10702_region_R4.h5ad')

# %%
# Attempt to load the .h5ad file
adata = None
try:
    adata = sc.read_h5ad(h5ad_file)
    print(f"Successfully loaded AnnData file: {h5ad_file}")
    print(adata)
except FileNotFoundError:
    print(f"AnnData file not found: {h5ad_file}. Will attempt to load individual files.")
except Exception as e:
    print(f"Error loading AnnData file {h5ad_file}: {e}. Will attempt to load individual files.")

# %%
print(adata.obs.index[:5])
print(adata.obs.volume.head())
print(adata.obs.center_x.head())
print(adata.obs.leiden.head())

# %%
keep_genes = [x for x in adata.var.index.tolist() if 'Blank' not in x]
print(len(keep_genes))
print(adata.shape[1])

# %%
min_expression = 25
ser_exp = adata.to_df().sum(axis=1)

keep_cells = ser_exp[ser_exp > min_expression].index.tolist()
print(len(keep_cells))
print(adata.shape[0])

# adata = adata[keep_cells]
# adata

# %%
adata_v2 = adata.copy()

# %% [markdown]
# # 1b Supplementary files

# %% [markdown]
# ## Count data

# %%
cell_by_gene_file = os.path.join(base_path, 'cell_by_gene.csv')

# %%
# Load gene expression data
counts_df = pd.read_csv(cell_by_gene_file, index_col=0) # Assuming first column is cell ID
print(f"Loaded {cell_by_gene_file}: {counts_df.shape[0]} cells, {counts_df.shape[1]} genes")
non_zero_values = counts_df.values[counts_df.values != 0]
top_5_values = sorted(non_zero_values, reverse=True)[:5]
print("Top 5 non-zero values:", top_5_values)

# %% [markdown]
# ## Metadata

# %%
cell_metadata_file = os.path.join(base_path, 'cell_metadata.csv')

# %%
# Load cell metadata
metadata_df = pd.read_csv(cell_metadata_file, index_col=0) # Assuming first column is cell ID
print(f"Loaded {cell_metadata_file}: {metadata_df.shape[0]} cells, {metadata_df.shape[1]} metadata columns")
metadata_df.head()

# %%
# Align indices (important!)
common_cells = counts_df.index.intersection(metadata_df.index)
counts_df = counts_df.loc[common_cells]
metadata_df = metadata_df.loc[common_cells]
print(f"Found {len(common_cells)} common cells between counts and metadata.")

if len(common_cells) == 0:
    raise ValueError("No common cells found between cell_by_gene.csv and cell_metadata.csv. Cannot create AnnData object.")

# %%
# Create AnnData object
# adata = ad.AnnData(X=counts_df.values, obs=metadata_df, var=pd.DataFrame(index=counts_df.columns))
# adata.X = adata.X.astype('float32') # Ensure X is float for scanpy operations
# print("Successfully created AnnData object from CSV files.")
# print(adata)

# %% [markdown]
# ### Cell boundaries

# %%
cell_boundaries_file = os.path.join(base_path, 'cell_boundaries.parquet')
cell_boundaries_gdf = None

# %%
cell_boundaries_gdf = gpd.read_parquet(cell_boundaries_file)
print(f"Loaded {cell_boundaries_file}. Shape: {cell_boundaries_gdf.shape}")

# %%
cell_boundaries_gdf.head()

# %%
cell_boundaries_gdf = cell_boundaries_gdf.set_index('EntityID', drop=False)
cell_boundaries_gdf.head()

# %%
# adata.obs.index = adata.obs.index.astype(str)
# cell_boundaries_gdf.index = cell_boundaries_gdf.index.astype(str)

# print(adata.obs.index[:5])
# print(cell_boundaries_gdf.index[:5])

# %%
# common_cells_boundaries = adata.obs.index.intersection(cell_boundaries_gdf.index)
# common_cells_boundaries[:5]

# %%
# adata.uns['cell_boundaries_gdf'] = cell_boundaries_gdf.loc[common_cells_boundaries]

# %% [markdown]
# ## Differentially expressed genes

# %%
differentially_expressed_genes_file = os.path.join(base_path, 'differentially_expressed_genes.csv')

# %%
degs_df = pd.read_csv(differentially_expressed_genes_file, index_col=0)
print(degs_df.shape)
print(adata.shape)

# %%
degs_df.head()

# %%
print(len(degs_df.gene.unique()))

# %%
degs_df = degs_df.set_index('gene', drop=False)
degs_df.shape

# %%
print(degs_df.index[:5])
print(adata.var.index[:5])

# %%
adata.var.index = adata.var.index.astype(str)
degs_df.index = degs_df.index.astype(str)

# %%
common_genes = adata.var.index.intersection(degs_df.index)
len(common_genes)

# %%
# adata.uns['cell_boundaries_gdf'] = degs_df.loc[common_genes]

# %% [markdown]
# ## Detected transcripts

# %%
detected_transcripts_file = os.path.join(base_path, 'detected_transcripts.parquet')

# %%
det_trans_df = pd.read_parquet(detected_transcripts_file)
det_trans_df.shape

# %%
det_trans_df.head()

# %% [markdown]
# ## Cell categories

# %%
cell_categories_file = os.path.join(base_path, 'cell_categories.csv')
cell_numeric_categories_file = os.path.join(base_path, 'cell_numeric_categories.csv')

# %%
cell_categories_df = pd.read_csv(cell_categories_file, index_col=0) # Assuming first column is cell ID
cell_numeric_categories_df = pd.read_csv(cell_numeric_categories_file, index_col=0) # Assuming first column is cell ID

# %%
cell_categories_df.head()

# %%
cell_numeric_categories_df.head()

# %% [markdown]
# # 2. Exploratory Data Analysis
# 
# Basic statistics and distributions of the data.

# %%
print(f"Number of cells: {adata.n_obs}")
print(f"Number of genes: {adata.n_vars}")

# %%
print("Calculating QC metrics (total_counts, n_genes_by_counts)...")
sc.pp.calculate_qc_metrics(adata, inplace=True)

# %%
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Plot for total_counts
sns.histplot(adata.obs['total_counts'], bins=50, kde=False, ax=axes[0])
axes[0].set_xlabel("Total transcripts per cell")
axes[0].set_ylabel("Number of cells")
axes[0].set_title("Distribution of Total Transcripts per Cell")

# Plot for n_genes_by_counts
sns.histplot(adata.obs['n_genes_by_counts'], bins=50, kde=False, ax=axes[1])
axes[1].set_xlabel("Number of genes detected per cell")
axes[1].set_ylabel("Number of cells")
axes[1].set_title("Distribution of Genes Detected per Cell")

plt.tight_layout()
plt.show()

# %%
coll_to_summary = 'leiden'

if adata.obs[coll_to_summary].nunique() < 30 and adata.obs[coll_to_summary].nunique() > 1:
    plt.figure(figsize=(8, max(4, adata.obs[coll_to_summary].nunique() * 0.3)))
    sns.countplot(y=adata.obs[coll_to_summary], order = adata.obs[coll_to_summary].value_counts(dropna=False).index)
    plt.title(f"Cell Counts by {coll_to_summary}")
    plt.xlabel("Number of Cells")
    plt.ylabel(coll_to_summary)
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## 3. Visualization

# %%
img = mpimg.imread(summary_image_file)
plt.figure(figsize=(10,10))
plt.imshow(img)
plt.axis('off') # Turn off axis numbers and ticks
plt.title("Experiment Summary Image (region_R1/summary.png)")
plt.show()

# %%
print(adata.obsm_keys())
print(list(adata.obs.columns))

# %%
adata.obsm['spatial'] = adata.obs[['center_x', 'center_y']].to_numpy()
spatial_coords_available = True

# %%
features = ['total_counts', 'n_genes_by_counts', 'leiden']
for color_by in features:
    plt.figure(figsize=(10, 10))
    sc.pl.spatial(adata, color=color_by, spot_size=30, show=False, frameon=False)
    plt.title(f"Spatial Plot of Cells (Colored by {color_by if color_by else 'default'})")
    plt.show()

# %%
print(list(adata.var.columns))
print(list(adata.uns))

# %%
sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5, flavor='cell_ranger')

# %%
import numpy as np
import scipy.sparse

print(f"adata.X data type: {adata.X.dtype}")

if scipy.sparse.issparse(adata.X):
    print(adata.X[:5, :5].toarray())
    if adata.X.nnz > 0: # nnz is number of stored_elements
        print(f"Min non-zero value in adata.X: {adata.X.data.min()}")
        print(f"Max non-zero value in adata.X: {adata.X.data.max()}")
        has_nan_sparse = np.isnan(adata.X.data).any()
        has_inf_sparse = np.isinf(adata.X.data).any()
        print(f"adata.X.data contains NaNs: {has_nan_sparse}")
        print(f"adata.X.data contains Infs: {has_inf_sparse}")

else: # Dense array
    print(adata.X[:5, :5])
    print(f"Min value in adata.X: {adata.X.min()}")
    print(f"Max value in adata.X: {adata.X.max()}")
    has_nan_dense = np.isnan(adata.X).any()
    has_inf_dense = np.isinf(adata.X).any()
    print(f"adata.X contains NaNs: {has_nan_dense}")
    print(f"adata.X contains Infs: {has_inf_dense}")

# %%
hvg_genes = adata.var_names[adata.var['highly_variable']].tolist()
if len(hvg_genes) > 0:
    num_genes_to_plot = min(len(hvg_genes), 3)
    genes_to_plot = hvg_genes[:num_genes_to_plot]
    print(f"Plotting spatial expression for HVGs: {genes_to_plot}")
    sc.pl.spatial(adata, color=genes_to_plot, spot_size=30, show=True, frameon=False, ncols=num_genes_to_plot)
else:
    print("No highly variable genes found after computation.")



# %%
# generate colors for categories by plotting
sc.pl.umap(adata, color="leiden", legend_loc='on data')
cats = adata.obs['leiden'].cat.categories.tolist()
colors = list(adata.uns['leiden_colors'])
cat_colors = dict(zip(cats, colors))

# colors for clustergrammer2
ser_color = pd.Series(cat_colors)
ser_color.name = 'color'
df_colors = pd.DataFrame(ser_color)
df_colors.index = ['Leiden-' + str(x) for x in df_colors.index.tolist()]

df_colors.loc[''] = 'white'

# %% [markdown]
# # 3. Alternative DEGs

# %%
resolution = 1.5

# Leiden Clustering
######################

# dividing by volume instead
sc.pp.normalize_total(adata_v2)
sc.pp.log1p(adata_v2)
sc.pp.scale(adata_v2, max_value=10)
sc.tl.pca(adata_v2, svd_solver='arpack')
sc.pp.neighbors(adata_v2, n_neighbors=10, n_pcs=20)
sc.tl.umap(adata_v2)
sc.tl.leiden(adata_v2, resolution=resolution)

# Calculate Leiden Signatures
#########################################df_pos.index = [str(x) for x in list(range(df_pos.shape[0]))]
ser_counts = adata_v2.obs['leiden'].value_counts()
ser_counts.name = 'cell counts'
meta_leiden = pd.DataFrame(ser_counts)

cat_name = 'leiden'
sig_leiden = pd.DataFrame(columns=adata_v2.var_names, index=adata_v2.obs[cat_name].cat.categories)
for clust in adata_v2.obs[cat_name].cat.categories:
    sig_leiden.loc[clust] = adata_v2[adata_v2.obs[cat_name].isin([clust]),:].X.mean(0)
sig_leiden = sig_leiden.transpose()
leiden_clusters = ['Leiden-' + str(x) for x in sig_leiden.columns.tolist()]
sig_leiden.columns = leiden_clusters
meta_leiden.index = sig_leiden.columns.tolist()
meta_leiden['leiden'] = pd.Series(meta_leiden.index.tolist(), index=meta_leiden.index.tolist())

# generate colors for categories by plotting
sc.pl.umap(adata_v2, color="leiden", legend_loc='on data')
cats = adata_v2.obs['leiden'].cat.categories.tolist()
colors = list(adata_v2.uns['leiden_colors'])
cat_colors = dict(zip(cats, colors))

# colors for clustergrammer2
ser_color = pd.Series(cat_colors)
ser_color.name = 'color'
df_colors = pd.DataFrame(ser_color)
df_colors.index = ['Leiden-' + str(x) for x in df_colors.index.tolist()]

df_colors.loc[''] = 'white'

# %% [markdown]
# # R. ROI

# %%
roi_file = "E:/Githubs/SPATIAL_data/202503071102_SESSA-p30-E165_VMSC10702/R4/p30_CTRL_R4/cells1.csv"
roi_data = pd.read_csv(roi_file)
cell1_ids = roi_data.iloc[:, 0].astype(str).tolist()
cell1_ids[:5]

# %%
roi_file = "E:/Githubs/SPATIAL_data/202503071102_SESSA-p30-E165_VMSC10702/R4/p30_CTRL_R4/cells2.csv"
roi_data = pd.read_csv(roi_file)
cell2_ids = roi_data.iloc[:, 0].astype(str).tolist()
cell2_ids[:5]

# %%
# roi1 = adata[adata.obs.index.isin(cell_ids), :]
# roi1

# %%
# adata[adata.obs.index.isin(cell_ids), :].obs["roi"] = "roi1"

# %%
# For multiple ROI assignments
roi_assignments = {
    "roi1": cell1_ids,
    "roi2": cell2_ids
}

# Initialize column
adata.obs["roi"] = "unassigned"

# Assign each ROI
for roi_name, cell_list in roi_assignments.items():
    mask = adata.obs.index.isin(cell_list)
    adata.obs.loc[mask, "roi"] = roi_name

# %%
color_by = "roi"
plt.figure(figsize=(10, 10))
sc.pl.spatial(adata, color=color_by, spot_size=30, show=False, frameon=False)
plt.title(f"Spatial Plot of Cells (Colored by {color_by if color_by else 'default'})")
plt.show()

# %%
import scanpy as sc
import pandas as pd

# First, subset to only cells in roi1 and roi2 (exclude unassigned)
adata_roi = adata[adata.obs["roi"].isin(["roi1", "roi2"])].copy()

# Run differential expression analysis
sc.tl.rank_genes_groups(
    adata_roi, 
    groupby='roi',  # Column containing your groups
    method='wilcoxon',  # or 't-test', 'logreg', 't-test_overestim_var'
    key_added='roi_deg',  # Key to store results
    reference='roi1',  # Compare roi2 vs roi1 (or use 'rest' for one-vs-rest)
    n_genes=None  # Calculate for all genes
)

# View top DEGs
sc.pl.rank_genes_groups(adata_roi, key='roi_deg', n_genes=25, sharey=False)

# Get results as a DataFrame
# Top genes upregulated in roi2 compared to roi1
result = sc.get.rank_genes_groups_df(adata_roi, group='roi2', key='roi_deg')
print(result.head(20))

# Save full results
result.to_csv('roi2_vs_roi1_DEGs.csv', index=False)

# Filter by significance and fold change
significant_degs = result[
    (result['pvals_adj'] < 0.05) &  # Adjusted p-value threshold
    (abs(result['logfoldchanges']) > 0.5)  # Log fold change threshold
]
print(f"Number of significant DEGs: {len(significant_degs)}")

# %% [markdown]
# # Two samples

# %%
import scanpy as sc
import pandas as pd

# # 1. Load both samples
# sample1 = sc.read_h5ad('sample1.h5ad')
# sample2 = sc.read_h5ad('sample2.h5ad')

# # 2. Subset to only roi1 cells from each sample
# # Assuming you've already assigned ROIs to each sample
# sample1_roi1 = sample1[sample1.obs['roi'] == 'roi1'].copy()
# sample2_roi1 = sample2[sample2.obs['roi'] == 'roi1'].copy()


# Load samples
sample1 = sc.read_h5ad('sample1.h5ad')
sample2 = sc.read_h5ad('sample2.h5ad')

# Load ROI cell IDs (assuming same format as before)
roi1_cells_sample1 = pd.read_csv('roi1_cells_sample1.csv', squeeze=True).tolist()
roi1_cells_sample2 = pd.read_csv('roi1_cells_sample2.csv', squeeze=True).tolist()

# Subset each sample
sample1_roi1 = sample1[sample1.obs.index.isin(roi1_cells_sample1)].copy()
sample2_roi1 = sample2[sample2.obs.index.isin(roi1_cells_sample2)].copy()

# Add metadata
sample1_roi1.obs['sample'] = 'sample1'
sample1_roi1.obs['roi'] = 'roi1'
sample2_roi1.obs['sample'] = 'sample2'
sample2_roi1.obs['roi'] = 'roi1'

# Continue with concatenation and DEG analysis as above...

# 3. Add sample information before merging
sample1_roi1.obs['sample'] = 'sample1'
sample2_roi1.obs['sample'] = 'sample2'

# 4. Concatenate the two subsets
adata_merged = sc.concat(
    [sample1_roi1, sample2_roi1],
    axis=0,  # Concatenate cells (observations)
    join='outer',  # Keep all genes
    label='sample',
    keys=['sample1', 'sample2'],
    index_unique='-'  # Add suffix to make cell names unique
)

# 5. Run differential expression analysis between samples
sc.tl.rank_genes_groups(
    adata_merged,
    groupby='sample',  # Compare by sample
    method='wilcoxon',
    reference='sample1',  # Compare sample2 vs sample1
    key_added='sample_deg'
)

# 6. Get and save results
results = sc.get.rank_genes_groups_df(adata_merged, group='sample2', key='sample_deg')
print(results.head(20))
results.to_csv('roi1_sample2_vs_sample1_DEGs.csv', index=False)

# Filter significant DEGs
significant_degs = results[
    (results['pvals_adj'] < 0.05) & 
    (abs(results['logfoldchanges']) > 0.5)
]
print(f"Number of significant DEGs: {len(significant_degs)}")

# %%
# Load samples
sample1 = sc.read_h5ad('sample1.h5ad')
sample2 = sc.read_h5ad('sample2.h5ad')

# Load ROI cell IDs (assuming same format as before)
roi1_cells_sample1 = pd.read_csv('roi1_cells_sample1.csv', squeeze=True).tolist()
roi1_cells_sample2 = pd.read_csv('roi1_cells_sample2.csv', squeeze=True).tolist()

# Subset each sample
sample1_roi1 = sample1[sample1.obs.index.isin(roi1_cells_sample1)].copy()
sample2_roi1 = sample2[sample2.obs.index.isin(roi1_cells_sample2)].copy()

# Add metadata
sample1_roi1.obs['sample'] = 'sample1'
sample1_roi1.obs['roi'] = 'roi1'
sample2_roi1.obs['sample'] = 'sample2'
sample2_roi1.obs['roi'] = 'roi1'

# Continue with concatenation and DEG analysis as above...

# %%
# 1. Check for batch effects
sc.pp.combat(adata_merged, key='sample')  # Batch correction if needed

# 2. Ensure proper normalization
# If samples were processed differently, re-normalize after merging
sc.pp.normalize_total(adata_merged, target_sum=1e4)
sc.pp.log1p(adata_merged)

# 3. Visualize to check for batch effects
sc.pp.pca(adata_merged)
sc.pl.pca(adata_merged, color=['sample'], title='PCA by sample')

# 4. Alternative: Include batch as covariate
# This requires using 'logreg' method
sc.tl.rank_genes_groups(
    adata_merged,
    groupby='sample',
    method='logreg',
    reference='sample1'
)

# %%
# Volcano plot
sc.pl.rank_genes_groups_volcano(
    adata_merged,
    key='sample_deg',
    group='sample2'
)

# Heatmap of top DEGs
top_genes = results.head(50)['names'].tolist()
sc.pl.heatmap(
    adata_merged,
    var_names=top_genes,
    groupby='sample',
    figsize=(8, 12),
    swap_axes=False,
    dendrogram=True
)

# Compare expression of specific genes
genes_of_interest = ['GENE1', 'GENE2', 'GENE3']
sc.pl.violin(
    adata_merged,
    keys=genes_of_interest,
    groupby='sample',
    rotation=45
)


