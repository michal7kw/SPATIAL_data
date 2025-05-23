# %% [markdown]
# # Vizgen MERSCOPE Vizualizer Cell Clustering
# 
# This notebook demonstrates how to use Scanpy to perform data pre-processing, dimensionality reduction, and single-cell Leiden clustering.
# 
# This notebook is designed to take the Anndata output from the MERSCOPE Vizualizer and return a new Anndata object with UMAP dimensionality reduction and Leiden clustering cell metadata.
# 
# Before running this notebook upload your data (using the Files tab on the left and upload button) to this Colab notebook (a specific filename is not required, but the file must end with `.hdf5`).
# 
# 

# %%
# %%capture
# !pip install -q numpy>=1.21
# !pip install -q fsspec
# !pip install -q gcsfs
# !pip install -q scanpy>=1.9.3
# !pip install -q anndata>=0.9.0
# !pip install -q leidenalg
# !pip install -q observable-jupyter
# !pip install -q clustergrammer2
# !pip install -q loompy

# %%
import os
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from matplotlib import pyplot as plt
import matplotlib.image as mpimg  # For displaying images
import seaborn as sns
import warnings
from observable_jupyter import embed
from clustergrammer2 import net, Network, CGM2
from scipy.stats.stats import pearsonr
from copy import deepcopy
from glob import glob

# Suppress FutureWarning messages
warnings.filterwarnings('ignore', category=FutureWarning)

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sc.settings.set_figure_params(dpi=100, frameon=True, figsize=(6, 6), facecolor='white')

# code for compressing data for visuals
# import zlib, json, base64
# def json_zip(j):
#     zip_json_string = base64.b64encode(
#         zlib.compress(
#             json.dumps(j).encode('utf-8')
#         )
#     ).decode('ascii')
#     return zip_json_string

# Avoids scroll-in-the-scroll in the entire Notebook
# from IPython.display import Javascript
# def resize_colab_cell():
#   display(Javascript('google.colab.output.setIframeHeight(0, true, {maxHeight: 5000})'))
# get_ipython().events.register('pre_run_cell', resize_colab_cell)

# %% [markdown]
# # Load Anndata

# %%
# Define file paths
base_path = 'region_R1/' # Ensure this path is correct relative to the notebook location
h5ad_file = os.path.join(base_path, '202503071102_SESSA-p30-E165_VMSC10702_region_R1.h5ad')
print(h5ad_file)

# Attempt to load the .h5ad file (same approach as region_R1_analysis.py)
ad_viz = None
try:
    ad_viz = sc.read_h5ad(h5ad_file)
    print(f"Successfully loaded AnnData file: {h5ad_file}")
    print(ad_viz)
except FileNotFoundError:
    print(f"AnnData file not found: {h5ad_file}. Will attempt to load individual files.")
except Exception as e:
    print(f"Error loading AnnData file {h5ad_file}: {e}. Will attempt to load individual files.")

# Fallback to CSV loading if h5ad failed (same as region_R1_analysis.py)
if ad_viz is None:
    print("\nAttempting to load data from individual CSV files...")
    cell_by_gene_file = os.path.join(base_path, 'cell_by_gene.csv')
    cell_metadata_file = os.path.join(base_path, 'cell_metadata.csv')
    
    try:
        # Load gene expression data
        counts_df = pd.read_csv(cell_by_gene_file, index_col=0)
        print(f"Loaded {cell_by_gene_file}: {counts_df.shape[0]} cells, {counts_df.shape[1]} genes")
        
        # Load cell metadata
        metadata_df = pd.read_csv(cell_metadata_file, index_col=0)
        print(f"Loaded {cell_metadata_file}: {metadata_df.shape[0]} cells, {metadata_df.shape[1]} metadata columns")
        
        # Align indices (important!)
        common_cells = counts_df.index.intersection(metadata_df.index)
        counts_df = counts_df.loc[common_cells]
        metadata_df = metadata_df.loc[common_cells]
        print(f"Found {len(common_cells)} common cells between counts and metadata.")

        if len(common_cells) == 0:
            raise ValueError("No common cells found between cell_by_gene.csv and cell_metadata.csv. Cannot create AnnData object.")

        # Create AnnData object
        ad_viz = ad.AnnData(X=counts_df.values, obs=metadata_df, var=pd.DataFrame(index=counts_df.columns))
        ad_viz.X = ad_viz.X.astype('float32')  # Ensure X is float for scanpy operations
        print("Successfully created AnnData object from CSV files.")
        print(ad_viz)
        
    except FileNotFoundError as e:
        print(f"Error: A required CSV file was not found: {e}. Cannot proceed with analysis.")
        raise
    except ValueError as e:
        print(f"Error creating AnnData object: {e}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred while loading CSV files: {e}")
        raise

# %%


# filter out blanks before clusering
keep_genes = [x for x in ad_viz.var.index.tolist() if 'Blank' not in x]
ad_viz = ad_viz[:, keep_genes]

# copy to cell_by_gene_matrix
cell_by_gene = deepcopy(ad_viz.to_df())

ad_viz


# %%
# ad_viz.__dict__['_raw'].__dict__['_var'] = ad_viz.__dict__['_raw'].__dict__['_var'].rename(columns={'_index': 'features'})

# %% [markdown]
# ### Filter Cells Based on Minimum Gene Expression Counts

# %%
# before filtering
ser_exp = ad_viz.to_df().sum(axis=1)
ser_exp.hist(bins=100)

# %%
min_expression = 25
keep_cells = ser_exp[ser_exp > min_expression].index.tolist()
ad_viz = ad_viz[keep_cells]
ad_viz

# %%
ser_exp = ad_viz.to_df().sum(axis=1)
ser_exp.hist(bins=100)

# %% [markdown]
# ### Filter Cells Based on Volume

# %%
# ser_volume = ad_viz.obs['volume']
# ser_volume.hist(bins=100)

# %%
# # filter cells based on volume
# min_volume = 50
# keep_cells = ser_volume[ser_volume > min_volume].index.tolist()
# ad_viz = ad_viz[keep_cells]
# ad_viz

# %%
# ser_volume = ad_viz.obs['volume']
# ser_volume.hist(bins=100)

# %% [markdown]
# # UMAP and Single-Cell Clustering

# %%
resolution = 1.5

# Leiden Clustering
######################

# dividing by volume instead
sc.pp.normalize_total(ad_viz)
sc.pp.log1p(ad_viz)
sc.pp.scale(ad_viz, max_value=10)
sc.tl.pca(ad_viz, svd_solver='arpack')
sc.pp.neighbors(ad_viz, n_neighbors=10, n_pcs=20)
sc.tl.umap(ad_viz)
sc.tl.leiden(ad_viz, resolution=resolution)

# Calculate Leiden Signatures
#########################################df_pos.index = [str(x) for x in list(range(df_pos.shape[0]))]
ser_counts = ad_viz.obs['leiden'].value_counts()
ser_counts.name = 'cell counts'
meta_leiden = pd.DataFrame(ser_counts)

cat_name = 'leiden'
sig_leiden = pd.DataFrame(columns=ad_viz.var_names, index=ad_viz.obs[cat_name].cat.categories)
for clust in ad_viz.obs[cat_name].cat.categories:
    sig_leiden.loc[clust] = ad_viz[ad_viz.obs[cat_name].isin([clust]),:].X.mean(0)
sig_leiden = sig_leiden.transpose()
leiden_clusters = ['Leiden-' + str(x) for x in sig_leiden.columns.tolist()]
sig_leiden.columns = leiden_clusters
meta_leiden.index = sig_leiden.columns.tolist()
meta_leiden['leiden'] = pd.Series(meta_leiden.index.tolist(), index=meta_leiden.index.tolist())

# generate colors for categories by plotting
sc.pl.umap(ad_viz, color="leiden", legend_loc='on data')
cats = ad_viz.obs['leiden'].cat.categories.tolist()
colors = list(ad_viz.uns['leiden_colors'])
cat_colors = dict(zip(cats, colors))

# colors for clustergrammer2
ser_color = pd.Series(cat_colors)
ser_color.name = 'color'
df_colors = pd.DataFrame(ser_color)
df_colors.index = ['Leiden-' + str(x) for x in df_colors.index.tolist()]

df_colors.loc[''] = 'white'

# %% [markdown]
# # Save AnnData
# After saving the anndata, use the Files tab to download the `.hdf5` to your local computer. This file can then be loaded into the MERSCOPE Vizualizer where the `leiden` clusters and UMAP dimensionality reduction embedding can be imported.

# %%
ad_viz

# %%
# Create clustered filename based on the original h5ad file
clustered_filename = h5ad_file.replace('.h5ad', '_clustered.h5ad')
ad_viz.write_h5ad(clustered_filename)
print(f"Saved clustered data to: {clustered_filename}")

# %%



