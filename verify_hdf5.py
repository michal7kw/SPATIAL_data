#!/usr/bin/env python3
"""
Verify and inspect the structure of the converted HDF5 file
"""

import h5py
import numpy as np
import pandas as pd
import os

def inspect_hdf5_structure(hdf5_file_path, max_depth=3):
    """
    Inspect the structure of an HDF5 file
    
    Parameters:
    -----------
    hdf5_file_path : str
        Path to the HDF5 file
    max_depth : int
        Maximum depth to traverse in the HDF5 structure
    """
    
    if not os.path.exists(hdf5_file_path):
        print(f"Error: File '{hdf5_file_path}' not found!")
        return
    
    print(f"Inspecting HDF5 file: {hdf5_file_path}")
    file_size = os.path.getsize(hdf5_file_path) / (1024**2)  # Size in MB
    print(f"File size: {file_size:.2f} MB")
    print("=" * 60)
    
    def print_structure(name, obj, depth=0):
        if depth > max_depth:
            return
        
        indent = "  " * depth
        
        if isinstance(obj, h5py.Group):
            print(f"{indent}{name}/ (Group)")
            if hasattr(obj, 'attrs') and len(obj.attrs) > 0:
                for attr_name, attr_value in obj.attrs.items():
                    print(f"{indent}  @{attr_name}: {attr_value}")
        elif isinstance(obj, h5py.Dataset):
            print(f"{indent}{name} (Dataset): shape={obj.shape}, dtype={obj.dtype}")
            if hasattr(obj, 'attrs') and len(obj.attrs) > 0:
                for attr_name, attr_value in obj.attrs.items():
                    print(f"{indent}  @{attr_name}: {attr_value}")
    
    with h5py.File(hdf5_file_path, 'r') as f:
        print("Root level attributes:")
        for attr_name, attr_value in f.attrs.items():
            print(f"  @{attr_name}: {attr_value}")
        print()
        
        print("File structure:")
        f.visititems(print_structure)

def load_basic_data_from_hdf5(hdf5_file_path):
    """
    Load basic data from the HDF5 file to demonstrate how to read it
    """
    print(f"\nLoading basic data from: {hdf5_file_path}")
    print("=" * 60)
    
    with h5py.File(hdf5_file_path, 'r') as f:
        # Load basic metadata
        n_obs = f.attrs['n_obs']
        n_vars = f.attrs['n_vars']
        print(f"Number of observations (cells): {n_obs}")
        print(f"Number of variables (genes): {n_vars}")
        
        # Load cell indices
        if '_index' in f['obs']:
            cell_indices = [idx.decode('utf-8') for idx in f['obs']['_index'][:10]]
            print(f"First 10 cell indices: {cell_indices}")
        
        # Load gene indices
        if '_index' in f['var']:
            gene_indices = [idx.decode('utf-8') for idx in f['var']['_index'][:10]]
            print(f"First 10 gene indices: {gene_indices}")
        
        # Check expression matrix format
        if 'X' in f:
            if isinstance(f['X'], h5py.Group):
                print(f"Expression matrix is sparse (CSR format)")
                print(f"  Data shape: {f['X']['data'].shape}")
                print(f"  Indices shape: {f['X']['indices'].shape}")
                print(f"  Indptr shape: {f['X']['indptr'].shape}")
                print(f"  Matrix shape: {f['X']['shape'][:]}")
            else:
                print(f"Expression matrix is dense: {f['X'].shape}")
        
        # Show available observation metadata
        print(f"\nObservation metadata columns:")
        for key in f['obs'].keys():
            if key != '_index':
                dataset = f['obs'][key]
                print(f"  {key}: shape={dataset.shape}, dtype={dataset.dtype}")
        
        # Show available obsm data (e.g., PCA, UMAP)
        if 'obsm' in f and len(f['obsm'].keys()) > 0:
            print(f"\nMulti-dimensional observations (obsm):")
            for key in f['obsm'].keys():
                dataset = f['obsm'][key]
                print(f"  {key}: shape={dataset.shape}, dtype={dataset.dtype}")

def example_data_loading(hdf5_file_path):
    """
    Example of how to load specific data from the HDF5 file
    """
    print(f"\nExample: Loading specific data from HDF5 file")
    print("=" * 60)
    
    with h5py.File(hdf5_file_path, 'r') as f:
        # Load UMAP coordinates if available
        if 'obsm' in f and 'X_umap' in f['obsm']:
            umap_coords = f['obsm']['X_umap'][:]
            print(f"UMAP coordinates loaded: shape={umap_coords.shape}")
            print(f"First 5 UMAP coordinates:")
            print(umap_coords[:5])
        
        # Load leiden clustering if available
        if 'obs' in f and 'leiden' in f['obs']:
            leiden_clusters = f['obs']['leiden'][:]
            print(f"\nLeiden clusters loaded: shape={leiden_clusters.shape}")
            print(f"First 10 cluster assignments: {leiden_clusters[:10]}")
            
            # Count clusters
            unique_clusters, counts = np.unique(leiden_clusters, return_counts=True)
            print(f"Number of clusters: {len(unique_clusters)}")
            print(f"Cluster sizes: {dict(zip(unique_clusters, counts))}")

if __name__ == "__main__":
    # Check the converted file
    hdf5_file = 'region_R3/202503071102_SESSA-p30-E165_VMSC10702_region_R3.hdf5'
    
    if os.path.exists(hdf5_file):
        inspect_hdf5_structure(hdf5_file)
        load_basic_data_from_hdf5(hdf5_file)
        example_data_loading(hdf5_file)
    else:
        print(f"HDF5 file not found: {hdf5_file}")
        print("Please run the conversion script first.") 