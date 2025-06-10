#!/usr/bin/env python3
import csv
import subprocess
import re
import os
import sys


SEL_REGION = "2"
REGION = f"R{SEL_REGION}"
SEL_PATCHES = "input.csv"
# DATA_PATH = "/beegfs/scratch/ric.broccoli/kubacki.michal/SPATIAL_data/data_p0-p7"
DATA_PATH = "/beegfs/scratch/ric.broccoli/kubacki.michal/SPATIAL_data/data_p30-E165"
OUTPUT_DIR = "/beegfs/scratch/ric.broccoli/kubacki.michal/SPATIAL_data/Segmentation/Vpt-segmentation/image_patches"

def parse_csv_row(row):
    """Parse a row from the CSV to extract filename, size, x, and y values."""
    # Split by commas and strip whitespace
    parts = [part.strip() for part in row.split(',')]
    
    # First part is the filename
    filename = parts[0]
    
    # Parse the key=value pairs
    params = {}
    for part in parts[1:]:
        if '=' in part:
            key, value = part.split('=', 1)
            params[key.strip()] = int(value.strip())
    
    return filename, params

def run_vpt_command(filename, size, x, y):
    """Construct and run the vpt command with the given parameters."""
    
    # Extract the base name without extension for the output file
    base_name = os.path.splitext(filename)[0]
    output_filename = f"patch_{base_name}_X_{x}_Y_{y}.png"
    
    # Construct the command
    cmd = [
        "vpt",
        "--verbose",
        "--log-level", "1",
        "extract-image-patch",
        "--input-images", f"{DATA_PATH}/{REGION}/images/",
        "--input-micron-to-mosaic", f"{DATA_PATH}/{REGION}/images/micron_to_mosaic_pixel_transform.csv",
        "--output-patch", os.path.join(OUTPUT_DIR, output_filename),
        "--center-x", str(x),
        "--center-y", str(y),
        "--size-x", str(size),
        "--size-y", str(size),
        "--input-z-index", "3",
        "--green-stain-name", "PolyT",
        "--blue-stain-name", "DAPI",
        "--normalization", "CLAHE"
    ]
    
    # Print the command for debugging
    print(f"\nRunning command for {filename}:")
    print(" ".join(cmd))
    
    # Run the command
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)  # shell=False is default and safer
        if result.returncode == 0:
            print(f"✓ Successfully processed {filename}")
        else:
            print(f"✗ Error processing {filename}")
            print(f"  Error output (stderr): {result.stderr}")
            print(f"  Output (stdout): {result.stdout}")
    except Exception as e:
        print(f"✗ Failed to run command for {filename}: {str(e)}")

def main():
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Check if input CSV exists
    if not os.path.exists(SEL_PATCHES):
        print(f"Error: {SEL_PATCHES} not found!")
        sys.exit(1)
    
    # Process each row in the CSV
    print(f"Processing entries from {SEL_PATCHES}...")
    
    with open(SEL_PATCHES, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:  # Skip empty lines
                continue
                
            try:
                filename, params = parse_csv_row(line)
                
                # Validate required parameters
                if 'size' not in params or 'x' not in params or 'y' not in params:
                    print(f"Warning: Line {line_num} missing required parameters (size, x, y). Skipping...")
                    continue
                
                # Run the command
                run_vpt_command(filename, params['size'], params['x'], params['y'])
                
            except Exception as e:
                print(f"Error processing line {line_num}: {str(e)}")
                continue
    
    print("\nProcessing complete!")

if __name__ == "__main__":
    main()