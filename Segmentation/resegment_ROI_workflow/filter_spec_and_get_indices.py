import json
import argparse
import copy

def filter_spec_and_get_indices(spec_path: str, roi_path: str, output_spec_path: str):
    """
    Filters a segmentation specification file to include only tiles that
    overlap with a given region of interest (ROI). It writes the filtered
    spec to a new file and prints the original indices of the filtered tiles.

    Args:
        spec_path: Path to the input segmentation_specification.json file.
        roi_path: Path to the roi_coords.json file.
        output_spec_path: Path to write the filtered segmentation_spec.json file.
    """
    with open(spec_path, 'r') as f:
        spec_data = json.load(f)

    with open(roi_path, 'r') as f:
        roi_coords = json.load(f)

    roi_x_min = roi_coords['x_min']
    roi_y_min = roi_coords['y_min']
    roi_x_max = roi_coords['x_max']
    roi_y_max = roi_coords['y_max']

    filtered_spec = copy.deepcopy(spec_data)
    filtered_windows = []
    original_indices = []

    for i, window in enumerate(spec_data['window_grid']['windows']):
        tile_x_min = window[0]
        tile_y_min = window[1]
        tile_x_max = tile_x_min + window[2]  # x_start + width
        tile_y_max = tile_y_min + window[3]  # y_start + height

        # Check for overlap
        if (tile_x_min < roi_x_max and tile_x_max > roi_x_min and
                tile_y_min < roi_y_max and tile_y_max > roi_y_min):
            filtered_windows.append(window)
            original_indices.append(i)

    filtered_spec['window_grid']['windows'] = filtered_windows
    filtered_spec['window_grid']['num_tiles'] = len(filtered_windows)

    with open(output_spec_path, 'w') as f:
        json.dump(filtered_spec, f, indent=4)

    # Print original indices to stdout
    for index in original_indices:
        print(index)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter a spec file and get original tile indices for an ROI.")
    parser.add_argument('--input-spec', type=str, required=True,
                        help='Path to the full segmentation_specification.json file.')
    parser.add_argument('--input-roi', type=str, required=True,
                        help='Path to the roi_coords.json file.')
    parser.add_argument('--output-spec', type=str, required=True,
                        help='Path to save the filtered segmentation_specification.json file.')

    args = parser.parse_args()

    filter_spec_and_get_indices(args.input_spec, args.input_roi, args.output_spec)