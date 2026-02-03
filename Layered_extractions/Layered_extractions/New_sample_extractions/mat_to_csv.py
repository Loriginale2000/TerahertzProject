import os
import scipy.io
import pandas as pd

# Function to convert .mat files to .csv
# Works for the file name convention from Ben's scans.

def convert_mat_to_csv(mat_file_path, output_dir=None):
    """
    Converts a .mat file containing 'tAxis' and 'sAxis' to a CSV file with
    columns 'Time_ps' and 'Signal'.

    Parameters:
    - mat_file_path (str): Full path to the .mat file.
    - output_dir (str, optional): Directory to save the CSV. Defaults to the same folder as the .mat file.
    """
    # Load the .mat file
    mat_data = scipy.io.loadmat(mat_file_path)
    
    # Extract time and signal axes
    try:
        time = mat_data['tAxis'].squeeze()
        signal = mat_data['sAxis'].squeeze()
    except KeyError as e:
        raise ValueError(f"Missing expected key in .mat file: {e}")
    
    # Build DataFrame
    df = pd.DataFrame({
        'Time_ps': time,
        'Signal': signal
    })

    # Determine output path
    if output_dir is None:
        output_dir = os.path.dirname(mat_file_path)
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(mat_file_path))[0]
    csv_file_path = os.path.join(output_dir, f"{base_name}.csv")
    
    # Save as CSV
    df.to_csv(csv_file_path, index=False)
    print(f"Saved CSV to: {csv_file_path}")