import os
import numpy as np
import pickle
import pandas as pd

# Function to process .npy files
def process_npy_files(input_folder, output_folder):
    # Get a list of all .npy files in the input folder
    npy_files = [f for f in os.listdir(input_folder) if f.endswith('.npy')]
    
    if not npy_files:
        print("No .npy files found in the input folder.")
        return

    # Load all files and find the minimum length
    lengths = []
    data_list = []

    for file in npy_files:
        
        data = pd.read_csv(os.path.join(input_folder, file))
        data_list.append(data)
        lengths.append(data.shape[0])

    # Determine the minimum length rounded down to the nearest multiple of 25
    min_length = min(lengths)
    rounded_min_length = (min_length // 25) * 25
    print(f"Minimum length: {min_length}, Rounded length: {rounded_min_length}")

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Process each file
    for i, (file, data) in enumerate(zip(npy_files, data_list)):
        
        data = data.to_numpy()
        truncated_data = data[:rounded_min_length]  # Truncate to the rounded minimum length
        reshaped_data = truncated_data.reshape(-1, 25, data.shape[1])  # Group rows into chunks of 25
        concatenated_data = reshaped_data.reshape(-1, 25 * data.shape[1])  # Flatten each group into one row
        
        new_file = file.replace(".npy", "")
        # Save to a new .npy file
        output_path = os.path.join(output_folder, f"processed_{new_file}.npy")
        np.save(output_path, concatenated_data)
        print(f"Processed file saved to: {output_path}")

# Example usage
input_folder = "final_data"
output_folder = "final_preprocess_output"
os.makedirs(output_folder, exist_ok=True)
process_npy_files(input_folder, output_folder)

