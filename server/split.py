import os
import numpy as np
import pandas as pd

exercises = {"09" : 0, "10" : 1, "11" : 2, "12" : 3} 

def process_npy_files_to_csv(input_folder, train_csv, val_csv, test_csv):
    # Initialize lists to collect data for training, validation, and testing
    train_data = []
    val_data = []
    test_data = []

    # Process each file in the input folder
    for file in os.listdir(input_folder):
        if file.endswith(".npy"):
            file_path = os.path.join(input_folder, file)

            # Load the .npy file
            data = np.load(file_path)
                        
            # Extract exercise number from the filename (modify this part as needed)
            exercise_number = exercises[file.split('_')[2]]  # Example: Assumes filename format is "exercise_X.npy"

            # Create a sliding window of size 2 rows (50 frames) with a step of 1 row (25 frames)
            window_size = 2  # Number of rows in each window
            step_size = 1    # Step size for sliding window

            # List to store the sliding window results for this file
            sliding_window_data = []

            for i in range(0, data.shape[0] - window_size + 1, step_size):
                window = data[i:i + window_size].flatten()  # Flatten rows into a single row
                sliding_window_data.append(window)

            # Convert the sliding window data into a DataFrame
            file_df = pd.DataFrame(sliding_window_data)
            
            # Add a column for the exercise number
            file_df['exercise'] = exercise_number

            # Split the data into training, validation, and testing sets
            num_rows = len(file_df)
            train_split = int(0.6 * num_rows)  # 60% for training
            val_split = train_split + int(0.2 * num_rows)  # 20% for validation

            train_data.append(file_df[:train_split])
            val_data.append(file_df[train_split:val_split])
            test_data.append(file_df[val_split:])

    # Concatenate all the collected data for training, validation, and testing
    train_df = pd.concat(train_data, ignore_index=True)
    val_df = pd.concat(val_data, ignore_index=True)
    test_df = pd.concat(test_data, ignore_index=True)

    # Save the DataFrames to CSV files
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    print(f"Training data saved to {train_csv}")
    print(f"Validation data saved to {val_csv}")
    print(f"Testing data saved to {test_csv}")

# Example usage
input_folder = "final_preprocess_output"  # Folder containing .npy files
train_csv = "old_training_data.csv"
val_csv = "old_validation_data.csv"
test_csv = "old_testing_data.csv"

process_npy_files_to_csv(input_folder, train_csv, val_csv, test_csv)