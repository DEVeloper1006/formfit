import os

def count_files_in_directory(directory_path):
    file_count = 0
    for root, dirs, files in os.walk(directory_path):
        file_count += len(files)
    return file_count

# Example usage
directory_path = 'refined_data'
print(f"Number of files in '{directory_path}': {count_files_in_directory(directory_path)}")