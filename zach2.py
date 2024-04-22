import pandas as pd
from pathlib import Path

# Set the path to the parent directory containing all the folders
parent_directory_path = Path.cwd() #Path("/path/to/parent/directory")

# Define the moving average window size
window_size = 10

# Function to process the EMG files
def process_emg_file(file_path):
    # Read the csv file into a DataFrame
    df = pd.read_csv(file_path)

    # Take the absolute value of all data starting from the second row
    df.iloc[1:] = df.iloc[1:].abs()

    # Calculate the moving average with the specified window size, skipping NaN values
    # that would appear at the beginning of the DataFrame due to `min_periods`.
    moving_avg = df.iloc[1:].rolling(window=window_size, min_periods=1).mean()

    # Downsample by selecting every tenth row, starting with row 2 (index 1).
    # This effectively skips the first moving average window that only averages one value.
    # Add 1 to the index to account for the zero-based index and reintroduce the header row.
    df_filtered = moving_avg.iloc[9::10].reset_index(drop=True)
    df_filtered_with_header = pd.concat([df.iloc[:1], df_filtered], ignore_index=True)

    # Write the modified DataFrame to a new CSV file with the suffix 'emg2.csv'
    new_file_path = file_path.parent / f"{file_path.stem}2.csv"
    df_filtered_with_header.to_csv(new_file_path, index=False)
    print(f"Processed and saved: {new_file_path}")

# counter = 0

# Traverse all subfolders and process the EMG files
for emg_file in parent_directory_path.rglob("*emg.csv"):
    # print(emg_file)
    # counter += 1
    # if counter > 2:
    #     break
    process_emg_file(emg_file)

print("Processing complete!")
