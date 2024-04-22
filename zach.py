#!/usr/bin/env python3

import pandas as pd
from pathlib import Path

# Define the specific folder path where the Processed_Data folder resides
parent_path = Path.cwd()  # Replace this with the path to your git repository if needed

# Define the relative path to Processed_Data from the parent path
processed_data_path = parent_path / "Processed_Data"

# Define the known suffixes for the .csv files and the columns you want to keep for each
csv_suffixes_columns = {
    "angle": ["time", "hip_flexion_r", "knee_angle_r","ankle_angle_r","hip_flexion_l"],  # Add your actual column names here
    "velocity": ["hip_flexion_velocity_r", "knee_velocity_r","ankle_velocity_r","hip_flexion_velocity_l"],  # Add your actual column names here
    "emg2": ["RTA","LRF","RRF","LBF","RBF","LGMED","RGMED","RMGAS","LVL","RVL","LGRAC","RGRAC","LGMAX","RGMAX"],  # Add your actual column names here
    "imu_real": ["RShank_ACCX","RShank_ACCY","RShank_ACCZ","RShank_GYROX","RShank_GYROY","RShank_GYROZ","LAThigh_ACCX",
                 "LAThigh_ACCY","LAThigh_ACCZ","LAThigh_GYROX","LAThigh_GYROY","LAThigh_GYROZ","RAThigh_ACCX","RAThigh_ACCY",
                 "RAThigh_ACCZ","RAThigh_GYROX","RAThigh_GYROY","RAThigh_GYROZ","LPThigh_ACCX","LPThigh_ACCY","LPThigh_ACCZ",
                 "LPThigh_GYROX","LPThigh_GYROY","LPThigh_GYROZ","RPThigh_ACCX","RPThigh_ACCY","RPThigh_ACCZ","RPThigh_GYROX",
                 "RPThigh_GYROY","RPThigh_GYROZ","LPelvis_ACCX","LPelvis_ACCY","LPelvis_ACCZ","LPelvis_GYROX","LPelvis_GYROY",
                 "LPelvis_GYROZ","RPelvis_ACCX","RPelvis_ACCY","RPelvis_ACCZ","RPelvis_GYROX","RPelvis_GYROY","RPelvis_GYROZ"],  # Add your actual column names here
    "moment_filt": ["knee_angle_l_moment", "ankle_angle_l_moment"],  # Add your actual column names here
    # ... other suffixes and their corresponding columns...
}

# Function to determine which rows to skip when downsampling
def should_skip_row(row_index):
    if row_index == 0:  # Keep the header row
        return False
    return (row_index - 1) % 10 != 0  # Then keep every tenth data row

# Function to process each activity subfolder
def process_activity_subfolder(activity_folder_path, csv_suffixes_columns):
    ab_name = activity_folder_path.parent.name
    activity_name = activity_folder_path.name

    dataframes = {}

    for suffix, columns in csv_suffixes_columns.items():
        csv_filename = f"{ab_name}_{activity_name}_{suffix}.csv"
        csv_path = activity_folder_path / csv_filename

        if csv_path.is_file():
            # if suffix == "emg":  # Check if this is the 'emg' CSV
            #     # Downsample the emg CSV file by reading every tenth row after the first
            #     df = pd.read_csv(csv_path, usecols=columns, skiprows=should_skip_row)
            # else:
            df = pd.read_csv(csv_path, usecols=columns)

            dataframes[suffix] = df

    if dataframes:
        combined_df = pd.concat(dataframes.values(), axis=1, ignore_index=False)
        combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]
        combined_df.to_parquet(activity_folder_path / f"{ab_name}_{activity_name}_combined_ang_vel_emg2_imu.parquet")    
    else:
        print(f"No CSV files found for: {ab_name}, {activity_name}")

# Loop through each ABXX directory and process the CSV files
# counter = 0
for ab_path in processed_data_path.glob("AB*"):
    # counter += 1
    # if counter > 2:
    #     break
    print("Starting ",ab_path)
    if ab_path.is_dir():
        for activity_path in ab_path.glob("*"):  # Replace "*" with a specific pattern if needed
            if activity_path.is_dir():
                process_activity_subfolder(activity_path, csv_suffixes_columns)

print("All done!")
