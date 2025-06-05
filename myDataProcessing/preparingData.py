import os
import numpy as np
import pandas as pd

input_data_path = 'era5_main_dataset_30.parquet'
output_root_dir = 'finetuning_1.40625deg'

train_dir = os.path.join(output_root_dir, 'train')
test_dir = os.path.join(output_root_dir, 'test')

columns = ['latitude', 'longitude', 'valid_time', 'stl1', 'slhf', 'u10', 'tclw', 'skt', 'msl', 'cvl',
           'v10', 'str', 'tcrw', 'sp', 'ssr', 'tcsw', 'cbh', 'sshf', 'tcc', 'pev', 'stl2', 'tcw',
           'd2m', 'tciw', 'tcwv', 'tp', 'cp', 'cvh', 'tco3', 't2m', 'e', 'lsp', 'fg10']

non_feature_cols = ['latitude', 'longitude', 'valid_time']
feature_cols = [col for col in columns if col not in non_feature_cols]

os.makedirs(output_root_dir, exist_ok=True)
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)


def pivot_to_array(df, var_list, latitudes, longitudes):
    """
    Create a dict of variable: 4D np.array shaped (T, 1, H, W)
    where T=time, H=lat, W=lon.
    """
    T = df['valid_time'].nunique()
    H = len(latitudes)
    W = len(longitudes)

    # Create empty arrays
    data_arrays = {}
    for var in var_list:
        data_arrays[var] = np.full((T, 1, H, W), np.nan, dtype=np.float32)

    # Map lat/lon to indices
    lat_to_idx = {lat: i for i, lat in enumerate(latitudes)}
    lon_to_idx = {lon: i for i, lon in enumerate(longitudes)}

    # Map time to index (sorted)
    times = sorted(df['valid_time'].unique())
    time_to_idx = {t: i for i, t in enumerate(times)}

    # Fill arrays
    for _, row in df.iterrows():
        t_idx = time_to_idx[row['valid_time']]
        lat_idx = lat_to_idx[row['latitude']]
        lon_idx = lon_to_idx[row['longitude']]
        for var in var_list:
            data_arrays[var][t_idx, 0, lat_idx, lon_idx] = row[var]

    return data_arrays, times


print("Loading dataset...")
df = pd.read_parquet(input_data_path)
df = df[columns]

# Filter train/test by valid_time (daily timestamps)
df['valid_time'] = pd.to_datetime(df['valid_time'])

train_df = df[(df['valid_time'] >= '2020-01-01') & (df['valid_time'] <= '2023-12-31')]
test_df = df[(df['valid_time'] >= '2024-01-01') & (df['valid_time'] <= '2024-12-31')]

print(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}")

# Get sorted unique lat/lon arrays
latitudes = np.sort(df['latitude'].unique())
longitudes = np.sort(df['longitude'].unique())

print("Processing train data...")
train_data, train_times = pivot_to_array(train_df, feature_cols, latitudes, longitudes)
print("Processing test data...")
test_data, test_times = pivot_to_array(test_df, feature_cols, latitudes, longitudes)

# Save train npz file
train_save_path = os.path.join(train_dir, 'train_data.npz')
np.savez_compressed(train_save_path, **train_data)
print(f"Saved training data to {train_save_path}")



# Save test npz file
test_save_path = os.path.join(test_dir, 'test_data.npz')
np.savez_compressed(test_save_path, **test_data)
print(f"Saved test data to {test_save_path}")



# Compute normalization (mean/std) using ClimaX's year-averaged method
normalize_mean = {var: [] for var in feature_cols}
normalize_std = {var: [] for var in feature_cols}

print("Calculating per-year normalization statistics...")

# Loop through each year in the training set
for year in range(2020, 2024):
    print(f"  Processing year {year}...")
    year_df = train_df[train_df['valid_time'].dt.year == year]
    year_data, _ = pivot_to_array(year_df, feature_cols, latitudes, longitudes)

    for var in feature_cols:
        arr = year_data[var]  # shape (T, 1, H, W)
        valid_values = arr[~np.isnan(arr)]
        if valid_values.size == 0:
            # Handle possible empty slice
            mean = 0.0
            std = 1.0
        else:
            mean = np.mean(valid_values)
            std = np.std(valid_values)

        normalize_mean[var].append(mean)
        normalize_std[var].append(std)

# Final normalization calculation
final_mean = {}
final_std = {}

for var in feature_cols:
    mean_array = np.array(normalize_mean[var], dtype=np.float32)  # (years,)
    std_array = np.array(normalize_std[var], dtype=np.float32)    # (years,)

    mean = np.mean(mean_array)
    variance = np.mean(std_array ** 2) + np.mean(mean_array ** 2) - mean ** 2
    std = np.sqrt(variance)

    # Save as 1D arrays with shape (1,) to avoid zero-dimensional scalar issue
    final_mean[var] = np.array([mean], dtype=np.float32)
    final_std[var] = np.array([std], dtype=np.float32)

# Save final normalization
np.savez(os.path.join(output_root_dir, 'normalize_mean.npz'), **final_mean)
np.savez(os.path.join(output_root_dir, 'normalize_std.npz'), **final_std)
print("Saved normalization mean/std files.")



# Save lat/lon arrays
np.save(os.path.join(output_root_dir, 'lat.npy'), latitudes)
np.save(os.path.join(output_root_dir, 'lon.npy'), longitudes)
print("Saved lat.npy and lon.npy")