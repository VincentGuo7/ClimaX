import os
import numpy as np
import pandas as pd

input_data_path = 'era5_main_dataset_30.parquet'
output_root_dir = 'finetuning_1.40625deg'

train_dir = os.path.join(output_root_dir, 'train')
val_dir = os.path.join(output_root_dir, 'val')
test_dir = os.path.join(output_root_dir, 'test')

columns = ['latitude', 'longitude', 'valid_time', 'stl1', 'slhf', 'u10', 'tclw', 'skt', 'msl', 'cvl',
           'v10', 'str', 'tcrw', 'sp', 'ssr', 'tcsw', 'cbh', 'sshf', 'tcc', 'pev', 'stl2', 'tcw',
           'd2m', 'tciw', 'tcwv', 'tp', 'cp', 'cvh', 'tco3', 't2m', 'e', 'lsp', 'fg10']

non_feature_cols = ['latitude', 'longitude', 'valid_time']
feature_cols = [col for col in columns if col not in non_feature_cols]

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)


def pivot_to_array(df, var_list, latitudes, longitudes):
    T = df['valid_time'].nunique()
    H = len(latitudes)
    W = len(longitudes)
    data_arrays = {var: np.full((T, 1, H, W), np.nan, dtype=np.float32) for var in var_list}
    lat_to_idx = {lat: i for i, lat in enumerate(latitudes)}
    lon_to_idx = {lon: i for i, lon in enumerate(longitudes)}
    times = sorted(df['valid_time'].unique())
    time_to_idx = {t: i for i, t in enumerate(times)}
    for _, row in df.iterrows():
        t_idx = time_to_idx[row['valid_time']]
        lat_idx = lat_to_idx[row['latitude']]
        lon_idx = lon_to_idx[row['longitude']]
        for var in var_list:
            data_arrays[var][t_idx, 0, lat_idx, lon_idx] = row[var]
    return data_arrays, times


def compute_climatology(data):
    return {var: np.nanmean(data[var], axis=0) for var in data}


def save_shards_and_climatology(df, years, out_dir, tag):
    all_data = []
    for year in years:
        year_df = df[df['valid_time'].dt.year == year]
        if len(year_df) == 0:
            continue
        # Split year into 2 shards
        year_df = year_df.sort_values('valid_time')
        unique_days = year_df['valid_time'].drop_duplicates().sort_values()
        mid = len(unique_days) // 2
        shard_dates = [unique_days[:mid], unique_days[mid:]]
        for i, dates in enumerate(shard_dates):
            shard_df = year_df[year_df['valid_time'].isin(dates)]
            data, _ = pivot_to_array(shard_df, feature_cols, latitudes, longitudes)
            np.savez_compressed(os.path.join(out_dir, f"{year}_{i}.npz"), **data)
            all_data.append(data)
    # Concatenate all for climatology
    merged_data = {var: np.concatenate([d[var] for d in all_data], axis=0) for var in feature_cols}
    climatology = compute_climatology(merged_data)
    np.savez_compressed(os.path.join(out_dir, 'climatology.npz'), **climatology)
    print(f"Saved {tag} shards and climatology to {out_dir}")


print("Loading dataset...")
df = pd.read_parquet(input_data_path)
df = df[columns]
df['valid_time'] = pd.to_datetime(df['valid_time'])

latitudes = np.sort(df['latitude'].unique())
longitudes = np.sort(df['longitude'].unique())

# Split by date
train_df = df[(df['valid_time'] >= '2020-01-01') & (df['valid_time'] < '2023-01-01')]
val_df = df[(df['valid_time'] >= '2023-01-01') & (df['valid_time'] < '2024-01-01')]
test_df = df[(df['valid_time'] >= '2024-01-01') & (df['valid_time'] <= '2024-12-31')]

save_shards_and_climatology(train_df, [2020, 2021, 2022], train_dir, "train")
save_shards_and_climatology(val_df, [2023], val_dir, "val")
save_shards_and_climatology(test_df, [2024], test_dir, "test")

# Normalization
normalize_mean = {var: [] for var in feature_cols}
normalize_std = {var: [] for var in feature_cols}

print("Calculating per-year normalization statistics...")
for year in [2020, 2021, 2022]:
    print(f"  Processing year {year}...")
    year_df = train_df[train_df['valid_time'].dt.year == year]
    year_data, _ = pivot_to_array(year_df, feature_cols, latitudes, longitudes)
    for var in feature_cols:
        arr = year_data[var]
        valid_values = arr[~np.isnan(arr)]
        mean = np.mean(valid_values) if valid_values.size > 0 else 0.0
        std = np.std(valid_values) if valid_values.size > 0 else 1.0
        normalize_mean[var].append(mean)
        normalize_std[var].append(std)

final_mean = {}
final_std = {}
for var in feature_cols:
    mean_array = np.array(normalize_mean[var], dtype=np.float32)
    std_array = np.array(normalize_std[var], dtype=np.float32)
    mean = np.mean(mean_array)
    variance = np.mean(std_array ** 2) + np.mean(mean_array ** 2) - mean ** 2
    std = np.sqrt(variance)
    final_mean[var] = np.array([mean], dtype=np.float32)
    final_std[var] = np.array([std], dtype=np.float32)

np.savez(os.path.join(output_root_dir, 'normalize_mean.npz'), **final_mean)
np.savez(os.path.join(output_root_dir, 'normalize_std.npz'), **final_std)
print("Saved normalization mean/std files.")

np.save(os.path.join(output_root_dir, 'lat.npy'), latitudes)
np.save(os.path.join(output_root_dir, 'lon.npy'), longitudes)
print("Saved lat.npy and lon.npy")