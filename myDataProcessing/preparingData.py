import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

input_data_path = 'era5_main_dataset_30.parquet'  # Update this path
output_root_dir = 'finetuning_1.40625deg'
train_dir = os.path.join(output_root_dir, 'train')
test_dir = os.path.join(output_root_dir, 'test')

columns = ['latitude', 'longitude', 'valid_time', 'stl1', 'slhf', 'u10', 'tclw', 'skt', 'msl', 'cvl', 
           'v10', 'str', 'tcrw', 'sp', 'ssr', 'tcsw', 'cbh', 'sshf', 'tcc', 'pev', 'stl2', 'tcw', 
           'd2m', 'tciw', 'tcwv', 'tp', 'cp', 'cvh', 'tco3', 't2m', 'e', 'lsp', 'fg10']

non_feature_cols = ['latitude', 'longitude', 'valid_time']
input_columns = [col for col in columns if col not in non_feature_cols]


os.makedirs(output_root_dir, exist_ok=True)


def save_coordinate_data(df, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    grouped = df.groupby(['latitude', 'longitude'])
    for (lat, lon), group in grouped:
        group = group.sort_values('valid_time')
        record = {}
        for col in input_columns:
            record[col] = group[col].to_numpy(dtype=np.float32)
        fname = f'lat_{lat:.5f}_lon_{lon:.5f}.npz'
        np.savez_compressed(os.path.join(save_dir, fname), **record)


# def save_coordinate_data(df, save_dir):
#     os.makedirs(save_dir, exist_ok=True)
#     grouped = df.groupby(['latitude', 'longitude'])
#     for (lat, lon), group in grouped:
#         group = group.sort_values('valid_time')
#         arrays = {
#             col: group[col].to_numpy(dtype=np.float32)
#             for col in group.columns
#             if col not in ['latitude', 'longitude', 'valid_time']
#         }
#         fname = f'lat_{lat:.5f}_lon_{lon:.5f}.npz'
#         np.savez_compressed(os.path.join(save_dir, fname), **arrays)

print('Loading data...')
df = pd.read_parquet(input_data_path)
df = df[columns]



print('Splitting coordinates...')
unique_coords = df[['latitude', 'longitude']].drop_duplicates()
train_coords, test_coords = train_test_split(unique_coords, test_size=0.2, random_state=42)

train_df = pd.merge(df, train_coords, on=['latitude', 'longitude'], how='inner')
test_df = pd.merge(df, test_coords, on=['latitude', 'longitude'], how='inner')

print(f'Train samples: {len(train_df)}, Test samples: {len(test_df)}')

print('Saving train data...')
save_coordinate_data(train_df, train_dir)
print('Saving test data...')
save_coordinate_data(test_df, test_dir)




# === Save normalization files (dummy placeholders, you should compute real mean/std) ===
print('Saving dummy normalization files...')



with open(os.path.join(output_root_dir, 'weather_features_titles.txt'), 'w') as f:
    for col in input_columns:
        f.write(f"{col}\n")


normalize_mean = {}
normalize_std = {}

grouped = train_df.groupby(['latitude', 'longitude'])
for col in input_columns:
    all_values = []
    for _, group in grouped:
        sorted_group = group.sort_values('valid_time')
        all_values.append(sorted_group[col].values.astype(np.float32))
    stacked = np.stack(all_values)  # shape: [num_locations, time]
    mean = np.array([stacked.mean()])  # ensure shape (1,)
    std = np.array([stacked.std()])    # ensure shape (1,)
    normalize_mean[col] = mean
    normalize_std[col] = std

np.savez(os.path.join(output_root_dir, 'normalize_mean.npz'), **normalize_mean)
np.savez(os.path.join(output_root_dir, 'normalize_std.npz'), **normalize_std)

print("Saved normalize_mean.npz and normalize_std.npz")




# === Save lat/lon arrays for indexing ===
print('Saving lat/lon arrays...')

latitudes = np.sort(df['latitude'].unique())
longitudes = np.sort(df['longitude'].unique())
np.save(os.path.join(output_root_dir, 'lat.npy'), latitudes)
np.save(os.path.join(output_root_dir, 'lon.npy'), longitudes)

print('Saved lat.npy and lon.npy')
