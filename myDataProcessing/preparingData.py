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


os.makedirs(output_root_dir, exist_ok=True)


def save_coordinate_data(df, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    grouped = df.groupby(['latitude', 'longitude'])
    for (lat, lon), group in grouped:
        group = group.sort_values('valid_time')
        data_array = group.drop(columns=['latitude', 'longitude', 'valid_time']).to_numpy(dtype=np.float32)
        fname = f'lat_{lat:.5f}_lon_{lon:.5f}.npz'
        np.savez_compressed(os.path.join(save_dir, fname), data=data_array)

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

non_feature_cols = ['latitude', 'longitude', 'valid_time']
input_columns = [col for col in df.columns if col not in non_feature_cols]

with open(os.path.join(output_root_dir, 'weather_features_titles.txt'), 'w') as f:
    for col in input_columns:
        f.write(f"{col}\n")


normalize_mean = df[input_columns].mean().to_numpy(dtype=np.float32)
normalize_std = df[input_columns].std().to_numpy(dtype=np.float32)

# Save to NPZ files
np.savez_compressed(os.path.join(output_root_dir, 'normalize_mean.npz'), mean=normalize_mean)
np.savez_compressed(os.path.join(output_root_dir, 'normalize_std.npz'), std=normalize_std)

print("Saved normalize_mean.npz and normalize_std.npz")




# === Save lat/lon arrays for indexing ===
print('Saving lat/lon arrays...')

latitudes = np.sort(df['latitude'].unique())
longitudes = np.sort(df['longitude'].unique())
np.save(os.path.join(output_root_dir, 'lat.npy'), latitudes)
np.save(os.path.join(output_root_dir, 'lon.npy'), longitudes)

print('Saved lat.npy and lon.npy')
