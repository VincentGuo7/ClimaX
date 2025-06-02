import os
import numpy as np

def check_npz_files(folder, expected_keys, num_features_check=True):
    npz_files = [f for f in os.listdir(folder) if f.endswith('.npz')]
    assert npz_files, f"No .npz files found in {folder}"
    
    for file in npz_files:
        path = os.path.join(folder, file)
        with np.load(path) as data:
            keys = set(data.files)
            missing = [k for k in expected_keys if k not in keys]
            if missing:
                raise ValueError(f"{file} is missing keys: {missing}")
            if num_features_check:
                for k in expected_keys:
                    if not isinstance(data[k], np.ndarray):
                        raise TypeError(f"{file} key {k} is not a numpy array")
                    if data[k].ndim != 1:
                        raise ValueError(f"{file} key {k} should be 1D time series, got shape {data[k].shape}")


def check_lat_lon(lat_path, lon_path):
    lat = np.load(lat_path)
    lon = np.load(lon_path)
    assert lat.ndim == 1, "Latitude should be 1D"
    assert lon.ndim == 1, "Longitude should be 1D"
    assert np.all(np.diff(lat) > 0), "Latitudes not sorted"
    assert np.all(np.diff(lon) > 0), "Longitudes not sorted"
    print(f"Lat shape: {lat.shape}, Lon shape: {lon.shape}")


def check_normalization_files(mean_path, std_path, expected_keys):
    mean = np.load(mean_path)
    std = np.load(std_path)

    assert set(mean.files) == set(expected_keys), "Mismatch in mean keys and feature keys"
    assert set(std.files) == set(expected_keys), "Mismatch in std keys and feature keys"

    for k in expected_keys:
        assert mean[k].shape == std[k].shape, f"Mismatch in shape for {k}: mean {mean[k].shape}, std {std[k].shape}"
        assert mean[k].ndim == 1, f"Normalization vector for {k} should be 1D"


def main(dataset_root):
    train_dir = os.path.join(dataset_root, "train")
    test_dir = os.path.join(dataset_root, "test")
    
    # Load expected feature names
    feature_file = os.path.join(dataset_root, "weather_features_titles.txt")
    assert os.path.exists(feature_file), "Missing weather_features_titles.txt"
    with open(feature_file, "r") as f:
        features = [line.strip() for line in f.readlines() if line.strip()]
    
    assert features, "No features found in weather_features_titles.txt"

    print(f"Checking {len(features)} features: {features}")
    
    # Check .npz files
    print("Checking training .npz files...")
    check_npz_files(train_dir, features)

    print("Checking test .npz files...")
    check_npz_files(test_dir, features)

    # Check lat/lon arrays
    print("Checking lat/lon...")
    check_lat_lon(
        os.path.join(dataset_root, "lat.npy"),
        os.path.join(dataset_root, "lon.npy")
    )

    # Check normalization files
    print("Checking normalization files...")
    check_normalization_files(
        os.path.join(dataset_root, "normalize_mean.npz"),
        os.path.join(dataset_root, "normalize_std.npz"),
        features
    )

    print("✅ All checks passed!")

if __name__ == "__main__":
    main("finetuning_1.40625deg")  # Replace with your root directory if different