import os
import numpy as np
import random


def sanity_check_npz_data(root_dir):
    train_dir = os.path.join(root_dir, 'train')
    test_dir = os.path.join(root_dir, 'test')
    mean_path = os.path.join(root_dir, 'normalize_mean.npz')
    std_path = os.path.join(root_dir, 'normalize_std.npz')

    def check_npz_structure(npz_file):
        data = np.load(npz_file)
        keys = data.files
        if "data" not in keys:
            print(f"❌ Missing 'data' in {npz_file}")
            return None, None
        if "valid_time" in keys:
            print(f"⚠️ Warning: 'valid_time' key found in {npz_file} — consider removing it.")
        arr = data["data"]
        if not isinstance(arr, np.ndarray):
            print(f"❌ 'data' is not a numpy array in {npz_file}")
            return None, None
        return arr.shape, arr.dtype

    print("🔍 Checking .npz files in train and test folders...")
    all_shapes = []
    for split in ['train', 'test']:
        split_dir = os.path.join(root_dir, split)
        for fname in os.listdir(split_dir):
            if fname.endswith('.npz'):
                fpath = os.path.join(split_dir, fname)
                shape, dtype = check_npz_structure(fpath)
                if shape:
                    all_shapes.append(shape)

    if not all_shapes:
        print("❌ No valid .npz files found.")
        return

    feature_dims = [s[1] for s in all_shapes]
    unique_dims = set(feature_dims)
    if len(unique_dims) > 1:
        print(f"❌ Inconsistent number of features across samples: {unique_dims}")
    else:
        print(f"✅ All samples have consistent feature dimension: {unique_dims.pop()}")

    print("🔍 Checking normalization files...")
    if not os.path.exists(mean_path) or not os.path.exists(std_path):
        print("❌ normalize_mean.npz or normalize_std.npz is missing.")
        return

    mean = np.load(mean_path)["mean"]
    std = np.load(std_path)["std"]

    expected_dim = feature_dims[0]
    if mean.shape[0] != expected_dim:
        print(f"❌ Mismatch: normalize_mean.npz has shape {mean.shape}, expected {expected_dim}")
    else:
        print("✅ normalize_mean.npz matches feature count")

    if std.shape[0] != expected_dim:
        print(f"❌ Mismatch: normalize_std.npz has shape {std.shape}, expected {expected_dim}")
    else:
        print("✅ normalize_std.npz matches feature count")


    # 🧪 Display first 5 rows of 4 random samples from train/test
    for split in ['train', 'test']:
        print(f"\n📦 Sampling 4 random .npz files from '{split}' directory...")
        split_dir = os.path.join(root_dir, split)
        npz_files = [f for f in os.listdir(split_dir) if f.endswith('.npz')]
        sample_files = random.sample(npz_files, min(4, len(npz_files)))
        for fname in sample_files:
            path = os.path.join(split_dir, fname)
            data = np.load(path)["data"]
            print(f"🔹 File: {fname} — shape: {data.shape}")


# Example usage
sanity_check_npz_data("finetuning_1.40625deg")  # Update this path if needed