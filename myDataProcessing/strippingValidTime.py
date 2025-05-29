import os
import numpy as np
from glob import glob

def strip_valid_time_from_npz(npz_dir):
    npz_files = glob(os.path.join(npz_dir, "*.npz"))
    print(f"Found {len(npz_files)} .npz files in {npz_dir}")

    for path in npz_files:
        data = np.load(path, allow_pickle=True)
        if 'valid_time' in data:
            new_data = {k: v for k, v in data.items() if k != 'valid_time'}
            np.savez_compressed(path, **new_data)
            print(f"✅ Stripped 'valid_time' from {os.path.basename(path)}")
        else:
            print(f"ℹ️ No 'valid_time' key in {os.path.basename(path)}")

# Example usage
strip_valid_time_from_npz("finetuning_1.40625deg/train")
strip_valid_time_from_npz("finetuning_1.40625deg/test")