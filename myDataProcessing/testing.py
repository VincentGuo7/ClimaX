import numpy as np
import sys

# def main():    
#     npz_path = "finetuning_1.40625deg/val/2023_0.npz"
#     data = np.load(npz_path)
#     print(f"Shapes of arrays in {npz_path}:")
#     for key in data.files:
#         arr = data[key]
#         print(f"  {key}: {arr.shape}")
#     data.close()

# if __name__ == "__main__":
#     main()


def check_npz_file(npz_path):
    print(f"Loading {npz_path} ...")
    data = np.load(npz_path)

    for key in data.files:
        arr = data[key]
        print(f"\nChecking array '{key}' with shape {arr.shape} and dtype {arr.dtype}:")

        has_nan = np.isnan(arr).any()
        has_inf = np.isinf(arr).any()

        if has_nan:
            print(f"  ⚠️ NaNs detected!")
        if has_inf:
            print(f"  ⚠️ Infs detected!")

        print(f"  min: {np.nanmin(arr)}")
        print(f"  max: {np.nanmax(arr)}")
        print(f"  mean: {np.nanmean(arr)}")
        print(f"  std: {np.nanstd(arr)}")

    print("\nCheck completed.")

def main():    
    npz_path = "finetuning_1.40625deg/train/2020_1.npz"
    check_npz_file(npz_path)


if __name__ == "__main__":
    main()