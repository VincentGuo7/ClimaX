import numpy as np
import sys

def main():    
    npz_path = "finetuning_1.40625deg/val/2023_0.npz"
    data = np.load(npz_path)
    print(f"Shapes of arrays in {npz_path}:")
    for key in data.files:
        arr = data[key]
        print(f"  {key}: {arr.shape}")
    data.close()

if __name__ == "__main__":
    main()