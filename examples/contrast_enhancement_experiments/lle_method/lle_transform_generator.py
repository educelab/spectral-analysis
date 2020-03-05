import os

if __name__ == '__main__':
    DATA_DIR = os.path.expandvars("$SCRATCH/2020-hyperspectral/data")
    OUTPUT_DIR = os.path.expandvars("$SCRATCH/2020-hyperspectral/outputs")
    NUM_COMPONENTS = 5
    LLE_FILENAME = os.path.join(OUTPUT_DIR, "fitted_lle.joblib")
    DATA_MASK_REFERENCE_FILE = os.path.join(DATA_DIR) + "data_source.txt"
    FILE_TYPE = "tiff"

    print("Writing to a file in the outside directory")
    with open(os.path.join(DATA_DIR, "new_file.txt"), 'w') as outfile:
        outfile.write("It worked!")
