import os

if __name__ == '__main__':
    DATA_DIR = "/data"
    OUTPUT_DIR = "/outputs"
    NUM_COMPONENTS = 5
    LLE_FILENAME = os.path.join(OUTPUT_DIR, "fitted_lle.joblib")
    DATA_MASK_REFERENCE_FILE = os.path.join(DATA_DIR) + "data_source.txt"
    FILE_TYPE = "tiff"

    print("Writing to a file in the outside directory")
    with open(os.path.join(DATA_DIR, "new_file.txt")) as outfile:
        outfile.write("It worked!")
