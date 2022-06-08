import os
import joblib

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def plot_principle_components(pca_like: PCA, output_dir):
    for i, component in enumerate(pca_like.components_):
        plt.figure(figsize=(24, 12))
        plt.bar(range(len(component)), component)
        plt.title(f"Component weights vs. band index for Principle Component {i}")
        plt.xlabel("Band index (from ultraviolet on left to infrared on right)")
        plt.ylabel(f"Band weight for component {i}")
        plt.savefig(os.path.join(output_dir, f"principle_component_{i}_loading_vector.jpg"))


if __name__ == "__main__":
    TRANSFORM_PATH = "../test_runs/PCA_test_masked_characters_floats/fitted_pca.joblib"
    OUTPUT_DIRECTORY = "/Volumes/HD-Daniel/PHerc118_ExploratoryPCA/TransformAnalysis"
    pca = joblib.load(TRANSFORM_PATH)
    joblib.dump(pca, os.path.join(OUTPUT_DIRECTORY, "fitted_pca.joblib"))
    plot_principle_components(pca, OUTPUT_DIRECTORY)
