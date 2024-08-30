import argparse
import os
import shutil

from clustering import ImageClustering
from data_loader import create_sub_dirs, read_image_paths_from_dir
from dim_reduction import DimReducer
from feature_extractor import FeatureExtractor


def main():
    parser = argparse.ArgumentParser(
        description="Image clustering using ResNet50 & Kmeans & PCA"
    )
    parser.add_argument("source_path", help="Path to source folder.")
    parser.add_argument("output_path", help="Path to output folder.")
    parser.add_argument("max_n_components", help="Maximum number of components.")
    args = parser.parse_args()

    print("Extracting features using ResNet50...")
    ftr_extractor = FeatureExtractor()
    features = ftr_extractor.extract_features(args.source_path)

    print("Applying dimension reduction to features extracted...")
    reducer = DimReducer(max_num_of_clusters=int(args.max_n_components), threshold=0.70)
    opt_num_comp = reducer.get_optimal_num_components(features)
    reducer.fit(opt_num_comp, features)
    features = reducer.transform(features)

    print("Clustering images...")
    cluster = ImageClustering(min_clusters=2, max_clusters=20)
    cluster.fit(features)

    print(f"Best Num. of Clusters: {cluster.best_n_clusters}")
    print(f"Best Clustering Score: {cluster.best_score}")

    image_paths = read_image_paths_from_dir(args.source_path)
    create_sub_dirs(cluster.best_n_clusters, args.output_path)

    for image_path, label in zip(image_paths, cluster.best_labels_):
        file_source_path = os.path.join(args.source_path, image_path)
        file_path = os.path.join(str(label), image_path)
        file_new_path = os.path.join(args.output_path, file_path)
        shutil.copy(file_source_path, file_new_path)


if __name__ == "__main__":
    main()
