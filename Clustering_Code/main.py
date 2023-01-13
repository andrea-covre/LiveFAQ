import argparse
import os
import numpy as np

from clustering_utils import KMeans as CustomKMeans
from sklearn.cluster import KMeans
from sklearn.metrics import homogeneity_score


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='data1')

    args = parser.parse_args()

    data_path = os.path.join(args.data_dir, 'img_data.npy')
    label_path = os.path.join(args.data_dir, 'label_data.npy')
    embed_path = os.path.join(args.data_dir, 'embed_data.npy')

    img_arr = np.load(data_path)
    label_arr = np.load(label_path)
    embed_arr = np.load(embed_path)

    # Reference KMeans implementation
    kmeans_ref = KMeans(n_clusters=10)
    kmeans_ref.fit(embed_arr)
    y_hat_ref = kmeans_ref.predict(embed_arr)

    # Custom KMeans with euclidean distance
    kmeans_1 = CustomKMeans(num_clusters=10)
    kmeans_1.fit(embed_arr)
    y_hat_1 = kmeans_1.predict(embed_arr)

    # Custom KMeans with cosine distance
    kmeans_2 = CustomKMeans(num_clusters=10, dist_fn='cosine')
    kmeans_2.fit(embed_arr)
    y_hat_2 = kmeans_2.predict(embed_arr)

    # Custom KMeans with dot-prod distance
    kmeans_3 = CustomKMeans(num_clusters=10, dist_fn='dot_prod')
    kmeans_3.fit(embed_arr)
    y_hat_3 = kmeans_3.predict(embed_arr)

    print('-' * 50)
    print(f'Reference H-score: {homogeneity_score(labels_true=label_arr, labels_pred=y_hat_ref)}')
    print(f'Euclidean KMeans H-score: {homogeneity_score(labels_true=label_arr, labels_pred=y_hat_1)}')
    print(f'Cosine KMeans H-score: {homogeneity_score(labels_true=label_arr, labels_pred=y_hat_2)}')
    print(f'Dot-prod KMeans H-score: {homogeneity_score(labels_true=label_arr, labels_pred=y_hat_3)}')
    print('-' * 50)

    return


if __name__ == '__main__':
    main()
