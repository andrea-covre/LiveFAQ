import argparse
import time
import os

import embedding_service
# import clustering_service

from sentence_transformers import SentenceTransformer
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import torch

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

from clustering_ensemble import EnsembleFuser

QUESTION_FILE_NAME = "incoming_questions.tmp"

def get_cli_args():
    parser = argparse.ArgumentParser(
                        description = 'LiveFAQ consumer consumes questions from Kafa.')

    parser.add_argument("-a", "--algorithm", type = str, 
                            default = "kmeans",
                            choices = ["kmeans", "gmm"],
                            help = "Specify Clustering Algorithm")

    parser.add_argument("-e", "--embedding", type = str, 
                            default = "ST1",
                            choices = ["ST1", "USE", "ENSEMBLE"],
                            help = "Specify Embedding")

    parser.add_argument("-c", "--clusters", type = int,
                            default = 10,
                            choices = range(1, 10),
                            help = "Specify number of clusters")

    args = parser.parse_args()
    # print(args)
    # print(args.algorithm)
    # print(args.embedding)
    # print(args.clusters)
    return args.algorithm, args.embedding, args.clusters

if __name__ == "__main__":
    # calling the main function
    algorithm, embedding, clusters = get_cli_args()

    print("\nLoading Model: "+ str(embedding))
    embedding_model = embedding_service.load_embedding_model(embedding)

    prev_time = 0
    while True:
        questions_received_so_far = []
        modified_time = time.ctime(os.path.getmtime(QUESTION_FILE_NAME))
        if modified_time != prev_time:
            prev_time = modified_time
            with open(QUESTION_FILE_NAME) as file:
                questions_received_so_far = [line.rstrip() for line in file]
            
            question_embeddings = embedding_service.get_embedding(questions_received_so_far, embedding_model, embedding)

            print(question_embeddings.shape)
            if question_embeddings.shape[0] > 2*clusters:
                # print(type(question_embeddings))
                # print((question_embeddings.shape))
                # print(type(clusters))
                # print(type(algorithm))
                clustering_fit =  KMeans(n_clusters=clusters, random_state=0).fit(question_embeddings)
                clustering_labels = clustering_fit.labels_
                # clustering_fit, clustering_labels = clustering_service.fit_clustering(question_embeddings, clusters, algorithm)
                group_sets = np.vstack((np.array(questions_received_so_far), clustering_labels)).T 
                print("\nClustering Results:")
                print(group_sets)
        time.sleep(2)
