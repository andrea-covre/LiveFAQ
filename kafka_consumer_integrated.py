import argparse
from kafka import KafkaConsumer
import time
from collections import defaultdict

import numpy as np
# from sklearn.mixture import GaussianMixture
# from sklearn.cluster import KMeans

# from sentence_transformers import SentenceTransformer
# import numpy as np
# import tensorflow as tf
# import tensorflow_hub as hub
# import torch

import embedding_service
import clustering_service
from clustering_ensemble import EnsembleFuser

# import os

# os.environ['KMP_DUPLICATE_LIB_OK']='True'

def get_cli_args():
    parser = argparse.ArgumentParser(
                        description = 'LiveFAQ consumer consumes questions from Kafka.')

    parser.add_argument("-a", "--algorithm", type = str, 
                            default = "gmm-diag",
                            choices = ["kmeans", "gmm-full", "gmm-diag"],
                            help = "Specify Clustering Algorithm")

    parser.add_argument("-e", "--embedding", type = str, 
                            default = "ST1",
                            choices = ["ST1", "USE", "ENSEMBLE"],
                            help = "Specify Embedding")

    parser.add_argument("-c", "--clusters", type = int,
                            default = 10,
                            choices = range(1, 10),
                            help = "Specify number of clusters")

    parser.add_argument("-t", "--timeout", type = int,
                            default = 10000,
                            choices = [10000, 20000],
                            help = "Specify number of clusters")

    args = parser.parse_args()
    # print(args)
    # print(args.algorithm)
    # print(args.embedding)
    # print(args.clusters)
    return args.algorithm, args.embedding, args.clusters, args.timeout

if __name__ == "__main__":
    # calling the main function
    algorithm, embedding, clusters, timeout_ms = get_cli_args()

    print("\nLoading Model: "+ str(embedding))
    embedding_model = embedding_service.load_embedding_model(embedding)

    topic_name = str("question-events")
    kafka_consumer = KafkaConsumer(topic_name, auto_offset_reset='earliest',
                                 bootstrap_servers=['localhost:9092'], api_version=(0, 10), consumer_timeout_ms=timeout_ms)
    print("Listening For Questions (Timeout set to " + str(timeout_ms) + " ms)")
    questions_received_so_far = []
    clusters_out = []
    for _ in range(clusters):
        clusters_out.append([])
    inference_time_output = open("inference_time.txt", "w")
    for msg in kafka_consumer:
        print ("\nReceived: " + str(msg.value.decode()))
        questions_received_so_far.append(str(msg.value.decode()))
        question_embeddings = embedding_service.get_embedding(questions_received_so_far, embedding_model, embedding)
        # # Simulate model execution
        # print("Simualting Model execution")
        # time.sleep(3)
        # print("Model execution complete, Results:  ")
        if question_embeddings.shape[0] > 2*clusters:
            # print(type(question_embeddings))
            # print(type(clusters))
            # print(type(algorithm))
            clustering_start = time.time()
            clustering_fit, clustering_labels = clustering_service.fit_clustering(question_embeddings, clusters, algorithm)
            grouping_dict = defaultdict(list)
            inference_time = time.time() - clustering_start
            print("Inference Time: " + str(inference_time))
            inference_time_output.write(str(inference_time) + "\n")
            group_sets = np.vstack((np.array(questions_received_so_far), clustering_labels)).T 
            for question in group_sets:
                grouping_dict[question[1]].append(question[0])
            print("\nClustering Results:")
            for cluster in sorted(list(grouping_dict.keys())):
                print("Cluster " + cluster + ":\n")
                for question in grouping_dict[cluster]:
                    print(question + "\n")
    inference_time_output.close()
    for question in group_sets:
        clusters_out[question[1].astype(np.int)].append(question[0]);
    trace_out = {"questions_added":questions_received_so_far,"final_clustering":clusters_out}
    print(trace_out)
    kafka_consumer.close()
