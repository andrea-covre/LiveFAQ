from sentence_transformers import SentenceTransformer
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import torch

# Needed to load the pth file
from clustering_ensemble import EnsembleFuser

def load_embedding_model(embedding_model_name):
    if embedding_model_name == "ST1":
        embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    elif embedding_model_name == "USE":
        module_url = "https://tfhub.dev/google/universal-sentence-encoder/4" #@param ["https://tfhub.dev/google/universal-sentence-encoder/4", "https://tfhub.dev/google/universal-sentence-encoder-large/5"]
        embedding_model = hub.load(module_url)
        print ("module %s loaded" % module_url)
    elif embedding_model_name == "ENSEMBLE":
        embedding_model_1 = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

        module_url = "https://tfhub.dev/google/universal-sentence-encoder/4" #@param ["https://tfhub.dev/google/universal-sentence-encoder/4", "https://tfhub.dev/google/universal-sentence-encoder-large/5"]
        embedding_model_2 = hub.load(module_url)

        # fused_models = ['ST1', 'ST3', 'USE']
        fused_models = ['ST1', 'USE']
        model_path = 'ensemble_fuser_' + '_'.join([x.lower() for x in fused_models]) + '.pth'

        embedding_model_3 = torch.load(model_path)
        embedding_model = [embedding_model_1, embedding_model_2, embedding_model_3]
    else:
        print("Unrecognized embedding_model. Sticking with ST1.")
        embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    return embedding_model


def get_embedding(question_list, embedding_model, embedding_model_name):
    if embedding_model_name == "ST1":
        questionEmbeddings = embedding_model.encode(np.array(question_list))
    elif embedding_model_name == "USE":
        questionEmbeddings = embedding_model(np.array(question_list))
        # def embed(input):
        #   return model(input)
    elif embedding_model_name == "ENSEMBLE":
        embeddings_list = list()
        embeddings_list.append(np.array(embedding_model[0].encode(np.array(question_list))))
        embeddings_list.append(np.array(embedding_model[1](np.array(question_list))))

        num_data = embeddings_list[0].shape[0]
        
        op_embed_list = list()
        
        with torch.no_grad():
            for ii in range(num_data):
                ip_list = [torch.FloatTensor(x[ii])[None, :] for x in embeddings_list]
                model_op = embedding_model[2](ip_list)[0]
                op_embed_list.append(model_op.cpu().detach().numpy())
        
        questionEmbeddings = np.stack(op_embed_list, axis=0)
    else:
        questionEmbeddings = embedding_model.encode(np.array(question_list))
    return questionEmbeddings