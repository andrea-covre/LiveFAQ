import torch
import numpy as np
import os

# Needed to load the pth file
from clustering_ensemble import EnsembleFuser

import sys


def load_model(fused_models):
    model_path = 'ensemble_fuser_' + '_'.join([x.lower() for x in fused_models]) + '.pth'

    return torch.load(model_path)


def main():
    # fused_models = ['ST1', 'ST3', 'USE']
    fused_models = ['ST1', 'USE']

    model = load_model(fused_models)
    model.eval()

    print(model)

    embeddings_list = list()

    for m_name in fused_models:
        embed_arr = np.load(os.path.join('SavedEmbeddings', f'qna_5500_embeddings_{m_name}.npy'))
        embeddings_list.append(embed_arr)
    
    op_embed_list = list()

    num_data = embeddings_list[0].shape[0]

    with torch.no_grad():
        for ii in range(num_data):
            ip_list = [torch.FloatTensor(x[ii])[None, :] for x in embeddings_list]

            model_op = model(ip_list)[0]

            op_embed_list.append(model_op.cpu().detach().numpy())
    
    op_embed_arr = np.stack(op_embed_list, axis=0)

    print('Embedding array shape:')
    print(op_embed_arr.shape)

    np.save(os.path.join('SavedEmbeddings', f'qna_5500_embeddings_' + 
            '_'.join(fused_models) + '.npy'), op_embed_arr)
    
    print('Done!')

    return


if __name__ == '__main__':
    main()

