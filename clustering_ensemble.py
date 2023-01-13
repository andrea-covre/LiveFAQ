import torch
import os
import random
import numpy as np
import torch.nn as nn

from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR

from matplotlib import pyplot as plt

import sys


NOISE_SIGMA = 1.e-3


class EnsembleFuser(nn.Module):
    def __init__(self, embed_dims, fused_dim=512) -> None:
        super().__init__()

        assert len(embed_dims) >= 2

        self.embed_dims = [x for x in embed_dims]
        self.fused_dim = fused_dim

        self.layers = nn.ModuleList()

        for e_dim in self.embed_dims:
            self.layers.append(
                nn.Sequential(
                    nn.Dropout(p=0.2),
                    nn.Linear(e_dim, self.fused_dim, bias=False),
                    nn.BatchNorm1d(self.fused_dim),
                )
            )
        
    def forward(self, x):
        assert isinstance(x, (list, tuple))

        res = 0
        for ii, layer in enumerate(self.layers):
            res += layer(x[ii])

        # To test single-model projection classifiers
        # res = self.layers[2](x[2])
        
        return res


def train(embeddings_list, label_arr, train_idxs, batch_size, model, loss, optimizer):
    model.train()

    np.random.shuffle(train_idxs)

    data_idx = 0

    train_loss = 0
    num_items = 0

    while data_idx < train_idxs.shape[0]:
        optimizer.zero_grad()

        batch_idxs = train_idxs[data_idx: data_idx + batch_size]

        ip_list = [torch.FloatTensor(x[batch_idxs, :]) for x in embeddings_list]

        if NOISE_SIGMA > 0:
            ip_list = [x + (NOISE_SIGMA * torch.rand_like(x)) - (NOISE_SIGMA / 2) for x in ip_list]

        label_t = torch.LongTensor(label_arr[batch_idxs])

        model_op = model(ip_list)

        loss_t = loss(model_op, label_t)

        loss_t.backward()
        optimizer.step()

        train_loss += loss_t.item() * label_t.size(0)
        num_items += label_t.size(0)
        
        data_idx += batch_size
    
    avg_loss = train_loss / num_items

    return avg_loss


def evaluate(embeddings_list, label_arr, val_idxs, batch_size, model, loss):
    model.eval()

    data_idx = 0

    val_loss = 0
    val_acc = 0
    num_items = 0

    while data_idx < val_idxs.shape[0]:
        batch_idxs = val_idxs[data_idx: data_idx + batch_size]

        ip_list = [torch.FloatTensor(x[batch_idxs, :]) for x in embeddings_list]

        label_t = torch.LongTensor(label_arr[batch_idxs])

        model_op = model(ip_list)

        loss_t = loss(model_op, label_t)

        pred_t = torch.argmax(model_op, dim=1)

        val_acc += int((pred_t == label_t).sum())

        val_loss += loss_t.item() * label_t.size(0)
        num_items += int(label_t.size(0))

        data_idx += batch_size

    val_acc = val_acc / num_items
    val_loss = val_loss / num_items

    return val_loss, val_acc
      

if __name__ == '__main__':
    NUM_CLUSTERS = 50
    FUSED_EMBED_DIM = 512
    BATCH_SIZE = 16
    NUM_EPOCHS = 100

    label_arr = np.load(os.path.join('SavedEmbeddings', f'qna_5500_labels_{NUM_CLUSTERS}_classes.npy'))
    num_data = label_arr.shape[0]

    fused_models = ['ST1', 'ST3', 'USE']

    embeddings_list = list()

    for m_name in fused_models:
        embed_arr = np.load(os.path.join('SavedEmbeddings', f'qna_5500_embeddings_{m_name}.npy'))
        assert embed_arr.shape[0] == num_data
        embeddings_list.append(embed_arr)
    
    embeddings_sizes = [x.shape[1] for x in embeddings_list]

    # print(embeddings_sizes)

    m1 = EnsembleFuser(embed_dims=embeddings_sizes, fused_dim=FUSED_EMBED_DIM)

    m2 = nn.Linear(FUSED_EMBED_DIM, NUM_CLUSTERS)
    model = nn.Sequential(m1, m2)
    
    loss = nn.CrossEntropyLoss()

    optimizer = AdamW(model.parameters(), lr=1.e-3, weight_decay=1.e-4)

    scheduler = ExponentialLR(optimizer, gamma=0.98)

    all_idxs = np.arange(num_data)

    for _ in range(random.randint(5, 10)):
        np.random.shuffle(all_idxs)

    num_train = int(0.7 * num_data)

    train_idxs = all_idxs[:num_train]

    all_idxs = all_idxs[num_train:]

    val_idxs = all_idxs

    print("Num-samples for:\nTrain | Val")
    print(train_idxs.shape, val_idxs.shape)

    print('-' * 60)

    best_val_acc = 0

    epoch_list = list()
    train_loss_list = list()
    val_loss_list = list()
    val_acc_list = list()

    for ii in range(NUM_EPOCHS):
        avg_train_loss = train(embeddings_list, label_arr, train_idxs, BATCH_SIZE, model, loss, optimizer)

        print(f"Epoch: {ii + 1} | Avg Train-loss: {avg_train_loss:.5f}")

        with torch.no_grad():
            avg_val_loss, avg_val_acc = evaluate(embeddings_list, label_arr, val_idxs, BATCH_SIZE, model, loss)

        print(f"Epoch: {ii + 1} | Avg Val-loss: {avg_val_loss:.5f} | Avg Val-acc: {avg_val_acc:.4f}")

        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            print("Saving best acc model!")
            torch.save(m1, "ensemble_fuser.pth")

        scheduler.step()

        epoch_list.append(1 + ii)
        train_loss_list.append(avg_train_loss)
        val_loss_list.append(avg_val_loss)
        val_acc_list.append(avg_val_acc)

        print('-' * 60)
        # sys.exit()

    plt.figure(figsize=(8, 8))
    plt.plot(epoch_list, train_loss_list, label='train_loss')
    plt.plot(epoch_list, val_loss_list, label='val_loss')
    plt.plot(epoch_list, val_acc_list, label='val_acc')
    plt.legend()
    plt.title("Loss Plot for ensemble-fuser classifier")
    plt.savefig('ensemble_fuser_classifier_plot.png', bbox_inches='tight')

    

