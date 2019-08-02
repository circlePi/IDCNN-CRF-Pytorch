import os
import torch
import torch.nn as nn
import numpy as np
from torch.nn import NLLLoss
import config.config as config
from util.embedding_util import get_embedding
from net.cnn import IDCNN
from net.crf import CRF
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from torch.nn import functional as F
torch.manual_seed(2018)
torch.cuda.manual_seed(2018)
torch.cuda.manual_seed_all(2018)
np.random.seed(2018)


class BILSTM_CRF(nn.Module):
    def __init__(self,
                 vocab_size,
                 word_embedding_dim,
                 word2id,
                 hidden_size, bi_flag,
                 num_layer, input_size,
                 cell_type, dropout,
                 num_tag,
                 checkpoint_dir):
        super(BILSTM_CRF, self).__init__()
        self.embedding = nn.Embedding(vocab_size, word_embedding_dim)
        
        self.embedding.weight.data.copy_(torch.from_numpy(get_embedding(vocab_size,
                                                                       word_embedding_dim,
                                                                       word2id)))
        self.idcnn = IDCNN(input_size=word_embedding_dim, filters=64)
        self.linear = nn.Linear(64, 256)
        self.out = nn.Linear(256, num_tag)

        self.crf = CRF(num_tag=num_tag)

        self.checkpoint_dir = checkpoint_dir

    def forward(self, inputs, length):
        embeddings = self.embedding(inputs)
        embeddings = nn.Dropout(0.2)(embeddings)
        out = self.idcnn(embeddings, length)
        out = self.linear(out)
        out = self.out(out)
        output = F.dropout(out, p=0.1, training=self.training)
        return output


    def loss_fn(self, rnn_output, labels, length):
        loss = self.crf.negative_log_loss(inputs=rnn_output, length=length, tags=labels)
        return loss

    def predict(self, rnn_output, length):
        best_path = self.crf.get_batch_best_path(rnn_output, length)
        return best_path

    def load(self):
        self.load_state_dict(torch.load(self.checkpoint_dir))

    def save(self):
        torch.save(self.state_dict(), self.checkpoint_dir)

    def evaluate(self, y_pred, y_true):
        y_true = y_true.cpu().numpy()
        y_pred = y_pred.numpy()
        f1 = f1_score(y_true, y_pred, labels=config.labels, average="macro")
        correct = np.sum((y_true==y_pred).astype(int))
        acc = correct/y_pred.shape[0]
        return (acc, f1)

    def class_report(self, y_pred, y_true):
        y_true = y_true.cpu().numpy()
        y_pred = y_pred.numpy()
        classify_report = classification_report(y_true, y_pred)
        print('\n\nclassify_report:\n', classify_report)
        print("confusion matrix:\n", confusion_matrix(y_true=y_true, y_pred=y_pred))




