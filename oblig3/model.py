from modeling_norbert import NorbertModel
import torch
from torch import nn
from transformers import BertModel
from useful_functions import get_avg_words_from_bert
import numpy as np
import torch.nn.functional as F
import torch.optim as optim


class BertLinear(nn.Module):

    def __init__(self, model_name, num_labels):
        super(BertLinear, self).__init__()
        self.num_labels = num_labels
        if model_name.split("-")[0][-1] == "3":
            self.bert = NorbertModel.from_pretrained(
                model_name, num_labels)
        else:
            self.bert = BertModel.from_pretrained(
                model_name, num_labels)
        config = self.bert.config

        self.linear_output = nn.Linear(
            config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, mapping_matrix):
        bert_output = self.bert(input_ids, attention_mask)

        summ = torch.einsum('bij, bjk -> bik', mapping_matrix,
                            bert_output.last_hidden_state)

        res = summ / \
            torch.clamp(torch.sum(mapping_matrix, dim=2, keepdim=True), 1)

        logits = self.linear_output(res)

        return logits


class BertGRU(nn.Module):

    def __init__(self, model_name, num_labels):
        super(BertGRU, self).__init__()
        self.num_labels = num_labels
        if model_name.split("-")[0][-1] == "3":
            self.bert = NorbertModel.from_pretrained(
                model_name, num_labels)
        else:
            self.bert = BertModel.from_pretrained(
                model_name, num_labels)
        config = self.bert.config

        self.gru = nn.GRU(config.hidden_size,
                          config.hidden_size, batch_first=True)

        self.linear_output = nn.Linear(
            config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, mapping_matrix):
        bert_output = self.bert(input_ids, attention_mask)

        summ = torch.einsum('bij, bjk -> bik', mapping_matrix,
                            bert_output.last_hidden_state)

        res = summ / \
            torch.clamp(torch.sum(mapping_matrix, dim=2, keepdim=True), 1)

        gru_output = self.gru(res)[0]

        logits = self.linear_output(gru_output)

        return logits


class BertLSTM(nn.Module):

    def __init__(self, model_name, num_labels):
        super(BertLSTM, self).__init__()
        self.num_labels = num_labels
        if model_name.split("-")[0][-1] == "3":
            self.bert = NorbertModel.from_pretrained(
                model_name, num_labels)
        else:
            self.bert = BertModel.from_pretrained(
                model_name, num_labels)
        config = self.bert.config

        self.bilstm = nn.LSTM(input_size=config.hidden_size,
                              hidden_size=config.hidden_size, bidirectional=True, batch_first=True)

        self.linear_output = nn.Linear(
            config.hidden_size*2, num_labels)

    def forward(self, input_ids, attention_mask, mapping_matrix):
        bert_output = self.bert(input_ids, attention_mask)

        summ = torch.einsum('bij, bjk -> bik', mapping_matrix,
                            bert_output.last_hidden_state)

        res = summ / \
            torch.clamp(torch.sum(mapping_matrix, dim=2, keepdim=True), 1)

        lstm_output = self.bilstm(res)[0]

        logits = self.linear_output(lstm_output)

        return logits


class LSTMTagger(nn.Module):
    def __init__(self, pretrained_embedding, hidden_dim, tagset_size):
        super(LSTMTagger, self).__init__()
        self.pretrained_embedding = pretrained_embedding
        self.hidden_dim = hidden_dim
        self.tagset_size = tagset_size
        self.word_embeddings = nn.Embedding.from_pretrained(torch.FloatTensor(pretrained_embedding.vectors), freeze=True)
        self.lstm = nn.LSTM(pretrained_embedding.vector_size, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def init_hidden(self, batch_size):
        return (torch.zeros(1, batch_size, self.hidden_dim),
                torch.zeros(1, batch_size, self.hidden_dim))

    def forward(self, sentence):
        batch_size = sentence.shape[0]
        seq_len = sentence.shape[1]
        mask = (sentence != self.pretrained_embedding.get_index("[PAD]"))  # create a mask to ignore padded labels
        self.hidden = self.init_hidden(batch_size)
        embeds = self.word_embeddings(sentence)
        lstm_out, self.hidden = self.lstm(embeds.transpose(0, 1), self.hidden)
        lstm_out = lstm_out.transpose(0, 1)
        lstm_out = lstm_out.masked_fill(~mask.unsqueeze(-1), -1e18)  # set masked labels to a very large negative value
        tag_outputs = self.hidden2tag(lstm_out)
        tag_scores = F.log_softmax(tag_outputs, dim=2)
        return tag_scores
