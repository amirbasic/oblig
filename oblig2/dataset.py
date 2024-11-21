import os
from typing import Optional, List
import torch
from torch.utils.data import Dataset
import pickle
from collections import Counter
from pandas import DataFrame


class TSVDataset(Dataset):
    def __init__(self, data, word2vec, label_vocab=None):
        self.unk_index = word2vec.get_index("[UNK]")
        self.tokens_premise = [
            [
                word2vec.get_index(token.lower(), default = self.unk_index)
                for token in document
            ] 
            for document in data[data.columns[1]].str.split(" ")
        ]
            
        self.tokens_hypothesis = [
            [
                word2vec.get_index(token.lower(), default = self.unk_index)
                for token in document
            ] 
            for document in data[data.columns[2]].str.split(" ")
        ]

        unk_tokens_premise = sum(token == self.unk_index for document in self.tokens_premise for token in document)
        unk_tokens_hypothesis = sum(token == self.unk_index for document in self.tokens_hypothesis for token in document)
        unk_tokens = unk_tokens_premise + unk_tokens_hypothesis
        n_tokens_premise = sum(len(document) for document in self.tokens_premise)
        n_tokens_hypothesis = sum(len(document) for document in self.tokens_hypothesis)
        n_tokens = n_tokens_premise + n_tokens_hypothesis
        print(f"Percentage of unknown tokens: {unk_tokens / n_tokens * 100.0:.2f}%")

        self.label = list(data['label'])
        self.label_vocab = label_vocab if label_vocab is not None else list(sorted(set(self.label)))
        self.num_labels = len(self.label_vocab)
        self.label_indexer = {i: n for n, i in enumerate(self.label_vocab)}

    def __getitem__(self, index):
        current_tokens_premise = self.tokens_premise[index]
        current_tokens_hypothesis = self.tokens_hypothesis[index]
        current_label = self.label[index]

        x_premise = torch.LongTensor(current_tokens_premise)
        x_hypothesis = torch.LongTensor(current_tokens_hypothesis)
        y = torch.LongTensor([self.label_indexer[current_label]])
        return x_premise, x_hypothesis, y

    def __len__(self):
        return len(self.tokens_premise)
