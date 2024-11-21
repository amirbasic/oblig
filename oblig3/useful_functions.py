import torch
import os
import random
import numpy as np
import pandas as pd
import gensim
import zipfile
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from typing import List, Tuple
import tqdm
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
import torch.nn as nn
import tqdm
import torch.nn.functional as F
from torch.optim import Optimizer
from conllu import parse
from itertools import chain
from eval_on_test import real_evaluation
from convert import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def seed_everything(seed_value=5550):
    "Set same seed to all random operations for reproduceability purposes"
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_embedding(modelfile):
    " Loading the file that is used as the embedding layer in the Neural Network"
    # Detect the model format by its extension:
    # Binary word2vec format:
    if modelfile.endswith(".bin.gz") or modelfile.endswith(".bin"):
        emb_model = gensim.models.KeyedVectors.load_word2vec_format(
            modelfile, binary=True, unicode_errors="replace"
        )
    # Text word2vec format:
    elif (
        modelfile.endswith(".txt.gz")
        or modelfile.endswith(".txt")
        or modelfile.endswith(".vec.gz")
        or modelfile.endswith(".vec")
    ):
        emb_model = gensim.models.KeyedVectors.load_word2vec_format(
            modelfile, binary=False, unicode_errors="replace"
        )
    # ZIP archive from the NLPL vector repository:
    elif modelfile.endswith(".zip"):
        with zipfile.ZipFile(modelfile, "r") as archive:
            stream = archive.open(
                "model.bin"  # or model.txt, if you want to look at the model
            )
            emb_model = gensim.models.KeyedVectors.load_word2vec_format(
                stream, binary=True, unicode_errors="replace"
            )
    else:  # Native Gensim format?
        emb_model = gensim.models.KeyedVectors.load(modelfile)
        #  If you intend to train the model further:
        # emb_model = gensim.models.Word2Vec.load(embeddings_file)
    return emb_model


class CollateFunctor:
    def __init__(self, padding_index: int, padding_label: int, max_length: int):
        self.padding_index = padding_index
        self.padding_label = padding_label
        self.max_length = max_length

    def __call__(self, samples: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        input_ids_sentencee = [s for s, y in samples]
        labels = [y for s, y in samples]

        input_ids_padded_sentence = torch.nn.utils.rnn.pad_sequence(
            input_ids_sentencee,
            batch_first=True,
            padding_value=self.padding_index
        )

        labels_padded = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=self.padding_label
        )

        input_ids_padded_sentence = input_ids_padded_sentence[:,
                                                              :self.max_length]

        labels_padded = labels_padded[:, :self.max_length]

        return input_ids_padded_sentence, labels_padded


def get_mapping_matrix_batched(offset_mapping, lengths, n_subwords: int, n_words: int):
    mapping = torch.zeros(len(lengths), n_words, n_subwords)

    for i_batch in range(len(lengths)):
        current_word, remaining_len = 0, lengths[i_batch][0]

        for i, (start, end) in enumerate(offset_mapping[i_batch]):
            if start == end:
                continue

            mapping[i_batch, current_word, i] = 1
            remaining_len -= end - start

            if remaining_len <= 0 and current_word < len(lengths[i_batch]) - 1:
                current_word += 1
                remaining_len = lengths[i_batch][current_word]

    return mapping


def get_avg_words_from_bert(dataloader, bert):
    for input_ids, attention_mask, y, offset_mapping, word_length in tqdm.tqdm(dataloader):
        offsets = offset_mapping
        word_lengths = word_length
        n_subwords = input_ids.size(1)
        n_words = max(len(words) for words in word_lengths)
        print("input ids:")
        print(input_ids)
        # if own bert setup is used: provide attention mask as well when calling upon it
        contextualized_embeddings = bert(input_ids)
        print("contextualized embeddings:")
        print(contextualized_embeddings)

        print("\t\tcontextualized embedding shape:",
              contextualized_embeddings.shape)

        mapping_matrix = get_mapping_matrix_batched(
            offsets, word_lengths, n_subwords, n_words)
        print("\t\tMapping matrix shape:", mapping_matrix.shape)

        summ = torch.einsum('bij, bjk -> bik', mapping_matrix,
                            contextualized_embeddings)
        res = summ / \
            torch.clamp(torch.sum(mapping_matrix, dim=2, keepdim=True), 1)
        print(
            "\t\tResult average pooling einsum of contextualized embedding on mapping matrix:")
        print(
            f"\t\t{res.shape} --> [batch_size, len_senteence, hidden dimension output size]")

        print("An example of \"get_mapping_matrix_batched\" done")

        return res

def invert_label(predictions, label_to_num_dict):
    inv_dict = {v: k for k, v in label_to_num_dict.items()}
    new_pred_label = list(map(inv_dict.get, predictions))
    return new_pred_label

def depad(pred, labels, sent_len):
    labels_unpad, pred_unpad = [], []
    for i in range(len(sent_len)):
        slice_idx = sent_len[i]
        labels_unpad.append(labels[i][: slice_idx])
        pred_unpad.append(pred[i][: slice_idx])

    return labels_unpad, pred_unpad

def train(model: nn.Module, train_iter: DataLoader, optimizer: Optimizer, scheduler: _LRScheduler):
    """ Training process of Neural-Network.
        Parameters
        ----------
        model: nn.Module - the neural network to train
        train_iter: torch.utils.data.DataLoader
        optimizer: torch.optim.Optimizer - Optimzer method
        scheduler: torch.optim.lr_scheduler._LRScheduler - Scheduler method

        Returns
        -------
        blank
    """
    model = model.to(device)
    model.train()
    for input_ids, attention_mask, y, offset_mapping, word_length, sent_len  in tqdm.tqdm(train_iter):
        input_ids, attention_mask, y, offset_mapping, word_length, sent_len  = input_ids.to(device), attention_mask.to(device), y.to(device), offset_mapping.to(device), word_length.to(device), sent_len.to(device) 
        optimizer.zero_grad()
        n_subwords = input_ids.size(1)
        n_words = max(len(words) for words in word_length)
        mapping_matrix = get_mapping_matrix_batched(offset_mapping, word_length, n_subwords, n_words).to(device)
        predictions = model(input_ids, attention_mask, mapping_matrix)
        predictions = predictions.permute(0, 2, 1)
        loss = F.cross_entropy(predictions, y)
        loss.backward()
        optimizer.step()
        scheduler.step()

@torch.no_grad()
def evaluate1(model: nn.Module, data_iter: DataLoader):
    """ Retrives the accuracy of the neural network 
        Parameters
        ----------
        model: nn.Module - the neural network used to predict
        data_iter: torch.utils.data.DataLoader

        Returns
        -------
        accuracy: float value
    """
    model.eval()
    labels_true, predictions = [], []
    for input_ids, attention_mask, y, offset_mapping, word_length, sent_len in tqdm.tqdm(data_iter):
        input_ids, attention_mask, y, offset_mapping, word_length, sent_len  = input_ids.to(device), attention_mask.to(device), y.to(device), offset_mapping.to(device), word_length.to(device), sent_len.to(device)
        n_subwords = input_ids.size(1)
        n_words = max(len(words) for words in word_length)
        mapping_matrix = get_mapping_matrix_batched(offset_mapping, word_length, n_subwords, n_words).to(device)
        pred = model(input_ids, attention_mask, mapping_matrix)
        pred = pred.argmax(dim=2).tolist()

        labels_unpad, pred_unpad = depad(pred, y.tolist(), sent_len)
        predictions += pred_unpad
        labels_true += labels_unpad

    all_pred = list(chain.from_iterable(predictions))
    all_true = list(chain.from_iterable(labels_true))

    #print(classification_report(all_pred, all_true))
    return predictions, (torch.tensor(all_pred) == torch.tensor(all_true)).float().mean() * 100.0


def real_evaluator(gold_path, predictions, dict_label):
    test_path = "IN5550/oblig3/data/train_pred.conllu.gz"
    correct_label_pred = []
    for i in predictions:
        correct_label_pred.append(invert_label(i, dict_label))

    pred = pd.DataFrame()
    pred["labels"] = correct_label_pred
    overwrite_conllu_labels(gold_path, pred, test_path)

    real_evaluation(gold_path, test_path)
    return

def train_epochs(args, model, optimizer, scheduler, train_iter, val_iter, dict_label):
    """ Calculating training and validation accuracy per epoch.

        Parameters
        ----------
        epochs: int
        args: argparse.ArgumentParser
        optimizer: torch.optim.Optimizer - Optimzer method
        scheduler: torch.optim.lr_scheduler._LRScheduler - Scheduler method
        train_iter: torch.utils.data.DataLoader
        val_iter: torch.utils.data.DataLoader

        Returns
        -------
        train_accuracy: float
        val_accuracy: float
    """
    for epoch in range(args.epochs):
        train(model, train_iter, optimizer, scheduler)

        #train_pred, train_accuracy = evaluate1(model, train_iter)
        val_pred, val_accuracy = evaluate1(model, val_iter)
        train_accuracy = 0

        print(f"epoch: {epoch + 1}")
        real_evaluator(args.gold_label_conllu_path, val_pred, dict_label)

    return train_accuracy, val_accuracy
    

def train_lstm_tagger(train_loader, val_loader, model, num_epochs, loss_fn, optimizer):
    """ Calculating training and validation accuracy per epoch.

        Parameters
        ----------
        args: argparse.ArgumentParser
        optimizer: torch.optim.Optimizer - Optimzer method
        loss_fn: torch.nn.NLLLoss
        train_loader: torch.utils.data.DataLoader
        val_loader: torch.utils.data.DataLoader

        Returns
        -------
        model: torch.model
    """
    for epoch in range(num_epochs):
        epoch_loss_train = 0
        predicted_list, true_list = [], []
        for i, (sentence, tags) in enumerate(train_loader):
            batch_size = sentence.shape[0]
            optimizer.zero_grad()
            tag_scores = model(sentence)
            tag_scores = tag_scores.permute(0, 2, 1)
            loss = loss_fn(tag_scores, tags)
            epoch_loss_train += loss.item()
            loss.backward()
            optimizer.step()
            
            _, predicted = torch.max(tag_scores, dim=1)
            
            for i in range(predicted.shape[0]):
                predictions_clean = predicted[i][tags[i] != -100]
                label_clean = tags[i][tags[i] != -100]
                true_list += label_clean.tolist()
                predicted_list += predictions_clean.tolist()
                
        num_correct_train = sum([1 for i in range(len(predicted_list)) if predicted_list[i] == true_list[i]])
        accuracy_train = (num_correct_train / len(predicted_list)) * 100 
            
        # pred16 = predicted_list.count(16) / len(predicted_list)
        # true16 = true_list.count(16) / len(true_list)
        # print(f"Percentage of 16 predictions (train): {(pred16 * 100):.2f}")
        # print(f"Percentage of true 16 (train): {(true16 * 100):.2f}")
                
        epoch_loss_validate = 0
        predicted_list, true_list = [], []
        with torch.no_grad():    
            for i, (sentence, tags) in enumerate(val_loader):
                optimizer.zero_grad()
                tag_scores = model(sentence)
                tag_scores = tag_scores.permute(0, 2, 1)
                loss = loss_fn(tag_scores, tags)
                epoch_loss_validate += loss.item()
                
                _, predicted = torch.max(tag_scores, dim=1)
                for i in range(predicted.shape[0]):
                    predictions_clean = predicted[i][tags[i] != -100]
                    label_clean = tags[i][tags[i] != -100]
                    true_list += label_clean.tolist()
                    predicted_list += predictions_clean.tolist()
                    
        num_correct_validate = sum([1 for i in range(len(predicted_list)) if predicted_list[i] == true_list[i]])
        accuracy_validate = (num_correct_validate / len(predicted_list)) * 100
        
        # pred16 = predicted_list.count(16) / len(predicted_list)
        # true16 = true_list.count(16) / len(true_list)
        # print(f"Percentage of 16 predictions (validate): {(pred16 * 100):.2f}")
        # print(f"Percentage of true 16 (validate): {(true16 * 100):.2f}")

        print(f"Epoch {epoch + 1} - Train Loss: {epoch_loss_train:.4f}, Train Acc: {accuracy_train:.4f}, Val Loss: {epoch_loss_validate:.4f}, Val Acc: {accuracy_validate:.4f}")
            
    return model

