import os
import torch
import random
import numpy as np
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
import torch.nn as nn
import tqdm
import torch.nn.functional as F
import pandas as pd
import gensim
import zipfile
import matplotlib.pyplot as plt
import matplotlib
import time

from generic_models import Model

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


def tag_or_untag(data: pd.DataFrame, tagged):
    """Extracts untagged or POS tagged sentences from a dataframe

        Parameters
        ----------
        data: pandas.DataFrame
        tagged: Boolean

        Returns
        -------
        data: pandas.DataFrame - The filtered dataframe based on tagged value.
    """
    if tagged:
        data = data.drop(["premise", "hypothesis"], axis=1)
    else:
        data = data.drop(["tagged_premise", "tagged_hypothesis"], axis=1)
    return data


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
    model.train()
    for p, h, label_true in tqdm.tqdm(train_iter):
        p, h, label_true = p.to(device), h.to(device), label_true.to(device)
        optimizer.zero_grad()
        label_pred = model(p, h)
        loss = F.cross_entropy(label_pred, label_true)
        loss.backward()
        optimizer.step()
        scheduler.step()


@torch.no_grad()
def evaluate(model: nn.Module, data_iter: DataLoader):
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
    for p, h, label_true in tqdm.tqdm(data_iter):
        p, h, label_true = p.to(device), h.to(device), label_true.to(device)
        output = model(p, h)
        predictions += output.argmax(dim=1).tolist()
        labels_true += label_true.tolist()

    return (torch.tensor(predictions) == torch.tensor(labels_true)).float().mean() * 100.0


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


def train_epochs(epochs, args, model, optimizer, scheduler, train_iter, val_iter):
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

        train_accuracy = evaluate(model, train_iter)
        val_accuracy = evaluate(model, val_iter)

        print(f"epoch: {epoch}\tTraining accuracy: {train_accuracy:.1f}%")
        print(f"Validation accuracy: {val_accuracy:.1f}%\n")
    return train_accuracy, val_accuracy


def run_single_model(args, word2vec, train_dataset, train_iter, val_iter):
    """ Calculating training and validation accuracy for a neural network.

        Parameters
        ----------
        args: argparse.ArgumentParser
        word2vec: torch.tensor - embedding layer
        train_dataset: TSVDataset 
        train_iter: torch.utils.data.DataLoader
        val_iter: torch.utils.data.DataLoader

        Returns
        -------
        train_accuracy: float
        val_accuracy: float
    """
    print("Starting model training ...")
    print(f"Model: {args.layer_type}")
    print(f"pooling: {args.pooling_type}")
    print(f"batch size: {args.batch_size}")
    print(f"lr: {args.lr}")
    print(f"hidden size {args.hidden_size}")
    print(f"num_layers: {args.num_layers}")
    print(f"dropout: {args.dropout}")
    model = Model(args, word2vec, len(train_dataset.label_vocab)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs * len(train_iter))

    train_epochs(args.epochs, args, model, optimizer,
                 scheduler, train_iter, val_iter)

    if args.save:
        model_name = "best_model.pt"
        state_dict = {
            'model': model.state_dict(),
            'vocab': train_dataset.label_vocab,
            "word2vec": word2vec,
            'training_args': args
        }
        torch.save(state_dict, model_name)
        print(f"Model saved as {model_name}")


def time_all_nn(args, word2vec, train_dataset, train_iter, val_iter):
    """ Plots training time of 4 different neural network architectures: feed-forward, Elman, GRU, Bi-LSTM.

        Parameters
        ----------
        args: argparse.ArgumentParser
        word2vec: torch.tensor - embedding layer
        train_dataset: TSVDataset 
        train_iter: torch.utils.data.DataLoader
        val_iter: torch.utils.data.DataLoader

        Returns
        -------
        Blank
    """
    print("Running and timing different architectures:")
    nn_archs = ["ff", "elman", "gru", "bilstm"]
    time_train = []
    train_acc = []
    val_acc = []

    for nn in nn_archs:
        args.layer_type = nn

        print("=================================================================")
        print(f"training {nn} model")

        model = Model(args, word2vec, len(
            train_dataset.label_vocab)).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=0.0)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, args.epochs * len(train_iter))

        start = time.time()

        train_accuracy, val_accuracy = train_epochs(
            args.epochs, args, model, optimizer, scheduler, train_iter, val_iter)

        print("=================================================================")

        end = time.time()
        elapsed = end - start
        time_train.append(elapsed)
        train_acc.append(train_accuracy)
        val_acc.append(val_accuracy)

    matplotlib.rcParams.update({'font.size': 22})

    plt.figure(figsize=(20, 14))
    plt.bar(nn_archs, time_train)
    plt.xlabel("Training time different NN architectures")
    plt.savefig(f"Training_time.png")
    plt.clf()

    X_axis = np.arange(len(nn_archs))
    plt.bar(X_axis - 0.2, train_acc, 0.4, label='Train')
    plt.bar(X_axis + 0.2, val_acc, 0.4, label='Val')

    plt.xticks(X_axis, nn_archs)
    plt.xlabel("NN architecture")
    plt.ylabel("accuacy")
    plt.title("Performance different NN architectures")
    plt.legend(loc="upper left")
    plt.savefig(f"Performance_nn_architecture.png")
    print("Timing is done, plots saved")


def grid_search(args, word2vec, train_dataset, train_iter, val_iter):
    """ Prints performance of a neural network with several different hyperparameters.

        Parameters
        ----------
        args: argparse.ArgumentParser
        word2vec: torch.tensor - embedding layer
        train_dataset: TSVDataset 
        train_iter: torch.utils.data.DataLoader
        val_iter: torch.utils.data.DataLoader

        Returns
        -------
        blank
    """
    hidden_size = [64, 128, 256]
    num_layers = [1, 2, 4]
    lr = [0.01, 0.001, 0.0001]
    print("Starting grid search on the following parameters: hidden_layers, num_layers & lr")
    for h in hidden_size:
        for n in num_layers:
            for l in lr:
                args.hidden_size = h
                args.num_layers = n
                args.lr = l

                print(
                    "=================================================================")
                print(
                    f"training model {args.layer_type} with hidden size {args.hidden_size}, lr: {args.lr}, number of layers: {args.num_layers}")

                model = Model(args, word2vec, len(
                    train_dataset.label_vocab)).to(device)
                optimizer = torch.optim.AdamW(
                    model.parameters(), lr=args.lr, weight_decay=0.0)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, args.epochs * len(train_iter))

                train_epochs(args.epochs, args, model, optimizer,
                             scheduler, train_iter, val_iter)

                print(
                    "=================================================================")
    print("Finished grid search")


def pool_search(args, word2vec, train_dataset, train_iter, val_iter):
    """ Prints performance of different neural network architectures using different pooling techniques.

        Parameters
        ----------
        args: argparse.ArgumentParser
        word2vec: torch.tensor - embedding layer
        train_dataset: TSVDataset 
        train_iter: torch.utils.data.DataLoader
        val_iter: torch.utils.data.DataLoader

        Returns
        -------
        blank
    """

    layer = ["elman, gru, bilstm"]
    pooling = ["sum", "avg", "max"]
    print("Starting pool search on RNN, GRU, LSTM with sum, max & avg pooling")
    for l in layer:
        for p in pooling:
            args.layer_type = l
            args.pooling_type = p

            print("=================================================================")
            print(
                f"training model {args.layer_type} with {args.pooling_type} pooling")

            model = Model(args, word2vec, len(
                train_dataset.label_vocab)).to(device)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=args.lr, weight_decay=0.0)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, args.epochs * len(train_iter))

            train_epochs(args.epochs, args, model, optimizer,
                         scheduler, train_iter, val_iter)

            print("=================================================================")
    print("Finished pool search")
