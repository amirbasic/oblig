import torch
import argparse
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import os
from transformers import AutoTokenizer
import pandas as pd
from model import BertLinear, LSTMTagger, BertGRU, BertLSTM
from useful_functions import *
from dataset import EmbDataset, open_and_read_path, drop_rows, TokenDataset
import torch.nn as nn
import torch.optim as optim
from smart_open import open
import time

import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model")
    parser.add_argument( "--data_path", default="IN5550/oblig3/data/norne-nb-in5550-train.conllu.gz")
    parser.add_argument("--gold_label_conllu_path",default="IN5550/oblig3/data/gold_label_val.conllu.gz")
    parser.add_argument( "--pretrained", default='/fp/projects01/ec30/models/norlm/norbert3-base/')
    parser.add_argument("--embedding_path", default="IN5550/oblig3/data/58/model.bin")
    parser.add_argument("--hidden_size", default=256)
    parser.add_argument("--num_layers", default=2)
    parser.add_argument("--batch_size", default=64)
    parser.add_argument("--lr", default=0.0001)
    parser.add_argument("--epochs", default=15)
    parser.add_argument("--split", default=0.8)
    parser.add_argument("--max_length", default=64)
    parser.add_argument("--dropout", default=0.1)
    parser.add_argument("--weight_decay", default=0.4)
    parser.add_argument("--seed", default=5550)
    parser.add_argument("--freeze", default=False)
    parser.add_argument("--save", type=bool, default=False)
    parser.add_argument("--task1", type=bool, default=False)
    parser.add_argument("--task2", type=bool, default=True)
    args = parser.parse_args()

    seed_everything(args.seed)

    print("running the following bert model:")
    print(args.pretrained)

    print("Loading data ...")
    data = open_and_read_path(args.data_path)
    args.max_length = max(data["sentence_length"])
    train_df, val_df = train_test_split(data, train_size=args.split, random_state=args.seed)
    train_df = drop_rows(train_df, min_labels=1)
    print("Loading data done")

    if args.task1:
        print("Loading embedding ...")
        embedding = load_embedding(args.embedding_path)
        embedding["[UNK]"] = torch.tensor(
            embedding.vectors).mean(dim=0).numpy()
        embedding["[PAD]"] = torch.zeros(embedding.vector_size).numpy()
        print("Loading embedding done")

    if args.task2:
        print("Loading model ...")
        tokenizer = AutoTokenizer.from_pretrained(args.pretrained)
        print("Loading model done")

    print("Splitting and processing data ...")

    if args.task1:
        print("Loading embedding data for Task 1 ...")
        emb_train_dataset = EmbDataset(train_df, embedding)
        emb_val_dataset = EmbDataset(val_df, embedding, label_vocab=emb_train_dataset.label_vocab)

        emb_train_dataloader = DataLoader(emb_train_dataset,
                                          batch_size=args.batch_size,
                                          shuffle=True,
                                          drop_last=True,
                                          num_workers=1,
                                          collate_fn=CollateFunctor(embedding.get_index("[PAD]"),
                                                                    -100, # Set padded label value different than non-entity label value to avoid confusion
                                                                    args.max_length)
                                          )

        emb_val_dataloader = DataLoader(emb_val_dataset,
                                        batch_size=args.batch_size,
                                        shuffle=False,
                                        drop_last=False,
                                        num_workers=1,
                                        collate_fn=CollateFunctor(embedding.get_index("[PAD]"),
                                                                  -100, # Set padded label value different than non-entity label value to avoid confusion
                                                                  args.max_length)
                                        )

        print("Loading embedding data for Task 1 done")

        model = LSTMTagger(embedding, 32, emb_train_dataset.num_labels)
        loss_fn = nn.NLLLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        start_time = time.time()
        model = train_lstm_tagger(emb_train_dataloader, emb_val_dataloader, model, args.epochs, loss_fn, optimizer)
        print("LSTM tagger training time: ", (time.time() - start_time))
        predicted_list, true_list = [], []
        with torch.no_grad():    
            for i, (sentence, tags) in enumerate(emb_val_dataloader):
                tag_scores = model(sentence)
                tag_scores = tag_scores.permute(0, 2, 1)
                _, predicted = torch.max(tag_scores, dim=1)
                for i in range(predicted.shape[0]):
                    predictions_clean = predicted[i][tags[i] != -100]
                    label_clean = tags[i][tags[i] != -100]
                    predicted_list.append(predictions_clean.tolist())
                    
        real_evaluator(args.gold_label_conllu_path, predicted_list, emb_train_dataset.label_indexer)
        assert 0

    if args.task2:
        print("Loading tokenized data for Task 2 ...")
        tok_train_dataset = TokenDataset(train_df, tokenizer, args)
        tok_val_dataset = TokenDataset(val_df, tokenizer, args, label_vocab=tok_train_dataset.label_vocab)

        tok_train_dataloader = DataLoader(tok_train_dataset,
                                          batch_size=args.batch_size,
                                          shuffle=True,
                                          drop_last=True,
                                          num_workers=1
                                          )

        tok_val_dataloader = DataLoader(tok_val_dataset,
                                        batch_size=args.batch_size,
                                        shuffle=False,
                                        drop_last=False,
                                        num_workers=1
                                        )

        print("Loading tokenized data for Task 2 done")

    print("Splitting and processing data done")

    print("Loading model ...")
    model = BertLSTM(args.pretrained, num_labels=17).to(device)
    freeze_model = args.freeze
    if freeze_model:
        for param in model.bert.parameters():
            param.requires_grad = False
    print("Loading model done")

    print("Training model ...")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs * len(tok_train_dataloader))
    train_epochs(args, model, optimizer, scheduler, tok_train_dataloader,
                 tok_val_dataloader, tok_train_dataset.label_indexer)
    print("Training model done")

    if args.save:
        model_name = "models/best_model_ass3.pt"
        state_dict = {
            "model": model.state_dict(),
            "label_vocab": tok_train_dataset.label_vocab,
            "label_dict": tok_train_dataset.label_indexer,
            "training_args": args
        }
        torch.save(state_dict, model_name)
        print(f"Model saved as {model_name}")

