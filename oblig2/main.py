from useful_functions import seed_everything, train, evaluate, tag_or_untag, load_embedding, time_all_nn, grid_search, run_single_model, pool_search
from generic_models import Model
from preprocess_text_generate_features import CollateFunctor
from dataset import TSVDataset
from torch.optim.lr_scheduler import _LRScheduler
import torch
import argparse
from preprocess_text_generate_features import *
import time
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.optim import Optimizer
import torch.nn.functional as F
import tqdm
from smart_open import open
import warnings
warnings.filterwarnings('ignore')


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model")
    parser.add_argument(
        "--data_path", default="IN5550/oblig2/data/mnli_train.tsv.gz")
    parser.add_argument(
        "--embedding_path", default="IN5550/oblig2/data/our_embedding_tag_untag.bin")
    parser.add_argument("--pos_tag", default=False)
    parser.add_argument("--hidden_size", default=4096)
    parser.add_argument("--num_layers", default=2)
    parser.add_argument("--batch_size", default=512)
    parser.add_argument("--lr", default=0.0005)
    parser.add_argument("--epochs", default=4)
    parser.add_argument("--split", default=0.7)
    parser.add_argument("--max_length", default=64)
    parser.add_argument("--dropout", default=0.0)
    parser.add_argument("--seed", default=5550)
    parser.add_argument("--pooling_type", default="max")  # "sum", "avg", "max"
    parser.add_argument("--freeze", default=True)
    # ff, gru, bilstm, elman
    parser.add_argument("--layer_type", default='bilstm')
    parser.add_argument("--grid_search", type=bool, default=False)
    parser.add_argument("--time_nn", type=bool, default=False)
    parser.add_argument("--pool_search", type=bool, default=False)
    parser.add_argument("--save", type=bool, default=True)
    args = parser.parse_args()

    seed_everything(args.seed)

    print("Loading data ...")
    lines = [line.strip().split('\t')
             for line in open(args.data_path, encoding="utf8")]
    df = pd.DataFrame(lines)
    df.columns = df.iloc[0]
    df = df[1:]
    df = tag_or_untag(df, tagged=args.pos_tag)
    print("Loading data done")

    print("Loading embedding ...")
    word2vec = load_embedding(args.embedding_path)
    word2vec["[UNK]"] = torch.tensor(word2vec.vectors).mean(dim=0).numpy()
    word2vec["[PAD]"] = torch.zeros(word2vec.vector_size).numpy()
    print("Loading embedding done")

    print("Splitting and processing data ...")
    train_df, val_df = train_test_split(
        df, train_size=args.split, random_state=args.seed)
    train_dataset = TSVDataset(train_df, word2vec)
    val_dataset = TSVDataset(
        val_df, word2vec, label_vocab=train_dataset.label_vocab)

    train_iter = DataLoader(train_dataset,
                            batch_size=args.batch_size,
                            shuffle=True,
                            drop_last=True,
                            num_workers=1,
                            collate_fn=CollateFunctor(word2vec.get_index("[PAD]"),
                                                      args.max_length)
                            )

    val_iter = DataLoader(val_dataset,
                          batch_size=args.batch_size,
                          shuffle=False,
                          drop_last=False,
                          num_workers=1,
                          collate_fn=CollateFunctor(word2vec.get_index("[PAD]"),
                                                    args.max_length)
                          )
    print("Splitting and processing data done")

    if args.grid_search:
        grid_search(args, word2vec, train_dataset, train_iter, val_iter)

    elif args.pool_search:
        pool_search(args, word2vec, train_dataset, train_iter, val_iter)

    elif args.time_nn:
        time_all_nn(args, word2vec, train_dataset, train_iter, val_iter)

    else:
        run_single_model(args, word2vec, train_dataset, train_iter, val_iter)
