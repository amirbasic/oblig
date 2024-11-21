import argparse
from generic_models import *
from useful_functions import *
import torch
from dataset import *
from preprocess_text_generate_features import *
import gzip
from smart_open import open
import shutil


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict on data") 
    parser.add_argument("--test", default="IN5550/oblig2/data/mnli_train.tsv.gz")
    parser.add_argument("--model_path", default="models/best_model_ass2.pt")
    args = parser.parse_args()

    print("Loading model ...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    saved_model = torch.load(args.model_path, map_location='cpu')
    training_args = saved_model['training_args']
    state_dict = saved_model['model']
    train_vocab = saved_model["vocab"]
    word2vec = saved_model["word2vec"]
    print("Loading model done")
    
    print("Loading data ...")
    lines = [line.strip().split('\t') for line in open(args.test, encoding="utf8")]
    df = pd.DataFrame(lines)
    df.columns = df.iloc[0]
    df = df[1:]
    df = tag_or_untag(df, tagged = training_args.pos_tag)
    
    dataset = TSVDataset(df, word2vec)
    label_indexies = dataset.label_indexer
      
    data_iter = DataLoader(dataset,
                           batch_size = training_args.batch_size,
                           shuffle = False,
                           num_workers = 1,
                           collate_fn = CollateFunctor(word2vec.get_index("[PAD]"),
                                                       training_args.max_length)
                            )
    print("Loading data done")


    print("Evaluating model ...")
    model = Model(training_args, word2vec, len(train_vocab)).to(device)
    model.load_state_dict(state_dict)
  
    model.eval()
    
    labels_true, predictions = [], []
    for p, h, label_true in tqdm.tqdm(data_iter):
        p, h, label_true = p.to(device), h.to(device), label_true.to(device)
        output = model(p, h)
        predictions += output.argmax(dim=1).tolist()
        labels_true += label_true.tolist()
        
    result = (torch.tensor(predictions) == torch.tensor(labels_true)).float().mean() * 100.0

    print("Evaluating model done")
    print(f"Result: Accuracy = {result:.2f}%")
        
    labels = []
    for i in predictions:
        _ = list(label_indexies.keys())[list(label_indexies.values()).index(i)]
        labels.append(_) 
        
    df = pd.DataFrame(lines)
    df.columns = df.iloc[0]
    df = df[1:]
    df['label'] = labels
    
    df.to_csv('predictions.tsv', sep="\t", index=False)
    with open('predictions.tsv', 'rb') as f_in:
        with gzip.open('predictions.tsv.gz', 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    