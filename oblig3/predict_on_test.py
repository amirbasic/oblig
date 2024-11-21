import argparse
from useful_functions import *
import torch
from dataset import *
from model import BertLinear, BertGRU, BertLSTM
from smart_open import open
from transformers import AutoTokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict on data") 
    parser.add_argument("--test", default = "IN5550/oblig3/data/gold_label_val.conllu.gz")
    parser.add_argument("--model_path", default = "models/best_model_ass3.pt")
    parser.add_argument("--output_path", default = "predictions.conllu.gz")
    args = parser.parse_args()

    print("Loading model ...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    saved_model = torch.load(args.model_path, map_location='cpu')
    training_args = saved_model['training_args']
    state_dict = saved_model['model']
    model_label_vocab = saved_model["label_vocab"]
    label_dict = saved_model["label_dict"]

    tokenizer = AutoTokenizer.from_pretrained(training_args.pretrained)
    model = BertLSTM(training_args.pretrained, num_labels = 17).to(device)
    model.load_state_dict(state_dict)
    print("Loading model done")
    
    print("Loading data ...")
    data = open_and_read_path(args.test)
    args.max_length = max(data["sentence_length"])

    dataset = TokenDataset(data, tokenizer, args, label_vocab=model_label_vocab)

    dataloader = DataLoader(dataset,
                            batch_size=training_args.batch_size,
                            shuffle=False,
                            drop_last=False,
                            num_workers=1
                            )

    print("Loading data done")

    print("Evaluating model ...")
    model.eval()
    predictions = []
    for input_ids, attention_mask, y, offset_mapping, word_length, sent_len in tqdm.tqdm(dataloader):
        input_ids, attention_mask, y, offset_mapping, word_length, sent_len  = input_ids.to(device), attention_mask.to(device), y.to(device), offset_mapping.to(device), word_length.to(device), sent_len.to(device)
        n_subwords = input_ids.size(1)
        n_words = max(len(words) for words in word_length)
        mapping_matrix = get_mapping_matrix_batched(offset_mapping, word_length, n_subwords, n_words).to(device)
        pred = model(input_ids, attention_mask, mapping_matrix)
        pred = pred.argmax(dim=2).tolist()

        labels_unpad, pred_unpad = depad(pred, y.tolist(), sent_len)
        predictions += pred_unpad
    print("Evaluating model done")

    print("Writing conllu file ...")
    correct_label_pred = []
    for i in predictions:
        correct_label_pred.append(invert_label(i, label_dict))
    pred = pd.DataFrame()
    pred["labels"] = correct_label_pred
    overwrite_conllu_labels(args.test, pred, args.output_path)
    print("Writing conllu file done")

    print("Mock evaluating ... ")
    real_evaluation(args.test, args.output_path)
    print("Mock evaluating done ")
    