import gzip
import io
import conllu
from sklearn.model_selection import train_test_split
from smart_open import open
import gzip

def overwrite_conllu_labels(conllu_path, df, prediction_path):
    with gzip.open(conllu_path, "rb") as f:
        data = io.TextIOWrapper(f, encoding="utf-8").read()
        sentences = conllu.parse(data)

        prediction_list = df['labels'].tolist()
  
        for i in range(len(prediction_list)):
            predictions = prediction_list[i]
            for j in range(len(sentences[i])):
                sentences[i][j]["misc"]["name"] = predictions[j]
                
    with open(prediction_path, "w") as f:
        for token_list in sentences:
            serialized = token_list.serialize()
            f.write(serialized)
            f.write('\n\n')
    return

def write_val_conllu(args):
    conllu_path = args.data_path
    val_gold_label_path = args.gold_label_conllu_path
    with gzip.open(conllu_path, "r") as f:
        data = io.TextIOWrapper(f, encoding="utf-8").read()
        parsed_data = conllu.parse(data)
         
        train_conllu, val_conllu = train_test_split(parsed_data, train_size=args.split, random_state=args.seed)

        with open(val_gold_label_path, "w") as f:
            for sent in val_conllu:
                serialized = sent.serialize()
                f.write(serialized)
                f.write('\n\n')
    return
