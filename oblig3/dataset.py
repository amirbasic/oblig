from torch.utils.data import Dataset
import torch
import pandas as pd
import conllu
import gzip
import io
#from datasets import ClassLabel

def open_and_read_path(data_path):
    # Load the CoNLL-U file    
    with gzip.open(data_path, "rb") as f:
        data = io.TextIOWrapper(f, encoding="utf-8").read()

    # Parse the CoNLL-U file using conllu library
    parsed_data = conllu.parse(data)

    # Extract the token and named entity label from the CoNLL-U file
    sentences = []
    tags = []
    labels = []
    metadata = []
    sentence_length = []
    for sentence in parsed_data:
        tokens = []
        tokens_tags = []
        token_label = []
        token_metadata = []
        for token in sentence:
            # Extract the token and named entity label
            tokens.append(token['form'])
            tokens_tags.append(token["upos"])
            token_label.append(token['misc']['name'])
            token_metadata.append(token['feats'])
        sentences.append(" ".join(tokens))
        tags.append(" ".join(tokens_tags))
        labels.append(" ".join(token_label))
        metadata.append(token_metadata)
        sentence_length.append(len(token_label))

    data_dict = {"sentence" : sentences,
                 "labels" : labels,
                 "sentence_length" : sentence_length}

    data = pd.DataFrame(data_dict)

    return data

def drop_rows(data, min_labels = 1):
    drop_idx = []
    data = data.reset_index()
    for row_idx in range(len(data.labels)):
        label_row = data.labels[row_idx].split(" ")
        filtered_labels = []
        for tag in label_row:
            if tag != "O":
                filtered_labels.append(tag)
                
        if min_labels > len(filtered_labels):
            drop_idx.append(row_idx)

    return data.drop(drop_idx, axis=0)

class EmbDataset(Dataset):
    def __init__(self, data, embedding, label_vocab = None):

        self.unk_index = embedding.get_index("[UNK]")
        self.sentences = [
            [
                embedding.get_index(token.lower(), default = self.unk_index)
                for token in document
            ] 
            for document in data["sentence"].str.split(" ")
        ]
            
        unk_tokens = sum(token == self.unk_index for document in self.sentences for token in document)
        n_tokens = sum(len(document) for document in self.sentences)
        print(f"\t\tPercentage of unknown tokens: {unk_tokens / n_tokens * 100.0:.2f}%")
        
        self.label = list(data['labels'])
        self.label_vocab = label_vocab if label_vocab is not None else list(sorted(set(" ".join(self.label).split(" "))))
        self.num_labels = len(self.label_vocab)
        self.label_indexer = {i: n for n, i in enumerate(self.label_vocab)}
        
    def __getitem__(self, index):
        current_tokens_sentence = self.sentences[index]
        current_labels = self.label[index]

        sentence = torch.LongTensor(current_tokens_sentence)

        labels = []
        for label in current_labels.split(" "):
            labels.append(self.label_indexer[label])
        
        y = torch.LongTensor(labels)
        return sentence, y

    def __len__(self):
        return len(self.sentences)


class TokenDataset(Dataset):
    def __init__(self, data, tokenizer, args, label_vocab = None):
        
        self.max_length = args.max_length
        
        self.train_texts = data["sentence"].to_list()
        self.labels = list(data["labels"])
        self.sentence_length = data["sentence_length"].to_list()
        
        self.encoding = tokenizer(self.train_texts,
                                 padding = True,
                                 truncation = True,
                                 return_offsets_mapping = True,
                                 max_length = self.max_length)
        
        self.input_ids = torch.tensor(self.encoding.input_ids)
        self.attention_mask = torch.tensor(self.encoding["attention_mask"])
        
        self.offsets = self.encoding.offset_mapping
        self.word_lengths = [[len(word) for word in sentence.split()] for sentence in self.train_texts]

        self.label_vocab = label_vocab if label_vocab is not None else list(sorted(set(" ".join(self.labels).split(" "))))
        self.num_labels = len(self.label_vocab)
        self.label_indexer = {i: n for n, i in enumerate(self.label_vocab)}
        
    def __getitem__(self, index):
        current_input_ids = self.input_ids[index]
        current_attention_mask = self.attention_mask[index]

        input_ids = torch.LongTensor(current_input_ids)
        attention_mask = torch.LongTensor(current_attention_mask)

        #print("sentence_length:", self.sentence_length)
        current_sent_len = self.sentence_length[index]
        #print("curr sent:", current_sent_len)

        def pad(l, content, width):
            l.extend([content] * (width - len(l)))
            l = l[:self.max_length]
            return l
            
        current_labels = self.labels[index]
        value = self.label_indexer["O"]
        labels = []
        for label in current_labels.split(" "):
            labels.append(self.label_indexer[label])
        labels = pad(labels, value, self.max_length)
        
        y = torch.LongTensor(labels)
        
        offset_mapping = torch.LongTensor(self.offsets[index])
        
        word_length = self.word_lengths[index]
        word_length = pad(word_length, 0, self.max_length)
        word_length = torch.LongTensor(word_length)

        return input_ids, attention_mask, y, offset_mapping, word_length, current_sent_len

    def __len__(self):
        return len(self.train_texts)


class NotebookTokenDataset(Dataset):
    def __init__(self, data, tokenizer, label_vocab = None):
        
        self.max_length = 64
        
        self.train_texts = data["sentence"].to_list()
        self.labels = list(data["labels"])
        
        self.encoding = tokenizer(self.train_texts,
                                 padding = True,
                                 truncation = True,
                                 return_offsets_mapping = True,
                                 max_length = self.max_length)
        
        self.input_ids = torch.tensor(self.encoding.input_ids)
        self.attention_mask = torch.tensor(self.encoding["attention_mask"])
        
        self.offsets = self.encoding.offset_mapping
        self.word_lengths = [[len(word) for word in sentence.split()] for sentence in self.train_texts]

        self.label_vocab = label_vocab if label_vocab is not None else list(sorted(set(" ".join(self.labels).split(" "))))
        self.num_labels = len(self.label_vocab)
        self.label_indexer = {i: n for n, i in enumerate(self.label_vocab)}
        
    def __getitem__(self, index):
        current_input_ids = self.input_ids[index]
        current_attention_mask = self.attention_mask[index]

        input_ids = torch.LongTensor(current_input_ids)
        attention_mask = torch.LongTensor(current_attention_mask)

        
        def pad(l, content, width):
            l.extend([content] * (width - len(l)))
            l = l[:self.max_length]
            return l
            
        current_labels = self.labels[index]
        value = self.label_indexer["O"]
        labels = []
        for label in current_labels.split(" "):
            labels.append(self.label_indexer[label])
        labels = pad(labels, value, self.max_length)
        
        y = torch.LongTensor(labels)
        
        offset_mapping = torch.LongTensor(self.offsets[index])

        word_length = self.word_lengths[index]
        word_length = pad(word_length, 0, self.max_length)
        word_length = torch.LongTensor(word_length)

        return input_ids, attention_mask, y, offset_mapping, word_length

    def __len__(self):
        return len(self.train_texts)
    
def align_label(texts, labels, tokenizer, labels_to_ids, label_all_tokens=False):
    tokenized_inputs = tokenizer(texts, padding='max_length', max_length=512, truncation=True)

    word_ids = tokenized_inputs.word_ids()

    previous_word_idx = None
    label_ids = []

    for word_idx in word_ids:

        if word_idx is None:
            label_ids.append(-100)

        elif word_idx != previous_word_idx:
            try:
                label_ids.append(labels_to_ids[labels[word_idx]])
            except:
                label_ids.append(-100)
        else:
            try:
                label_ids.append(labels_to_ids[labels[word_idx]] if label_all_tokens else -100)
            except:
                label_ids.append(-100)
        previous_word_idx = word_idx

    return label_ids

class DataSequence(torch.utils.data.Dataset):

    def __init__(self, df, tokenizer, labels_to_ids):

        lb = [i.split() for i in df['labels'].values.tolist()]
        txt = df['sentence'].values.tolist()
        self.texts = [tokenizer(str(i),
                               padding='max_length', max_length = 512, truncation=True, return_tensors="pt") for i in txt]
        self.labels = [align_label(i,j, tokenizer, labels_to_ids) for i,j in zip(txt, lb)]

    def __len__(self):

        return len(self.labels)

    def get_batch_data(self, idx):

        return self.texts[idx]

    def get_batch_labels(self, idx):

        return torch.LongTensor(self.labels[idx])

    def __getitem__(self, idx):

        batch_data = self.get_batch_data(idx)
        batch_labels = self.get_batch_labels(idx)

        return batch_data, batch_labels
    
    
    
    