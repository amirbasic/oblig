import gensim
import logging
import multiprocessing
import argparse
from os import path
import pandas as pd
import re
import nltk

# This script trains a word2vec word embedding model using Gensim
# Example corpus: /fp/projects01/ec30/corpora/enwiki
# To run this on Fox, load this module:
# nlpl-gensim/4.2.0-foss-2021a-Python-3.9.5

def text_preprocessing(
    text:list,
    punctuations = r'''!()-[]{};:'"\,<>./?@#$%^&*_“~''',
    #stop_words=['and', 'a', 'is', 'the', 'in', 'be', 'will']
    )->list:
    """
    A method to preproces text
    """
    for x in text.lower(): 
        if x in punctuations: 
            text = text.replace(x, "")

    # Removing words that have numbers in them
    text = re.sub(r'\w*\d\w*', '', text)

    # Removing digits
    text = re.sub(r'[0-9]+', '', text)

    # Cleaning the whitespaces
    text = re.sub(r'\s+', ' ', text).strip()

    # Setting every word to lower
    text = text.lower()

    # Converting all our text to a list 
    text = text.split(' ')

    # Droping empty strings
    text = [x for x in text if x!='']

    return text

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("--corpus", help="Path to a training corpus (can be compressed)", required=True)
    arg("--cores", default=False, help="Limit on the number of cores to use")
    arg("--sg", default=0, type=int, help="Use Skipgram (1) or CBOW (0)")
    arg("--window", default=5, type=int, help="Size of context window")
    arg("--vocab", default=100000, type=int, help="Max vocabulary size")
    args = parser.parse_args()

    data = pd.read_csv(args.corpus, sep = "\t", header = 0, on_bad_lines='skip')
    data = data.dropna()
    premise = data['premise'].values.tolist()
    hypothesis = data['hypothesis'].values.tolist()
    corpus = premise + hypothesis
    for i, text in enumerate(corpus):
        corpus[i] = text_preprocessing(text)

    data = gensim.models.word2vec.LineSentence(corpus)

    if args.cores:
        # Use the number of cores we are told to use (in a SLURM file, for example):
        cores = int(args.cores)
    else:
        # Use all cores we have access to except one
        cores = (
            multiprocessing.cpu_count() - 1
        )

    skipgram = args.sg
    # Context window size (e.g., 2 words to the right and to the left)
    window = args.window
    # How many words types we want to be considered (sorted by frequency)?
    vocabsize = args.vocab

    vectorsize = 300  # Dimensionality of the resulting word embeddings.

    # For how many epochs to train a model (how many passes over corpus)?
    iterations = 4
    
    model = gensim.models.Word2Vec(
        data,
        vector_size=vectorsize,
        window=window,
        workers=cores,
        sg=skipgram,
        max_final_vocab=vocabsize,
        epochs=iterations,
        sample=0.001,
    )

    filename = path.basename(corpus).replace(".txt.gz", ".model")
    #logger.info(filename)

    # Save the model without the output vectors (what you most probably want):
    filename = "trained_word2vec.model"
    model.wv.save(filename)









