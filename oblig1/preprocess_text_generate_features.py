import pandas as pd 
import torch
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn import metrics, preprocessing
from torch import nn
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch.nn.functional as Fun
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.corpus import stopwords
import string
import re
import warnings
from sklearn import preprocessing

# List of words to be ignored in preprocessing of the text data
ignore_words = ["sep", "be", "min", "say", "in", "hel", "co", "de", "la",
             "th", "cut", "much", "www", "com", "tsx", "asa", "49", "have",
             "far", "same", "pv", "us", "all", "well", "42", "own", "name",
             "many", "still", "click", "here", "read", "news", "even", "world",
             "we", "on", "day", "now", "only", "run", "big", "how", "go", "to",
             "like", "comment", "search", "man", "Texa", "sept", "29", "prnewswire",
             "new", "use", "two", "just", "next", "step", "make", "easy", "get",
             "need", "set", "oem", "marie", "smith", "pr", "end", "June",
             "re", "per", "10", "fix", "clos", "lay", "mill", "such", "non",
             "promot", "ed", "ph", "put", "for", "1st", "ky", "cumberland",
             "list", "arixona", "phoenix", "chigaco", "illinois", "idaho",
             "boston", "massachusett", "city", "seattle", "washington", "omaha",
             "nebraska", "chicago", "oregon", "philadelphia", "pennsylvania",
             "johnson", "tennesse","tennessee", "george", "fox", "indiana", "mercy", "virginia",
             "nashville", "michigan", "nortwest", "sam", "houston", "alabama",
             "florida", "cincinnati", "ohio", "florida", "minnesota", "minneapolis",
             "lincoln", "enland", "maine", "carolina", "los", "angele", "georgia",
             "see", "site", "about", "sacramento", "putt", "band", "back", "gm",
             "too", "foot", "bit", "as", "cas", "suight", "ad", "till", "richard",
             "m2", "ga", "ga", "az", "do", "zack", "michael", "connecticut", "york",
             "kenneth", "show", "203", "christina", "09", "24", "Roy", "of",
             "colorado", "chicago", "arizona", "michel", "montreal", "quebec", "mario",
             "james", "tsx", "sub", "gmbh", "ebitda", "old", "new", "tri", "steve", "arizona",
             "chicago", "tennessee", "george", "sam", "columbia", "california", "tampa",
             "way", "abnewswire", "stamford", "04", "21", "22", "15", "25", "4k", "___", 
             "nv", "gam", "upcom", "expo", "combin", "john", "up", "xml", "json", "at", "san",
             "jose", "kumar", "rt", "di", "jr", "ko", "pac", "andre", "mgm", "nevada", "et",
              "vow", "then", "ali", "spr", "four", "fla", "17th", "st", "10th", "fda", "nda",
              "tim", "describ", "brian", "david", "matt", "or", "caqr", "acte", "bas", "llc",
              "ge", "jordan", "11", "vancouver", "toronto", "issu", "undue", "la_propn", "nv_propn",
             "vega_propn", "gam_verb", "combin_verb", "john_propn", "excit_adj", "tailore_verb",
             "set_verb", "s_verb", "gam_adj", "com_noun", "xml_propn", "json_propn", "at_adv",
             "jose_noun", "san_propn", "california_propn", "kumar_propn", "rt_propn", "up_adv",
             "set_noun", "go_verb", "be_verb", "here_adv", "get_propn", "nevada_propn", "v_noun",
             "tv_noun", "et_propn", "george_propn", "jack_propn", "put_verb", "then_adv", "vow_verb",
             "s_propn", "ali_propn", "spr_verb", "get_verb", "just_adv", "29_adj", "all_adv", "fla_propn",
             "j_propn", "d_propn", "st_noun", "top", "fda_propn", "also_adv", "_sym", "fda_noun", "tim_noun",
             "appro_verb", "determin_verb", "k_propn", "d_verb", "matt_propn", "brian_propn", "david_propn",
             "llc_propn", "k_noun", "james_propn", "jordan_propn"]

def preprocessing_step(text):
    """ Preprocess the raw text data using our "Basic text preprocessing" approach.

        Parameters
        ----------
        text: string or string-like object - Raw strings of data

        Returns
        -------
        text: string or string-like object - Raw strings of data
    """
    # changing all "NUM" POS tags to whitespace
    text = re.sub(r'[\w.,]*[\w.,]*_NUM\S*', ' ', text)
    text = re.sub(r'[^\w]', ' ', text)
    text = re.sub("^\d+\s|\s\d+\s|\s\d+$", " ", text)
    text = text.lower()
    
    querywords = text.split()
    
    resultwords  = [word for word in querywords]
   
    text = ' '.join(resultwords)

    return text

def preprocessing_step_ignore(text):
    """ Preprocess the raw text data using our "Ignoring 'irrelevant' word" approach.

        Parameters
        ----------
        text: string or string-like object - Raw strings of data

        Returns
        -------
        text: string or string-like object - Raw strings of data
    """
    text = re.sub(r'[\w.,]*[\w.,]*_NUM\S*', ' ', text)
    text = re.sub(r'[^\w]', ' ', text)
    text = re.sub("^\d+\s|\s\d+\s|\s\d+$", " ", text)
    text = text.lower()
    
    querywords = text.split()
    
    resultwords  = [word for word in querywords if word not in ignore_words]
   
    text = ' '.join(resultwords)

    return text

def fit_bow(x, ignore_words_bool, max_features):
    """ Vectorizing to create the BoW features by using one of our preprocessing steps

        Parameters
        ----------
        x: string or string-like object - Raw strings of data
        ignore_words_bool: Boolean - True if preprocessing_step_ignore() is used or False if preprocessing_step() is used.
        max_features: int - Number of max features in our BoW.

        Returns
        -------
        x: torch.tensor - BoW features created.
        vectorizer: sklearn.feature_extraction.text.TfidfVectorizer - The vectorizer method used to create our BoW.
        vocab: dictionary - A mapping of terms to feature indices.
    """
    if ignore_words_bool:
        preprocessor = preprocessing_step_ignore
    else:
        preprocessor = preprocessing_step
        
    vectorizer = TfidfVectorizer(dtype = np.float32,
                                   ngram_range=(1,2),
                                   preprocessor = preprocessor,
                                   max_features = max_features,
                                  )
    
    x = vectorizer.fit_transform(x)
    vocab = vectorizer.vocabulary_
    x = x.toarray()
    torch.from_numpy(x)
    
    return x, vectorizer, vocab

def make_bow(x, fitted_vectorizer):
    """ Vectorizing to create the BoW features by using one of our preprocessing steps

        Parameters
        ----------
        x: string or string-like object - Raw strings of data
        fitted_vectorizer: sklearn.feature_extraction.text.TfidfVectorizer - Pre-fitted vectorizer

        Returns
        -------
        x: torch.tensor - tranformed text data
    """

    x = fitted_vectorizer.transform(x)
    x = x.toarray()
    torch.from_numpy(x)
    
    return x

def preprocess_data(path_to_data, ignore_words_bool = True,max_features=10000, path_to_test_data=None, random_state=42, evaluate=False):
    """ Entire preprocessing procedure

        Parameters
        ----------
        path_to_data: string - Path to training dataset.
        ignore_words_bool: Boolean - True if preprocessing_step_ignore() is used or False if preprocessing_step() is used.
        max_features: int - Number of max features in our BoW.
        path_to_test_data: string - Path to test dataset.
        random_state: int - random state number.
        evaluate: Boolean - Set to True if test data is used.

        Returns
        -------
        X_train: torch.tensor - tensor of BoW features.
        X_val: torch.tensor - transformed tensor based on BoW features.
        y_train: torch.tensor - transformed label outputs based on preprocessing.LabelEncoder.
        y_val: torch.tensor - transformed label outputs based on preprocessing.LabelEncoder.
        vocab: dictionary - A mapping of terms to feature indices.
        num_classes: int - number of output classes.
    """
    data = pd.read_csv(path_to_data, sep = "\t")
    
    X = data.text
    y = data.source
    
    X_train, X_val, y_train, y_val = train_test_split(X,
                                                  y,
                                                  test_size = 0.2,
                                                  random_state = random_state,
                                                  stratify = y
                                                  )
    
    print("Fitting BoW")
    X_train, fitted_vectorizer, vocab = fit_bow(X_train,  ignore_words_bool, max_features=max_features)
    
    X_val = make_bow(X_val, fitted_vectorizer)
    
    label_dict =  preprocessing.LabelEncoder()
    label_dict.fit(y_train)
    y_train = label_dict.transform(y_train)
    y_val = label_dict.transform(y_val)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_val = torch.tensor(y_val, dtype=torch.long)
    num_classes = len(label_dict.classes_)
    
    if evaluate:
        pd.read_csv(path_to_test_data, sep = "\t")
        X_test = data.text
        y_test = data.source
        
        X_test = make_bow(X_test, fitted_vectorizer)
        y_test = label_dict.transform(y_test)
        y_test = torch.tensor(y_test)
        
        return X_test, y_test, vocab, num_classes
    
    return X_train, X_val, y_train, y_val, vocab, num_classes
    