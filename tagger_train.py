# python3.7 tagger_train.py <train_file_absolute_path> <model_file_absolute_path>

from random import uniform
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle

import sys
import numpy as np
import torch
import random
from gensim.models import Word2Vec
from multiprocessing import cpu_count

all_chars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .!?:,\'%-\(\)/$|&;[]"{}#*=@`'


def prepare_sequence(seq, to_ix):
    idxs = []
    for word in seq:
        if to_ix.get(word) is not None:
            idxs.append(to_ix[word])
        else:
            idxs.append(random.randrange(len(to_ix)))
    # idxs = [to_ix[w] for w in seq]

    return torch.tensor(idxs, dtype=torch.long)


def read_data(path):
    f = open(path, "r")
    training_data = []
    tags_set = set()
    for line in f:
        sentence = line.split()
        words = []
        tags = []
        for item in sentence:
            splitted = item.split("/")
            if len(splitted) > 2:
                for i in range(len(splitted) - 1):
                    words.append(splitted[i])
                    tags.append(splitted[-1])
            else:
                words.append(splitted[0])
                tags.append(splitted[1])
        for tag in tags:
            tags_set.add(tag)
        training_data.append(tuple([words, tags]))
    return training_data, tags_set


def load_embeddings(data, word_dict, size=50):
    model = Word2Vec(data, min_count=1, size=size, workers=cpu_count(), window=3, sg=1)
    word_embeddings = np.zeros((len(word_dict), size))
    for word in word_dict:
        word_embeddings[word_dict[word]] = model[word]
        #print(word)
        #print(model[word])

    # print(model.most_similar('gunshot')[:3])
    return word_embeddings


def char_embedding():
    '''embeddings = np.array([hash(c) for c in s])
    embeddings /= np.max(embeddings)
    embeddings = {s[i]: embeddings[i] for i in range(len(embeddings))}'''

    char_embeddings = dict()
    for i, c in enumerate(all_chars):
        char_embeddings[c] = [0]*len(all_chars)
        char_embeddings[c][i] = 1
    for c in char_embeddings.keys():
        print(c)
        print(char_embeddings[c])
        print('\n')
    return char_embeddings


def train_model(train_file, model_file=0):
    # write your code here. You can add functions as well.
    # use torch library to save model parameters, hyperparameters, etc. to model_file
    training_data, tags_set = read_data(train_file)
    chars = dict()
    word_to_ix = {}
    for sent, tags in training_data:
        for word in sent:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)

    word_embeddings = load_embeddings([i[0] for i in training_data], word_to_ix)
    char_embeddings = char_embedding()
    tag_to_ix = {}
    ix_to_tag = {}
    count = 0
    for tag in tags_set:
        tag_to_ix[tag] = count
        ix_to_tag[count] = tag
        count += 1

    print('Finished...')



if __name__ == "__main__":
    # make no changes here
    train_file = sys.argv[1]
    model_file = sys.argv[2]
    train_model(train_file, model_file)
