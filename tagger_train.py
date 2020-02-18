# python3.7 tagger_train.py <train_file_absolute_path> <model_file_absolute_path>

import pickle
import sys
from random import uniform

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from gensim.models import KeyedVectors


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


def load_embeddings(word_dict):
    word_embeddings = np.zeros((len(word_dict), 300))
    print("load")
    model = KeyedVectors.load_word2vec_format('embeddings/GoogleNews-vectors-negative300.bin', binary=True)
    print("start")
    for word in word_dict:
        word_embeddings[word_dict[word]] = model[word]
        print(word)
        print(model[word])

    return word_embeddings


def train_model(train_file, model_file):
    # write your code here. You can add functions as well.
    # use torch library to save model parameters, hyperparameters, etc. to model_file
    training_data, tags_set = read_data(train_file)

    word_to_ix = {}
    for sent, tags in training_data:
        for word in sent:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)

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
    # train_file = sys.argv[1]
    # model_file = sys.argv[2]
    training_data, tags_set = read_data("./corpus.train")
    print("finished")
    word_to_ix = {}
    print(len(training_data))
    for sent, tags in training_data:
        for word in sent:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
    print("start")
    load_embeddings(word_to_ix)
    # train_model(train_file, model_file)
