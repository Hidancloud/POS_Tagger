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


def char_embedding(chars, emb_size=10):
    '''embeddings = np.array([hash(c) for c in s])
    embeddings /= np.max(embeddings)
    embeddings = {s[i]: embeddings[i] for i in range(len(embeddings))}'''

    char_embeddings = dict()
    for i, c in enumerate(chars):
        char_embeddings[c] = [0]*len(chars)
        char_embeddings[c][i] = 1
    # torch.nn.Embedding(len(char_embeddings.keys()), emb_size)

    return char_embeddings


def train_model(train_file, model_file='0'):
    maxlen = -float('inf')
    training_data, tags_set = read_data(train_file)
    chars = dict()
    word_to_ix = {}
    for sent, tags in training_data:
        print(sent)
        for word in sent:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
            if len(word) > maxlen:
                maxlen = len(word)

    ''' unused part, skip
    word_embeddings = load_embeddings([i[0] for i in training_data], word_to_ix)
    char_embeddings = char_embedding(all_chars) '''

    # parameters from the task
    l = 5
    w_emb = 50
    d_char = 10
    k = 5
    torch_word_embedding = nn.Embedding(len(word_to_ix), w_emb)
    torch_char_embedding = nn.Embedding(len(all_chars), d_char)

    for sent, _ in training_data:
        for word in sent:
            # create embedings for the word and each char in it
            this_word_embedding = torch_word_embedding(torch.tensor([word_to_ix[word]], dtype=torch.long))
            char_embeddings = [torch_char_embedding(torch.tensor([all_chars.index(char)], dtype=torch.long)) for char in word]
            # pad zeros to the left and to the right to correctly apply method on bounds
            char_embeddings.append([[0]*d_char]*int((k-1)/2))
            char_embeddings = [[0]*d_char]*int((k-1)/2) + char_embeddings
            # create X vector from the window size K
            X = [char_embeddings[i-int((k-1)/2):i+int((k-1)/2)] for i in range(int((k-1)/2), len(char_embeddings) - int((k-1)/2))]
            # apply g(XU + b) from the task **what is g by the way?**
            linear = nn.Linear(k*d_char, l, bias=True)
            proceed = np.array([torch.nn.functional.relu(linear(x_i)) for x_i in X])
            # calculate c using the formula from the task
            c = np.array([max(proceed[:, k]) for k in range(l)])
            # final representation of the word, concat word embedding and c
            x = [*this_word_embedding, *c]
            # apply bidirectional lstm, to extract both hidden states two models are used, can be replaced with one
            # with flag *bidirectional* = True
            lstm_forward = nn.LSTM(len(x), hidden_size=50, num_layers=1, bias=True, batch_first=False, dropout=0, bidirectional=False)
            _, hidden_forward = lstm_forward(x)
            lstm_backward = nn.LSTM(len(x), hidden_size=50, num_layers=1, bias=True, batch_first=False, dropout=0, bidirectional=False)
            x.reverse()
            _, hidden_backward = lstm_backward(x)
            h = np.array(*hidden_forward, *hidden_backward)
            # projecting vector of bidirectional lstm with size 50+50=100 on the correct space with dim=45 using linear
            # layer
            last_layer = nn.Linear(2*50, 45, bias=True)
            answer = torch.nn.functional.softmax(last_layer(h))

            ### TODO: use Adam classifier to optimize everything with nn.NLLLoss, calculate accuracy and save the model
            ### TODO: also test everything

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
