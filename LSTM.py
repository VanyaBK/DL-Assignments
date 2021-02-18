# -*- coding: utf-8 -*-
"""LSTM_Word_Embeddings.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1D9PDpEG6I_dvxeM_1wTbVnWrpKRyQt3m
"""

from __future__ import print_function
import gensim.downloader as api # package to download text corpus
import nltk # text processing
from nltk.corpus import stopwords
import string
import numpy as np
import collections
# download stopwords
nltk.download('stopwords')

# download textcorpus
data = api.load('text8')
print(data)

# collect all words to be removed
stop = stopwords.words('english') + list(string.punctuation)

actual_words = []
cleaned_words = []
unique_words = set()

word_counts = collections.defaultdict(int)
for words in data:
    for word in words:
        word_counts[word]+=1

# remove stop words
print('removing stop words from text corpus')
for words in data:
    current_nonstop_words = [w for w in words if w not in stop]
    cleaned_words+=(current_nonstop_words)
    #cleaned_words += current_nonstop_words
    actual_words += words

    for ns in current_nonstop_words:
        unique_words.add(ns)

# print statistics
print(len(actual_words), 'words BEFORE cleaning stop words and punctuations')
print(len(cleaned_words), 'words AFTER cleaning stop words and punctuations')
print('vocabulary size: ', len(unique_words))

# 'cleaned_words' and 'unique_words' to create a word2vec model

#Generate Training Data
import collections
word_counts = collections.defaultdict(int)
for word in cleaned_words:
  word_counts[word] += 1

v_count = len(word_counts.keys())

# GENERATE LOOKUP DICTIONARIES
words_list = sorted(list(word_counts.keys()),reverse=False)
word_index = dict((word, i) for i, word in enumerate(words_list))
index_word = dict((i, word) for i, word in enumerate(words_list))
one_hot_encoding = []

training_data = []
for ind in range(len(cleaned_words)-11):
    w_context=[]
    for i in range(10):
        w_context.append(word_index[cleaned_words[ind+i]])
    w_target = word_index[cleaned_words[ind+10]]
    temp={}
    temp['text']=w_context
    temp['label']=w_target
    training_data.append(temp)

import torch
from torch import nn

class Model(nn.Module):
    def __init__(self, n_vocab):
        super(Model, self).__init__()
        self.lstm_size = 128
        self.embedding_dim = 128
        self.num_layers = 1

        self.embedding = nn.Embedding(
            num_embeddings=n_vocab,
            embedding_dim=self.embedding_dim,
        )
        self.lstm = nn.LSTM(
            input_size=self.lstm_size,
            hidden_size=self.lstm_size,
            num_layers=self.num_layers,
            dropout=0.2,
            
        )
        self.fc = nn.Linear(self.lstm_size, n_vocab)

    def forward(self, x, prev_state):
        x=torch.LongTensor(x)
        embed = self.embedding(x)
        embed.unsqueeze_(0)
        output, state = self.lstm(embed, prev_state)
        logits = self.fc(output)

        return logits, state

    def init_state(self, sequence_length):
        return (torch.zeros(self.num_layers, sequence_length, self.lstm_size),
                torch.zeros(self.num_layers, sequence_length, self.lstm_size))

import argparse
import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader


def train(training_data,model, sequence_length,max_epochs,BATCH_SIZE):
    model.train()
    dataloader = DataLoader(training_data, batch_size=1)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    ind=0
    for epoch in range(max_epochs):
        state_h, state_c = model.init_state(sequence_length)
        for batch in dataloader:
            
            optimizer.zero_grad()

            y_pred, (state_h, state_c) = model(batch['text'], (state_h, state_c))
           
            x = batch['label'].item()
            y = torch.zeros(len(unique_words))
            y[x] = 1
            
            loss = criterion(y_pred.transpose(1,2).squeeze(0), y.long())

            state_h = state_h.detach()
            state_c = state_c.detach()

            loss.backward()
            optimizer.step()
           
            #print({ 'epoch': epoch, 'batch': batch, 'loss': loss.item() })

model = Model(len(unique_words))
train_iterator = DataLoader(training_data, batch_size=256)

train(training_data,model,10,1,256)
#Save word embeddings
torch.save(model.embedding.weight,"LSTM_EMBEDDING.pt")