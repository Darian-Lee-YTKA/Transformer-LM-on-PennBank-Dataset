from datasets import load_dataset

"""from datasets import load_dataset
import pandas as pd
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import torch
import torch.nn as nn
import torch.optim as optim

ptb = load_dataset('ptb-text-only/ptb_text_only', trust_remote_code=True)


train = pd.DataFrame(ptb['train'])
val = pd.DataFrame(ptb['validation'])
test = pd.DataFrame(ptb['test'])
print(train.head())



print("starting")






import re
import unicodedata


def remove_accents(text):
    text = "<S> " + text + " <.>"
    return ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    )

train["sentence"] = train["sentence"].apply(remove_accents)
val["sentence"] = val["sentence"].apply(remove_accents)
test["sentence"] = test["sentence"].apply(remove_accents)

print(train.head())

import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import f1_score
# define dataset

print("defining the dataset")
class Language_Model(Dataset):

    def __init__(self, data, token_vocab=None, training= False):
        if training:
            self.token_vocab = {"<PAD>": 0, "<unk>": 1}

            for utterance in data["sentence"]:
                tokens = utterance.split(" ")
                for token in tokens:
                    if token in self.token_vocab:
                        continue
                    else:
                        self.token_vocab[token] = len(self.token_vocab) # this will just be the index inwhich it was added

        else:
            assert token_vocab is not None
            self.token_vocab = token_vocab
        self.corpus_x_ids = []
        self.corpus_y_ids = []
        self.corpus_pad_ids = []

        for utterance in data["sentence"]:
            token_list = utterance.split(" ")

            x_ids = [self.token_vocab.get(token, self.token_vocab['<unk>']) for token in token_list]
            self.corpus_x_ids.append(torch.tensor(x_ids))

            y_ids = [self.token_vocab.get(token, self.token_vocab['<unk>']) for token in token_list[1:]]
            y_ids.append(self.token_vocab['<PAD>'])  # Add <PAD> token to match length
            self.corpus_y_ids.append(torch.tensor(y_ids))
            # will contain things like (writing in text although they will be in numbers)
            # self.corpus_x_ids = [["<START>", "the", "french", "ferret", "<STOP>], ...]
            # self.corpus_y_ids = [["the", "french", "ferret", "<STOP>, "<PAD>"], ...]

            pad_ids = [1] * (len(x_ids) - 1) + [0]
            self.corpus_pad_ids.append(torch.tensor(pad_ids))

    def __len__(self):
            #Returns the number of sentences.
            return len(self.corpus_x_ids)

    def __getitem__(self, idx):

           # Returns the x, y, and padding mask tensors for a given index.

           # Args:
           #     idx (int): Index of the sentence.

           # Returns:
            #    tuple: (x_ids, y_ids, pad_ids)
             #   example = (["<START>", "the", "french", "ferret", "<STOP>], ["the", "french", "ferret", "<STOP>", "<PAD>"], [1,1,1,1,0]

            print("inputs shape ", self.corpus_x_ids[idx].shape, "output shape ", self.corpus_y_ids[idx].shape)
            print("inputs: ", self.corpus_x_ids[idx], "outputs ", self.corpus_y_ids[idx])

            return self.corpus_x_ids[idx], self.corpus_y_ids[idx], self.corpus_pad_ids[idx]

train_dataset = Language_Model(train, training=True)
val_dataset = Language_Model(val, token_vocab=train_dataset.token_vocab,
                             training=False)
test_dataset = Language_Model(test, token_vocab=train_dataset.token_vocab,
                             training=False)

# collate token_ids and tag_ids to make mini-batches
print("collate_fn")
from torch.nn.utils.rnn import pad_sequence
import torch

import numpy as np


def collate_fn(batch):

   # Args:
    #    batch (list of tuples): Each tuple contains (x_ids, y_ids, pad_mask).

   # Returns:
   #     tuple: padded tensors (x_batch, y_batch, pad_batch).


    x_ids, y_ids, pad_masks = zip(*batch)
    max_len = max(len(x) for x in x_ids)

    x_batch = torch.stack(
        [torch.cat([x, torch.tensor([train_dataset.token_vocab["<PAD>"]] * (max_len - len(x)))]) for x in
         x_ids]).long()  # Ensure long type
    y_batch = torch.stack(
        [torch.cat([y, torch.tensor([train_dataset.token_vocab["<PAD>"]] * (max_len - len(y)))]) for y in
         y_ids]).long()  # Ensure long type
    pad_batch = torch.stack(
        [torch.cat([pad, torch.tensor([train_dataset.token_vocab["<PAD>"]] * (max_len - len(pad)))]) for pad in
         pad_masks]).long()  # Ensure long type
    print("x_batch shape: ", x_batch.shape, " y_batch shape ", y_batch.shape)
    print("x batch")
    print(x_batch)
    print("y batch")
    print(y_batch)
    return x_batch, y_batch, pad_batch






class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table == nn.Embedding(vocab_size, vocab_size)
    def forward(self, idx, targets):
        logits = self.token_embedding_table(idx)
        B, T, C = logits.shape
        logits = logits.view(B*T, C)
        targets = targets.view(B*T)
        loss = F.cross_entropy()

        return logits, loss

m = BigramLanguageModel(len(train_dataset.token_vocab))"""

import torch
import numpy as np
import random


def set_random_seed(seed):
    torch.manual_seed(seed)  # Set seed for CPU
    torch.cuda.manual_seed_all(seed)  # Set seed for all GPUs (if using CUDA)
    np.random.seed(seed)  # Set seed for NumPy
    random.seed(seed)  # Set seed for Python's random module
    torch.backends.cudnn.deterministic = True  # Ensure deterministic operations for reproducibility
    torch.backends.cudnn.benchmark = False  # Disable benchmark to avoid non-deterministic behavior


set_random_seed(6)

from datasets import load_dataset
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import re
import unicodedata

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load the PTB dataset
ptb = load_dataset('ptb-text-only/ptb_text_only', trust_remote_code=True)

train = pd.DataFrame(ptb['train'])
val = pd.DataFrame(ptb['validation'])
test = pd.DataFrame(ptb['test'])

train = train

print(train.shape)
val = val

print(val.shape)
test = test

print(test.shape)

import unicodedata
import string


def remove_accents(text):
    text = text.lower()

    # Remove accents
    text = ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    )

    # Remove punctuation using re.sub
    text = re.sub(r'[^\w\s]', '', text)  # Remove anything that is not a word character or whitespace

    # Add special tokens
    text = "<S> " + text + " <.>"

    return text


train["sentence"] = train["sentence"].apply(remove_accents)
val["sentence"] = val["sentence"].apply(remove_accents)
test["sentence"] = test["sentence"].apply(remove_accents)


# Define the Dataset class
class Language_Model(Dataset):
    def __init__(self, data, token_vocab=None, training=False):
        if training:
            self.token_vocab = {"<PAD>": 0, "<unk>": 1}
            for utterance in data["sentence"]:
                tokens = utterance.split(" ")
                for token in tokens:
                    if token not in self.token_vocab:
                        self.token_vocab[token] = len(self.token_vocab)
        else:
            assert token_vocab is not None
            self.token_vocab = token_vocab
        self.token_vocab_inv = {v: k for k, v in self.token_vocab.items()}

        self.corpus_x_ids = []
        self.corpus_y_ids = []
        self.corpus_pad_ids = []

        for utterance in data["sentence"]:
            token_list = utterance.split(" ")

            x_ids = [self.token_vocab.get(token, self.token_vocab['<unk>']) for token in token_list]
            self.corpus_x_ids.append(torch.tensor(x_ids))

            y_ids = [self.token_vocab.get(token, self.token_vocab['<unk>']) for token in token_list[1:]]
            y_ids.append(self.token_vocab['<PAD>'])  # Add <PAD> token to match length
            self.corpus_y_ids.append(torch.tensor(y_ids))

            pad_ids = [1] * (len(x_ids) - 1) + [0]
            self.corpus_pad_ids.append(torch.tensor(pad_ids))

    def __len__(self):
        return len(self.corpus_x_ids)

    def __getitem__(self, idx):
        return self.corpus_x_ids[idx], self.corpus_y_ids[idx], self.corpus_pad_ids[idx]


print(train.head())
train_dataset = Language_Model(train, training=True)
val_dataset = Language_Model(val, token_vocab=train_dataset.token_vocab, training=False)
test_dataset = Language_Model(test, token_vocab=train_dataset.token_vocab, training=False)
max_seq_length_t = max(len(x) for x in train_dataset.corpus_x_ids)
max_seq_length_v = max(len(x) for x in val_dataset.corpus_x_ids)
max_seq_length_test = max(len(x) for x in test_dataset.corpus_x_ids)

max_seq_length = max(max_seq_length_v, max_seq_length_test, max_seq_length_t)

print("max sequence length :", max_seq_length)
print(train_dataset.token_vocab)


# Collate function for batching
def collate_fn(batch):
    x_ids, y_ids, pad_masks = zip(*batch)
    max_len = max(len(x) for x in x_ids)

    x_batch = torch.stack(
        [torch.cat([x, torch.tensor([train_dataset.token_vocab["<PAD>"]] * (max_len - len(x)))]) for x in x_ids]).long()
    y_batch = torch.stack(
        [torch.cat([y, torch.tensor([train_dataset.token_vocab["<PAD>"]] * (max_len - len(y)))]) for y in y_ids]).long()
    pad_batch = torch.stack(
        [torch.cat([pad, torch.tensor([train_dataset.token_vocab["<PAD>"]] * (max_len - len(pad)))]) for pad in
         pad_masks]).long()

    return x_batch, y_batch, pad_batch


n_embd = 30
block_size = max_seq_length
head_size = 6



best_perplexity = float('inf')
class SeqTagger(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim,
                            num_layers=num_layers, batch_first=True, bidirectional=False)

        self.dropout = nn.Dropout(0.4)

        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)

        lstm_out = self.dropout(lstm_out)

        out = self.fc(lstm_out)
        return out




EMBEDDING_DIM = 300
HIDDEN_DIM = 1000
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 45
token_vocab_inv = {id_: tag for tag, id_ in train_dataset.token_vocab.items()}

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
model = SeqTagger(
    vocab_size=len(train_dataset.token_vocab),
    embedding_dim=EMBEDDING_DIM,
    hidden_dim=HIDDEN_DIM

)


train_loader = DataLoader(train_dataset, batch_size=32, collate_fn=collate_fn, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=32, collate_fn=collate_fn)


optimizer = torch.optim.AdamW(model.parameters(), lr=.001)  # new

import os

best_val_loss = float("inf")


loss_fn = nn.CrossEntropyLoss(ignore_index=train_dataset.token_vocab['<PAD>'])
def train(model, train_loader, optimizer, epoch):
    print("Starting train...")
    model.train()
    total_loss = 0.0
    for batch_idx, (x_batch, y_batch, pad_batch) in enumerate(train_loader):
        optimizer.zero_grad()

        # Forward pass through the model
        outputs = model(x_batch)

        # Flatten the predictions and labels for loss computation
        outputs = outputs.view(-1, outputs.shape[-1])  # [batch_size * seq_len, vocab_size]
        y_batch = y_batch.view(-1)  # Flatten targets [batch_size * seq_len]

        # Compute loss (cross-entropy loss)
        loss = loss_fn(outputs, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch}, Train Loss: {avg_loss}")


"""def evaluate(model, val_loader):
    model.eval()
    total_loss = 0.0  # Declare `total_loss` here
    with torch.no_grad():
        for x_batch, y_batch, pad_batch in val_loader:
            logits, loss = model(x_batch, y_batch)
            total_loss += loss.item()

    avg_loss = total_loss / len(val_loader)
    print(f"Validation Loss: {avg_loss}")
    return avg_loss  # Return the computed average loss"""




def evaluate(model, val_loader):
    best_perplexity = float('inf')
    model.eval()
    total_loss = 0.0
    total_perplexity = 0.0

    with torch.no_grad():
        for x_batch, y_batch, pad_batch in val_loader:
            print(y_batch)
            print(y_batch.size())
            batch_size, seq_len = y_batch.size()
            outputs = model(x_batch)

            outputs = outputs.view(-1, outputs.shape[-1])
            y_batch = y_batch.view(-1)

            loss = loss_fn(outputs, y_batch)
            total_loss += loss.item()


            log_probs = torch.nn.functional.log_softmax(outputs, dim=-1)



            selected_log_probs = log_probs.view(-1, log_probs.size(-1))[
                torch.arange(batch_size * seq_len), y_batch.view(-1)]


            cross_entropy = -(selected_log_probs * pad_batch.view(-1)).sum() / pad_batch.sum()


            perplexity = torch.exp(cross_entropy)

            total_perplexity += perplexity.item()

    avg_loss = total_loss / len(val_loader)
    avg_perplexity = total_perplexity / len(val_loader)
    if avg_perplexity < best_perplexity:
        best_perplexity = avg_perplexity
        print("🎾🎾New best model saved at epoch: ", epoch)
        torch.save(model.state_dict(), 'best_model_perplexed_lstm.pth')
        print("Saved best model")

    print(f"Validation Loss: {avg_loss}")
    print(f"Validation Perplexity: {avg_perplexity}")

    return avg_loss, avg_perplexity


def calculate_perplexity(model, x_ids, y_ids, pad_ids):
    model.eval()
    with torch.no_grad():
        # Forward pass
        outputs = model(x_ids.unsqueeze(0))  # Assuming x_ids is a single sequence
        logits = outputs.view(-1, outputs.shape[-1])  # Flatten outputs to 2D: [seq_len, vocab_size]

        # Flatten y_ids and pad_ids to match logits
        y_ids_flat = y_ids.view(-1)  # Flatten y_ids to 1D: [seq_len]
        pad_ids_flat = pad_ids.view(-1)  # Flatten pad_ids to 1D: [seq_len]

        # Compute log probabilities
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        # Select log probabilities corresponding to true targets
        selected_log_probs = log_probs[
            torch.arange(y_ids_flat.size(0)), y_ids_flat
        ]

        # Compute cross-entropy loss using padding mask
        cross_entropy = -(selected_log_probs * pad_ids_flat).sum() / pad_ids_flat.sum()

        # Calculate perplexity
        perplexity = torch.exp(cross_entropy)

    return perplexity.item()


for epoch in range(3):
    print("Starting epoch:", epoch)

    train(model, train_loader, optimizer, epoch)
    evaluate(model, val_loader)




model = SeqTagger(
    vocab_size=len(train_dataset.token_vocab),
    embedding_dim=EMBEDDING_DIM,
    hidden_dim=HIDDEN_DIM

)
model_path = "/Users/darianlee/PycharmProjects/243_hw3/best_model_perplexed_lstm.pth"

state_dict = torch.load(model_path)
model.load_state_dict(state_dict, strict=False)
model.eval()


import torch

perplexities = []
for idx in range(len(test_dataset)):
    x_ids, y_ids, pad_ids = test_dataset[idx]
    perplexity = calculate_perplexity(model, x_ids, y_ids, pad_ids)
    perplexities.append(perplexity)

average_perplexity = sum(perplexities) / len(perplexities)

print(f"Average Perplexity: {average_perplexity:.4f}")
print("max: ", max(perplexities))
submission_df = pd.DataFrame({
    "ID": range(len(perplexities)),
    "ppl": perplexities
})

submission_path = "/Users/darianlee/PycharmProjects/243_hw3/submission_lstm.csv"
submission_df.to_csv(submission_path, index=False)

print(f"Submission saved to {submission_path}")

