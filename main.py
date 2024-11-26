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


class Layer_norm:  # this is a custom version of layer normalization from the attention paper
    # normalization is super important because it helps stabilize and speed up training.
    # without it, the model might struggle to learn effectively due to exploding or vanishing gradients.
    # it also makes the training process less sensitive to weight initialization and improves generalization.
    def __init__(self, feature_dim, epsilon=1e-5, momentum=0.1):
        # sets up the layer norm with the given params

        # Args:
        # - feature_dim (int): how many features are in the input
        # - epsilon (float, optional): small number added to variance to avoid division by 0. default is 1e-5.
        # - momentum (float, optional): used for batch norm stuff, not really needed here. default is 0.1.
        self.epsilon = epsilon  # this keeps us from dividing by 0 when calculating variance
        self.gamma = torch.ones(feature_dim)  # this is the "scale" parameter, starts at 1
        self.beta = torch.zeros(feature_dim)  # this is the "shift" parameter, starts at 0

    def __call__(self, x):
        # call makes it part of the commputational graph, so gamma and beta will be optimized
        # forward pass.
        # takes in x (torch.Tensor): input tensor (batch_size, feature_dim)
        # outputs: torch.Tensor: output tensor after normalization

        # basically, we calculate the mean for each feature across all batches (dim=1 is the feature axis)
        feature_mean = x.mean(1, keepdim=True)  # shape: (batch_size, 1)

        # now calculate the variance for each feature across all batches
        feature_var = x.var(1, keepdim=True)  # shape: (batch_size, 1)

        # normalize the input by subtracting the mean and dividing by the standard deviation
        # this makes each feature have a mean of 0 and a standard deviation of 1
        normalized_x = (x - feature_mean) / torch.sqrt(feature_var + self.epsilon)  # shape: (batch_size, feature_dim)

        # now we scale and shift the normalized data using gamma and beta (learnable params)
        self.output = self.gamma * normalized_x + self.beta  # shape: (batch_size, feature_dim)

        # just return the final output
        return self.output

    def parameters(self):
        # this returns the learnable params (gamma and beta) that will get updated during training

        return [self.gamma, self.beta]


class Attention_head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.dropout = nn.Dropout(.3)
        self.key = nn.Linear(n_embd, head_size, bias=False)  # just a linear layer representing the key of dim
        # will be (batch_size, sequence_length, head_size)

        # for each batch, there will be a matrix of shape (sequence_length, head_size) for each sequence.
        # where each row is a token and each column is the key vector
        # so if
        #   batch_size = 2
        #   sequence_length = 3
        #   head_size = 4
        # the output will be:
        # Output shape: (2, 3, 4)
        # Batch 1:
        #   [[k1_1, k1_2, k1_3, k1_4],  # Token 1 in Sequence 1 (Key vector of size 4)
        #   [k2_1, k2_2, k2_3, k2_4],  # Token 2 in Sequence 1 (Key vector of size 4)
        #   [k3_1, k3_2, k3_3, k3_4]]   # Token 3 in Sequence 1 (Key vector of size 4)
        #
        # Batch 2:
        #   [[k1_1, k1_2, k1_3, k1_4],  # Token 1 in Sequence 2 (Key vector of size 4)
        #   [k2_1, k2_2, k2_3, k2_4],  # Token 2 in Sequence 2 (Key vector of size 4)
        #   [k3_1, k3_2, k3_3, k3_4]]   # Token 3 in Sequence 2 (Key vector of size 4)
        #
        self.query = nn.Linear(n_embd, head_size, bias=False)  # just a linear layer representing the query of dim
        # will be (batch_size, sequence_length, head_size)
        self.value = nn.Linear(n_embd, head_size, bias=False)  # just a linear layer representing the value of dim
        # will be (batch_size, sequence_length, head_size)
        self.register_buffer('triangle_mask', torch.tril(torch.ones(block_size, block_size)))
        # torch.ones(block_size, block_size) creates a square matrix filled with ones of shape (block_size, block_size).
        # torch.tril() then converts this matrix to a lower triangular matrix, where all elements above the main diagonal are set to 0.
        # For example, if block_size = 4, the output would look like this:
        #
        # [[1, 0, 0, 0],  # First row: 1 at the diagonal and below
        # [1, 1, 0, 0],  # Second row: 1's at the diagonal and below
        # [1, 1, 1, 0],  # Third row: 1's at the diagonal and below
        # [1, 1, 1, 1]]   # Fourth row: all 1's (triangle including diagonal)
        # This matrix is useful for causal attention, where each token can only attend to itself and previous tokens, not future tokens.
        # By using a lower triangular matrix, we mask out the future tokens (set to 0) during the attention computation.

    def forward(self, x):
        B, T, C = x.shape

        k = self.key(x)  # b t c

        q = self.query(x)  # btc

        # to compute attention
        attention_weights = q @ k.transpose(-2, -1) * C ** -0.5  # will be (B, T, C) @ (B, C, T) -> B, T, T
        # Explanation of the components:
        # 1. q: The query matrix. Shape: (B, T, C)
        #    - B: Batch size
        #    - T: Sequence length (number of tokens)
        #    - C: Head size (dimension of each query vector)
        #    - Represents the query vectors for each token in the sequence for all batches.
        #
        # 2. k: The key matrix. Shape: (B, T, C)
        #    - Has the same dimensions as the query matrix.
        #    - Represents the key vectors for each token in the sequence for all batches.
        #
        # 3. k.transpose(-2, -1): Transposes the last two dimensions of k, making it Shape: (B, C, T)
        #    - The sequence length dimension (T) and the head size dimension (C) are swapped.
        #    - This is necessary for the matrix multiplication with q to compute dot products.
        #
        # 4. q @ k.transpose(-2, -1):
        #    - Matrix multiplication between q (Shape: B, T, C) and k.transpose(-2, -1) (Shape: B, C, T).
        #    - The result is of Shape: (B, T, T).
        #      - B: Batch size remains unchanged.
        #      - T: Sequence length in the first and second dimensions indicates attention scores between all pairs of tokens in the sequence.
        #      - Each element in the output matrix represents the unnormalized attention score between a query (row) and a key (column).
        #
        # 5. C ** -0.5:
        #    - This scales the dot product by the square root of the head size (C) to prevent large values.
        #    - Scaling helps stabilize gradients during training and ensures numerical stability, especially for large values of C.
        #    - The formula for scaled dot-product attention is:
        #         Attention_Score[i, j] = (q[i] ⋅ k[j]) / sqrt(C)
        #      - This scaling is inspired by the transformer paper ("Attention Is All You Need").
        #
        # The output:
        # weights: Shape: (B, T, T)
        #    - B: Batch size.
        #    - T: Sequence length.
        #    - Represents the scaled dot-product attention scores for all token pairs in the sequence.
        #    - Each row corresponds to a token in the sequence, and each column corresponds to another token in the sequence.
        #    - For example, weights[b, t1, t2] represents the attention score for token t1 attending to token t2 in the b-th batch.

        masked_attention_weights = attention_weights.masked_fill(self.triangle_mask[:T, :T] == 0,
                                                                 float('-inf'))  # B T T
        # Explanation of the components:
        #
        # 1. `attention_weights`: Shape (B, T, T)
        #    - This is the attention weights matrix computed from scaled dot-product attention.
        #    - Each row corresponds to a query token, and each column corresponds to a key token with the values filled out as attention scores

        # 2. `self.tril[:T, :T]`:
        # just a lower trianglur mask matrix
        # if you remember, b is the number of sequences in the batch. each b contains tokens at that index in the sequence
        # this will be applied to each sequence seprately
        # this comes from this line self.register_buffer('triangle_mask', torch.tril(torch.ones(block_size, block_size)))

        # 4. `masked_fill(mask, float('-inf'))`:
        #    - `masked_fill` replaces values in `attention_weights` at positions where the mask is `True` with `float('-inf')`.
        #    - This ensures that tokens cannot attend to future positions (enforcing causality in the attention mechanism).
        #      - For example, in sequence generation, token 3 should not "see" token 4.
        #    - `float('-inf')` is used because, when passed through the softmax function in the next step, it results in a value of 0 probability for those positions.
        #
        # Output:
        # `masked_attention_weights`: Shape (B, T, T)
        #    - Same shape as `attention_weights`.
        #    - For each sentence in batch, rows represent query tokens, and columns represent key tokens.
        #    - Upper triangular positions (future tokens) are set to `-inf`, preventing attention to these positions.
        #
        # Example:
        # For T = 4, a single batch of unmasked attention weights might look like:
        #    attention_weights =
        #    [[0.2, 0.5, 0.3, 0.7],
        #     [0.4, 0.6, 0.8, 0.9],
        #     [0.1, 0.3, 0.7, 0.2],
        #     [0.5, 0.4, 0.6, 0.8]]
        #
        # After applying the mask:
        #    masked_attention_weights =
        #    [[0.2,  -inf,  -inf,  -inf],
        #     [0.4,  0.6,   -inf,  -inf],
        #     [0.1,  0.3,   0.7,   -inf],
        #     [0.5,  0.4,   0.6,    0.8]
        softmaxed_attention_weights = F.softmax(masked_attention_weights, dim=1)
        softmaxed_attention_weights = self.dropout(softmaxed_attention_weights)
        # Applies the softmax function along dimension 1 (the sequence length dimension).
        # This converts the masked attention weights into probabilities, ensuring that
        # the attention scores for each query token sum to 1 across all key tokens.
        # The shape of the result remains (B, T, T), where:
        # - Each row (corresponding to a query token) contains normalized probabilities
        #   for attending to all key tokens.
        # - The mask ensures positions with '-inf' become zero after softmax.
        v = self.value(x)
        out = softmaxed_attention_weights @ v  # b t t @ b t c = b t c. ( b t head_size)

        # basically this multiplication is happening sepreately for each sentence in the batch
        return out


# Define the Bigram Language Model
class MultiHead_Attention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Attention_head(head_size) for _ in range(num_heads)])

        # nn.ModuleList is a container that registers submodules (e.g., layers) as part of the model,
        # allowing PyTorch to track their parameters, handle device placement, and include them in backpropagation.
        # Basically, this creates a list of `num_heads` Attention_head modules, each initialized with `head_size`,
        # and registers them as part of the model so that PyTorch can track their parameters for training and device management.
        self.projection = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(.3)

        # just a linear layer applied to the output

    def forward(self, x):
        # Concatenates the outputs of each attention head along the channel dimension (the dimension that has the representations)
        # Each head processes the input `x` independently, and their results are concatenated to form a larger tensor.
        # # If each head outputs a tensor of shape (B, T, head_size), where:
        #     #   B = batch size
        #     #   T = sequence length
        #     #   head_size = the size of each attention head's output
        #     # Then after concatenation along dim=-1, the output will have the shape (B, T, num_heads * head_size),
        #     # where `num_heads` is the number of attention heads.
        # for example
        # assume this is head 1 output shape (1,2,3)
        # [[[0.1, 0.2, 0.3],  # Token 1, Batch 1
        #   [0.4, 0.5, 0.6]]]  # Token 2, Batch 1
        # and this is head 2 output shape (1,2,3)
        # [[[0.7, 0.8, 0.9],  # Token 1, Batch 1
        # [1.0, 1.1, 1.2]]]  # Token 2, Batch 1
        # the concat output would be 1 2 6
        # [[[0.1, 0.2, 0.3, 0.7, 0.8, 0.9],  # Token 1, Batch 1
        # [0.4, 0.5, 0.6, 1.0, 1.1, 1.2]]]  # Token 2, Batch 1
        out = torch.cat([h(x) for h in self.heads], dim=-1)

        out = self.dropout(self.projection(out))

        return (out)


class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.linear = nn.Linear(n_embd, 4 * n_embd)  # they multipled by 4 in attention is all you need paper
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(4 * n_embd, n_embd)
        self.dropout = nn.Dropout(.3)  # new code

    def forward(self, x):
        x = self.dropout(x)
        x = self.linear(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


class Attention_then_feedForward_block(nn.Module):
    def __init__(self, n_embd, num_heads):
        super().__init__()
        self.attention = MultiHead_Attention(num_heads, n_embd // num_heads)
        self.feed_forward = FeedForward(n_embd)
        self.layer_normalization1 = Layer_norm(n_embd)
        self.layer_normalization2 = Layer_norm(n_embd)

    def forward(self, x):
        x = self.layer_normalization1(x)

        x = x + self.attention(x)  # x has dimensaions b t c
        x = self.layer_normalization2(x)
        x = x + self.feed_forward(x)  # these x + is done to make optimization easier
        # Residual connections help gradients flow more easily through deep networks, preventing vanishing/exploding gradients.
        # They allow the model to learn both transformations and retain original information, improving optimization.
        return x


class LanguageModel(nn.Module):
    def __init__(self, vocab_size, num_heads, num_blocks):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)  # will give us token embeddings
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)  # go from embeddings to outputs. Made sure that this is the vocab size in order to not have errors where the perplexity is artifically low
        self.attention_head = MultiHead_Attention(num_heads,
                                                  n_embd // num_heads)  # these will concat to give us n embed because the 4 heads will be concatinated by 4 in the end to give n_embd
        self.feedForward = FeedForward(n_embd)
        self.blocks = nn.Sequential(
            *[Attention_then_feedForward_block(n_embd, num_heads) for _ in range(num_blocks)]
        )

    def forward(self, idx, targets):
        B, T = idx.shape  # the batch and seq lebn

        token_embeddings = self.token_embedding_table(idx)  # batch, seq len, embedding size
        possitional_embeddings = self.position_embedding_table(torch.arange(T, device=device))  # t by c
        x = token_embeddings + possitional_embeddings  # b t c
        # so if the token embedding for a sentence is [0, 1] and the positional embedding is [.01, .02]
        # we would get [0+.01, 1+.02] = [.01, 1.02]
        x = self.blocks(x)
        logits = self.lm_head(
            x)  # B T C where b is batch t is time (seq len) and c is channels in this case the vocab size
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

            return logits, loss
        else:
            return logits, None  # for generating

    def generate(self, start, max_len=5):
        # Assuming `start` is a list of token IDs or a tensor
        # Convert start to tensor if it is not already
        start = torch.tensor(start, device=device).unsqueeze(0)  # Add batch dimension if needed (B=1)

        generated_tokens = start  # To store the tokens generated so far

        stop_token_id = train_dataset.token_vocab.get('<.>',
                                                      None)  # Get the stop token ID (assuming '<.>' is the stop token)

        if stop_token_id is None:
            raise ValueError("Stop token '<.>' not found in token_vocab")

        # Generate max_len-1 tokens (one less, since we want the last one to be the stop token)
        for _ in range(max_len - 1):
            # Forward pass through the model (only feed the current generated sequence)
            logits, _ = self.forward(generated_tokens, targets=None)  # We don't need targets for generation

            # Apply softmax to get the probabilities
            probs = F.softmax(logits[:, -1, :], dim=-1)  # Get the distribution for the last token

            # Exclude <PAD> token by setting its probability to 0 (assuming <PAD> is 0)
            pad_token_id = train_dataset.token_vocab.get('<PAD>', 0)  # Get the <PAD> token index, assuming it's 0
            probs[:, pad_token_id] = 0  # Set the probability of <PAD> to 0 to exclude it

            # Normalize the probabilities to ensure they sum to 1
            probs = probs / probs.sum(dim=-1, keepdim=True)

            # Sample the next token based on the distribution
            next_token = torch.multinomial(probs, num_samples=1)  # Sample a token from the distribution

            # Append the predicted token to the sequence
            generated_tokens = torch.cat([generated_tokens, next_token], dim=1)  # Add to sequence

        # Add the stop token as the last token in the sequence
        generated_tokens = torch.cat([generated_tokens, torch.tensor([[stop_token_id]], device=device)], dim=1)

        # Convert the generated tokens to words using the token_vocab
        generated_ids = generated_tokens.squeeze(0).cpu().numpy()  # Remove batch dimension and convert to numpy array
        generated_words = [train_dataset.token_vocab_inv[token_id] for token_id in
                           generated_ids]  # Decode tokens to words

        return generated_words


# Initialize the model
model = LanguageModel(len(train_dataset.token_vocab), 6, 5)

# Set up the DataLoader
train_loader = DataLoader(train_dataset, batch_size=3, collate_fn=collate_fn, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=3, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=3, collate_fn=collate_fn)

# Define optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=.001)  # new

import os

best_val_loss = float("inf")



def train(model, train_loader, optimizer, epoch):
    print("starting train")
    model.train()
    total_loss = 0.0
    for batch_idx, (x_batch, y_batch, pad_batch) in enumerate(train_loader):
        optimizer.zero_grad()
        logits, loss = model(x_batch, y_batch)
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

best_perplexity = float('inf')


def evaluate(model, val_loader):
    model.eval()
    total_loss = 0.0
    total_perplexity = 0.0

    with torch.no_grad():
        for x_batch, y_batch, pad_batch in val_loader:
            logits, loss = model(x_batch, y_batch)
            total_loss += loss.item()

            # Calculate log probabilities
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

            # Compute selected log probabilities for the true targets
            batch_size, seq_len = y_batch.shape
            selected_log_probs = log_probs.view(-1, log_probs.size(-1))[
                torch.arange(batch_size * seq_len), y_batch.view(-1)]

            # Compute cross-entropy
            cross_entropy = -(selected_log_probs * pad_batch.view(-1)).sum() / pad_batch.sum()

            # Calculate perplexity
            perplexity = torch.exp(cross_entropy)

            total_perplexity += perplexity.item()

    avg_loss = total_loss / len(val_loader)
    avg_perplexity = total_perplexity / len(val_loader)

    print(f"Validation Loss: {avg_loss}")
    print(f"Validation Perplexity: {avg_perplexity}")

    return avg_loss, avg_perplexity


def calculate_perplexity(model, x_ids, y_ids, pad_ids):
    model.eval()
    with torch.no_grad():
        outputs = model(x_ids.unsqueeze(0), targets=None)
        if isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        # Select log probabilities for the true targets
        selected_log_probs = log_probs[0, torch.arange(len(y_ids)), y_ids]

        # Compute cross-entropy
        cross_entropy = -(selected_log_probs * pad_ids).sum() / pad_ids.sum()

        # Calculate perplexity
        perplexity = torch.exp(cross_entropy)
    return perplexity.item()


for epoch in range(8):
    print("Starting epoch:", epoch)

    model.train()  # Set model to training mode
    total_train_loss = 0.0

    for x_batch, y_batch, pad_batch in train_loader:
        logits, loss = model(x_batch, y_batch)
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    print(f"Epoch {epoch} - Training Loss: {avg_train_loss}")



    val_perplexities = []
    for idx in range(len(val_dataset)):
        x_ids, y_ids, pad_ids = val_dataset[idx]
        val_perplexity = calculate_perplexity(model, x_ids, y_ids, pad_ids)
        val_perplexities.append(val_perplexity)

    avg_val_perplexity = sum(val_perplexities) / len(val_perplexities)

    print(f"Average Perplexity for val per sample 🎾: {avg_val_perplexity:.4f}")

    if avg_val_perplexity < best_perplexity:
        best_perplexity = avg_val_perplexity
        print("🎾🎾New best model saved at epoch: ", epoch)
        torch.save(model.state_dict(), 'best_model_perplexed_small.pth')
        print("Saved best model")

model_path = "/Users/darianlee/PycharmProjects/243_hw3/best_model_perplexed_small.pth"
model = LanguageModel(len(train_dataset.token_vocab), 6, 5)
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

submission_path = "/Users/darianlee/PycharmProjects/243_hw3/submission.csv"
submission_df.to_csv(submission_path, index=False)

print(f"Submission saved to {submission_path}")

