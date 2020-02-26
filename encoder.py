import torch.nn as nn
from namedtensor import ntorch, NamedTensor

import string

num_char_types = len(string.printable[:-4]) + 1
char_embed_dim = 20
column_encoding_dim = 32
kernel_size = 5
num_dense_layers = 10
button_embed_dim = 32

class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()

        self.char_embedding = ntorch.nn.Embedding(
            num_char_types, char_embed_dim
        ).spec("stateLoc", "charEmb")

        # input string
        # output string
        # commited string: the output of the expressions synthesized so far
        # scratch string: the previous partial results
        # masks: for each chracter position which transformations may occur (length is fixed to 7)
        self.column_encoding = ntorch.nn.Conv1d(
            in_channels = 4 * char_embed_dim + 7,
            out_channels = column_encoding_dim,
            kernel_size = kernel_size,
            padding = int((kernel_size - 1) / 2)
        ).spec("inFeatures", "strLen", "E")
        ).spec("inFeatures", "E")
