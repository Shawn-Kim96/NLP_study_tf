import torch
import torch.nn as nn


class SGNS(nn.Module):

    def __init__(self, vocab_size, embedding_dim):
        super(SGNS, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=embedding_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, word_input, context_input):
        word_embedding = self.embedding(word_input)
        context_embedding = self.embedding(context_input)
        output = torch.sum(word_embedding * context_embedding, dim=-1)
        output = torch.squeeze(output)
        #         output = self.sigmoid(output)
        return output

