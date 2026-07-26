import torch

class Word2Vec(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.input_embedding_layer = torch.nn.Embedding(vocab_size, embedding_dim)
        self.output_embedding_layer = torch.nn.Embedding(vocab_size, embedding_dim)
        
    def forward(self, x, y):
        e1 = self.input_embedding_layer(x)
        e2 = self.output_embedding_layer(y)
        
        return (e1 * e2).sum(dim=1)