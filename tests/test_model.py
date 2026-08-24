import torch
from word2vec_trainer.word2vec_model import Word2Vec

def test_shape():
    model = Word2Vec(10, 10)
    input_embeddings = model.state_dict()['input_embedding_layer.weight']
    output_embeddings = model.state_dict()['output_embedding_layer.weight']

    assert input_embeddings.shape == (10, 10)
    assert output_embeddings.shape == (10, 10)    

def test_forward():
    model = Word2Vec(10, 10)
    x = torch.tensor([[1], [0], [4]])
    y = torch.tensor([[2], [4], [8]])
    z = model(x, y)

    assert z.shape == (3, 10)
