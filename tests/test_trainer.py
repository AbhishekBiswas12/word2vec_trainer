import os
from word2vec_trainer.trainer import Trainer

def test_trainer():
    trainer = Trainer(
        vocab_size = 10,
        embedding_dim=2,
        path_to_train='tests/test_inputs/train_w2v.csv',
        path_to_val='tests/test_inputs/valid_w2v.csv',
        path_to_neg_dist = 'tests/test_inputs/neg_distribution_test.npy',
        epochs=10,
        lr=0.01,
        path_to_saved_model='tests/test_inputs/model.bin'
    )

    assert trainer.model != None
