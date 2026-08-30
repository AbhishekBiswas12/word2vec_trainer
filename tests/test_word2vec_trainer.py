import os
from word2vec_trainer.word2vec_trainer import Word2Vec_Trainer

def test_word2vec_trainer():
    word2vec_trainer = Word2Vec_Trainer(
      cleaned_data_file_path='tests/test_inputs/example_data.txt',
      embedding_dim=20,
      batch_size_train=10,
      batch_size_val=10,
      train_path='train_test.csv',
      valid_path='valid_test.csv',
      test_path='test_test.csv',
      neg_distribution_path = 'neg_distribution_test.npy',
      path_to_saved_model='tests/model_w2v.bin'
    )
    word2vec_trainer.preprocessing()
    assert os.path.exists('train_test.csv') == True
    assert os.path.exists('valid_test.csv') == True
    assert os.path.exists('test_test.csv') == True

    word2vec_trainer.train()
    assert os.path.exists('tests/model_w2v.bin') == True
