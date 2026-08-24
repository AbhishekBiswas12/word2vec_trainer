import os
from word2vec_trainer.preprocessing import Preprocessing

def test_preprocessing():
    preprocessor = Preprocessing('tests/test_inputs/example_data.txt',
                                subsampled_file_path = 'tests/test_inputs/subsampled_data.txt',
                                vocabulary_file = 'tests/test_inputs/vocabulary.csv',
                                pos_example = 'tests/test_inputs/pos_examples.csv',
                                train_path = 'tests/test_inputs/train_w2v.csv',
                                valid_path = 'tests/test_inputs/valid_w2v.csv',
                                test_path = 'tests/test_inputs/test_w2v.csv')
    preprocessor.run()

    assert os.path.exists(preprocessor.train_path) == True
    assert os.path.exists(preprocessor.valid_path) == True
    assert os.path.exists(preprocessor.test_path) == True
    assert os.path.exists(preprocessor.subsampled_file_path) == True
    assert os.path.exists(preprocessor.vocabulary_file) == True
    assert os.path.exists(preprocessor.pos_examples) == True
    
