from word2vec_trainer.dataset_loader import DatasetLoader

def test_dataset_loader():
    dataset = DatasetLoader('tests/test_inputs/train_w2v.csv', 10, 'tests/test_inputs/neg_distribution_test.npy')

    assert dataset.file_path == 'tests/test_inputs/train_w2v.csv'
    with open('tests/test_inputs/train_w2v.csv', 'r') as f:
        train_size = len(f.readlines()) - 1
    assert dataset.__len__() == int(train_size * (1+dataset.neg_num))
    
