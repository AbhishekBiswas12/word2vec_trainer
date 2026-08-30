from word2vec_trainer.preprocessing import Preprocessor
from word2vec_trainer.trainer import Trainer

class Word2Vec_Trainer:
    def __init__(
        self,
        # Preprocessor params
        cleaned_data_file_path = 'cleaned_data.txt',
        subsampled_file_path = 'subsampled_data.txt',
        vocabulary_file = 'vocabulary.csv',
        pos_example = 'pos_examples.csv',
        train_path = 'train_w2v.csv',
        train_ratio = 0.95,
        valid_path = 'valid_w2v.csv',
        valid_ratio = 0.025,
        test_path = 'test_w2v.csv',
        test_ratio = 0.025,
        window = 5,
        # Trainer params
        embedding_dim = 300,
        batch_size_train = 1500000,
        batch_size_val = 50000,
        new_model = True,
        path_to_saved_model = "",
        epochs = 50,
        lr = 0.01
    ):
        # Preprocessor params
        self.cleaned_data_file_path = cleaned_data_file_path
        self.subsampled_file_path = subsampled_file_path
        self.vocabulary_file = vocabulary_file
        self.pos_example = pos_example
        self.train_path = train_path
        self.train_ratio = train_ratio
        self.valid_path = valid_path
        self.valid_ratio = valid_ratio
        self.test_path = test_path
        self.test_ratio = test_ratio
        self.window = window
        # Trainer params
        self.embedding_dim = embedding_dim
        self.batch_size_train = batch_size_train
        self.batch_size_val = batch_size_val
        self.new_model = new_model
        self.path_to_saved_model = path_to_saved_model!="" if path_to_saved_model!="" else "word2vec_model.bin"
        self.epochs = epochs
        self.lr = lr
        preprocessor = None
        trainer = None

    def preprocessing(self):
        self.preprocessor = Preprocessor(
            self.cleaned_data_file_path,
            self.subsampled_file_path,
            self.vocabulary_file,
            self.pos_example,
            self.train_path,
            self.train_ratio,
            self.valid_path,
            self.valid_ratio,
            self.test_path,
            self.test_ratio,
            self.window
        )

        self.preprocessor.run()

    def train(self):
        self.trainer(
            vocab_size = self.preprocessor.vocabulary_size,
            embedding_dim = self.embedding_dim,
            batch_size_train = self.batch_size_train,
            batch_size_val = self.batch_size_val,
            new_model = self.new_model,
            path_to_train = self.path_to_train,
            path_to_val = self.path_to_val,
            path_to_saved_model = self.path_to_saved_model,
            path_to_neg_dist = self.path_to_neg_dist,
            epochs = self.epochs,
            lr = self.lr
        )

        self.trainer.run()



        




        
