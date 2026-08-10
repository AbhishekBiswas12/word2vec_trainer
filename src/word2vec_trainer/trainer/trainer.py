import torch
import torch.nn.functional as F
import torch.optim as op
from torch.utils.data import IterableDataset
from torch.utils.data import DataLoader
import numpy as np
import math
from tqdm.auto import tqdm
from src.dataset.DatasetLoader import DatasetLoader
from src.model.word2vec_model import Word2Vec

class Trainer:
    def __init__(
        self,
        vocab_size = 508205,
        embedding_dim = 300,
        batch_size_train = 1500000,
        batch_size_val = 50000,
        new_model = True,
        path_to_train = "train.csv",
        path_to_val = "val_csv",
        path_to_saved_model = "",
        path_to_neg_dist = "neg_distribution.npy"
    ):
        # assigning variable values
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.batch_size_train = batch_size_train
        self.batch_size_val = batch_size_val
        self.new_model = new_model
        self.path_to_train = path_to_train
        self.path_to_val = path_to_val
        self.path_to_saved_model = path_to_saved_model
        self.path_to_neg_dist = path_to_neg_dist

        # creating dataset and model objects
        self.train = DatasetLoader(
                    path_to_train,
                    batch_size_train,
                    path_to_neg_dist=self.path_to_neg_dist,
                    neg_num=2)
        self.val = DatasetLoader(
                    path_to_val,
                    batch_size_val,
                    path_to_neg_dist=self.path_to_neg_dist,
                    neg_num=2)
        self.model = self.model_creation()
        self.model.requires_grad_(True)

        # Learning rate, optimizer, loss selection -> TODO: assign from user choice
        self.lr = 0.01
        self.optimizer = op.Adagrad(self.model.parameters(), lr=self.lr)
        self.loss = torch.nn.BCEWithLogitsLoss()

    def run(self):
        self.train_model()
        
        
    def model_creation(self):
        if self.new_model:
            print("Creating train_losses.txt")
            open('train_losses.txt', 'w').close()
            print("Creating val_losses.txt")
            open('val_losses.txt', 'w').close()
            return Word2Vec(self.vocab_size, self.embedding_dim)            
        else:
            return torch.load(self.path_to_saved_model, weights_only=False) 

    def validate_model(self, device='cpu'):
        val_loss = 0.0
        val_size = 0
        
        train_pos_probs = 0.0
        train_neg_probs = 0.0
        train_pos_size = 0
        train_neg_size = 0
        val_pos_probs = 0.0
        val_neg_probs = 0.0
        val_pos_size = 0
        val_neg_size = 0
        
        train_loss = 0.0
        train_size = 0
        self.model.eval()
        
        with torch.no_grad():
            for x in tqdm(self.train, desc=f"Computing Train Loss", total=self.train.total_batches):
                context = x[:, 0].to(device)
                target = x[:, 1].to(device)
                labels = x[:, 2].to(device)
                scores = self.model(context, target)
                probs = torch.sigmoid(scores)
                
                train_pos_probs += probs[labels == 1].sum().item()
                train_neg_probs += probs[labels == 0].sum().item()
                
                train_pos_size += probs[labels==1].size()[0]
                train_neg_size += probs[labels==0].size()[0]
                
                l = self.loss(scores, labels.float())
                train_loss += l.item() * context.shape[0]
                train_size += context.shape[0]
            
            for y in tqdm(self.val, desc=f"Computing Validation Loss", total=self.val.total_batches):
                context = y[:, 0].to(device)
                target = y[:, 1].to(device)
                labels = y[:, 2].to(device)
                
                scores = self.model(context, target)
                probs = torch.sigmoid(scores)
                
                val_pos_probs += probs[labels == 1].sum().item()
                val_neg_probs += probs[labels == 0].sum().item()
                
                val_pos_size += probs[labels==1].size()[0]
                val_neg_size += probs[labels==0].size()[0]
                
                l = self.loss(scores, labels.float())
                val_loss += l.item() * context.shape[0]
                val_size += context.shape[0]
            
        train_loss /= train_size
        val_loss /= val_size
        
        train_pos_probs /= train_pos_size
        train_neg_probs /= train_neg_size
        
        val_pos_probs /= val_pos_size
        val_neg_probs /= val_neg_size
        
        self.model.train()
        return train_loss, train_pos_probs, train_neg_probs, val_loss, val_pos_probs, val_neg_probs

    def train_model(self):
        epochs = 50
        epoch = len(open('train_losses.txt', 'r').readlines())
        print("starting training...")
        for epoch in range(epoch, epochs):
            for x in tqdm(self.train, desc=f"Epoch {epoch+1}", total=self.train.total_batches):
                context = x[:, 0]
                target = x[:, 1]
                labels = x[:, 2]
                
                batch_size = target.shape[0]
                
                # Forward
                scores = self.model(context, target)
                
                l = self.loss(scores, labels.float())
                
                # Backprop
                l.backward()
            self.optimizer.zero_grad()
            self.optimizer.step()
            train_loss, train_pos_probs, train_neg_probs, val_loss, val_pos_probs, val_neg_probs = self.validate_model()
            with open('train_losses.txt', 'a') as losses, open('val_losses.txt', 'a') as val_losses:
                losses.write(f"{train_loss}\t{train_pos_probs}\t{train_neg_probs}\n")
                val_losses.write(f"{val_loss}\t{val_pos_probs}\t{val_neg_probs}\n")
                print(f"Epoch {epoch+1}, Loss: {train_loss}, Positive probabs: {train_pos_probs}, Negative probabs: {train_neg_probs}\n Val_loss: {val_loss}, Positive probabs {val_pos_probs}, Negative probabs: {val_neg_probs}")  
                if self.path_to_saved_model != '':
                    torch.save(self.model, )
                    print(f"Model saved after {epoch+1} epochs")
