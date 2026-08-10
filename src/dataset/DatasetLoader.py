import torch
from torch.utils.data import IterableDataset
from torch.utils.data import DataLoader
import numpy as np
import math
from tqdm.auto import tqdm

class DatasetLoader(IterableDataset):
    self.total_batches = 0
    def __init__(self, file_path, batch_size, path_to_neg_dist="neg_distribution.npy", neg_num=5):
        self.file_path = file_path
        self.batch_size = batch_size
        self.neg_num = neg_num
        with open(file_path) as f:
            self.length = sum(1 for _ in f) - 1
        self.neg_dist = np.load(path_to_neg_dist)
        self.total_batches=math.ceil(self.length * (1 + self.neg_num))/self.batch_size)

    def __len__(self):
        return self.length * (1+self.neg_num)

    def gen_negatives(self, contexts, targets, batch):
      negatives = np.random.choice(
        len(self.neg_dist),
        size=(len(contexts), self.neg_num),
        p=self.neg_dist
      )
      for n, t, c in zip(negatives, targets, contexts):
        for i in range(len(n)):
          while n[i] == t:
            n[i] = np.random.choice(
              len(self.neg_dist),
              p=self.neg_dist
            )
        batch.extend(
            (c, neg, 0) for neg in n
        )
      return batch

    def __iter__(self):
        with open(self.file_path) as f:
            next(f)  # skip header
            np.random.seed(42)
            batch = []
            targets = []
            contexts = []

            pbar = tqdm(
              total=math.ceil(
                  (self.length * (1 + self.neg_num))/self.batch_size
              ),
              desc="Loading batches",
              unit="batches"
            )

            for line in f:
              context, target, label = line.strip().split(',')
              context = int(context)
              target = int(target)
              label = int(label)
              contexts.append(context)
              targets.append(target)
              batch.append((context, target, label))
              if len(contexts)==10000:
                batch = self.gen_negatives(contexts, targets, batch)
                contexts = []
                targets = []
                while len(batch) >= self.batch_size:
                  pbar.update(1)
                  yield torch.tensor(
                      batch[:self.batch_size]
                  )

                  batch = batch[self.batch_size:]
            if contexts:
              batch = self.gen_negatives(contexts, targets, batch)

            if batch:
              pbar.update(1)
              yield torch.tensor(batch)
            pbar.close()
