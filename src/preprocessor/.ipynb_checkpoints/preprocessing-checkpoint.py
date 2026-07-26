# preprocessing.py
import regex as re
import pandas as pd
from tqdm.auto import tqdm
import math
import random
import numpy as np
import csv

class Preprocessing:
  def __init__(self, 
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
               window = 10):
    self.cleaned_data_file_path = cleaned_data_file_path
    self.subsampled_file_path = subsampled_file_path
    self.vocabulary_file = vocabulary_file
    self.pos_examples = pos_example
    self.window = window
    self.train_path = train_path
    self.train_ratio = train_ratio
    self.valid_path = valid_path
    self.valid_ratio = valid_ratio
    self.test_path = test_path
    self.test_ratio = test_ratio
    with open(self.pos_examples, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(('int', 'target', 'pred'))

  def run(self):
    count_word = self.vocabulary_subsampling()
    self.negative_distribution_calc()
    self.positive_example_gen(count_word)
    print('Generated Positive examples.')
    print('Splitting data into train, test and valid sets.')
    self.train_test_val_split()

  def vocabulary_subsampling(self):
    cleaned_text = open(self.cleaned_data_file_path, 'r').read()
    print('Creating Vocabulary...')
    vocabulary = pd.DataFrame(pd.DataFrame(cleaned_text.split(" ")).value_counts())[pd.DataFrame(pd.DataFrame(cleaned_text.split(" ")).value_counts())['count'] >= 5]
    vocabulary.reset_index(inplace=True)
    print('Columns of vocabulary:', vocabulary.columns)
    vocabulary.loc[len(vocabulary)] = [r'<UNK>', 0]
    vocabulary.rename(columns={0: 'Word'}, inplace=True)
    print("Created Vocabulary.")
    vocabulary.to_csv(self.vocabulary_file)
    print('Saved vocabulary.')

    # subsampling
    word_int = {v:k for k, v in vocabulary['Word'].to_dict().items()}
    word_freqs = vocabulary["count"].to_dict()
    cleaned_text_int = []
    c=0
    unk = 0
    for word in tqdm(cleaned_text.split(" "), desc='Converting words to integers'):
      if word in word_int.keys():
        cleaned_text_int.append(word_int[word])
      else:
        cleaned_text_int.append(word_int[r'<UNK>'])
        unk+=1
      c+=1

    t = 1e-5
    keep_probs = {}
    for word, freq in word_freqs.items():
      if word == len(vocabulary)-1:
        keep_prob = (math.sqrt((unk/c) / t) + 1) * (t / (unk/c))
      else:
        keep_prob = (math.sqrt((freq/c) / t) + 1) * (t / (freq/c))
      keep_prob = min(1.0, keep_prob)
      keep_probs[word] = keep_prob

    count_word = 0
    with open(self.subsampled_file_path, 'w') as f:
      for i in tqdm(cleaned_text_int, desc="Subsampling common words"):
        if random.random() >= keep_probs[i]:
          continue
        f.write(str(i) + ' ')
        count_word+=1
    print('Subsampling finished')
    return count_word

  def negative_distribution_calc(self):
    text = open(self.subsampled_file_path, 'r').read().split(" ")[:-1]
    counts = np.bincount(text)
    scaled_counts = counts**0.75
    neg_dist = (scaled_counts/np.sum(scaled_counts))
    np.save('neg_distribution_test.npy', neg_dist)

  def positive_example_gen(self, counts):
    sampled_text = open(self.subsampled_file_path, 'r').read().split(" ")[:-1]
    window = self.window
    pairs = {
        'int': [],
        'target': [],
        'pred': []
    }
    for i in tqdm(range(self.window, counts-window-1), desc='Computing positive pairs'):
      dic = {
          'int': [sampled_text[i]]*(window*2),
          'target':sampled_text[i-window:i] + sampled_text[i+1:i+window+1],
          'pred': [1]*(window*2)
      }
      pairs['int'].extend(dic['int'])
      pairs['target'].extend(dic['target'])
      pairs['pred'].extend(dic['pred'])

      if i%1000000==0:
        with open(self.pos_examples, "a", newline="", encoding="utf-8") as f:
          writer = csv.writer(f)
          writer.writerows(zip(*pairs.values()))
          pairs = {
              'int': [],
              'target': [],
              'pred': []
          }
  
  def train_test_val_split(self):
    train_count = 0
    test_count = 0
    valid_count = 0

    with open(self.pos_examples, "r", newline="", encoding="utf-8") as infile, \
        open(self.train_path, "w", newline="", encoding="utf-8") as train_out, \
        open(self.valid_path, "w", newline="", encoding="utf-8") as valid_out, \
        open(self.test_path, "w", newline="", encoding="utf-8") as test_out:
      reader = csv.reader(infile)
      train_writer = csv.writer(train_out)
      valid_writer = csv.writer(valid_out)
      test_writer = csv.writer(test_out)

      header = next(reader)

      train_writer.writerow(header)
      valid_writer.writerow(header)
      test_writer.writerow(header)
      for row in tqdm(reader, desc='Splitting data into train, test and valid sets'):
        r = random.random()
        if r < self.train_ratio:
          train_writer.writerow(row)
          train_count+=1
        elif r < self.train_ratio + self.valid_ratio:
          valid_writer.writerow(row)
          valid_count+=1
        else:
          test_writer.writerow(row)
          test_count+=1
      print('Final Train count: ',train_count)
      print('Final Valid count: ', valid_count)
      print('Final Test count: ', test_count)
