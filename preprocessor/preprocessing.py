import regex as re
import pandas as pd
from tqdm.auto import tqdm
import math
import random

class Preprocessing:
  def __init__(self, cleaned_data_file_path = 'cleaned_data.txt', subsampled_file_path = 'subsampled_data.txt', vocabulary_file = 'vocabulary.csv'):
    self.cleaned_data_file_path = cleaned_data_file_path
    self.subsampled_file_path = subsampled_file_path
    self.vocabulary_file = vocabulary_file

  def run(self):
    self.vocabulary()

  def vocabulary(self):
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

    with open(self.subsampled_file_path, 'w') as f:
      for i in tqdm(cleaned_text_int, desc="Subsampling common words"):
        if random.random() >= keep_probs[i]:
          continue
        f.write(str(i) + ' ')
    print('Subsampling finished')