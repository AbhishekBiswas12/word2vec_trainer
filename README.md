# Word2Vec Trainer

A PyTorch implementation of **Word2Vec** built as a learning project to understand how word embeddings are trained.

## About

I built this project while learning about **Word2Vec, Skip-gram, negative sampling, and word embeddings**.

The goal was to implement the main components myself and understand how they work rather than treating Word2Vec as a black box.

The project includes:

- Text preprocessing
- Vocabulary creation
- Training pair generation
- Skip-gram model
- Negative sampling
- Model training with PyTorch

## Project Structure

```text
word2vec_trainer/
│
├── pyproject.toml
├── README.md
├── LICENSE
│
├── src/
│   └── word2vec_trainer/
│       ├── dataset_loader.py
│       ├── word2vec_model.py
│       ├── preprocessing.py
│       └── trainer.py
│
├── tests/
│   ├── test_dataset.py
│   ├── test_preprocessor.py
│   ├── test_model.py
│   └── test_trainer.py
│
└── examples/
    └── basic_training.py
```

## Training Pipeline

```text
Text without commas or special characters
   ↓
Preprocessing
   ↓
Vocabulary
   ↓
Training Pairs
   ↓
Negative Sampling
   ↓
Skip-gram Model
   ↓
Learned Word Embeddings
```

## Installation

Clone the repository:

```bash
git clone https://github.com/AbhishekBiswas12/word2vec_trainer.git
cd word2vec_trainer
```

Install in editable mode:

```bash
python -m pip install -e .
```

## Usage

A basic training example is available in:

```text
examples/basic_training.py
```

The public API is still evolving as the project develops.

## Testing

Run the test suite with:

```bash
pytest
```

## AI-Assisted Development

This project was built primarily as a learning exercise.

During development, I occasionally used **ChatGPT** and **Google Colab's coding assistant** when I got stuck with implementation details, debugging, or understanding concepts.

I used these tools as learning and development assistance while working to understand the underlying implementation.

## Status

🚧 **Work in progress**

Future improvements include:

- Improve the public API
- Expand test coverage
- Add more examples
- Add embedding evaluation and visualization
- Benchmark different training configurations
- Publish the package to PyPI

## References

- [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781)
- [Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546)
- [The Illustrated Word2Vec](https://jalammar.github.io/illustrated-word2vec/)

## License

MIT License. See [LICENSE](LICENSE) for details.
