import torch
from datasets import load_dataset
import nltk
from collections import Counter
import numpy as np

nltk.download('punkt')

def load_glove_embeddings(glove_file, word_to_index, vocab_size, embedding_dim=100):
    """Load GloVe embeddings and create embedding matrix."""
    embeddings_index = {}
    with open(glove_file, encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype="float32")
            embeddings_index[word] = vector
    
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for word, i in word_to_index.items():
        if i < vocab_size:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
            else:
                embedding_matrix[i] = np.random.normal(0, 0.1, embedding_dim)  # Random for OOV
    return embedding_matrix

def prepare_data(seq_length=5, vocab_size=10000):
    """Load and preprocess WikiText-2 dataset."""
    # Load dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    
    # Tokenize
    def tokenize(text):
        return nltk.word_tokenize(text.lower())
    
    train_text = dataset['train']['text'][:10000]
    valid_text = dataset['validation']['text'][:10000]
    test_text = dataset['test']['text'][:10000]
    print(f"Loaded dataset: train {len(train_text)} lines, valid {len(valid_text)} lines, test {len(test_text)} lines")
    
    # Filter empty strings and tokenize
    train_tokens = [token for line in train_text if line.strip() for token in tokenize(line)]
    print(f"Tokenized train: {len(train_tokens)} tokens")
    valid_tokens = [token for line in valid_text if line.strip() for token in tokenize(line)]
    print(f"Tokenized valid: {len(valid_tokens)} tokens")
    test_tokens = [token for line in test_text if line.strip() for token in tokenize(line)]
    print(f"Tokenized test: {len(test_tokens)} tokens")
    
    # Build vocabulary
    word_counts = Counter(train_tokens)
    vocab = ['<PAD>', '<UNK>'] + [word for word, _ in word_counts.most_common(vocab_size - 2)]
    word_to_index = {word: i for i, word in enumerate(vocab)}
    index_to_word = {i: word for word, i in word_to_index.items()}
    print(f"Vocabulary size: {len(vocab)}")
    
    # Convert tokens to indices
    def tokens_to_indices(tokens, word_to_index):
        return [word_to_index.get(token, word_to_index['<UNK>']) for token in tokens]
    
    train_indices = tokens_to_indices(train_tokens, word_to_index)
    print(f"Converted train tokens to indices: {len(train_indices)}")
    valid_indices = tokens_to_indices(valid_tokens, word_to_index)
    print(f"Converted valid tokens to indices: {len(valid_indices)}")
    test_indices = tokens_to_indices(test_tokens, word_to_index)
    print(f"Converted test tokens to indices: {len(test_indices)}")
    
    # Create input-output pairs
    def create_sequences(indices, seq_length):
        inputs, targets = [], []
        for i in range(len(indices) - seq_length):
            inputs.append(indices[i:i + seq_length])
            targets.append(indices[i + seq_length])
        return torch.tensor(inputs, dtype=torch.long), torch.tensor(targets, dtype=torch.long)
    
    train_inputs, train_targets = create_sequences(train_indices, seq_length)
    valid_inputs, valid_targets = create_sequences(valid_indices, seq_length)
    test_inputs, test_targets = create_sequences(test_indices, seq_length)
    
    return (train_inputs, train_targets), (valid_inputs, valid_targets), (test_inputs, test_targets), word_to_index, index_to_word

def get_data_loaders(train_data, valid_data, test_data, batch_size=32):
    """Create DataLoader objects."""
    train_inputs, train_targets = train_data
    valid_inputs, valid_targets = valid_data
    test_inputs, test_targets = test_data
    
    train_dataset = torch.utils.data.TensorDataset(train_inputs, train_targets)
    valid_dataset = torch.utils.data.TensorDataset(valid_inputs, valid_targets)
    test_dataset = torch.utils.data.TensorDataset(test_inputs, test_targets)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, valid_loader, test_loader