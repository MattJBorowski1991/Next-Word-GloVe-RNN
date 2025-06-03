import torch
import torch.nn as nn
import torch.optim as optim
from data.data import prepare_data, get_data_loaders, load_glove_embeddings
from models.model import NextWordGloVeRNN
import math

import nltk

def train_model(model, train_loader, valid_loader, device, epochs=5, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        valid_loss = 0
        with torch.no_grad():
            for inputs, targets in valid_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                valid_loss += loss.item()
        
        train_loss /= len(train_loader)
        valid_loss /= len(valid_loader)
        train_perplexity = math.exp(train_loss)
        valid_perplexity = math.exp(valid_loss)
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Perplexity: {train_perplexity:.2f}, "
              f"Valid Loss: {valid_loss:.4f}, Valid Perplexity: {valid_perplexity:.2f}")
    
    return model

def evaluate_model(model, test_loader, device):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
    
    test_loss /= len(test_loader)
    test_perplexity = math.exp(test_loss)
    print(f"Test Loss: {test_loss:.4f}, Test Perplexity: {test_perplexity:.2f}")
    return test_perplexity

def predict_next_word(model, input_sequence, word_to_index, index_to_word, device, seq_length=5):
    model.eval()
    tokens = nltk.word_tokenize(input_sequence.lower())
    indices = [word_to_index.get(token, word_to_index['<UNK>']) for token in tokens[-seq_length:]]
    input_tensor = torch.tensor([indices], dtype=torch.long).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        predicted_index = torch.argmax(probs, dim=1).item()
        predicted_word = index_to_word[predicted_index]
    
    return predicted_word

def main():
    # Hyperparameters
    seq_length = 5
    vocab_size = 10000
    embedding_dim = 100
    hidden_dim = 128
    num_layers = 2
    batch_size = 32
    epochs = 100
    lr = 0.001
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Prepare data
    (train_inputs, train_targets), (valid_inputs, valid_targets), (test_inputs, test_targets), word_to_index, index_to_word = prepare_data(seq_length, vocab_size)
    train_loader, valid_loader, test_loader = get_data_loaders((train_inputs, train_targets), (valid_inputs, valid_targets), (test_inputs, test_targets), batch_size)
    
    # Load GloVe embeddings
    glove_file = "embeddings/glove.6B.100d.txt"  # Download from https://nlp.stanford.edu/data/glove.6B.zip
    embedding_matrix = load_glove_embeddings(glove_file, word_to_index, vocab_size, embedding_dim)
    
    # Initialize model
    model = NextWordGloVeRNN(vocab_size, embedding_dim, hidden_dim, num_layers, embedding_matrix).to(device)
    
    # Train model
    model = train_model(model, train_loader, valid_loader, device, epochs, lr)
    
    # Evaluate model
    test_perplexity = evaluate_model(model, test_loader, device)
    
    # Sample predictions
    sample_inputs = [
        "Valkyria Chronicles III is a",
        "The game was released in",
        "It is a truth universally"
    ]
    print("\nSample Predictions:")
    for input_seq in sample_inputs:
        predicted_word = predict_next_word(model, input_seq, word_to_index, index_to_word, device, seq_length)
        print(f"Input: {input_seq} → Predicted: {predicted_word}")

if __name__ == "__main__":
    main()