# NextWordGloVeRNN
A Recurrent Neural Network (RNN) for next-word prediction using pre-trained GloVe embeddings. Trained on WikiText-2, this model predicts the next word in a sequence, leveraging two LSTM layers for sequence modeling.

## Features
- Uses pre-trained GloVe 100D embeddings, fine-tuned during training.
- Two-layer LSTM architecture (128 hidden units).
- Evaluates performance with perplexity and sample predictions.
- Trained on WikiText-2 dataset.

## Setup
1. **Clone the repository**:
   ```bash
   git clone https://github.com/MattBorowski1911/next-word-glove-rnn.git
   cd next-word-glove-rnn

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt

3. **Download GloVe embeddings**:
   ```bash
   wget https://nlp.stanford.edu/data/glove.6B.zip
   unzip glove.6B.zip -d embeddings/
mv embeddings/glove.6B/glove.6B.100d.txt embeddings/
rm -rf embeddings/glove.6B/

4. **Run the script**:
   ```bash
   python main.py