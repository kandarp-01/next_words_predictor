# Next Word Predictor using LSTM

A Streamlit application that uses a trained LSTM model to predict the next word from seed text and generate a continuation one word at a time.

## Technologies

- Python
- NumPy and Pandas
- TensorFlow/Keras
- Streamlit
- Pickle for saving and loading the tokenizer and maximum sequence length

## Model Training

The training workflow uses `qoute_dataset.csv` and its `quote` column:

1. Load the dataset with Pandas, extract the quotes, lowercase the text, and remove punctuation with `string.punctuation` and `str.translate()`.
2. Fit a `tensorflow.keras.preprocessing.text.Tokenizer` with `num_words=10000` and convert the text to integer sequences.
3. Create examples from each sequence, using preceding words as input and the following word as the target.
4. Pre-pad input sequences with zeros to `max_len`; one-hot encode target indices with `to_categorical()` using 10,000 classes.
5. Train an Embedding/recurrent/Dense softmax network. Both experiments use a 10,000-word vocabulary and 50-dimensional embedding:
   - **SimpleRNN:** 128 recurrent units, 10 epochs.
   - **LSTM:** 128 recurrent units, 100 epochs.
6. Compile with Adam, categorical cross-entropy, and accuracy; train with batch size 128 and validation split 0.1.

The LSTM model is saved as `lstm_model.h5` and is the model used by the application. The notebook and `qoute_dataset.csv` are not present in the current project directory, so the training details above are not independently verifiable from those files here.

## Project Files

- `app.py` — Streamlit interface and next-word generation logic.
- `lstm_model.h5` — trained LSTM model.
- `tokenizer.pkl` — fitted tokenizer.
- `max_len.pkl` — maximum input sequence length.
- `requirements.txt` — Python dependencies for the application.

The training dataset and Jupyter Notebook are not currently included in the project directory.

## Installation and Run

From the project root, create and activate a virtual environment:

```bash
python -m venv .venv
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies and start the app:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Keep `lstm_model.h5`, `tokenizer.pkl`, and `max_len.pkl` in the project root, where the app expects them.

## Usage

1. Launch the app using the command above.
2. Enter seed text, for example `how are you`.
3. Choose the number of words to generate.
4. Click **Generate text**.
5. Read the generated continuation displayed by the app.

For each step, the app lowercases the input, removes punctuation, converts it to token IDs, truncates excess tokens from the beginning, and pre-pads shorter inputs with zeros. It selects the highest-scoring next word with `argmax`, appends it to the text, and repeats for the requested number of words.

Generated text depends on the training data and is not guaranteed to be grammatically correct or factually accurate.
