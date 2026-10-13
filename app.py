
from pathlib import Path
import pickle
import string

import numpy as np
import streamlit as st
from keras.layers import Embedding, Dense
from keras.models import load_model


# -----------------------------
# File Paths
# -----------------------------
APP_DIR = Path(__file__).resolve().parent

MODEL_PATH = APP_DIR / "lstm_model.h5"
MAX_LEN_PATH = APP_DIR / "max_len.pkl"
TOKENIZER_PATH = APP_DIR / "tokenizer.pkl"


# -----------------------------
# Keras Compatibility Classes
# -----------------------------
class CompatibleEmbedding(Embedding):
    @classmethod
    def from_config(cls, config):
        config = config.copy()
        config.pop("quantization_config", None)
        return cls(**config)


class CompatibleDense(Dense):
    @classmethod
    def from_config(cls, config):
        config = config.copy()
        config.pop("quantization_config", None)
        return cls(**config)


# -----------------------------
# Text Preprocessing
# -----------------------------
translator = str.maketrans("", "", string.punctuation)


# -----------------------------
# Load Model and Assets
# -----------------------------
@st.cache_resource
def load_prediction_assets():

    with MAX_LEN_PATH.open("rb") as file:
        max_len = pickle.load(file)

    with TOKENIZER_PATH.open("rb") as file:
        tokenizer = pickle.load(file)

    model = load_model(
        MODEL_PATH,
        compile=False,
        custom_objects={
            "Embedding": CompatibleEmbedding,
            "Dense": CompatibleDense,
        },
    )

    return model, tokenizer, max_len


# -----------------------------
# Validate Model Input Length
# -----------------------------
def get_input_length(model, max_len):

    if (
        not isinstance(max_len, (int, np.integer))
        or isinstance(max_len, (bool, np.bool_))
        or max_len < 1
    ):
        raise ValueError(
            "max_len.pkl must contain a positive integer."
        )

    input_shape = model.input_shape

    if isinstance(input_shape, list):
        raise ValueError(
            "This application supports single-input models only."
        )

    model_length = input_shape[-1]

    if (
        model_length is not None
        and int(model_length) != int(max_len)
    ):
        raise ValueError(
            f"max_len.pkl contains {max_len}, "
            f"but the model expects {model_length} tokens."
        )

    return int(max_len)


# -----------------------------
# Predict Next Word
# -----------------------------
def predictor(model, tokenizer, text, max_len):

    input_length = get_input_length(model, max_len)

    text = text.lower()
    text = text.translate(translator)

    token_sequence = tokenizer.texts_to_sequences([text])[0]

    if not token_sequence:
        return ""

    index_word = getattr(tokenizer, "index_word", None)

    if not index_word:
        index_word = {
            index: word
            for word, index in tokenizer.word_index.items()
        }

    # Keep the most recent tokens.
    token_sequence = token_sequence[-input_length:]

    # Pre-pad with zeros.
    padded_sequence = (
        [0] * (input_length - len(token_sequence))
        + token_sequence
    )

    model_input = np.array(
        [padded_sequence],
        dtype=np.int32
    )

    prediction = model.predict(
        model_input,
        verbose=0
    )

    predicted_index = int(np.argmax(prediction))

    return index_word.get(predicted_index, "")


# -----------------------------
# Generate Text
# -----------------------------
def generate_text(model, tokenizer, seed_text, n_words, max_len):

    generated_text = seed_text

    for _ in range(n_words):

        next_word = predictor(
            model,
            tokenizer,
            generated_text,
            max_len
        )

        if not next_word:
            break

        generated_text += " " + next_word

    return generated_text


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(
    page_title="Next Word Predictor",
    page_icon="✍️",
    layout="centered"
)

st.title("✍️ Next Word Predictor")

st.write(
    "Enter seed text and generate a continuation "
    "using your trained LSTM model."
)

seed_text = st.text_area(
    "Seed text",
    placeholder="Enter your starting sentence...",
    height=120
)

n_words = st.number_input(
    "Number of words to generate",
    min_value=1,
    max_value=100,
    value=10,
    step=1
)

if st.button("Generate text", type="primary"):

    if not seed_text.strip():

        st.warning(
            "Enter some text before requesting a prediction."
        )

    else:

        try:
            with st.spinner(
                "Loading model and generating text..."
            ):

                model, tokenizer, max_len = load_prediction_assets()

                generated_text = generate_text(
                    model,
                    tokenizer,
                    seed_text.strip(),
                    int(n_words),
                    max_len
                )

            if generated_text == seed_text.strip():

                st.warning(
                    "No known tokens were found, or the model "
                    "could not generate a next word."
                )

            else:

                st.subheader("Generated text")
                st.write(generated_text)

        except Exception as exc:

            st.error(
                f"Unable to load the model or generate a prediction: {exc}"
            )
