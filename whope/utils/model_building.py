import gc
import json
from typing import Any, Dict, List, Tuple

import keras_tuner as kt
import numpy as np
import pandas as pd
import tensorflow as tf
from cleantext import clean
from datasets import DatasetDict, load_dataset
from kagglehub import dataset_download
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.data import AUTOTUNE, Dataset
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, History
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.layers import (
    LSTM,
    BatchNormalization,
    Bidirectional,
    Dense,
    Dropout,
    GlobalAveragePooling1D,
    Input,
    Layer,
    LayerNormalization,
    MultiHeadAttention,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from transformers import BertTokenizer, TFBertModel

EMBED_DIM: int = 128  # (google tiny bert embeddings)
VOCAB_SIZE: int = 30522  # (bert tokenizer)
SEQ_LEN: int = 100
EXAMPLE_LIMIT: int = 50000
TOPIC_NUMBER: int = 6
EMOTIONS_NUMBER: int = 5
BATCH_SIZE: int = 32
MENTAL_SIZE: int = 10
MIN_DENSE_UNITS: int = 32
MAX_DENSE_UNITS: int = 128
DENSE_UNITS_STEP: int = 16
MIN_DROP_RATE: float = 0.1
MAX_DROP_RATE: float = 0.5
DROP_RATE_STEP: float = 0.1
MIN_LEARNING_RATE: float = 1e-4
MAX_LEARNING_RATE: float = 1e-2
MIN_RECURRENT_UNITS: int = 8
MAX_RECURRENT_UNITS: int = 64
RECURRENT_UNITS_STEP: int = 8
MIN_HEADS: int = 2
MAX_HEADS: int = 8
HEADS_STEP: int = 2
MAX_TUNER_TRIALS: int = 1
MAX_PATIENCE: int = 5
TRAIN_EPOCHS: int = 20
GOOGLE_BERT_NAME: str = "google/bert_uncased_L-2_H-128_A-2"
RANDOM_STATE: int = 1
TEST_SIZE: float = 0.25


def read_dataset_emotions():
    path = kagglehub.dataset_download("simaanjali/emotion-analysis-based-on-text")
    dataset = pd.read_csv(f"{path}/emotion_sentimen_dataset.csv")
    # rename text to statement and Emotion to status
    dataset = dataset.rename(columns={"text": "statement", "Emotion": "status"})
    dataset = dataset.drop(columns=["Unnamed: 0"])

    emotions_to_keep = {"sadness", "hate", "love", "happiness", "anger"}
    dataset = dataset[dataset["status"].isin(emotions_to_keep)]

    min_count = dataset["status"].value_counts().min()
    balanced = dataset.groupby("status").sample(n=min_count, random_state=RANDOM_STATE)

    balanced = balanced.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    return balanced


def lemmatize(text):

    # Tokenize the text
    tokens = word_tokenize(text)

    # Lemmatize and remove stopwords
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens]

    return " ".join(lemmatized_tokens)


def preprocess(df):
    df["statement"] = df["statement"].apply(lambda x: clean(x, clean_all=True))
    df["statement"] = df["statement"].apply(lambda x: " ".join([word for word in x.split() if word.lower() not in stop_words]))
    df["statement"] = df["statement"].apply(lemmatize)
    return df


def convert_to_tf_dataset(x: Dict[str, Any], y: np.ndarray, batch_size: int) -> Dataset:
    # Cast inputs to tf.int32 to match BERT's requirements.
    inputs: Dict[str, Any] = {"input_ids": tf.cast(x["input_ids"], tf.int32), "attention_mask": tf.cast(x["attention_mask"], tf.int32)}
    dataset: Dataset = Dataset.from_tensor_slices((inputs, y))

    dataset: Dataset = dataset.batch(batch_size).prefetch(AUTOTUNE)
    return dataset


class BertModelWrapper(Layer):
    def __init__(self, bert_model: TFBertModel, **kwargs: Any):
        super().__init__(**kwargs)
        self.bert_model: TFBertModel = bert_model

    def call(self, inputs: List[tf.Tensor]) -> tf.Tensor:
        # inputs comes as a list of [input_ids, attention_mask]
        input_ids: tf.Tensor = inputs[0]
        attention_mask: tf.Tensor = inputs[1]
        # Now call the underlying BERT model.
        # Make sure to use return_dict=False so that the output is a tuple.
        bert_outputs: Tuple[tf.Tensor] = self.bert_model(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
        # return just the last_hidden_state
        return bert_outputs[0]

    def get_config(self):
        config: Dict[str, Any] = super().get_config()
        config.update({"bert_model_name": getattr(self.bert_model, "name_or_path", None)})
        return config

    @classmethod
    def from_config(cls: Type["BertModelWrapper"], config: Dict[str, Any]) -> "BertModelWrapper":
        config.pop("bert_model_name", None)
        return cls(BERT_MODEL, **config)


class CLSExtractor(Layer):
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return inputs[:, 0, :]


def build_model_dense(hp: kt.HyperParameters, bert_model: TFBertModel) -> Model:
    output_dim: int = EMOTIONS_NUMBER
    dense_units: int = hp.Int("dense_units", min_value=MIN_DENSE_UNITS, max_value=MAX_DENSE_UNITS, step=DENSE_UNITS_STEP)
    drop_rate: float = hp.Float("drop_rate", min_value=MIN_DROP_RATE, max_value=MAX_DROP_RATE, step=DROP_RATE_STEP)

    # Input layers for BERT (the names are important because BERT expect those names)
    # we have to set dtype because BERT expects that type, and otherwise it would be casted to float32
    input_ids = Input(shape=(SEQ_LEN,), name="input_ids", dtype=tf.int32)
    attention_mask = Input(shape=(SEQ_LEN,), name="attention_mask", dtype=tf.int32)

    # Get BERT embeddings
    last_hidden_state = BertModelWrapper(bert_model)([input_ids, attention_mask])

    # Use the custom layer to extract the [CLS] token representation
    sequence_output = CLSExtractor()(last_hidden_state)

    # Classification head
    x = Dense(dense_units, activation="relu", kernel_initializer=HeNormal())(sequence_output)
    x = BatchNormalization()(x)
    x = Dropout(drop_rate)(x)

    output = Dense(output_dim, activation="softmax")(x)

    # Define the model
    model: Model = Model(inputs=[input_ids, attention_mask], outputs=output)

    model.compile(optimizer=Adam(learning_rate=hp.Float("learning_rate", min_value=MIN_LEARNING_RATE, max_value=MAX_LEARNING_RATE, sampling="LOG")), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    return model


def build_model_recurrent(hp: kt.HyperParameters, bert_model: TFBertModel) -> Model:
    output_dim: int = EMOTIONS_NUMBER

    rnn_units: int = hp.Int("rnn_units", min_value=MIN_RECURRENT_UNITS, max_value=MAX_RECURRENT_UNITS, step=RECURRENT_UNITS_STEP)
    dense_units: int = hp.Int("dense_units", min_value=MIN_DENSE_UNITS, max_value=MAX_DENSE_UNITS, step=DENSE_UNITS_STEP)
    drop_rate: float = hp.Float("drop_rate", min_value=MIN_DROP_RATE, max_value=MAX_DROP_RATE, step=DROP_RATE_STEP)

    # Input layers for BERT (the names are important because BERT expect those names)
    # we have to set dtype because BERT expects that type, and otherwise it would be casted to float32
    input_ids = Input(shape=(SEQ_LEN,), name="input_ids", dtype=tf.int32)
    attention_mask = Input(shape=(SEQ_LEN,), name="attention_mask", dtype=tf.int32)

    # Get BERT embeddings
    last_hidden_state = BertModelWrapper(bert_model)([input_ids, attention_mask])

    # BiLSTM processing of BERT embeddings
    lstm_out = Bidirectional(LSTM(rnn_units, dropout=drop_rate, recurrent_dropout=drop_rate))(last_hidden_state)
    x = Dense(dense_units, activation="relu", kernel_initializer=HeNormal())(lstm_out)
    x = BatchNormalization()(x)
    x = Dropout(drop_rate)(x)

    output = Dense(output_dim, activation="softmax")(x)

    # Define the model
    model: Model = Model(inputs=[input_ids, attention_mask], outputs=output)

    model.compile(optimizer=Adam(learning_rate=hp.Float("learning_rate", min_value=MIN_LEARNING_RATE, max_value=MAX_LEARNING_RATE, sampling="LOG")), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    return model


def positional_encoding(seq_len: int, embed_dim: int) -> np.ndarray:
    # Create a range of positions (0, 1, 2, ..., seq_len-1) and then reshape it to (seq_len, 1)
    position: np.ndarray = np.arange(seq_len)[:, np.newaxis]
    # Calculate the division term for the sinusoidal functions, np.arange(0, embed_dim, 2) creates a sequence of even numbers (0, 2, 4, ..., seq_len-2)
    div_term: np.ndarray = np.exp(np.arange(0, embed_dim, 2) * -(np.log(10000.0) / embed_dim))
    # Create an empty matrix to store the positional encoding values
    pos_encoding: np.ndarray = np.zeros((seq_len, embed_dim))
    # Apply sine function for the even indices (0, 2, 4, ...)
    pos_encoding[:, 0::2] = np.sin(position * div_term)
    # Apply cosine function for the odd indices (1, 3, 5, ...)
    pos_encoding[:, 1::2] = np.cos(position * div_term)
    return pos_encoding


class PositionalEncodingAdder(Layer):
    def __init__(self, pos_encoding: np.ndarray, **kwargs) -> None:
        super().__init__(**kwargs)
        # Convert the numpy encoding to a constant tensor
        self.pos_encoding: tf.Tensor = tf.constant(pos_encoding, dtype=tf.float32)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        # inputs shape is (batch, SEQ_LEN, EMBED_DIM)
        # pos_encoding shape is (SEQ_LEN, EMBED_DIM)
        # Necesitamos expandir pos_encoding para que sea compatible con inputs
        # Agregando una dimensión en el eje del batch: (1, SEQ_LEN, EMBED_DIM)
        expanded_encoding = tf.expand_dims(self.pos_encoding, axis=0)

        # Ahora TensorFlow hará broadcast automáticamente en la dimensión del batch
        return inputs + expanded_encoding

    def get_config(self) -> Dict[str, Any]:
        config: Dict[str, Any] = super().get_config()
        config.update({"pos_encoding": self.pos_encoding.numpy().tolist()})
        return config


def build_model_transformer(hp: kt.HyperParameters, bert_model: TFBertModel) -> Model:
    output_dim: int = EMOTIONS_NUMBER

    num_heads: int = hp.Int("num_heads", min_value=MIN_HEADS, max_value=MAX_HEADS, step=HEADS_STEP)
    dense_units: int = hp.Int("dense_units", min_value=MIN_DENSE_UNITS, max_value=MAX_DENSE_UNITS, step=DENSE_UNITS_STEP)
    drop_rate: float = hp.Float("drop_rate", min_value=MIN_DROP_RATE, max_value=MAX_DROP_RATE, step=DROP_RATE_STEP)

    # Positional encoding
    pos_encoding: np.ndarray = positional_encoding(SEQ_LEN, EMBED_DIM)

    # BERT input layers (the names are important because BERT expects those names)
    # we have to set dtype because BERT expects that type, and otherwise it would be casted to float32
    input_ids = Input(shape=(SEQ_LEN,), name="input_ids", dtype=tf.int32)
    attention_mask = Input(shape=(SEQ_LEN,), name="attention_mask", dtype=tf.int32)

    # Get BERT embeddings
    last_hidden_state = BertModelWrapper(bert_model)([input_ids, attention_mask])

    # Add positional encoding using the custom layer instead of a Lambda
    x = PositionalEncodingAdder(pos_encoding)(last_hidden_state)

    # Multi-head attention
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=EMBED_DIM)(x, x, attention_mask=attention_mask[:, None, None, :])
    attn_output = Dropout(drop_rate)(attn_output)
    x_attn = LayerNormalization(epsilon=1e-6)(x + attn_output)

    # Global pooling and classification
    x_pooled = GlobalAveragePooling1D()(x_attn)
    outputs = Dense(output_dim, activation="softmax")(x_pooled)

    # Create model
    model: Model = Model(inputs=[input_ids, attention_mask], outputs=outputs)

    model.compile(optimizer=Adam(learning_rate=hp.Float("learning_rate", min_value=MIN_LEARNING_RATE, max_value=MAX_LEARNING_RATE, sampling="LOG")), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    return model


def build_tuner(model_type: str, bert_model: TFBertModel) -> kt.Tuner:
    model_builder: Any = None

    match model_type:
        case "dense":
            model_builder = lambda hp: build_model_dense(hp, bert_model)
        case "recurrent":
            model_builder = lambda hp: build_model_recurrent(hp, bert_model)
        case "transformer":
            model_builder = lambda hp: build_model_transformer(hp, bert_model)
        case _:
            raise ValueError("Invalid model type")

    # default trials is 10
    tuner: kt.Tuner = kt.BayesianOptimization(
        model_builder,
        objective="val_loss",
        max_trials=MAX_TUNER_TRIALS,
        overwrite=True,
    )

    return tuner


def get_hps(train_dataset: Dataset, val_dataset: Dataset, tuner: kt.Tuner) -> kt.HyperParameters:
    tuner.search(train_dataset, validation_data=val_dataset)
    best_hps: kt.HyperParameters = tuner.get_best_hyperparameters()[0]
    return best_hps


def train_model(train_dataset: Dataset, val_dataset: Dataset, tuner: kt.Tuner, best_hps: kt.HyperParameters) -> Tuple[Model, History]:
    model: Model = tuner.hypermodel.build(best_hps)
    early_stopping: EarlyStopping = EarlyStopping(monitor="val_loss", patience=MAX_PATIENCE, restore_best_weights=True)

    history: History = model.fit(train_dataset, epochs=TRAIN_EPOCHS, validation_data=val_dataset, callbacks=[early_stopping])

    return model, history


def save_model(model: Model, model_type: str) -> None:
    model_name: str = f"{model_type}_model.keras"
    model.save(model_name)


def save_history(history: History, model_type: str) -> None:
    model_name: str = f"{model_type}_history.json"
    with open(model_name, "w") as f:
        json.dump(history.history, f)


def get_models_and_histories(model_types: List[str]) -> List[Tuple[Model, History]]:
    dataset = read_dataset_emotions()

    dataset = preprocess(dataset)

    tokenizer: BertTokenizer = BertTokenizer.from_pretrained(GOOGLE_BERT_NAME)
    bert_model: TFBertModel = TFBertModel.from_pretrained(GOOGLE_BERT_NAME)
    encoded_inputs: Dict[str, Any]
    labels: np.ndarray
    encoded_inputs, labels = transform_dataset_kaggle(dataset, tokenizer)

    x_train: Dict[str, Any]
    x_val: Dict[str, Any]
    y_train: np.ndarray
    y_val: np.ndarray

    encoded_inputs_np = {key: val.numpy() for key, val in encoded_inputs.items()}

    input_ids_train, input_ids_val, y_train, y_val = train_test_split(encoded_inputs_np["input_ids"], labels, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    attention_mask_train, attention_mask_val = train_test_split(encoded_inputs_np["attention_mask"], test_size=TEST_SIZE, random_state=RANDOM_STATE)

    x_train = {"input_ids": input_ids_train, "attention_mask": attention_mask_train}
    x_val = {"input_ids": input_ids_val, "attention_mask": attention_mask_val}

    models_and_histories: List[Tuple[Model, History]] = []
    for model_type in model_types:
        train_dataset: Dataset = convert_to_tf_dataset(x_train, y_train, BATCH_SIZE)
        val_dataset: Dataset = convert_to_tf_dataset(x_val, y_val, BATCH_SIZE)
        tuner: kt.Tuner = build_tuner(model_type, bert_model)
        best_hps: kt.HyperParameters = get_hps(train_dataset, val_dataset, tuner)
        model, history = train_model(train_dataset, val_dataset, tuner, best_hps)
        models_and_histories.append((model, history))
        save_model(model, model_type)
        save_history(history, model_type)
        K.clear_session()
        gc.collect()

    return models_and_histories
