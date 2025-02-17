from kagglehub import dataset_download
from datasets import load_dataset, DatasetDict
import pandas as pd
import numpy as np
from cleantext import clean
from transformers import BertTokenizer, TFBertModel
import tensorflow as tf
from tensorflow.data import AUTOTUNE, Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.layers import Dense, GlobalAveragePooling1D, BatchNormalization, Dropout, Input, Bidirectional, LSTM, MultiHeadAttention, LayerNormalization, Add, Activation, Layer
from tensorflow.keras import backend as K
import keras_tuner as kt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, History
import json
import gc
from typing import Any, Dict, Tuple, List

# EMBED_DIM: int = 768 # (bert embeddings)
EMBED_DIM: int = 128 # (google tiny bert embeddings)
VOCAB_SIZE: int = 30522 # (bert tokenizer)
SEQ_LEN: int = 100
EXAMPLE_LIMIT: int = 50000
TOPIC_NUMBER: int = 6
EMOTIONS_NUMBER: int = 13
BATCH_SIZE: int = 32

def read_dataset_hf() -> DatasetDict:
    dataset: DatasetDict = load_dataset("community-datasets/yahoo_answers_topics")
    return dataset

def read_dataset_kaggle() -> pd.DataFrame:
    path: str = dataset_download("simaanjali/emotion-analysis-based-on-text")
    dataset: pd.DataFrame = pd.read_csv(f"{path}/emotion_sentimen_dataset.csv")
    return dataset

def preprocess(text: str) -> str:
    regex_url: str = r'http\S+|www\S+|https\S+'
    regex_mentions: str = r'#\w+'
    regex_hashtags: str = r'#\w+'
    regex_non_alphanumeric: str = r'[^A-Za-z0-9\s]'
    regex_combined: str = r'|'.join((regex_url, regex_mentions, regex_hashtags, regex_non_alphanumeric))

    cleaned_text: str = clean(text, clean_all=True, reg=regex_combined, reg_replace='')

    return cleaned_text

def transform_dataset_hf(dataset: DatasetDict, tokenizer: BertTokenizer) -> Tuple[Dict[str, Any], np.ndarray]:
    # combine train and test split from huggingface
    train_df: pd.DataFrame = dataset["train"].to_pandas()
    test_df: pd.DataFrame = dataset["test"].to_pandas()
    dataset: pd.DataFrame = pd.concat([train_df, test_df], ignore_index=True)[:EXAMPLE_LIMIT]

    # drop columns: id, question_title, question_content
    dataset: pd.DataFrame = dataset.drop(columns=["id", "question_title", "question_content"])
    # drop rows where "topic" value is NOT in [0,2,3,6,8,9]
    valid_topics: set = {0, 2, 3, 6, 8, 9}
    dataset: pd.DataFrame = dataset[dataset["topic"].isin(valid_topics)]
    # change topic 2 to 1, 3 to 2, 6 to 3, 8 to 4 and 9 to 5
    dataset["topic"] = dataset["topic"].replace({2: 1, 3: 2, 6: 3, 8: 4, 9: 5})
    # now the topics are {0, 1, 2, 3, 4, 5}

    dataset["best_answer"] = dataset["best_answer"].apply(preprocess)
    encoded_inputs: Dict[str, Any] = tokenizer(dataset["best_answer"].tolist(), padding="max_length", truncation=True, max_length=SEQ_LEN, return_tensors="tf")
    topics: np.ndarray = dataset["topic"].values

    return encoded_inputs, topics

def transform_dataset_kaggle(dataset: pd.DataFrame, tokenizer: BertTokenizer) -> Tuple[Dict[str, Any], np.ndarray]:
    # "dataset" is a pandas dataset
    dataset: pd.DataFrame = dataset[:EXAMPLE_LIMIT]
    # remove column named "#"
    dataset: pd.DataFrame = dataset.drop(columns=["Unnamed: 0"])
    dataset["text"] = dataset["text"].apply(preprocess)

    encoded_inputs: Dict[str, Any] = tokenizer(dataset["text"].tolist(), padding="max_length", truncation=True, max_length=SEQ_LEN, return_tensors="tf")
    encoder: LabelEncoder = LabelEncoder()
    dataset["Emotion"] = encoder.fit_transform(dataset["Emotion"])
    emotions: np.ndarray = dataset["Emotion"].values

    return encoded_inputs, emotions

def convert_to_tf_dataset(x: Dict[str, Any], y: np.ndarray, batch_size: int) -> Dataset:
    # Cast inputs to tf.int32 to match BERT's requirements.
    inputs: Dict[str, Any] = {
        "input_ids": tf.cast(x["input_ids"], tf.int32), 
        "attention_mask": tf.cast(x["attention_mask"], tf.int32)
    }
    dataset: Dataset = Dataset.from_tensor_slices((inputs, y))

    dataset: Dataset = dataset.batch(batch_size).prefetch(AUTOTUNE)
    return dataset

class BertModelWrapper(Layer):
    def __init__(self, bert_model, **kwargs):
        super().__init__(**kwargs)
        self.bert_model = bert_model

    def call(self, inputs):
        # inputs comes as a list of [input_ids, attention_mask]
        input_ids, attention_mask = inputs
        # Now call the underlying BERT model.
        # Make sure to use return_dict=False so that the output is a tuple.
        bert_outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
        # return just the last_hidden_state
        return bert_outputs[0]
    
    def get_config(self):
        config = super().get_config()
        config.update({"bert_model_name": getattr(self.bert_model, "name_or_path", None)})
        return config

    @classmethod
    def from_config(cls, config):
        bert_model_name = config.pop("bert_model_name", None)
        if bert_model_name is not None:
            bert_model = TFBertModel.from_pretrained(bert_model_name)
        else:
            bert_model = None
        return cls(bert_model, **config)

class CLSExtractor(Layer):
    def call(self, inputs):
        return inputs[:, 0, :]

def build_model_dense(hp: kt.HyperParameters, task: str, bert_model: TFBertModel) -> Model:
    output_dim: int = EMOTIONS_NUMBER if task == "emotions" else TOPIC_NUMBER
    dense_units: int = hp.Int('dense_units', min_value=32, max_value=128, step=16)
    drop_rate: float = hp.Float('drop_rate', min_value=0.1, max_value=0.5, step=0.1)

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

    model.compile(
        optimizer=Adam(learning_rate=hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')),
        loss="sparse_categorical_crossentropy",
        metrics=['accuracy']
    )

    return model

def build_model_recurrent(hp: kt.HyperParameters, task: str, bert_model: TFBertModel) -> Model:
    output_dim: int = EMOTIONS_NUMBER if task == "emotions" else TOPIC_NUMBER

    rnn_units: int = hp.Int('rnn_units', min_value=8, max_value=64, step=8)
    dense_units: int = hp.Int('dense_units', min_value=8, max_value=64, step=8)
    drop_rate: float = hp.Float('drop_rate', min_value=0.1, max_value=0.5, step=0.1)

    # Input layers for BERT (the names are important because BERT expect those names)
    # we have to set dtype because BERT expects that type, and otherwise it would be casted to float32
    input_ids = Input(shape=(SEQ_LEN,), name="input_ids", dtype=tf.int32)
    attention_mask = Input(shape=(SEQ_LEN,), name="attention_mask", dtype=tf.int32)

    # Get BERT embeddings
    last_hidden_state = BertModelWrapper(bert_model)([input_ids, attention_mask])

    # BiLSTM processing of BERT embeddings
    lstm_out = Bidirectional(LSTM(rnn_units, dropout=drop_rate, recurrent_dropout=drop_rate))(last_hidden_state)
    x = Dense(dense_units, activation='relu', kernel_initializer=HeNormal())(lstm_out)
    x = BatchNormalization()(x)
    x = Dropout(drop_rate)(x)

    output = Dense(output_dim, activation='softmax')(x)

    # Define the model
    model: Model = Model(inputs=[input_ids, attention_mask], outputs=output)

    model.compile(
        optimizer=Adam(learning_rate=hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')),
        loss="sparse_categorical_crossentropy",
        metrics=['accuracy']
    )

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
    def __init__(self, pos_encoding: np.ndarray, **kwargs):
        super().__init__(**kwargs)
        # Convert the numpy encoding to a constant tensor
        self.pos_encoding = tf.constant(pos_encoding, dtype=tf.float32)
    def call(self, inputs):
        # Add the positional encoding for the [CLS] position.
        # Since inputs shape is (batch, EMBED_DIM) and pos_encoding is (SEQ_LEN, EMBED_DIM),
        # we add only the first row.
        return inputs + self.pos_encoding[0]

    def get_config(self):
        config = super().get_config()
        config.update({"pos_encoding": self.pos_encoding.numpy().tolist()})
        return config

def build_model_transformer(hp: kt.HyperParameters, task: str, bert_model: TFBertModel) -> Model:
    output_dim: int = EMOTIONS_NUMBER if task == "emotions" else TOPIC_NUMBER

    num_heads: int = hp.Int('num_heads', min_value=2, max_value=8, step=2)
    dense_units: int = hp.Int('dense_units', min_value=32, max_value=128, step=16)
    drop_rate: float = hp.Float('drop_rate', min_value=0.1, max_value=0.5, step=0.1)

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
    x_attn = Dropout(drop_rate)(x_attn)
    x_attn = LayerNormalization(epsilon=1e-6)(x_attn)

    # Two dense layers to form a feed-forward network.
    ffn_output = Dense(dense_units, activation="relu", kernel_initializer=HeNormal())(x_attn)
    ffn_output = Dense(EMBED_DIM, activation="relu", kernel_initializer=HeNormal())(ffn_output)
    ffn_output = Dropout(drop_rate)(ffn_output)
    # Add the residual connection and apply layer normalization.
    x_ffn = LayerNormalization(epsilon=1e-6)(x_attn + ffn_output)

    # Global pooling and classification
    x_pooled = GlobalAveragePooling1D()(x_ffn)
    x_dense = Dense(dense_units, activation="relu", kernel_initializer=HeNormal())(x_pooled)
    outputs = Dense(output_dim, activation="softmax")(x_dense)

    # Create model
    model: Model = Model(inputs=[input_ids, attention_mask], outputs=outputs)

    model.compile(
        optimizer=Adam(learning_rate=hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')),
        loss="sparse_categorical_crossentropy",
        metrics=['accuracy']
    )

    return model

def build_tuner(model_type: str, task: str, bert_model: TFBertModel) -> kt.Tuner:
    model_builder: Any = None

    match model_type:
        case "dense":
            model_builder = lambda hp : build_model_dense(hp, task, bert_model)
        case "recurrent":
            model_builder = lambda hp : build_model_recurrent(hp, task, bert_model)
        case "transformer":
            model_builder = lambda hp : build_model_transformer(hp, task, bert_model)
        case _:
            raise ValueError("Invalid model type")

    # default trials is 10
    tuner: kt.Tuner = kt.BayesianOptimization(
        model_builder,
        objective='val_loss',
        overwrite=True,
    )

    return tuner

def get_hps(train_dataset: Dataset, val_dataset: Dataset, tuner: kt.Tuner) -> kt.HyperParameters:
    tuner.search(train_dataset, validation_data=val_dataset)
    best_hps: kt.HyperParameters = tuner.get_best_hyperparameters()[0]
    return best_hps

def train_model(train_dataset: Dataset, val_dataset: Dataset, tuner: kt.Tuner, best_hps: kt.HyperParameters) -> Tuple[Model, History]:
    model: Model = tuner.hypermodel.build(best_hps)
    early_stopping: EarlyStopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    history: History = model.fit(
        train_dataset,
        epochs=20,
        validation_data=val_dataset,
        callbacks=[early_stopping]
        )

    return model, history

def save_model(model: Model, model_type: str, task: str) -> None:
    model_name: str = f"{task}_{model_type}_model.keras"
    model.save(model_name)

def save_history(history: History, model_type: str, task: str) -> None:
    model_name: str = f"{task}_{model_type}_history.json"
    with open(model_name, "w") as f:
        json.dump(history.history, f)

def get_models_and_histories(model_types: List[str], task: str) -> List[Tuple[Model, History]]:
    dataset: Any = read_dataset_kaggle() if task == "emotions" else read_dataset_hf()
    model_name: str = "google/bert_uncased_L-2_H-128_A-2"
    tokenizer: BertTokenizer = BertTokenizer.from_pretrained(model_name)
    bert_model: TFBertModel = TFBertModel.from_pretrained(model_name)
    encoded_inputs: Dict[str, Any]
    labels: np.ndarray
    encoded_inputs, labels = transform_dataset_kaggle(dataset, tokenizer) if task == "emotions" else transform_dataset_hf(dataset, tokenizer)
    
    x_train: Dict[str, Any]
    x_val: Dict[str, Any]
    y_train: np.ndarray
    y_val: np.ndarray

    encoded_inputs_np = {key: val.numpy() for key, val in encoded_inputs.items()}

    input_ids_train, input_ids_val, y_train, y_val = train_test_split(
        encoded_inputs_np["input_ids"], labels, test_size=0.3, random_state=42
    )

    attention_mask_train, attention_mask_val = train_test_split(
        encoded_inputs_np["attention_mask"], test_size=0.3, random_state=42
    )

    x_train = {"input_ids": input_ids_train, "attention_mask": attention_mask_train}
    x_val   = {"input_ids": input_ids_val, "attention_mask": attention_mask_val}

    models_and_histories: List[Tuple[Model, History]] = []
    for model_type in model_types:
        train_dataset: Dataset = convert_to_tf_dataset(x_train, y_train, BATCH_SIZE)
        val_dataset: Dataset = convert_to_tf_dataset(x_val, y_val, BATCH_SIZE)
        tuner: kt.Tuner = build_tuner(model_type, task, bert_model)
        best_hps: kt.HyperParameters = get_hps(train_dataset, val_dataset, tuner)
        model, history = train_model(train_dataset, val_dataset, tuner, best_hps)
        models_and_histories.append((model, history))
        save_model(model, model_type, task)
        save_history(history, model_type, task)
        K.clear_session()
        gc.collect()

    return models_and_histories

model_types: List[str] = ["dense", "recurrent", "transformer"]

models_and_histories_emotion: List[Tuple[Model, History]] = get_models_and_histories(model_types, "emotions")
models_and_histories_topics: List[Tuple[Model, History]] = get_models_and_histories(model_types, "topics")
