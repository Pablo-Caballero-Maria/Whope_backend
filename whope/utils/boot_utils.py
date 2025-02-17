import os
from pathlib import Path
from typing import List, Tuple, Any, Dict, Type
import tensorflow as tf
import numpy as np
from cryptography.hazmat.primitives.asymmetric.rsa import (
    RSAPrivateKey,
    RSAPublicKey,
    generate_private_key,
)
from cryptography.hazmat.primitives.serialization import (
    BestAvailableEncryption,
    Encoding,
    PrivateFormat,
    PublicFormat,
)
from motor.motor_asyncio import (
    AsyncIOMotorClient,
    AsyncIOMotorCollection,
    AsyncIOMotorDatabase,
)
from pymongo import ASCENDING
from tensorflow.keras.layers import Layer
from transformers import BertTokenizer, TFBertModel

BASE_DIR: Path = Path(__file__).resolve().parent.parent


def generate_asymmetric_keys(password: bytes) -> Tuple[bytes, bytes]:
    private_key: RSAPrivateKey = generate_private_key(public_exponent=65537, key_size=2048)
    public_key: RSAPublicKey = private_key.public_key()

    private_key_bytes: bytes = private_key.private_bytes(encoding=Encoding.PEM, format=PrivateFormat.PKCS8, encryption_algorithm=BestAvailableEncryption(password))

    public_key_bytes: bytes = public_key.public_bytes(encoding=Encoding.PEM, format=PublicFormat.SubjectPublicKeyInfo)

    return private_key_bytes, public_key_bytes


async def init_db() -> None:
    MONGODB_URI: str = os.getenv("MONGODB_URI")
    MONGODB_CLIENT: AsyncIOMotorClient = AsyncIOMotorClient(MONGODB_URI)
    # this calling of the database is just done to create the indexes on startup, but then when the database is needed,
    # it will be called with get_motor_db to ensure that its async loop is the same as the one of the method that calls it
    DATABASE: AsyncIOMotorDatabase = MONGODB_CLIENT["whope"]
    USERS: AsyncIOMotorCollection = DATABASE["users"]
    await USERS.create_index([("username", ASCENDING)], unique=True)
    await USERS.create_index([("is_worker", ASCENDING), ("status", ASCENDING)])


def load_nlm_rules() -> str:
    path: Path = Path(os.path.join(BASE_DIR, "utils/nlm_rules.pip"))
    with open(path, "r") as f:
        return f.read()


def load_knowledge_base() -> List[str]:
    directory: Path = Path(os.path.join(BASE_DIR, "knowledge_base"))
    knowledge_base = []
    for file in directory.iterdir():
        with open(file, "r") as f:
            knowledge_base.append(f.read())

    return knowledge_base


MODEL_NAME: str = "google/bert_uncased_L-2_H-128_A-2"
BERT_MODEL: TFBertModel = TFBertModel.from_pretrained(MODEL_NAME)
BERT_TOKENIZER: BertTokenizer = BertTokenizer.from_pretrained(MODEL_NAME)


def get_custom_objects():
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

    class PositionalEncodingAdder(Layer):
        def __init__(self, pos_encoding: np.ndarray, **kwargs) -> None:
            super().__init__(**kwargs)
            # Convert the numpy encoding to a constant tensor
            self.pos_encoding: tf.Tensor = tf.constant(pos_encoding, dtype=tf.float32)

        def call(self, inputs: tf.Tensor) -> tf.Tensor:
            # Add the positional encoding for the [CLS] position.
            # Since inputs shape is (batch, EMBED_DIM) and pos_encoding is (SEQ_LEN, EMBED_DIM),
            # we add only the first row.
            return inputs + self.pos_encoding[0]

        def get_config(self) -> Dict[str, Any]:
            config: Dict[str, Any]= super().get_config()
            config.update({"pos_encoding": self.pos_encoding.numpy().tolist()})
            return config

    custom_objects = {"BertModelWrapper": BertModelWrapper, "PositionalEncodingAdder": PositionalEncodingAdder, "CLSExtractor": CLSExtractor}

    return custom_objects
