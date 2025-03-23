import os
from pathlib import Path
from typing import Any, Dict, List, Tuple, Type

import numpy as np
import tensorflow as tf
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


async def load_knowledge_base() -> None:
    from whope.settings import get_motor_db

    # call it only if collection "vector_documents" does not exist
    db: AsyncIOMotorDatabase = await get_motor_db()
    collections: List[str] = await db.list_collection_names()
    await setup_vector_database() if "vector_documents" not in collections else None

    directory: Path = Path(os.path.join(BASE_DIR, "knowledge_base"))
    for file in directory.iterdir():
        with open(file, "r") as f:
            await store_document_with_embedding(f.read())


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
            config: Dict[str, Any] = super().get_config()
            config.update({"pos_encoding": self.pos_encoding.numpy().tolist()})
            return config

    custom_objects = {"BertModelWrapper": BertModelWrapper, "PositionalEncodingAdder": PositionalEncodingAdder, "CLSExtractor": CLSExtractor}

    return custom_objects


async def setup_vector_database():
    from whope.settings import get_motor_db

    db: AsyncIOMotorDatabase = await get_motor_db()

    # Store documents with their embeddings
    await db.create_collection("vector_documents")
    # Create vector index
    await db.command({"createIndexes": "vector_documents", "indexes": [{"name": "vector_index", "key": {"embedding": 1}}]})


async def store_document_with_embedding(document):
    from utils.nlp_utils import preprocess

    from whope.settings import BERT_MODEL, BERT_TOKENIZER, get_motor_db

    db: AsyncIOMotorDatabase = await get_motor_db()
    cleaned_document: str = preprocess(document)
    encoded_document: str = BERT_TOKENIZER(cleaned_document, padding=True, truncation=True, max_length=100, return_tensors="tf")
    embeddings: list[float] = BERT_MODEL(encoded_document["input_ids"]).last_hidden_state[:, 0, :].numpy().tolist()[0]

    await db.vector_documents.insert_one({"content": document, "embedding": embeddings})
