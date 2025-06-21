from typing import Any, Dict, List
import math

from motor.motor_asyncio import AsyncIOMotorDatabase
from sentence_transformers import util
from utils.nlp_utils import get_emotions_from_message
import torch
from whope.settings import SENTENCE_SIMILARITY_MODEL, get_motor_db


async def get_activating_phrases_percentage(last_message: Dict[str, str]) -> float:
    db: AsyncIOMotorDatabase = await get_motor_db()
    activating_embeddings = db.activating.find({}, {"content": 1, "embedding": 1})
    print("total number of activating embeddings: ", await db.activating.count_documents({}))
    matching_activating_embeddings: int = 0
    total_activating_embeddings: int = 0


    embedding_2: List[float] = SENTENCE_SIMILARITY_MODEL.encode(last_message["message"], convert_to_tensor=True)

    async for embedding_doc in activating_embeddings:
        embedding_1: List[float] = torch.tensor(embedding_doc["embedding"])

        print("type of embedding 1: ", type(embedding_1))
        print("type of embedding 2: ", type(embedding_2))
        print("embedding_1 shape:", embedding_1.shape)
        print("embedding_2 shape:", embedding_2.shape)

        similarity: float = util.pytorch_cos_sim(embedding_1, embedding_2).item()
        if similarity > 0.7:
            matching_activating_embeddings += 1
        total_activating_embeddings += 1

    percentage_act_phrases: float = matching_activating_embeddings / total_activating_embeddings if total_activating_embeddings > 0 else 0
    return percentage_act_phrases


def get_negative_emotions_punctation(prediction: List[float], emotion_mapping: Dict[int, str]) -> float:

    negative_emotions: List[str] = ["sadness", "hate", "anger"]
    negative_emotions_punctation: float = sum(prediction[i] for i in range(len(prediction)) if emotion_mapping[i] in negative_emotions)

    return negative_emotions_punctation


def get_top_3_emotions(prediction: List[float], emotion_mapping: Dict[int, str]) -> Dict[str, float]:
    # now i need to get the top 3 emotions and do a dict like {emotion1: value1, emotion2: value2, emotion3: value3}
    top_3_emotions: Dict[str, float] = {}
    for i in range(len(prediction)):
        top_3_emotions[emotion_mapping[i]] = prediction[i]
    top_3_emotions = dict(sorted(top_3_emotions.items(), key=lambda item: item[1], reverse=True)[:3])
    return top_3_emotions


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


async def get_risk_from_message(prediction: List[float], emotion_mapping: Dict[int, str], last_message: Dict[str, str], old_risk: float) -> float:
    negative_emotions_punctation: float = get_negative_emotions_punctation(prediction, emotion_mapping)
    activating_phrases_percentage: float = await get_activating_phrases_percentage(last_message)

    new_risk: float = 0.8 * negative_emotions_punctation + 0.2 * activating_phrases_percentage
    # TODO: accumulate risk over time
    risk: float = 0.2 * old_risk + 0.8 * new_risk
   
    return sigmoid(risk)  # risk must be kept between 0 and 1


async def get_enriched_message(messages: List[Dict[str, str]]) -> Dict[str, Any]:
    # messages is the list of full message objects
    last_message: Dict[str, str] = messages[-1]  # <-- not enriched
    past_messages: List[Dict[str, str]] = messages[0:-1]  # <-- enriched
    prediction: List[float] = get_emotions_from_message(last_message["message"])

    emotion_mapping: Dict[int, str] = {0: "sadness", 1: "hate", 2: "love", 3: "happiness", 4: "anger"}

    # enriched message has fields: message (content of the last message), emotions (list of accumulated emotions), risk (accumulated risk),
    # past_questions (list of already asked questions), last_emotion (last emotion)
    # now, we have to enrich the last message

    top_3_emotions: Dict[str, float] = get_top_3_emotions(prediction, emotion_mapping)

    old_risk: float = past_messages[0]["risk"] if past_messages else 0.0
    risk: float = await get_risk_from_message(prediction, emotion_mapping, last_message, old_risk)

    last_emotion_index: int = prediction.index(max(prediction))
    last_emotion: str = emotion_mapping[last_emotion_index]

    enriched_message: Dict[str, Any] = {
        "message": last_message["message"],
        "emotions": top_3_emotions,
        "risk": risk,
        "past_questions": past_messages[0]["past_questions"] if past_messages else [],
        "last_emotion": last_emotion,
    }
    print("printing enriched message from management.py", enriched_message)
    return enriched_message
