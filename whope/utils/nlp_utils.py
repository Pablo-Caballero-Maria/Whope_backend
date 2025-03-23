from typing import Any, Dict, List, Tuple

import numpy as np
from cleantext import clean
from huggingface_hub import InferenceClient
from motor.motor_asyncio import AsyncIOMotorCursor, AsyncIOMotorDatabase
from sklearn.metrics.pairwise import cosine_similarity

from whope.settings import (
    BERT_MODEL,
    BERT_TOKENIZER,
    EMOTION_MODEL,
    HF_TOKEN,
    MENTALL_ILLNESS_MODEL,
    TOPIC_MODEL,
)


def preprocess(text: str) -> str:
    regex_url: str = r"http\S+|www\S+|https\S+"
    regex_mentions: str = r"#w+"
    regex_hashtags: str = r"@\w+"
    regex_non_alphanumeric: str = r"[^A-Za-z0-9\s]"
    regex_combined: str = r"|".join((regex_url, regex_mentions, regex_hashtags, regex_non_alphanumeric))

    cleaned_text: str = clean(text, clean_all=True, reg=regex_combined, reg_replace="")

    return cleaned_text


def predict_from_message(message: str, model: Any) -> List[float]:
    cleaned_message: str = preprocess(message)
    encoded_inputs = BERT_TOKENIZER(cleaned_message, padding="max_length", truncation=True, max_length=100, return_tensors="tf")

    prediction: List[float] = model.predict([encoded_inputs["input_ids"], encoded_inputs["attention_mask"]])[0]

    return prediction


def get_topics_from_message(message: str) -> List[str]:
    topics_mapping: Dict[int, str] = {0: "society", 1: "health", 2: "education", 3: "business", 4: "family", 5: "politics"}
    # topic_ids: List[int] = predict_from_message(message, TOPIC_MODEL)
    # topics: List[str] = [topics_mapping[topic_id] for topic_id in topic_ids]
    prediction: List[float] = predict_from_message(message, TOPIC_MODEL)

    return prediction


def get_emotions_from_message(message: str) -> List[str]:
    emotions_mapping: Dict[int, str] = {0: "empty", 1: "sadness", 2: "enthusiasm", 3: "neutral", 4: "worry", 5: "surprise", 6: "love", 7: "fun", 8: "hate", 9: "happiness", 10: "boredom", 11: "relief", 12: "anger"}
    # emotion_ids: List[int] = predict_from_message(message, EMOTION_MODEL)
    # emotions: List[str] = [emotions_mapping[emotion_id] for emotion_id in emotion_ids]
    prediction: List[float] = predict_from_message(message, EMOTION_MODEL)
    return prediction


def get_mental_illnesses_from_message(message: str) -> List[str]:
    # 'BPD', 'bipolar', 'depression', 'Anxiety', 'schizophrenia', 'Suicidal', 'Stress'
    mental_illness_mapping: Dict[int, str] = {0: "BPD", 1: "bipolar", 2: "depression", 3: "Anxiety", 4: "schizophrenia", 5: "Suicidal", 6: "Stress"}
    # mental_illness_ids: List[int] = predict_from_message(message, MENTALL_ILLNESS_MODEL)
    # mental_illness: List[str] = [mental_illness_mapping[illness_id] for illness_id in mental_illness_ids]
    prediction: List[float] = predict_from_message(message, MENTALL_ILLNESS_MODEL)
    return prediction


def calculate_similarity_encoded_message_doc(encoded_message: np.ndarray, doc_embedding: np.ndarray) -> float:
    return cosine_similarity(encoded_message, doc_embedding)[0][0]


async def get_relevant_documents_vector_search(message: str, top_n: int) -> List[str]:
    from whope.settings import get_motor_db

    db: AsyncIOMotorDatabase = await get_motor_db()
    cleaned_message: str = preprocess(message)
    encoded_message: np.ndarray = BERT_TOKENIZER(cleaned_message, padding=True, truncation=True, max_length=100, return_tensors="tf")
    query_embedding: np.ndarray = BERT_MODEL(encoded_message["input_ids"]).last_hidden_state[:, 0, :].numpy()[0]

    # Get all documents with their embeddings
    documents: AsyncIOMotorCursor = db.vector_documents.find({}, {"content": 1, "embedding": 1})

    # Calculate similarity scores
    similarities: Dict[str, float] = {document["content"]: calculate_similarity_encoded_message_doc(np.array(query_embedding).reshape(1, -1), np.array(document["embedding"]).reshape(1, -1)) for document in await documents.to_list(length=None)}

    # Sort by similarity score in descending order and take top_n
    sorted_similarities: List[Tuple[str, float]] = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    top_results: List[Tuple[str, float]] = sorted_similarities[:top_n]

    return [content for content, _ in top_results]


async def generate_answer_from_prompt(prompt: str, message: str) -> str:
    # "message" is the unencrypted message content, needed for RAG
    client: InferenceClient = InferenceClient(api_key=HF_TOKEN)
    model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
    relevant_documents: List[str] = await get_relevant_documents_vector_search(message, 1)
    print("Relevant documents: ", relevant_documents)
    full_prompt: str = f"{prompt} \n Context documents: \n {' '.join(relevant_documents)}"
    messages = [{"role": "user", "content": full_prompt}]
    completion: str = client.chat.completions.create(model=model_name, messages=messages, max_tokens=500)
    response: str = completion.choices[0].message.content
    print("Deepseek response: ", response)
    response_clean: str = response.split("</think>")[1]
    return response_clean
