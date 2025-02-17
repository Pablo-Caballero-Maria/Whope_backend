from typing import Any, Dict, List

import numpy as np
from cleantext import clean
from huggingface_hub import InferenceClient
from sklearn.metrics.pairwise import cosine_similarity

from whope.settings import (
    BERT_MODEL,
    EMOTION_MODEL,
    HF_TOKEN,
    KNOWLEDGE_DOCUMENTS,
    TOKENIZER,
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


def predict_from_message(message: str, model: Any) -> List[int]:
    cleaned_message: str = preprocess(message)
    encoded_inputs = TOKENIZER(cleaned_message, padding="max_length", truncation=True, max_length=100, return_tensors="tf")

    prediction: List[float] = model.predict([encoded_inputs["input_ids"], encoded_inputs["attention_mask"]])
    # prediction = [0.6, 0.1, 0.3]
    # get indexes of those values greater than 0.5
    threshold: float = 0.5
    prediction_indexes: List[int] = [i for i, x in enumerate(prediction[0]) if x > threshold]
    return prediction_indexes


def get_topics_from_message(message: str) -> List[str]:
    topics_mapping: Dict[int, str] = {0: "society", 1: "health", 2: "education", 3: "business", 4: "family", 5: "politics"}
    topic_ids: List[int] = predict_from_message(message, TOPIC_MODEL)
    topics: List[str] = [topics_mapping[topic_id] for topic_id in topic_ids]

    return topics


def get_emotions_from_message(message: str) -> List[str]:
    emotions_mapping: Dict[int, str] = {0: "empty", 1: "sadness", 2: "enthusiasm", 3: "neutral", 4: "worry", 5: "surprise", 6: "love", 7: "fun", 8: "hate", 9: "happiness", 10: "boredom", 11: "relief", 12: "anger"}
    emotion_ids: List[int] = predict_from_message(message, EMOTION_MODEL)
    emotions: List[str] = [emotions_mapping[emotion_id] for emotion_id in emotion_ids]

    return emotions


def get_relevant_documents(message: str, top_n: int) -> List[str]:
    relevant_documents: Dict[str, float] = {}
    cleaned_message: str = preprocess(message)
    encoded_message: Dict[str, Any] = TOKENIZER(cleaned_message, padding=True, truncation=True, max_length=100, return_tensors="tf")
    message_embeddings: np.ndarray = BERT_MODEL(encoded_message["input_ids"]).last_hidden_state[:, 0, :].numpy()

    for document in KNOWLEDGE_DOCUMENTS:
        cleaned_document: str = preprocess(document)
        encoded_document: Dict[str, Any] = TOKENIZER(cleaned_document, padding=True, truncation=True, max_length=100, return_tensors="tf")
        document_embeddings: np.ndarray = BERT_MODEL(encoded_document["input_ids"]).last_hidden_state[:, 0, :].numpy()
        similarity: float = cosine_similarity(message_embeddings, document_embeddings)[0][0]
        relevant_documents[document] = similarity

    return sorted(relevant_documents, key=relevant_documents.get, reverse=True)[:top_n]


def generate_answer_from_prompt(prompt: str, message: str) -> str:
    # "message" is the unencrypted message content, needed for RAG
    client: InferenceClient = InferenceClient(api_key=HF_TOKEN)
    model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
    relevant_documents: List[str] = get_relevant_documents(message, 1)
    full_prompt: str = f"{prompt} \n Context documents: \n {' '.join(relevant_documents)}"
    messages = [{"role": "user", "content": full_prompt}]
    completion: str = client.chat.completions.create(model=model_name, messages=messages, max_tokens=500)
    response: str = completion.choices[0].message.content
    print("Deepseek response: ", response)
    response_clean: str = response.split("</think>")[1]
    return response_clean
