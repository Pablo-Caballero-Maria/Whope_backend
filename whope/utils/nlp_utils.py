import random
from typing import Any, Dict, List

import tensorflow as tf
import torch
from cleantext import clean
from huggingface_hub import InferenceClient
from motor.motor_asyncio import AsyncIOMotorDatabase
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sentence_transformers import util
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from utils.db_utils import save_message_AI

from whope.settings import (
    BERT_TOKENIZER,
    CONTEXT_DOCS,
    EMOTION_MODEL,
    EMOTION_TOKENIZER,
    HF_TOKEN,
    QUESTIONS,
    SENTENCE_SIMILARITY_MODEL,
    STOP_WORDS,
    get_motor_db,
)


def lemmatize(text):

    # Tokenize the text
    tokens = word_tokenize(text)

    # Lemmatize and remove stopwords
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens]

    return " ".join(lemmatized_tokens)


def preprocess(text: str) -> str:
    cleaned_text: str = clean(text, clean_all=True)

    # remove words that are in STOP_WORDS
    cleaned_text = " ".join(word for word in cleaned_text.split() if word.lower() not in STOP_WORDS)
    # lemmatize the text
    cleaned_text = lemmatize(cleaned_text)

    return cleaned_text


def get_emotions_from_message(message: str) -> List[float]:
    cleaned_message: str = preprocess(message)
    encoded_inputs = BERT_TOKENIZER(cleaned_message, padding="max_length", truncation=True, max_length=100, return_tensors="tf")

    prediction: List[float] = EMOTION_MODEL.predict([encoded_inputs["input_ids"], encoded_inputs["attention_mask"]])[0]

    return prediction.tolist()


async def generate_answer_from_prompt(prompt: str, message: str) -> str:
    # "message" is the unencrypted message content, needed for RAG
    client: InferenceClient = InferenceClient(api_key=HF_TOKEN)
    model_name: str = "deepseek-ai/DeepSeek-R1"
    relevant_docs: List[str] = [doc for doc in CONTEXT_DOCS if util.pytorch_cos_sim(SENTENCE_SIMILARITY_MODEL.encode(doc, convert_to_tensor=True), SENTENCE_SIMILARITY_MODEL.encode(message, convert_to_tensor=True)) > 0.7]

    full_prompt: str = f"{prompt} \n Context documents: \n {' '.join(relevant_docs)}"
    messages = [{"role": "user", "content": full_prompt}]
    completion: str = client.chat.completions.create(model=model_name, messages=messages, max_tokens=500)
    response: str = completion.choices[0].message.content

    # sometimes the response is like this: "<think>...</think>"
    response_clean: str = response.split("</think>")[1] if "</think>" in response else response

    return response_clean


def has_risk_increased(past_messages: List[Dict[str, Any]]) -> bool:
    if len(past_messages) < 3:
        return False

    # Get the last 3 messages to check trend
    n = min(3, len(past_messages))
    recent_messages = past_messages[-n:]

    # Count how many consecutive increases we have
    increases = 0
    total_comparisons = len(recent_messages) - 1

    for i in range(1, len(recent_messages)):
        if recent_messages[i]["risk"] > recent_messages[i - 1]["risk"]:
            increases += 1

    # Consider it trending upward if more than half the comparisons show increase
    return increases > total_comparisons / 2


def is_containment_done(past_questions: List[str]) -> bool:
    return any(question.startswith("CONTAINMENT:") for question in past_questions)


def get_stage(enriched_message: Dict[str, Any], past_messages: List[Dict[str, Any]], past_questions: List[str], last_emotion: str, negative_emotions: List[str]) -> str:
    # remove first element from past_messages
    past_messages = past_messages[1:] if len(past_messages) > 0 else []
    risk: float = enriched_message["risk"]
    risk_increased: bool = has_risk_increased(past_messages)
    containment_done: bool = is_containment_done(past_questions)
    last_emotion_negative: bool = last_emotion in negative_emotions

    stage = ""

    if last_emotion_negative:
        if risk < 0.5:
            stage = "CONTAINMENT"
        else:
            stage = "RISK"

    else:
        if risk_increased:
            stage = "RISK"

        else:
            if risk < 0.5 and containment_done:
                stage = "EXPLORATION"

            elif risk > 0.2 and risk < 0.8:
                stage = "CONTAINMENT"

            else:
                stage = "RESOURCE"

    return stage


def are_last_n_messages_similar(last_n_messages: List[str]) -> bool:
    bool_list: List[bool] = []

    for i in range(1, len(last_n_messages)):
        for j in range(1, len(last_n_messages)):
            similarity: float = util.pytorch_cos_sim(last_n_messages[i], last_n_messages[j])
            bool_list.append(similarity > 0.7)

    return all(bool_list)


async def is_closure(message: str) -> bool:
    db: AsyncIOMotorDatabase = await get_motor_db()
    closure_embeddings = await db.closure.find({}, {"content": 1, "embedding": 1}).to_list(length=None)

    message_embedding = SENTENCE_SIMILARITY_MODEL.encode(message, convert_to_tensor=True)

    return any(util.pytorch_cos_sim(torch.tensor(doc["embedding"]), message_embedding).item() > 0.7 for doc in closure_embeddings)


async def is_end(past_messages: List[Dict[str, Any]], enriched_message: Dict[str, Any]) -> bool:
    last_n_messages: List[str] = [message["message"] for message in past_messages[-3:]]
    last_n_messages_similar: bool = are_last_n_messages_similar(last_n_messages)
    are_last_n_messages_short: bool = all(len(message) < 10 for message in last_n_messages)
    closure: bool = await is_closure(enriched_message["message"])

    if closure or last_n_messages_similar or are_last_n_messages_short:
        return True
    return False


async def generate_answer_from_enriched_message(enriched_message: Dict[str, Any], past_messages: List[Dict[str, Any]], user_id) -> str:
    negative_emotions: List[str] = ["sadness", "hate", "anger"]
    questions: List[str] = QUESTIONS
    last_emotion: str = enriched_message["last_emotion"]
    past_questions: List[str] = enriched_message["past_questions"]  # each question is like: "RISK: How are you feeling today?"

    stage: str = get_stage(enriched_message, past_messages, past_questions, last_emotion, negative_emotions)
    if is_end(past_messages, enriched_message):
        return "END"

    potential_questions: List[str] = [question for question in questions if question.startswith(stage) and question not in past_questions]
    prompt: str = random.choice(potential_questions) if potential_questions else "END"
    enriched_message["past_questions"].append(prompt)

    await save_message_AI(enriched_message, user_id)

    return await generate_answer_from_prompt(prompt, enriched_message["message"])
