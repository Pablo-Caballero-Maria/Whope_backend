from whope.settings import TOPIC_MODEL, EMOTION_MODEL, TOKENIZER, HF_TOKEN
from utils.crypto_utils import encrypt_with_symmetric_key, decrypt_with_symmetric_key
from typing import List, Dict, Any
from cleantext import clean
from transformers import BertTokenizer
from huggingface_hub import InferenceClient

def preprocess(text: str) -> str:
  regex_url: str = r'http\S+|www\S+|https\S+'
  regex_mentions: str  = r'#w+'
  regex_hashtags: str = r'@\w+'
  regex_non_alphanumeric: str = r'[^A-Za-z0-9\s]'
  regex_combined: str = r'|'.join((regex_url, regex_mentions, regex_hashtags, regex_non_alphanumeric))

  cleaned_text: str = clean(text, clean_all=True, reg=regex_combined, reg_replace='')

  return cleaned_text

def predict_from_message(message: str, model: any) -> str:
    cleaned_message: str = preprocess(message)
    encoded_inputs = TOKENIZER(cleaned_message, padding=True, truncation=True, max_length=100, return_tensors="tf")

    prediction: List[float]  = model.predict(encoded_inputs["input_ids"])
    # prediction = [0.6, 0.1, 0.3]
    # get indexes of those values greater than 0.5
    threshold: float = 0.5
    prediction_indexes = [i for i, x in enumerate(prediction[0]) if x > threshold]
    return prediction_indexes


def get_topics_from_message(message: str) -> List[str]:
    topics_mapping: Dict[int, str] = {0: "society", 1: "health", 2: "education", 3: "business", 4: "family", 5: "politics"}
    topic_ids: int = predict_from_message(message, TOPIC_MODEL)
    topics: str = [topics_mapping[topic_id] for topic_id in topic_ids] 

    return topics

def get_emotions_from_message(message: str) -> List[str]:
    emotions_mapping: Dict[int, str] = {0: "empty", 1: "sadness", 2: "enthusiasm", 3: "neutral", 4: "worry", 5: "surprise", 6: "love", 7: "fun",     8: "hate", 9: "happiness", 10: "boredom", 11: "relief", 12: "anger"}
    emotion_ids: int = predict_from_message(message, EMOTION_MODEL)
    emotions: str = [emotions_mapping[emotion_id] for emotion_id in emotion_ids]

    return emotions

def generate_answer_from_prompt(prompt: str) -> str:
    client: InferenceClient = InferenceClient(api_key=HF_TOKEN)
    model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
    messages = [{"role": "user", "content": prompt}]
    completion: str = client.chat.completions.create(model=model_name, messages=messages, max_tokens=500)
    response: str = completion.choices[0].message.content
    print("DeepSeek response: ", response)
    response_clean: str = response.split("</think>")[1]
    return response_clean



