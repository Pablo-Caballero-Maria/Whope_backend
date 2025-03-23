import random
from typing import Dict, List, Tuple

from utils.nlp_utils import (
    generate_answer_from_prompt,
    get_emotions_from_message,
    get_mental_illnesses_from_message,
)

# all emotions are: {0: "empty", 1: "sadness", 2: "enthusiasm", 3: "neutral", 4: "worry", 5: "surprise", 6: "love", 7: "fun", 8: "hate", 9: "happiness", 10: "boredom", 11: "relief", 12: "anger"}
# all mental illnesses are: {0: "BPD", 1: "bipolar", 2: "depression", 3: "Anxiety", 4: "schizophrenia", 5: "Suicidal", 6: "Stress"}
# Mapping dictionaries for converting keys to list indices
EMOTION_INDEX: Dict[str, int] = {
    "empty": 0,
    "sadness": 1,
    "enthusiasm": 2,
    "neutral": 3,
    "worry": 4,
    "surprise": 5,
    "love": 6,
    "fun": 7,
    "hate": 8,
    "happiness": 9,
    "boredom": 10,
    "relief": 11,
    "anger": 12,
}

MENTAL_ILLNESS_INDEX: Dict[str, int] = {
    "BPD": 0,
    "bipolar": 1,
    "depression": 2,
    "Anxiety": 3,
    "schizophrenia": 4,
    "Suicidal": 5,
    "Stress": 6,
}


class Node:
    def __init__(self, name: str, emotions: Dict[str, float], mental_illnesses: Dict[str, float], transitions: Dict[str, Tuple[str, float]]):
        self.name: str = name
        self.emotions: Dict[str, float] = emotions
        self.mental_illnesses: Dict[str, float] = mental_illnesses
        # For each transition, we store a tuple of (question, edge weight)
        self.transitions: Dict[str, Tuple[str, float]] = transitions


GRAPH: Dict[str, Node] = {
    "start": Node(
        name="start",
        emotions={},
        mental_illnesses={},
        transitions={"sadness": ("Ask the user if he feels sad", 0.5), "depression": ("Ask the user if he feels depressed", 0.5)},
    ),
    "sadness": Node(
        name="sadness",
        emotions={"sadness": 1.0},
        mental_illnesses={},
        transitions={"sadness": ("Ask the user if he feels sad", 0.5), "depression": ("Ask the user if he feels depressed", 0.5)},
    ),
    "depression": Node(
        name="depression",
        emotions={},
        mental_illnesses={"depression": 1.0},
        transitions={"sadness": ("Ask the user if he feels sad", 0.5), "depression": ("Ask the user if he feels depressed", 0.5)},
    ),
}


def calculate_score(transition: str, current_node: Node, emotions: List[float], mental_illnesses: List[float]) -> float:
    edge_weight = current_node.transitions[transition][1] if transition in current_node.transitions else 0.0
    emotion_score = emotions[EMOTION_INDEX[transition]] if transition in EMOTION_INDEX else 0.0
    illness_score = mental_illnesses[MENTAL_ILLNESS_INDEX[transition]] if transition in MENTAL_ILLNESS_INDEX else 0.0
    return edge_weight * emotion_score * illness_score


async def get_answer_from_emotions_mental_illnesses(last_message: str, past_messages: List[str]) -> str:
    # remove last message
    # past_messages = past_messages[:-1]
    print("Past messages: ", past_messages)
    current_node: Node = GRAPH["start"]
    next_node_name: str = ""
    for message in past_messages:
        emotions: List[float] = get_emotions_from_message(message)
        mental_illnesses: List[float] = get_mental_illnesses_from_message(message)
        next_node_name = max(current_node.transitions, key=lambda transition: calculate_score(transition, current_node, emotions, mental_illnesses))
        current_node = GRAPH[next_node_name]

    answer: str = current_node.transitions[next_node_name][0]
    bot_answer: str = await generate_answer_from_prompt(answer, last_message)
    return bot_answer
