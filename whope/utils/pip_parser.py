import ast
import random
from typing import List

from whope.settings import NLM_RULES


def do_terms_match(present_terms: List[str], rule: str) -> bool:
    bool_words: List[str] = ["AND", "OR", "NOT"]
    rule_terms: List[str] = [term for term in rule.split(" ") if term not in bool_words]
    terms: List[str] = list(map(lambda x: x.lower() if x in bool_words else "True" if x in present_terms else "False", rule_terms))
    full_boolean_string: str = " ".join(terms)
    return eval(full_boolean_string)


def does_history_match(history: List[str], rule: str) -> bool:
    history_clean: List[str] = [rule_id for rule_id in history if rule_id != "-1"]
    histories_body: str = rule.split(" AND HISTORY IS ")[1].split(" THEN ")[0]
    possible_histories: List[List[int]] = [ast.literal_eval(history) for history in histories_body.split(" OR ")]

    return True if history_clean in possible_histories else False


def get_rule_prompt_from_topics_emotions_history(topics: List[str], emotions: List[str], history: List[str]) -> tuple[str, str]:

    possible_rule_ids: List[str] = []
    possible_prompts: List[str] = []
    rules: List[str] = NLM_RULES.splitlines()
    for rule in rules:
        if rule[0] == "#":
            continue
        rule_id: str = rule.split(":")[0]
        rule_body: str = rule.split(":")[1]
        conditions: str = rule_body.split(" THEN ")[0]
        prompt: str = rule_body.split(" THEN ")[1]
        topic_body: str = conditions.split("IF TOPIC IS ")[1].split(" AND EMOTION IS")[0]
        do_topics_match: bool = do_terms_match(topics, topic_body)

        emotion_body: str = conditions.split(" AND EMOTION IS ")[1].split(" AND HISTORY IS ")[0]
        do_emotions_match: bool = do_terms_match(emotions, emotion_body)

        does_history_match_: bool = does_history_match(history, rule_body)

        does_rule_apply: bool = do_topics_match and do_emotions_match and does_history_match_
        possible_rule_ids.append(rule_id) if does_rule_apply else None
        possible_prompts.append(prompt) if does_rule_apply else None

    if possible_rule_ids != []:
        random_rule_id: int = random.choice(range(len(possible_rule_ids)))
        return possible_rule_ids[random_rule_id], possible_prompts[random_rule_id]
    else:
        default_prompt: str = "Ask the user if he can repeat the message, wording it differently"
        return "-1", default_prompt
