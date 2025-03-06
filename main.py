import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, List

app = FastAPI()

class UserData(BaseModel):
    user_id: str
    topic_scores: Dict[str, int]
    total_time: int
    wrong_questions_data: Dict[str, List[Dict[str, int]]]

raw_score = ctrl.Antecedent(np.arange(0, 51, 1), 'raw_score')
penalty = ctrl.Antecedent(np.arange(0, 51, 1), 'penalty')
adjusted_score = ctrl.Consequent(np.arange(0, 51, 1), 'adjusted_score')

raw_score['low'] = fuzz.trimf(raw_score.universe, [0, 0, 20])
raw_score['medium'] = fuzz.trimf(raw_score.universe, [10, 25, 40])
raw_score['high'] = fuzz.trimf(raw_score.universe, [30, 50, 50])

penalty['low'] = fuzz.trimf(penalty.universe, [0, 0, 15])
penalty['medium'] = fuzz.trimf(penalty.universe, [10, 20, 30])
penalty['high'] = fuzz.trimf(penalty.universe, [25, 50, 50])

adjusted_score['low'] = fuzz.trimf(adjusted_score.universe, [0, 0, 20])
adjusted_score['medium'] = fuzz.trimf(adjusted_score.universe, [15, 25, 35])
adjusted_score['high'] = fuzz.trimf(adjusted_score.universe, [30, 50, 50])

rules = [
    ctrl.Rule(raw_score['high'] & penalty['low'], adjusted_score['high']),
    ctrl.Rule(raw_score['high'] & penalty['medium'], adjusted_score['medium']),
    ctrl.Rule(raw_score['high'] & penalty['high'], adjusted_score['low']),

    ctrl.Rule(raw_score['medium'] & penalty['low'], adjusted_score['medium']),
    ctrl.Rule(raw_score['medium'] & penalty['medium'], adjusted_score['low']),
    ctrl.Rule(raw_score['medium'] & penalty['high'], adjusted_score['low']),

    ctrl.Rule(raw_score['low'], adjusted_score['low'])
]

adjusted_ctrl = ctrl.ControlSystem(rules)
adjusted_sim = ctrl.ControlSystemSimulation(adjusted_ctrl)

def compute_fuzzy_adjusted_score(score, easy, medium, hard):
    total_wrong = easy + medium + hard
    penalty_val = ((easy * 1 + medium * 2 + hard * 4) / max(1, total_wrong)) ** 1.5

    if score >= 35:
        penalty_val *= 0.75

    adjusted_sim.input['raw_score'] = score
    adjusted_sim.input['penalty'] = penalty_val
    adjusted_sim.compute()

    return adjusted_sim.output['adjusted_score']

def determine_fuzzy_level(adj_score):
    if adj_score >= 35:
        return "Advanced"
    elif adj_score >= 20:
        return "Intermediate"
    else:
        return "Beginner"

@app.post("/compute-user-levels/")
def compute_user_levels(user_data: UserData):
    levels = {}
    total_level = 0
    num_topics = len(user_data.topic_scores)

    for topic, score in user_data.topic_scores.items():
        wrong_data = user_data.wrong_questions_data.get(topic, [])
        easy = sum(1 for q in wrong_data if q["point"] == 5)
        medium = sum(1 for q in wrong_data if q["point"] == 10)
        hard = sum(1 for q in wrong_data if q["point"] == 20)

        adj = compute_fuzzy_adjusted_score(score, easy, medium, hard)
        lvl = determine_fuzzy_level(adj)
        levels[topic] = lvl

        total_level += {"Beginner": 1, "Intermediate": 2, "Advanced": 3}[lvl]

    avg_level = total_level / num_topics
    if avg_level >= 2.6:
        total_level_str = "Advanced"
    elif avg_level >= 1.6:
        total_level_str = "Intermediate"
    else:
        total_level_str = "Beginner"

    return {"topic_levels": levels, "overall_level": total_level_str}
