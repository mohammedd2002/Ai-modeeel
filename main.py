from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import skfuzzy as fuzz
import skfuzzy.control as ctrl

app = FastAPI()

class UserPerformance(BaseModel):
    total_score: float
    total_time: float
    topic_scores: dict

raw_score = ctrl.Antecedent(np.arange(0, 71, 1), 'raw_score')
total_score = ctrl.Antecedent(np.arange(0, 351, 1), 'total_score')
total_time = ctrl.Antecedent(np.arange(0, 10000, 1), 'total_time')
adjusted_score = ctrl.Consequent(np.arange(0, 71, 1), 'adjusted_score')

raw_score.automf(3)
total_score.automf(3)
total_time.automf(3)

adjusted_score['low'] = fuzz.trimf(adjusted_score.universe, [0, 0, 20])
adjusted_score['medium'] = fuzz.trimf(adjusted_score.universe, [20, 35, 50])
adjusted_score['high'] = fuzz.trimf(adjusted_score.universe, [47, 70, 70])

rules = [
    ctrl.Rule(raw_score['good'] & total_score['good'] & total_time['poor'], adjusted_score['high']),
    ctrl.Rule(raw_score['average'] & total_score['average'] & total_time['average'], adjusted_score['medium']),
    ctrl.Rule(raw_score['poor'] | total_time['good'], adjusted_score['low']),
    ctrl.Rule(raw_score['good'] & total_time['average'], adjusted_score['high']),
    ctrl.Rule(raw_score['average'] & total_time['poor'], adjusted_score['medium']),
    ctrl.Rule(raw_score['average'] & total_score['good'] & total_time['average'], adjusted_score['medium']),
    ctrl.Rule(raw_score['good'] & total_score['average'] & total_time['average'], adjusted_score['medium']),
    ctrl.Rule(raw_score['good'] & total_score['good'] & total_time['average'], adjusted_score['high']),
    ctrl.Rule(raw_score['average'] & total_score['average'] & total_time['poor'], adjusted_score['medium']),
    ctrl.Rule(raw_score['poor'] & total_score['average'] & total_time['poor'], adjusted_score['low']),
    ctrl.Rule(raw_score['average'] & total_score['good'] & total_time['poor'], adjusted_score['high']),
    ctrl.Rule(raw_score['good'] & total_score['average'] & total_time['poor'], adjusted_score['high']),
    ctrl.Rule(raw_score['average'] & total_score['average'] & total_time['good'], adjusted_score['low']),
    ctrl.Rule(raw_score['poor'] & total_score['good'] & total_time['average'], adjusted_score['medium']),
]

adjusted_ctrl = ctrl.ControlSystem(rules)
adjusted_sim = ctrl.ControlSystemSimulation(adjusted_ctrl)

def calculate_wrong_questions(score, max_topic_score=70):
    missing_score = max_topic_score - score
    easy = medium = hard = 0
    while missing_score >= 20 and hard < 2:
        hard += 1
        missing_score -= 20
    while missing_score >= 10 and medium < 2:
        medium += 1
        missing_score -= 10
    while missing_score >= 5 and easy < 2:
        easy += 1
        missing_score -= 5
    return easy, medium, hard

def compute_penalty(easy, medium, hard, total_score, total_time, score):
    total_mistakes = easy + medium + hard
    if total_mistakes == 0:
        return 0
    penalty_weight = easy * 0.4 + medium * 0.8 + hard * 1.5
    if total_score >= 250:
        penalty_weight *= 0.3
    elif total_score >= 200:
        penalty_weight *= 0.5
    if total_time > 6000 and score < 25:
        penalty_weight += 1
    return min(penalty_weight, 5)

def apply_total_score_adjustment(adj_score, total_score):
    if total_score >= 250:
        return min(adj_score + 5, 70)
    elif total_score >= 200:
        return min(adj_score + 3, 70)
    elif total_score < 150:
        return max(adj_score - 3, 0)
    return adj_score

def compute_fuzzy_adjusted_score(score, total_score_val, total_time_val):
    easy, medium, hard = calculate_wrong_questions(score)
    penalty_val = compute_penalty(easy, medium, hard, total_score_val, total_time_val, score)
    adjusted_sim.input['raw_score'] = score
    adjusted_sim.input['total_score'] = total_score_val
    adjusted_sim.input['total_time'] = total_time_val
    adjusted_sim.compute()
    final_score = adjusted_sim.output['adjusted_score'] - penalty_val
    adjusted_score_with_total = apply_total_score_adjustment(final_score, total_score_val)
    return max(0, adjusted_score_with_total), (easy, medium, hard)

def determine_fuzzy_level(adj_score):
    return "Advanced" if adj_score >= 50 else "Intermediate" if adj_score >= 25 else "Beginner"

def compute_user_levels(user_data):
    levels = {}
    num_topics = len(user_data["topic_scores"])
    for topic, score in user_data["topic_scores"].items():
        adj, (easy, medium, hard) = compute_fuzzy_adjusted_score(score, user_data["total_score"], user_data["total_time"])
        levels[topic] = {"level": determine_fuzzy_level(adj), "adjusted_score": adj}
    level_values = {"Beginner": 1, "Intermediate": 2, "Advanced": 3}
    avg_level = sum(level_values[levels[topic]["level"]] for topic in user_data["topic_scores"]) / num_topics
    return "Advanced" if avg_level >= 2.5 else "Intermediate" if avg_level >= 1.5 else "Beginner"

@app.post("/evaluate")
def evaluate_performance(performance: UserPerformance):
    overall_level = compute_user_levels(performance.dict())
    return {"level": overall_level}
