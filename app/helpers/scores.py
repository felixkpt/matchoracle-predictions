import os
import json
from app.configs.settings import basepath

def scores(occurrences=None, ignore=None):
    
    occurrences = [int(key) for key in occurrences.keys()]
    
    filename = os.path.abspath(os.path.join(
        basepath(), "helpers/scores.json"))

    scores_list = []

    try:
        with open(filename, 'r') as file:
            raw_scores = json.load(file)

            if occurrences is not None:
                for item in raw_scores:
                    if item["number"] in occurrences and (ignore is None or item["number"] not in ignore):
                        scores_list.append(item) 
            else:
                scores_list = raw_scores

    except FileNotFoundError:
        print('Scores file not found.')
        scores_list = []

    return scores_list
