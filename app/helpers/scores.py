import os
import json
from configs.settings import API_BASE_URL, basepath


def scores(occurrences=None, ignore=None):
    
    occurrences = [int(key) for key in occurrences.keys()]

    print(occurrences)

    filename = os.path.abspath(os.path.join(
        basepath(), "app/helpers/scores.json"))

    filtered_scores = {}
    try:
        with open(filename, 'r') as file:
            raw_scores = json.load(file)

            if occurrences is not None:
                # Filter scores based on occurrences if occurrences are provided
                for item in raw_scores:
                    if item["number"] in occurrences and (ignore is None or item["number"] not in ignore):
                        filtered_scores[item["number"]] = item
            else:
                filtered_scores = {item["number"]: item for item in raw_scores}

    except FileNotFoundError:
        filtered_scores = {}

    return filtered_scores
