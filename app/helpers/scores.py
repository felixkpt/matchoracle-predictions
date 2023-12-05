import json
import os


def scores(occurrences, min_threshold=0.2, bts_min_threshold=1):
    filename = os.path.abspath(f"app/helpers/scores.json")

    with open(filename, "r") as file:
        scores_list = json.load(file)

        # Filter the occurrences only for those with a value greater than min_threshold
        filtered_occurrences = [key for key,
                                value in occurrences.items() if value >= min_threshold]

        # Filter scores based on occurrences
        filtered_scores = [score for score in scores_list if score["cs"]
                           in filtered_occurrences] if filtered_occurrences else filtered_scores

    return filtered_scores
