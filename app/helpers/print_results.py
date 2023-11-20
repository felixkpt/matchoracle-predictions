from app.train_predictions.includes.functions import accuracy_score_precision_score, confusion_matrix
from app.matches.update_backend import update_backend
from collections import Counter


def print_hda_predictions(target, matches_frame, preds, predict_proba, scoring, occurances=None, update_backend_with_preds=False):

    # Get the total number of predictions
    total_predictions = len(preds)

    print(f"\nBELOW ARE THE MODEL RESULTS FOR {total_predictions} PREDS:\n")

    # Calculate the counts of each class label in the predictions
    class_counts = Counter(preds)

    # Calculate the percentages
    percentage_h = round((class_counts[0] / total_predictions) * 100, 0)
    percentage_d = round((class_counts[1] / total_predictions) * 100, 0)
    percentage_a = round((class_counts[2] / total_predictions) * 100, 0)

    # Print the percentages
    print(f"Percentage of Home Win (H): {percentage_h}%")
    print(f"Percentage of Draw (D): {percentage_d}%")
    print(f"Percentage of Away Win (A): {percentage_a}%")
    print(f"")

    accuracy_score_precision_score(matches_frame, target, preds, scoring)

    # Print the percentages for each match
    # for i, match_data in enumerate(predict_proba):
    #     h_percentage, d_percentage, a_percentage = match_data
    #     h_percentage = round(h_percentage)
    #     d_percentage = round(d_percentage)
    #     a_percentage = round(a_percentage)

    #     print(
    #         f"Match {i + 1}: H: {h_percentage}%, D: {d_percentage}%, A: {a_percentage}%")
    # print(f"")

    # confusion_matrix(matches_frame, target, preds)

    # print(f"Predictions: {preds}")

    # if update_backend_with_preds:
    #     update_backend(target, occurances,
    #                    matches_frame, preds, predict_proba)


def print_bts_predictions(target, matches_frame, preds, predict_proba, scoring, occurances=None, update_backend_with_preds=False):

    print(f"\nBELOW ARE THE MODEL RESULTS:\n")

    # Get the total number of predictions
    total_predictions = len(preds)

    # Calculate the counts of each class label in the predictions
    class_counts = Counter(preds)

    # Calculate the percentages
    percentage_ng = round((class_counts[0] / total_predictions) * 100, 0)
    percentage_gg = round((class_counts[1] / total_predictions) * 100, 0)

    # Print the percentages
    print(f"Percentage of NO (0): {percentage_ng}%")
    print(f"Percentage of YES (1): {percentage_gg}%")
    print(f"")

    accuracy_score_precision_score(matches_frame, target, preds, scoring)

    confusion_matrix(matches_frame, target, preds)

    # # Print the percentages for each match
    # for i, match_data in enumerate(normalized_probas):
    #     gg_percentage, ng_percentage = match_data
    #     gg_percentage = round(gg_percentage)
    #     ng_percentage = round(ng_percentage)

    #     print(
    #         f"Match {i + 1}: GG: {gg_percentage}%, NG: {ng_percentage}%")
    # print(f"")

    # print(f"Predictions: {preds}")

    # if update_backend_with_preds:
    #     update_backend(target, occurances,
    #                    matches_frame, preds, predict_proba)


def print_over25_predictions(target, matches_frame, preds, predict_proba, scoring, occurances=None, update_backend_with_preds=False):

    print(f"\nBELOW ARE THE MODEL RESULTS:\n")

    # Get the total number of predictions
    total_predictions = len(preds)

    # Calculate the counts of each class label in the predictions
    class_counts = Counter(preds)

    # Calculate the percentages
    percentage_under = round((class_counts[0] / total_predictions) * 100, 0)
    percentage_over = round((class_counts[1] / total_predictions) * 100, 0)

    # Print the percentages
    print(f"Percentage of UNDER (0): {percentage_over}%")
    print(f"Percentage of OVER (1): {percentage_under}%")
    print(f"")

    accuracy_score_precision_score(matches_frame, target, preds, scoring)

    confusion_matrix(matches_frame, target, preds)

    # normalized_probas = normalizer(predict_proba)

    # # Print the percentages for each match
    # for i, match_data in enumerate(normalized_probas):
    #     gg_percentage, ng_percentage = match_data
    #     gg_percentage = round(gg_percentage)
    #     ng_percentage = round(ng_percentage)

    #     print(
    #         f"Match {i + 1}: OVER: {gg_percentage}%, UNDER: {ng_percentage}%")
    # print(f"")

    # print(f"Predictions: {preds}")

    # if update_backend_with_preds:
    #     update_backend(target, occurances,
    #                    matches_frame, preds, predict_proba)


def print_cs_predictions(target, total_matches, test_frame, preds, predict_proba, scoring, occurances=None, print_minimal=False, update_backend_with_preds=False):

    print(f"\nBELOW ARE THE MODEL RESULTS:\n")

    accuracy_score_precision_score(test_frame, target, preds, scoring)

    confusion_matrix(test_frame, target, preds)

    if not print_minimal:
        # Print the percentages for each match
        for i, pred in enumerate(preds):
            proba = max(predict_proba[i])
            print(
                f"Match {i + 1}: CS: {pred} ({proba}%)")
        print(f"")

    
        print(f"Predictions: {preds}")

    if update_backend_with_preds:
        update_backend(target, occurances,
                       test_frame, preds, predict_proba)
