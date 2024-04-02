from app.helpers.functions import preds_score, confusion_matrix
from collections import Counter


def header(compe_data, preds):

    # Get the total number of predictions
    total_preds = len(preds)

    print(
        f"\nBELOW ARE THE MODEL RESULTS FOR {compe_data['id']} ({total_preds} PREDS):\n")

    # Calculate the counts of each class label in the predictions
    class_counts = Counter(preds)
    return total_preds, class_counts


def print_preds_update_hyperparams(user_token, target, compe_data, preds, predict_proba, train_frame, test_frame=None, print_minimal=False):

    if target.endswith('hda_target'):
        print_hda_predictions(user_token, target, compe_data, preds,
                              predict_proba, train_frame, test_frame, print_minimal)
    if target == 'bts_target':
        print_bts_predictions(user_token, target, compe_data, preds,
                              predict_proba, train_frame, test_frame, print_minimal)
    if target == 'over15_target' or target == 'over25_target' or target == 'over35_target':
        print_over_predictions(user_token, target, compe_data, preds,
                                 predict_proba, train_frame, test_frame, print_minimal)
    if target == 'cs_target':
        print_cs_predictions(user_token, target, compe_data, preds,
                             predict_proba, train_frame, test_frame, print_minimal)


def print_hda_predictions(user_token, target, compe_data, preds, predict_proba, train_frame, test_frame=None, print_minimal=False):

    total_predictions, class_counts = header(compe_data, preds)

    # Calculate the percentages
    y_pred_0 = round((class_counts[0] / total_predictions) * 100, 2)
    y_pred_1 = round((class_counts[1] / total_predictions) * 100, 2)
    y_pred_2 = round((class_counts[2] / total_predictions) * 100, 2)

    predicted = {0: y_pred_0, 1: y_pred_1, 2: y_pred_2}

    # Print the percentages
    print(f"Percentage of Home Win (0): {y_pred_0}%")
    print(f"Percentage of Draw (1): {y_pred_1}%")
    print(f"Percentage of Away Win (2): {y_pred_2}%")
    print(f"")


    if compe_data and 'is_training' in compe_data and compe_data['is_training']:
        compe_data['predicted'] = predicted
        compe_data['train_counts'] = len(train_frame)
        compe_data['test_counts'] = len(test_frame)

    preds_score(user_token, target, test_frame, preds, compe_data)

    if not len(test_frame) > 0:
        confusion_matrix(test_frame, target, preds)

    if not print_minimal:
        # Print the percentages for each match
        for i, match_data in enumerate(predict_proba):
            y_pred_0, y_pred_1, y_pred_2 = match_data
            y_pred_0 = round(y_pred_0)
            y_pred_1 = round(y_pred_1)
            y_pred_2 = round(y_pred_2)

            print(
                f"Match {i + 1}: H: {y_pred_0}%, D: {y_pred_1}%, A: {y_pred_2}%")
        print(f"")

    print(f"Predictions: {preds}")


def print_bts_predictions(user_token, target, compe_data, preds, predict_proba, train_frame, test_frame=None, print_minimal=False):

    total_predictions, class_counts = header(compe_data, preds)

    # Calculate the percentages
    y_pred_0 = round((class_counts[0] / total_predictions) * 100, 2)
    y_pred_1 = round((class_counts[1] / total_predictions) * 100, 2)

    predicted = {0: y_pred_0, 1: y_pred_1}

    # Print the percentages
    print(f"Percentage of No (0): {y_pred_0}%")
    print(f"Percentage of Yes (1): {y_pred_1}%")
    print(f"")

    if compe_data and 'is_training' in compe_data and compe_data['is_training']:
        compe_data['predicted'] = predicted
        compe_data['train_counts'] = len(train_frame)
        compe_data['test_counts'] = len(test_frame)

    preds_score(user_token, target, test_frame, preds, compe_data)

    if not len(test_frame) > 0:
        confusion_matrix(test_frame, target, preds)

    if not print_minimal:
        # Print the percentages for each match
        for i, match_data in enumerate(predict_proba):
            y_pred_0, y_pred_1 = match_data
            y_pred_1 = round(y_pred_1)
            y_pred_0 = round(y_pred_0)

            print(f"Match {i + 1}: No: {y_pred_0}%, Yes: {y_pred_1}%")
        print(f"")

        print(f"Predictions: {preds}")


def print_over_predictions(user_token, target, compe_data, preds, predict_proba, train_frame, test_frame=None, print_minimal=False):

    total_predictions, class_counts = header(compe_data, preds)

    # Calculate the counts of each class label in the predictions
    class_counts = Counter(preds)

    # Calculate the percentages
    y_pred_0 = round((class_counts[0] / total_predictions) * 100, 2)
    y_pred_1 = round((class_counts[1] / total_predictions) * 100, 2)

    predicted = {0: y_pred_0, 1: y_pred_1}

    # Print the percentages
    print(f"Percentage of UNDER (0): {y_pred_0}%")
    print(f"Percentage of OVER (1): {y_pred_1}%")
    print(f"")

    if compe_data and 'is_training' in compe_data and compe_data['is_training']:
        compe_data['predicted'] = predicted
        compe_data['train_counts'] = len(train_frame)
        compe_data['test_counts'] = len(test_frame)

    preds_score(user_token, target, test_frame, preds, compe_data)

    if not len(test_frame) > 0:
        confusion_matrix(test_frame, target, preds)

    if not print_minimal:
        # Print the percentages for each match
        for i, match_data in enumerate(predict_proba):
            y_pred_0, y_pred_1 = match_data
            y_pred_1 = round(y_pred_1)
            y_pred_0 = round(y_pred_0)

            print(f"Match {i + 1}: UN: {y_pred_0}%, OV: {y_pred_1}%")
        print(f"")

        print(f"Predictions: {preds}")


def print_cs_predictions(user_token, target, compe_data, preds, predict_proba, train_frame, test_frame=None, print_minimal=False):

    total_predictions, class_counts = header(compe_data, preds)

    if not len(test_frame) > 0:
        confusion_matrix(test_frame, target, preds)

    print_minimal = False

    predicted = {}

    if not print_minimal:
        # Print the percentages for each match
        for i, pred in enumerate(preds):
            proba = max(predict_proba[i])

            cs = int(pred)
            print(
                f"Match {i + 1}: CS: {cs} ({proba}%)")

            if cs in predicted:
                predicted[cs] = predicted[cs] + 1
            else:
                predicted[cs] = 1

        preds_len = len(preds)
        for cs in (predicted):
            predicted[cs] = round(predicted[cs] / preds_len * 100, 2)

        predicted = dict(
        sorted(predicted.items(), key=lambda x: int(x[0])))
        print(f"")

        print(f"Predictions: {preds}")

    if compe_data and 'is_training' in compe_data and compe_data['is_training']:
        # filter more than 0 only
        __occurrences = {}
        for k in compe_data['occurrences']:
            if compe_data['occurrences'][k] > 0:
                __occurrences[k] = compe_data['occurrences'][k]

        compe_data['occurrences'] = __occurrences

        compe_data['predicted'] = predicted
        compe_data['train_counts'] = len(train_frame)
        compe_data['test_counts'] = len(test_frame)

    preds_score(user_token, target, test_frame, preds, compe_data)
