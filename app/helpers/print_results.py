from app.helpers.functions import preds_score_percentage, confusion_matrix, get_predicted_hda, get_predicted, get_predicted_cs
import numpy as np


def header(model_type, compe_data, preds):

    print(f"--- Model Results: {model_type} | Compe: #{compe_data.get('id','UNKNOWN')} | Season: #{compe_data.get('season_id','UNKNOWN')} | Total Predictions: {len(preds)} ---")


def print_preds_hyperparams(target, model_type, compe_data, preds, predict_proba, test_frame=None, print_minimal=False):

    if target.endswith('hda_target'):
        print_hda_predictions(target, model_type, compe_data, preds,
                              predict_proba, test_frame, print_minimal)
    if target == 'bts_target':
        print_bts_predictions(target, model_type, compe_data, preds,
                              predict_proba, test_frame, print_minimal)
    if target == 'over15_target' or target == 'over25_target' or target == 'over35_target':
        print_over_predictions(target, model_type, compe_data, preds,
                               predict_proba, test_frame, print_minimal)
    if target == 'cs_target':
        print_cs_predictions(target, model_type, compe_data, preds,
                             predict_proba, test_frame, print_minimal)


def print_hda_predictions(target, model_type, compe_data, preds, predict_proba, test_frame=None, print_minimal=False):

    header(model_type, compe_data, preds)

    predicted = get_predicted_hda(preds)
    y_pred_0 = predicted[0]
    y_pred_1 = predicted[1]
    y_pred_2 = predicted[2]

    # Print the percentages
    print(f"Percentage of Home Win (0): {y_pred_0}%")
    print(f"Percentage of Draw (1): {y_pred_1}%")
    print(f"Percentage of Away Win (2): {y_pred_2}%")
    print(f"")

    preds_score_percentage(target, test_frame, preds)

    if not len(test_frame) > 0:
        confusion_matrix(test_frame, target, preds)

    if not print_minimal:
        matches = np.array(test_frame)
        # Print the percentages for each match
        for i, match_data in enumerate(predict_proba):
            y_pred_0, y_pred_1, y_pred_2 = match_data
            y_pred_0 = round(y_pred_0)
            y_pred_1 = round(y_pred_1)
            y_pred_2 = round(y_pred_2)

            print(
                f"Match {i + 1}, #{matches[i][0]}: H: {y_pred_0}%, D: {y_pred_1}%, A: {y_pred_2}%")
        print(f"")

    print(f"Predictions: {preds}")

def print_bts_predictions(target, model_type, compe_data, preds, predict_proba, test_frame=None, print_minimal=False):

    header(model_type, compe_data, preds)

    predicted = get_predicted(preds)
    y_pred_0 = predicted[0]
    y_pred_1 = predicted[1]
    
    # Print the percentages
    print(f"Percentage of No (0): {y_pred_0}%")
    print(f"Percentage of Yes (1): {y_pred_1}%")
    print(f"")

    preds_score_percentage(target, test_frame, preds)

    if not len(test_frame) > 0:
        confusion_matrix(test_frame, target, preds)

    if not print_minimal:
        matches = np.array(test_frame)
        # Print the percentages for each match
        for i, match_data in enumerate(predict_proba):
            y_pred_0, y_pred_1 = match_data
            y_pred_1 = round(y_pred_1)
            y_pred_0 = round(y_pred_0)

            print(
                f"Match {i + 1}, #{matches[i][0]}: NO: {y_pred_0}%, YES: {y_pred_1}%")

        print(f"")

        print(f"Predictions: {preds}")


def print_over_predictions(target, model_type, compe_data, preds, predict_proba, test_frame=None, print_minimal=False):

    header(model_type, compe_data, preds)

    predicted = get_predicted(preds)
    y_pred_0 = predicted[0]
    y_pred_1 = predicted[1]

    # Print the percentages
    print(f"Percentage of UNDER (0): {y_pred_0}%")
    print(f"Percentage of OVER (1): {y_pred_1}%")
    print(f"")

    preds_score_percentage(target, test_frame, preds)

    if not len(test_frame) > 0:
        confusion_matrix(test_frame, target, preds)

    if not print_minimal:
        matches = np.array(test_frame)
        # Print the percentages for each match
        for i, match_data in enumerate(predict_proba):
            y_pred_0, y_pred_1 = match_data
            y_pred_1 = round(y_pred_1)
            y_pred_0 = round(y_pred_0)

            print(
                f"Match {i + 1}, #{matches[i][0]}: UN: {y_pred_0}%, OV: {y_pred_1}%")
        print(f"")

        print(f"Predictions: {preds}")

# Updated original function using the extracted function
def print_cs_predictions(target, model_type, compe_data, preds, predict_proba, test_frame=None, print_minimal=False):
    
    header(model_type, compe_data, preds)

    # Use the extracted function
    predicted, match_details = get_predicted_cs(preds, test_frame, predict_proba)

    if not len(test_frame) > 0:
        confusion_matrix(test_frame, target, preds)

    if not print_minimal:
        # Print the percentages for each match
        for match_id, cs, proba in match_details:
            print(f"Match {match_id}: CS: {cs} ({proba}%)")
        
        print(f"")
        print(f"Predictions: {preds}")
        
    preds_score_percentage(target, test_frame, preds)
