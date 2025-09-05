import os
import json
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, f1_score
from sklearn.metrics import confusion_matrix as c_matrix
from app.configs.settings import COMMON_FEATURES, basepath
from collections import Counter
import numpy as np

def natural_occurrences(possible_outcomes, train_frame, test_frame, target, print_output=True):
    # Combine train and test frames
    combined_frame = pd.concat([train_frame, test_frame], ignore_index=True)

    # Calculate the percentages & return occurrences
    occurrences = {}
    percentage_counts = combined_frame[target].value_counts()
    total_matches = len(combined_frame)

    for outcome in possible_outcomes:
        count = percentage_counts.get(outcome, -1)
        percentage = (count / total_matches) * 100

        # Set values close to zero to zero
        percentage = round(0 if percentage < 0.01 else percentage, 2)

        occurrences[outcome] = percentage
        if print_output:
            print(f"Natural Percentage of {outcome}: {percentage}%")

    return occurrences


def natural_occurrences_grid(possible_outcomes, train_frame, target, without_target_frame):
    # Calculate the percentages & return occurrences
    occurrences = {}

    _train_frame = train_frame

    train_frame = []
    for x in (without_target_frame):
        # for y in enumerate(_train_frame):
        print(x)
        # if x['id'] == y['id']:
        #     train_frame.append(y)

    percentage_counts = train_frame[target].value_counts()
    total_matches = len(train_frame)

    for outcome in possible_outcomes:
        count = percentage_counts.get(outcome, 0)
        percentage = round((count / total_matches) * 100, 2)
        occurrences[outcome] = percentage
        print(f"Natural Percentage of {outcome}: {percentage}%")

    return occurrences


def save_model(model_type, model, target, compe_data):
    COMPETITION_ID = compe_data['id']
    SEASON_ID = compe_data['season_id']
    PREDICTION_TYPE = compe_data['prediction_type']

    # Create the directory if it doesn't exist
    directory = os.path.abspath(
        os.path.join(basepath(), f"trained_models/{PREDICTION_TYPE}/{COMPETITION_ID}/{SEASON_ID}/"))
    os.makedirs(directory, exist_ok=True)

    # Save the model
    filename = os.path.abspath(f"{directory}/{target}_model.joblib")

    joblib.dump(model, filename)

    # Save metadata (model_type)
    meta_filename = os.path.join(directory, f"{target}_model_meta.json")
    with open(meta_filename, "w") as f:
        json.dump({"model_type": model_type}, f)


def get_model(target, compe_data):
    COMPETITION_ID = compe_data['id']
    SEASON_ID = compe_data['season_id']
    PREDICTION_TYPE = compe_data['prediction_type']
    # Load model
    model_filename = os.path.join(
        basepath(), f"trained_models/{PREDICTION_TYPE}/{COMPETITION_ID}/{SEASON_ID}/{target}_model.joblib")
    model = joblib.load(model_filename)

    # Load metadata
    meta_filename = os.path.join(
        basepath(), f"trained_models/{PREDICTION_TYPE}/{COMPETITION_ID}/{SEASON_ID}/{target}_model_meta.json")
    model_type = None
    if os.path.exists(meta_filename):
        with open(meta_filename, "r") as f:
            meta = json.load(f)
            model_type = meta.get("model_type")

    return model, model_type


def preds_score_percentage(target, test_frame, preds, should_print=True):
    # Calculate accuracy and precision for the target variable
    accuracy = accuracy_score(test_frame[target], preds)
    precision = precision_score(
        test_frame[target], preds, average='weighted', zero_division=0)
    f1 = f1_score(test_frame[target], preds,
                  average='weighted', zero_division=0)
    # Calculate the percentages
    average_score = int((1/3 * accuracy + 1/3 * precision + 1/3 * f1) * 100)
    accuracy = int((accuracy / 1) * 100)
    precision = int((precision / 1) * 100)
    f1 = int((f1 / 1) * 100)

    if should_print:
        print(f"Accuracy: {accuracy}%")
        print(f"Precision: {precision}%")
        print(f"F1 score: {f1}%")
        print(f"AVG score: {average_score}%")
        print(f"")

    return accuracy, precision, f1, average_score


def confusion_matrix(test_frame, target, preds):
    # Calculate the confusion matrix
    confusion = c_matrix(test_frame[target], preds)
    # Print the confusion matrix
    print("Confusion Matrix:")
    print(confusion)
    print("\n")


def feature_importance(model, model_type, compe_data, target, FEATURES, show=True, threshold=0.009):
    COMPETITION_ID = compe_data['id']
    SEASON_ID = compe_data['season_id']
    PREDICTION_TYPE = compe_data['prediction_type']

    directory = os.path.abspath(os.path.join(
        basepath(), f"trained_models/{PREDICTION_TYPE}/{COMPETITION_ID}/{SEASON_ID}/{model_type}/"
    ))
    os.makedirs(directory, exist_ok=True)

    importances = None
    best_features = []

    # Try to use native tree-based importances; if unavailable, keep all features.
    try:
        importances = model.feature_importances_
    except AttributeError:
        if show:
            print("feature_importances_ not available for this model; keeping all FEATURES unchanged.")
        best_features = list(FEATURES)

    if importances is not None:
        if show:
            print(importances)

        for i, val in enumerate(importances):
            if val > threshold:
                best_features.append(FEATURES[i])

    if show:
        print("FEATURES: ", len(FEATURES), len(best_features), best_features)

    filename = os.path.abspath(f"{directory}/{target}_features.json")
    with open(filename, 'w') as file:
        json.dump(best_features, file, indent=4)

    return best_features

def get_features(model_type, compe_data, target):
    COMPETITION_ID = compe_data['id']
    SEASON_ID = compe_data['season_id']
    PREDICTION_TYPE = compe_data['prediction_type']

    features = COMMON_FEATURES
    has_features = False

    try:
        # Load hyperparameters data
        filename = os.path.abspath(os.path.join(basepath(),
                                   f"trained_models/{PREDICTION_TYPE}/{COMPETITION_ID}/{SEASON_ID}/{model_type}/{target}_features.json"))

        try:
            with open(filename, 'r') as file:
                features_data = parse_json(json.load(file))
        except:
            FileNotFoundError

        # Get the hyperparameters for compe id
        features = features_data if len(features_data) > 0 else features
        has_features = len(features) > 0

    except:
        KeyError

    return features, has_features


def parse_json(json_data):
    if isinstance(json_data, dict):
        parsed_data = {}
        for key, value in json_data.items():
            parsed_key = int(key) if key.isdigit() else key
            parsed_value = parse_json(value)
            parsed_data[parsed_key] = parsed_value
        return parsed_data
    elif isinstance(json_data, list):
        return [parse_json(item) for item in json_data]
    else:
        return json_data

def get_predicted_hda(preds):
    # Calculate the counts of each class label in the predictions
    class_counts = Counter(preds)
    total_predictions = len(preds)

    # Calculate the percentages
    y_pred_0 = round((class_counts[0] / total_predictions) * 100, 2)
    y_pred_1 = round((class_counts[1] / total_predictions) * 100, 2)
    y_pred_2 = round((class_counts[2] / total_predictions) * 100, 2)
    return {0: y_pred_0, 1: y_pred_1, 2:y_pred_2}


def get_predicted(preds):
    # Calculate the counts of each class label in the predictions
    class_counts = Counter(preds)
    total_predictions = len(preds)

    # Calculate the percentages
    y_pred_0 = round((class_counts[0] / total_predictions) * 100, 2)
    y_pred_1 = round((class_counts[1] / total_predictions) * 100, 2)
    return {0: y_pred_0, 1: y_pred_1}


def get_predicted_cs(preds, test_frame, predict_proba):
    predicted = {}
    match_details = []
    
    matches = np.array(test_frame)
    
    # Collect match details and count CS frequencies
    for i, pred in enumerate(preds):
        proba = max(predict_proba[i])
        cs = int(pred)
        match_id = matches[i][0] if len(matches[i]) > 0 else i + 1
        
        match_details.append((match_id, cs, proba))
        
        if cs in predicted:
            predicted[cs] += 1
        else:
            predicted[cs] = 1
    
    # Calculate percentages
    preds_len = len(preds)
    for cs in predicted:
        predicted[cs] = round(predicted[cs] / preds_len * 100, 2)
    
    # Sort by CS value
    predicted = dict(sorted(predicted.items(), key=lambda x: int(x[0])))
    
    return predicted, match_details

import numpy as np

import numpy as np

def bound_probabilities(probas, min_val=7, max_val=90):
    """
    Clamp probabilities within a specified range, preserving the top class,
    convert to integers, and normalize to sum 100%. Ensures all values >= min_val.

    Parameters:
        probas (list or np.ndarray): Probabilities (0-100 scale or 0-1 scale)
        min_val (int, optional): Minimum allowed probability for any class. Default is 10.
        max_val (int, optional): Maximum allowed probability for any class. Default is 90.

    Returns:
        np.ndarray: Integer probabilities summing to 100, all >= min_val
    """
    probas = np.array(probas, dtype=float)

    # Scale if in 0-1 range
    if probas.max() <= 1.0:
        probas = probas * 100

    # Remember the index of the top class
    top_idx = np.argmax(probas)

    # Clamp top class to max_val
    probas[top_idx] = min(probas[top_idx], max_val)

    # Raise other classes below min_val
    for i in range(len(probas)):
        if i != top_idx and probas[i] < min_val:
            probas[i] = probas[i] + min_val

    # Normalize to sum 100
    total = probas.sum()
    scale = 100 / total
    probas = probas * scale

    # Convert to integers
    probas = np.floor(probas).astype(int)

    # Adjust any rounding difference to top class
    diff = 100 - probas.sum()
    probas[top_idx] += diff

    return probas
