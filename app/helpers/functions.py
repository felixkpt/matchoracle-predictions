import os
import json
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, f1_score
from sklearn.metrics import confusion_matrix as c_matrix
from app.train_predictions.hyperparameters.hyperparameters import save_hyperparameters
from configs.settings import COMMON_FEATURES, basepath


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


def save_model(model, train_frame, test_frame, FEATURES, target, compe_data):
    COMPETITION_ID = compe_data['id']
    PREDICTION_TYPE = compe_data['prediction_type']

    matches = train_frame
    model.fit(matches[FEATURES], matches[target])

    # Create the directory if it doesn't exist
    directory = os.path.abspath(
        os.path.join(basepath(), f"trained_models/{PREDICTION_TYPE}/{COMPETITION_ID}/"))
    os.makedirs(directory, exist_ok=True)

    # Save the model
    filename = os.path.abspath(f"{directory}/{target}_model.joblib")

    joblib.dump(model, filename)


def get_model(target, compe_data):
    COMPETITION_ID = compe_data['id']
    PREDICTION_TYPE = compe_data['prediction_type']
    # Save the model
    filename = os.path.abspath(
        os.path.join(basepath(), f"trained_models/{PREDICTION_TYPE}/{COMPETITION_ID}/{target}_model.joblib"))
    return joblib.load(filename)


def preds_score(user_token, target, test_frame, preds, compe_data):
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

    print(f"Accuracy: {accuracy}%")
    print(f"Precision: {precision}%")
    print(f"F1 score: {f1}%")
    print(f"AVG score: {average_score}%")
    print(f"")

    scores = accuracy, precision, f1, average_score

    if compe_data and 'is_training' in compe_data and compe_data['is_training']:
        compe_data['scores'] = scores
        save_hyperparameters(compe_data, target, user_token)


def confusion_matrix(test_frame, target, preds):
    # Calculate the confusion matrix
    confusion = c_matrix(test_frame[target], preds)
    # Print the confusion matrix
    print("Confusion Matrix:")
    print(confusion)
    print("\n")


def feature_importance(model, compe_data, target, FEATURES, show=True, threshold=0.009):
    feature_importance = model.feature_importances_

    if show:
        print(feature_importance)

    best_features = []
    for i, val in enumerate(feature_importance):
        if val > threshold:
            best_features.append(FEATURES[i])
    if show:
        print(len(FEATURES), len(best_features), best_features)

    COMPETITION_ID = compe_data['id']
    PREDICTION_TYPE = compe_data['prediction_type']

    # Create the directory if it doesn't exist
    directory = os.path.abspath(os.path.join(basepath(),
                                             f"configs/important_features/{PREDICTION_TYPE}/{COMPETITION_ID}/"))
    os.makedirs(directory, exist_ok=True)

    # Save the features
    filename = os.path.abspath(f"{directory}/{target}_features.json")
    # Save the sorted data back to the JSON file
    with open(filename, 'w') as file:
        json.dump(best_features, file, indent=4)
    
    return best_features


def get_features(compe_data, target):
    COMPETITION_ID = compe_data['id']
    PREDICTION_TYPE = compe_data['prediction_type']

    features = COMMON_FEATURES
    has_features = False

    try:
        # Load hyperparameters data
        filename = os.path.abspath(os.path.join(basepath(),
                                   f"configs/important_features/{PREDICTION_TYPE}/{COMPETITION_ID}/{target}_features.json"))

        try:
            with open(filename, 'r') as file:
                features_data = parse_json(json.load(file))
        except:
            FileNotFoundError

        # Get the hyperparameters for compe id
        features = features_data
        has_features = True

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
