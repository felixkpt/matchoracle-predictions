import numpy as np
import os
import json
import joblib
from sklearn.metrics import accuracy_score, precision_score, f1_score
from sklearn.metrics import confusion_matrix as c_matrix
from datetime import datetime
from dateutil.relativedelta import relativedelta


def natural_occurances(possible_outcomes, train_frame, target, print_output=True):
    # Calculate the percentages & return occurances
    occurances = {}
    percentage_counts = train_frame[target].value_counts()
    total_matches = len(train_frame)

    for outcome in possible_outcomes:
        count = percentage_counts.get(outcome, -1)
        percentage = round((count / total_matches) * 100, 2)
        occurances[outcome] = percentage
        if print_output:
            print(f"Natural Percentage of {outcome}: {percentage}%")

    return occurances


def natural_occurances_grid(possible_outcomes, train_frame, target, without_target_frame):
    # Calculate the percentages & return occurances
    occurances = {}

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
        occurances[outcome] = percentage
        print(f"Natural Percentage of {outcome}: {percentage}%")

    return occurances


def save_model(model, train_frame, test_frame, PREDICTORS, target, COMPETITION_ID):
    matches = train_frame
    model.fit(matches[PREDICTORS], matches[target])
    # Save the model
    filename = os.path.abspath(
        f"trained_models/{COMPETITION_ID}/{target}_model.joblib")

    joblib.dump(model, filename)


def get_model(target, COMPETITION_ID):
    # Save the model
    filename = os.path.abspath(
        f"trained_models/{COMPETITION_ID}/{target}_model.joblib")
    return joblib.load(filename)


def accuracy_score_precision_score(test_frame, target, preds, scoring):
    # Calculate accuracy and precision for the target variable
    accuracy = accuracy_score(test_frame[target], preds)
    precision = precision_score(
        test_frame[target], preds, average=scoring, zero_division=0)
    f1 = f1_score(test_frame[target], preds,
                  average='weighted', zero_division=0)
    # Calculate the percentages
    mean = int((1/3 * accuracy + 1/3 * precision + 1/3 * f1) * 100)
    accuracy = (int((accuracy / 1) * 100))
    precision = (int((precision / 1) * 100))
    f1 = (int((f1 / 1) * 100))

    print(f"Accuracy: {accuracy}%")
    print(f"Precision: {precision}%")
    print(f"F1 score: {f1}%")
    print(f"Mean score: {mean}%")
    print(f"")


def confusion_matrix(test_frame, target, preds):
    # Calculate the confusion matrix
    confusion = c_matrix(test_frame[target], preds)
    # Print the confusion matrix
    print("Confusion Matrix:")
    print(confusion)
    print("\n")


def hyperparameters_array_generator(train_frame, class_weight_counts=14, rest_counts=6):
    len_train = len(train_frame)

    _min = 10
    # Setting the range for params
    _n_estimators = np.linspace(
        100, len_train / 6 if len_train > 100 else 100, rest_counts)
    class_weight = np.linspace(1.0, 2.0, class_weight_counts)
    _min_samples_splits = np.linspace(
        _min, len_train / 40 if len_train > _min else _min, rest_counts)
    _max_depth = np.linspace(
        _min, len_train / 10 if len_train > _min else _min, rest_counts)

    n_estimators = []
    for x in _n_estimators:
        x = int(x)
        n_estimators.append(x)

    min_samples_split = [2, 4, 6, 8]
    for x in _min_samples_splits:
        x = int(x)
        min_samples_split.append(x)

    max_depth = []
    for x in _max_depth:
        x = int(x)
        max_depth.append(x)

    return [n_estimators, min_samples_split, class_weight]


def save_hyperparameters(COMPETITION_ID, target, best_params, train_counts, test_counts):
    # Load existing hyperparameters data
    filename = os.path.abspath(
        f"app/train_predictions/tuning/hyperparameters/{target}_hyperparameters.json")

    try:
        with open(filename, 'r') as file:
            hyperparameters_data = parse_json(json.load(file))
    except FileNotFoundError:
        hyperparameters_data = {}

    current_datetime = datetime.today()
    now = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    add = {
        'train_counts': train_counts,
        'test_counts': test_counts,
    }

    # Check if COMPETITION_ID already exists
    if COMPETITION_ID in hyperparameters_data:
        # If it exists, update only the 'updated_at' timestamp
        crated_at = hyperparameters_data[COMPETITION_ID]['crated_at']
        hyperparameters_data[COMPETITION_ID] = {
            **best_params,
            **add,
            **{"crated_at": crated_at, 'updated_at': now}
        }
    else:
        # If it doesn't exist, add a new entry with 'created_at' and 'updated_at' timestamps
        hyperparameters_data[COMPETITION_ID] = {
            **best_params,
            **add,
            **{"crated_at": now, 'updated_at': now}
        }

    # Sort the dictionary by keys
    sorted_hyperparameters = dict(
        sorted(hyperparameters_data.items(), key=lambda x: int(x[0])))

    # Save the sorted data back to the JSON file
    with open(filename, 'w') as file:
        json.dump(sorted_hyperparameters, file, indent=4)


def get_hyperparameters(COMPETITION_ID, target, outcomes=None):

    has_weights = False
    n_estimators = 100
    min_samples_split = 2
    transformed_dict = {key: 1 for key in outcomes or [0, 1]}
    class_weight = transformed_dict
    min_samples_leaf = 1
    max_features = 'log2'

    try:
        # Load hyperparameters data
        filename = os.path.abspath(
            f"app/train_predictions/tuning/hyperparameters/{target}_hyperparameters.json")

        try:
            with open(filename, 'r') as file:
                hyperparameters_data = parse_json(json.load(file))
        except:
            FileNotFoundError

        # Get the hyperparameters for COMPETITION_ID
        best_params = hyperparameters_data.get(COMPETITION_ID, None)

        hyper_params = best_params
        n_estimators = hyper_params['n_estimators']
        min_samples_split = hyper_params['min_samples_split']
        class_weight = hyper_params['class_weight']
        min_samples_leaf = hyper_params['min_samples_leaf']
        max_features = hyper_params['max_features']
        has_weights = True
    except:
        KeyError

    hyper_params = [
        n_estimators, min_samples_split, class_weight,
        min_samples_leaf, max_features,
    ]

    return [hyper_params, has_weights]


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
