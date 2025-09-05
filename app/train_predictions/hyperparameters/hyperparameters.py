from app.configs.settings import basepath
import numpy as np
import os
import json
from app.matches.update_backend import update_backend
from datetime import datetime


def hyperparameters_array_generator(train_frame, class_weight_counts=14, class_weight_max=2.0, rest_counts=6):
    len_train = len(train_frame)

    # Setting the range for params
    _n_estimators = np.linspace(150, 300, rest_counts)
    class_weight = np.linspace(1.0, class_weight_max, class_weight_counts)
    _min_samples_splits = np.linspace(10, 50, rest_counts)
    _max_depth = np.linspace(
        2, len_train / 10 if len_train > 2 else 2, rest_counts)

    n_estimators = []
    for x in _n_estimators:
        x = int(x)
        n_estimators.append(x)

    min_samples_split = []
    for x in _min_samples_splits:
        x = int(x)
        min_samples_split.append(x)

    max_depth = [None]
    for x in _max_depth:
        x = int(x)
        max_depth.append(x)

    return [n_estimators, min_samples_split, class_weight]

def get_param_grid(model_type, overrides=None):
    if overrides is None:
        overrides = {}

    if model_type in ["RandomForest", "BalancedRandomForestClassifier", "ExtraTrees"]:
        param_grid = {
            "n_estimators": [100, 150, 300],
            "min_samples_split": [10, 20],
            "min_samples_leaf": [1, 3, 5],
            "class_weight": ["balanced", "balanced_subsample"],
            "max_features": [50, 100],
            "max_depth": [15]
        }
    elif model_type in ["GradientBoosting"]:
        param_grid = {
            "n_estimators": [100, 150, 300],
            "max_depth": [3, 5],
            "min_samples_split": [2, 5, 10],
            "learning_rate": [0.05, 0.1],
        }
    elif model_type in ["HistGB"]:
        param_grid = {
            "learning_rate": [0.05, 0.1],
            "max_depth": [3, 5],
        }
    elif model_type == "LogReg":
        param_grid = {
            "C": [0.01, 0.1, 1, 5],
            "solver": ["saga"],
            "max_iter": [8000],
            "class_weight": ["balanced", None],
        }

    else:
        param_grid = {}

    # Merge defaults with overrides
    return {**param_grid, **overrides}

def save_hyperparameters(model_type, compe_data, target, user_token):
    print(f'Saving {target} hyperparameters...')
    id = compe_data['id']
    prediction_type = compe_data['prediction_type']
    best_params = compe_data['best_params']
    train_counts = compe_data['train_counts']
    test_counts = compe_data['test_counts']
    occurrences = compe_data['occurrences']
    predicted = compe_data['predicted']
    scores = compe_data['scores']
    from_date = compe_data['from_date']
    to_date = compe_data['to_date']

    # Load existing hyperparameters data
    directory = os.path.abspath(
        os.path.join(basepath(), f"train_predictions/hyperparameters/saved/{compe_data['prediction_type']}/{model_type}/"))
    os.makedirs(directory, exist_ok=True)

    filename = os.path.abspath(f"{directory}/{target}_hyperparams.json")

    try:
        with open(filename, 'r') as file:
            hyperparameters_data = parse_json(json.load(file))
    except FileNotFoundError:
        hyperparameters_data = {}

    current_datetime = datetime.today()
    now = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    accuracy_score, precision_score, f1_score, average_score = scores
    main_object = {
        'prediction_type': prediction_type,
        'train_counts': train_counts,
        'test_counts': test_counts,
        'occurrences': occurrences,
        'last_predicted': predicted,
        'accuracy_score': accuracy_score,
        'precision_score': precision_score,
        'f1_score': f1_score,
        'average_score': average_score,
        'from_date': from_date,
        'to_date': to_date,
    }

    print("hyperparameters_data",hyperparameters_data)
    print("best_params",best_params)
    check = f"{id}-{compe_data['season_id']}"
    # Check if id already exists
    if check in hyperparameters_data:
        # If it exists, update only the 'updated_at' timestamp
        created_at = hyperparameters_data[check]['created_at']
        hyperparameters_data[check] = {
            **best_params,
            **main_object,
            **{"created_at": created_at, 'updated_at': now}
        }
    else:
        # If it doesn't exist, add a new entry with 'created_at' and 'updated_at' timestamps
        hyperparameters_data[check] = {
            **best_params,
            **main_object,
            **{"created_at": now, 'updated_at': now}
        }

    # Sort the dictionary by keys
    sorted_hyperparameters = dict(
        sorted(
            hyperparameters_data.items(),
            key=lambda x: tuple(map(int, x[0].split('-')))
        )
    )

    # Save the sorted data back to the JSON file
    with open(filename, 'w') as file:
        json.dump(sorted_hyperparameters, file, indent=4)
    
    print(f'Hyperparameters saved!\n')

    update_backend(user_token, id, target, main_object)


def get_hyperparameters(model_type, compe_data, target, outcomes=None):

    has_weights = False
    hyper_params = {}

    try:
        # Load hyperparameters data
        filename = os.path.abspath(
            os.path.join(basepath(), f"train_predictions/hyperparameters/saved/{compe_data['prediction_type']}/{model_type}/{target}_hyperparams.json"))

        try:
            with open(filename, 'r') as file:
                hyperparameters_data = parse_json(json.load(file))
        except:
            FileNotFoundError

        check = f"{id}-{compe_data['season_id']}"
        # Get the hyperparameters for compe id
        best_params = hyperparameters_data.get(check, None)

        hyper_params = best_params
        has_weights = True
    except:
        KeyError

    return [hyper_params, has_weights]


def get_occurrences(model_type, compe_data, target, min_threshold=0.0):
    id = str(compe_data.get('id'))
    try:
        # Load hyperparameters data
        filename = os.path.abspath(
            os.path.join(basepath(), f"train_predictions/hyperparameters/saved/{compe_data['prediction_type']}/{model_type}/{target}_hyperparams.json"))

        try:
            with open(filename, 'r') as file:
                hyperparameters_data = json.load(file)
        except FileNotFoundError:
            hyperparameters_data = {}

        check = f"{id}-{compe_data['season_id']}"

        # Get the hyperparameters for compe id
        best_params = hyperparameters_data.get(check, None)

        occurrences = best_params.get("occurrences", {}) if best_params else {}

        # Filter occurrences based on min_threshold
        filtered_occurrences = {
            key: value for key, value in occurrences.items() if value >= min_threshold}

    except (KeyError, TypeError):
        filtered_occurrences = {}

    return filtered_occurrences


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
