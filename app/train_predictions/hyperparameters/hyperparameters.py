import numpy as np
import os
import json
from app.matches.update_backend import update_backend
from datetime import datetime


def hyperparameters_array_generator(train_frame, class_weight_counts=14, class_weight_max=2.0, rest_counts=6):
    len_train = len(train_frame)

    # Setting the range for params
    _n_estimators = np.linspace(150, 250, rest_counts)
    class_weight = np.linspace(1.0, class_weight_max, class_weight_counts)
    _min_samples_splits = np.linspace(2, 6, rest_counts)
    _min_samples_leaf = np.linspace(
        2, 6 if len_train > 2 else 2, rest_counts)
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

    min_samples_leaf = []
    for x in _min_samples_leaf:
        x = int(x)
        min_samples_leaf.append(x)

    max_depth = [None]
    for x in _max_depth:
        x = int(x)
        max_depth.append(x)

    return {
        'n_estimators': [150],
        'min_samples_split': min_samples_split,
        'class_weight': class_weight,
        'min_samples_leaf': min_samples_leaf,
        'max_depth': max_depth,
    }


def save_hyperparameters(compe_data, target, user_token):
    print('Saving hyperparameters...\n')
    print('-----------\n')
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
        f"app/train_predictions/hyperparameters/{compe_data['prediction_type']}/")
    os.makedirs(directory, exist_ok=True)

    name = target[0]+'_multiple' if type(target) == list else target

    filename = os.path.abspath(
        f"{directory}/{name}{'_normalizer' if 'is_normalizer' in compe_data and compe_data['is_normalizer'] else ''}_hyperparams.json")

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

    # Check if id already exists
    if id in hyperparameters_data:
        # If it exists, update only the 'updated_at' timestamp
        created_at = hyperparameters_data[id]['created_at']
        hyperparameters_data[id] = {
            **best_params,
            **main_object,
            **{"created_at": created_at, 'updated_at': now}
        }
    else:
        # If it doesn't exist, add a new entry with 'created_at' and 'updated_at' timestamps
        hyperparameters_data[id] = {
            **best_params,
            **main_object,
            **{"created_at": now, 'updated_at': now}
        }

    # Sort the dictionary by keys
    sorted_hyperparameters = dict(
        sorted(hyperparameters_data.items(), key=lambda x: int(x[0])))

    # Save the sorted data back to the JSON file
    with open(filename, 'w') as file:
        json.dump(sorted_hyperparameters, file, indent=4)

    update_backend(user_token, id, target, main_object)


def get_hyperparameters(compe_data, target, is_normalizer=False):

    has_weights = False
    hyper_params = {
        'random_state': 1,
        'criterion': 'gini',
        'max_depth': None,
        'n_estimators': 150,
        'min_samples_split': 20,
        'class_weight': 'balanced',
        'min_samples_leaf': 4,
        'max_leaf_nodes': None,
        'max_features': None,
        'bootstrap': False,
    }

    name = target[0]+'_multiple' if type(target) == list else target

    try:
        # Load hyperparameters data
        filename = os.path.abspath(
            f"app/train_predictions/hyperparameters/{compe_data['prediction_type']}/{name}{'_normalizer' if is_normalizer else ''}_hyperparams.json")

        try:
            with open(filename, 'r') as file:
                hyperparameters_data = parse_json(json.load(file))
        except:
            FileNotFoundError

        # Get the hyperparameters for compe id
        best_params = hyperparameters_data.get(compe_data['id'], None)

        has_weights = True
        for key in best_params:
            if key in hyper_params:
                hyper_params[key] = best_params[key]

    except:
        KeyError

    return [hyper_params, has_weights]


def get_occurrences(compe_data, target):

    occurrences = []
    name = target[0]+'_multiple' if type(target) == list else target

    try:
        # Load hyperparameters data
        filename = os.path.abspath(
            f"app/train_predictions/hyperparameters/{compe_data['prediction_type']}/{name}_hyperparams.json")

        try:
            with open(filename, 'r') as file:
                hyperparameters_data = parse_json(json.load(file))
        except:
            FileNotFoundError

        # Get the hyperparameters for compe id
        best_params = hyperparameters_data.get(compe_data['id'], None)

        occurrences = best_params['occurrences']

    except:
        KeyError

    return occurrences


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


def best_parms_to_fractions(best_params, train_frame):

    if 'min_samples_split' in best_params:
        min_samples_split = best_params['min_samples_split'] / len(train_frame)
        best_params['min_samples_split'] = round(min_samples_split, 3)

    if 'min_samples_leaf' in best_params:
        min_samples_leaf = best_params['min_samples_leaf'] / len(train_frame)
        best_params['min_samples_leaf'] = round(min_samples_leaf, 3)

    if 'n_estimators' in best_params:
        n_estimators = best_params['n_estimators'] / len(train_frame)
        best_params['n_estimators_fraction'] = round(n_estimators, 3)

    return best_params
