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


def save_hyperparameters(compe_data, target, user_token):
    print('Saving hyperparameters...')
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
    
    filename = os.path.abspath(f"{directory}/{name}_hyperparams.json")

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


def get_hyperparameters(compe_data, target, outcomes=None):

    has_weights = False
    n_estimators = 100
    min_samples_split = 2
    transformed_dict = {key: 1 for key in outcomes or [0, 1]}
    class_weight = transformed_dict
    min_samples_leaf = 1

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

        hyper_params = best_params
        n_estimators = hyper_params['n_estimators']
        min_samples_split = hyper_params['min_samples_split']
        min_samples_leaf = hyper_params['min_samples_leaf']
        class_weight = hyper_params['class_weight']
        has_weights = True
    except:
        KeyError

    hyper_params = {
        'n_estimators': n_estimators,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        'class_weight': class_weight,
    }

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
