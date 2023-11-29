from sklearn.metrics import accuracy_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import ClassifierChain
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from app.train_predictions.hyperparameters.hyperparameters import hyperparameters_array_generator


def grid_search(model, train_frame, FEATURES, targets, occurrences, is_random_search=False):
    target = targets

    print(
        f"SearchCV Strategy: {'Randomized' if is_random_search else 'GridSearch'}")

    n_estimators, min_samples_split, class_weight, min_samples_leaf = hyperparameters_array_generator(
        train_frame, 5, 1.3, 4)

    _class_weight = []
    for i, x in enumerate(class_weight):
        for j in class_weight:
            res = {0: round(class_weight[i], 3), 1: round(j, 3)}
            _class_weight.append(res)

    # filtering based on the fact that our model struggles at making 0 and 1 preds, 1 being the worst
    class_weight = []
    for x in _class_weight:
        if x[0] > 1 and x[0] == x[1]:
            continue
        if x[0] < 2 and x[1] < 2:
            class_weight.append(x)

    # Creating a dictionary grid for grid search
    param_grid = {
        'base_estimator__random_state': [1],
        'base_estimator__n_estimators': n_estimators,
        'base_estimator__min_samples_split': min_samples_split,
        'base_estimator__min_samples_leaf': [3, 5, 7],
        'base_estimator__class_weight': [None],
    }

    # Create a base RandomForestClassifier
    base_classifier = model

    # Create a ClassifierChain
    model = ClassifierChain(base_classifier)

    if not is_random_search:
        # Create the RandomizedSearchCV object
        gridsearch = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=5,
            scoring=lambda estimator, X, y_true: scorer(
                estimator, X, y_true),
            verbose=2,
            n_jobs=-1,
        ).fit(train_frame[FEATURES], train_frame[target])
    else:
        gridsearch = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            n_iter=10,
            cv=5,
            scoring=lambda estimator, X, y_true: scorer(
                estimator, X, y_true),
            random_state=42,
            verbose=3,
        ).fit(train_frame[FEATURES], train_frame[target])

    best_params = gridsearch.best_params_
    best_score = gridsearch.best_score_

    # Extract and print the best parameters and score
    print(f"")
    print(f"Best params: {format_best_params(best_params)}\n")

    print(f"Best score: {best_score}\n")

    return format_best_params(best_params)


def scorer(estimator, X, y_true):
    y_pred = estimator.predict(X)
    totals = len(y_true)

    # Assuming y_true and y_pred are binary
    y_pred_0 = round(np.sum(y_pred[:, 1] == 0) / totals * 100)

    return y_pred_0


def format_best_params(best_params):
    formatted_params = {
        f'{key.replace("base_estimator__", "")}': value if key != 'base_estimator__class_weight' else (value if value == None else {k: v for k, v in value.items()})
        for key, value in best_params.items()
    }

    return formatted_params
