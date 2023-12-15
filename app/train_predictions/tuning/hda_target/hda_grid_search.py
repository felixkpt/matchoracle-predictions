from app.helpers.functions import combined_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, StratifiedKFold, RandomizedSearchCV
from app.train_predictions.hyperparameters.hyperparameters import hyperparameters_array_generator, best_parms_to_fractions

import numpy as np

# Set a random seed for reproducibility
np.random.seed(42)

# Function for conducting grid search on a model


def grid_search(model, train_frame, FEATURES, target, occurrences, is_random_search=False, score_weights=None):
    if not score_weights:
        score_weights = [0, 0.4, 0.3, 0.3]

    print(
        f"SearchCV Strategy: {'Randomized' if is_random_search else 'GridSearch'}")

    # Generate hyperparameters array for the grid search
    hyper_params = hyperparameters_array_generator(
        train_frame, 10, 2.0, 5)
    n_estimators = hyper_params['n_estimators']
    min_samples_split = hyper_params['min_samples_split']
    class_weight = hyper_params['class_weight']
    min_samples_leaf = hyper_params['min_samples_leaf']
    max_depth = hyper_params['max_depth']

    # Create class_weight combinations
    _class_weight = []
    for i, x in enumerate(class_weight):
        for j in class_weight:
            for k in class_weight:
                res = {0: round(class_weight[i], 3),
                       1: round(j, 3), 2: round(k, 3)}
                _class_weight.append(res)

    # Filtering class_weight based on specified conditions
    class_weight = []
    for x in _class_weight:
        if x[0] < 1.3 and x[1] > 1.6 and x[2] < 1.7:
            class_weight.append(x)

    # Define the grid for grid search
    param_grid = {
        'random_state': [1],
        'criterion': ['gini'],
        'max_depth': [None],
        'n_estimators': [200],
        'min_samples_split': min_samples_split,
        'class_weight': ['balanced'],
        'min_samples_leaf': min_samples_leaf,
        'max_leaf_nodes': [None],
        'max_features': ['sqrt'],
        'bootstrap': [True],
    }

    # Fitting grid search to the train data
    n_splits = 3
    if not is_random_search:
        gridsearch = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=StratifiedKFold(n_splits=n_splits),
            scoring=lambda estimator, X, y_true: scorer(
                estimator, X, y_true, occurrences, score_weights),
            verbose=3,
            n_jobs=-1,
        ).fit(train_frame[FEATURES], train_frame[target])
    else:
        gridsearch = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            n_iter=10,
            cv=n_splits,
            scoring=lambda estimator, X, y_true: scorer(
                estimator, X, y_true, occurrences, score_weights),
            random_state=42,
            verbose=3,
        ).fit(train_frame[FEATURES], train_frame[target])

    # Extract and print the best class weight and score
    best_params = gridsearch.best_params_
    best_score = gridsearch.best_score_
    print(f"Best params: {best_params}")
    print(f"Best score: {best_score}")

    return {
        'best_params': best_parms_to_fractions(best_params, train_frame),
        'best_score': best_score,
        'score_weights': score_weights,
    }

# Custom scorer function for the grid search


def scorer(estimator, X, y_true, occurrences, score_weights):
    occurance_outcome_0 = occurrences.get(0, 0)
    occurance_outcome_1 = occurrences.get(1, 0)
    occurance_outcome_2 = occurrences.get(2, 0)

    # Assuming estimator is a RandomForestClassifier
    y_pred = estimator.predict(X)

    # Calculate the proportions of predicted occurrences
    totals = len(X)
    y_pred_0 = round(sum(1 for p in y_pred if p == 0) / totals * 100)
    y_pred_1 = round(sum(1 for p in y_pred if p == 1) / totals * 100)
    y_pred_2 = round(sum(1 for p in y_pred if p == 2) / totals * 100)

    # If none of the penalization conditions are met, return a combined score
    # Calculate the absolute differences between predicted and actual occurrences
    diff_outcome_0 = abs(y_pred_0 - occurance_outcome_0)
    diff_outcome_1 = abs(y_pred_1 - occurance_outcome_1)
    diff_outcome_2 = abs(y_pred_2 - occurance_outcome_2)

    # Calculate a similarity score based on the differences
    max_diff = max(diff_outcome_0, diff_outcome_1, diff_outcome_2)
    natural_score = 1 - (1.5 * max_diff / 100)  # Normalize to [0, 1]
    natural_score = natural_score if natural_score >= 0 else 0

    combined = combined_score(y_true, y_pred, score_weights, natural_score)

    return combined


def plot_grid_search_results(weigh_data):
    # Set up the plot
    sns.set_style('whitegrid')
    plt.figure(figsize=(12, 8))

    # Create a 3D plot for class weights vs. F1 score
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(weigh_data['weight_0'], weigh_data['weight_1'],
               weigh_data['weight_2'], c=weigh_data['score'], cmap='viridis', s=100)

    # Add labels and ticks to the plot
    ax.set_xlabel('Weight for class 0')
    ax.set_ylabel('Weight for class 1')
    ax.set_zlabel('Weight for class 2')
    ax.set_title('Scoring for different class weights', fontsize=24)

    # Show the plot
    plt.show()
