from app.helpers.functions import combined_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, StratifiedKFold, RandomizedSearchCV
from app.train_predictions.hyperparameters.hyperparameters import hyperparameters_array_generator, best_parms_to_fractions
import numpy as np
from itertools import product

# Set a random seed for reproducibility
np.random.seed(42)


def recursive(class_weight, outcomes, j):
    res = {}

    try:
        key = outcomes[j+1]

        for k in class_weight:
            _res = {key: round(k, 3)}
            res = {**res, **_res}
            return recursive(class_weight, outcomes, k)

    except:
        KeyError
        return res


def grid_search(model, train_frame, FEATURES, target, occurrences, is_random_search=False, score_weights=None):
    if not score_weights:
        score_weights = [0, 0.4, 0.3, 0.3]

    print(
        f"SearchCV Strategy: {'Randomized' if is_random_search else 'GridSearch'}")

    hyper_params = hyperparameters_array_generator(
        train_frame, 4, 1.3, 4)
    n_estimators = hyper_params['n_estimators']
    min_samples_split = hyper_params['min_samples_split']
    class_weight = hyper_params['class_weight']
    min_samples_leaf = hyper_params['min_samples_leaf']
    max_depth = hyper_params['max_depth']

    outcomes = train_frame[target].unique()

    print(len(outcomes))

    # Count the occurrences of each class in cs_target
    class_counts = train_frame['cs_target'].value_counts()
    # Set a threshold for the minimum number of instances per class
    min_instances_threshold = 5

    # Identify classes with counts below the threshold
    classes_to_drop = class_counts[class_counts <
                                   min_instances_threshold].index

    # Filter out instances with classes to drop
    filtered_train_frame = train_frame[~train_frame['cs_target'].isin(
        classes_to_drop)]

    class_weight = np.linspace(1.0, 1.2, 2)
    __class_weight = []

    for combination in product(class_weight, repeat=int(2)):
        res = {key: round(x, 3) for key, x in zip(outcomes, combination)}
        __class_weight.append(res)

    class_weight = __class_weight

    # Creating a dictionary grid for grid search
    param_grid = {
        'random_state': [1],
        'criterion': ['gini'],
        'max_depth': [None],
        'n_estimators': n_estimators,
        'min_samples_split': min_samples_split,
        'class_weight': ['balanced'],
        'min_samples_leaf': min_samples_leaf,
        'max_leaf_nodes': [None],
        'max_features': [None],
        'bootstrap': [True],
    }

    n_splits=5
    # Fitting grid search to the train data
    if not is_random_search:
        gridsearch = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=StratifiedKFold(n_splits),
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
    # Assuming estimator is a RandomForestClassifier
    y_pred = estimator.predict(X)
    natural_score = 0

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
