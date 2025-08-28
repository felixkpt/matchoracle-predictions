from sklearn.metrics import f1_score, precision_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, StratifiedKFold, RandomizedSearchCV
from app.train_predictions.hyperparameters.hyperparameters import hyperparameters_array_generator
import numpy as np
from app.configs.settings import GRID_SEARCH_N_SPLITS, GRID_SEARCH_VERBOSE, TRAIN_MAX_CORES

# Set a random seed for reproducibility
np.random.seed(42)


def grid_search(model, train_frame, FEATURES, target, occurrences, is_random_search=False):
    print(
        f"SearchCV Strategy: {'Randomized' if is_random_search else 'GridSearch'}")

    n_estimators, min_samples_split, class_weight = hyperparameters_array_generator(
        train_frame, 10, 2.0, 4)

    _class_weight = []
    for i, x in enumerate(class_weight):

        for j in class_weight:
            for k in class_weight:
                res = {0: round(class_weight[i], 3),
                       1: round(j, 3), 2: round(k, 3)}
                _class_weight.append(res)

    # filtering based on the fact that our model struggles at making 1 and 2 preds, 1 being the worst
    class_weight = []
    for x in _class_weight:
        if x[0] < 1.3 and x[1] > 1.4 and x[2] < 1.7:
            class_weight.append(x)

    # Creating a dictionary grid for grid search
    param_grid = {
        'random_state': [1],
        'n_estimators': n_estimators,
        'min_samples_split': min_samples_split,
        'class_weight': class_weight + ['balanced'],
        'min_samples_leaf': [4, 7],
        'max_features': [None]
    }

    grid_search_n_splits = 2 if len(train_frame) < 50 else GRID_SEARCH_N_SPLITS
    # Fitting grid search to the train data
    if not is_random_search:
        gridsearch = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=StratifiedKFold(n_splits=grid_search_n_splits),
            scoring=lambda estimator, X, y_true: scorer(
                estimator, X, y_true, occurrences),
            verbose=GRID_SEARCH_VERBOSE,
            n_jobs=TRAIN_MAX_CORES,
        ).fit(train_frame[FEATURES], train_frame[target])
    else:
        gridsearch = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            n_iter=10,
            cv=grid_search_n_splits,
            scoring=lambda estimator, X, y_true: scorer(
                estimator, X, y_true, occurrences),
            random_state=42,
            verbose=GRID_SEARCH_VERBOSE,
            n_jobs=TRAIN_MAX_CORES,
        ).fit(train_frame[FEATURES], train_frame[target])

    # Extract and print the best class weight and score
    best_params = gridsearch.best_params_
    best_score = gridsearch.best_score_
    print(f"Best params: {best_params}")
    print(f"Best score: {best_score}")

    # Create a DataFrame to store the grid search results
    # weigh_data = pd.DataFrame({
    #     'score': gridsearch.cv_results_['mean_test_score'],
    #     'weight_0': [x[0] for x in class_weight],
    #     'weight_1': [x[1] for x in class_weight],
    #     'weight_2': [x[2] for x in class_weight]
    # })

    # Call the plotting method
    # plot_grid_search_results(weigh_data)

    return best_params


def scorer(estimator, X, y_true, occurrences):
    y_pred = estimator.predict(X)

    # Natural score = how close predicted class distribution is to actual
    totals = len(X)
    occ_0, occ_1, occ_2 = [occurrences.get(i, 0) for i in [0, 1, 2]]
    pred_0 = round(sum(p == 0 for p in y_pred) / totals * 100)
    pred_1 = round(sum(p == 1 for p in y_pred) / totals * 100)
    pred_2 = round(sum(p == 2 for p in y_pred) / totals * 100)

    max_diff = max(abs(pred_0 - occ_0), abs(pred_1 - occ_1), abs(pred_2 - occ_2))
    natural_score = max(0, 1 - (1.5 * max_diff / 100))

    # Standard metrics
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    f1_draw = f1_score(y_true, y_pred, labels=[1], average='macro')

    # Weighted combination
    combined_score = (
        0.25 * precision +
        0.25 * natural_score +
        0.2 * f1_macro +
        0.15 * f1_weighted +
        0.15 * f1_draw
    )
    return combined_score

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
