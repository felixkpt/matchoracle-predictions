from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, RandomizedSearchCV
from app.train_predictions.hyperparameters.hyperparameters import hyperparameters_array_generator, get_param_grid
import numpy as np
from app.configs.settings import GRID_SEARCH_N_SPLITS, GRID_SEARCH_VERBOSE, TRAIN_MAX_CORES

# Set a random seed for reproducibility
np.random.seed(42)


def grid_search(model, train_frame, FEATURES, target, occurrences, is_random_search=False, model_type="RandomForest"):
    print(
        f"SearchCV Strategy: {'Randomized' if is_random_search else 'GridSearch'}")

    n_estimators, min_samples_split, class_weight = hyperparameters_array_generator(
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

    # Get dictionary grid for grid search
    param_grid = get_param_grid(model_type, n_estimators, min_samples_split, class_weight=['balanced'], max_feature=[None, 'sqrt'])

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

    return {
        "best_estimator": gridsearch.best_estimator_,   # fitted model
        "best_params": best_params,                     # dict of best hyperparameters
        "best_score": best_score,                       # best CV score
        "cv_results": gridsearch.cv_results_,           # full results from all runs
    }

    # Create a DataFrame to store the grid search results
    # weigh_data = pd.DataFrame({
    #     'score': gridsearch.cv_results_['mean_test_score'],
    #     'weight': [1 - x for x in class_weight]
    # })

    # Set up the plot
    # sns.set_style('whitegrid')
    # plt.figure(figsize=(12, 8))

    # # Create the line plot for class weight vs. F1 score
    # sns.lineplot(x=weigh_data['weight'], y=weigh_data['score'])

    # # Add labels and ticks to the plot
    # plt.xlabel('Weight for class 1')
    # plt.ylabel('F1 score')
    # plt.xticks([round(i / 10, 1) for i in range(0, 11, 1)])
    # plt.title('Scoring for different class weights', fontsize=24)

    # # Show the plot
    # plt.show()


def scorer(estimator, X, y_true, occurrences):
    occurance_outcome_0 = occurrences.get(0, 0)
    occurance_outcome_1 = occurrences.get(1, 0)

    # Assuming estimator is a RandomForestClassifier
    y_pred = estimator.predict(X)

    # Calculate the proportions of predicted occurrences
    totals = len(X)
    y_pred_0 = round(sum(1 for p in y_pred if p == 0) / totals * 100)
    y_pred_1 = round(sum(1 for p in y_pred if p == 1) / totals * 100)

    # Define a threshold for penalization
    max_diff_threshold = 10

    # Penalize if predicted occurrences exceed specific natural occurrences by more than max_diff_threshold
    if y_pred_0 - max_diff_threshold > occurance_outcome_0:
        return 0
    if y_pred_1 - max_diff_threshold > occurance_outcome_1:
        return 0

    # If none of the penalization conditions are met, return a combined score
    # Calculate the absolute differences between predicted and actual occurrences
    diff_outcome_0 = abs(y_pred_0 - occurance_outcome_0)
    diff_outcome_1 = abs(y_pred_1 - occurance_outcome_1)

    # Calculate a similarity score based on the differences
    max_diff = max(diff_outcome_0, diff_outcome_1)
    natural_score = 1 - (max_diff / 100)
    natural_score = natural_score if natural_score > 0 else 0

    # print(f"NATURAL: UNDER: {occurance_outcome_0}, OVER: {occurance_outcome_1}")
    # print(f"PREDICT: UNDER: {y_pred_0}, OVER: {y_pred_1}")
    # print(f"TOTALS: {totals}, Ntrl score: {round(natural_score, 3)}")

    # Calculate other required scorers and combine
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    score = 0.4 * f1 + 0.7 * natural_score

    return score
