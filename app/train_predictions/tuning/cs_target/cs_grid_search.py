from sklearn.metrics import f1_score, precision_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, StratifiedKFold, RandomizedSearchCV
from app.train_predictions.hyperparameters.hyperparameters import hyperparameters_array_generator, get_param_grid
import numpy as np
from itertools import product
from app.configs.settings import GRID_SEARCH_N_SPLITS, GRID_SEARCH_VERBOSE, TRAIN_MAX_CORES

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


def grid_search(model, train_frame, FEATURES, target, occurrences, is_random_search=False, model_type="RandomForest"):
    print(
        f"SearchCV Strategy: {'Randomized' if is_random_search else 'GridSearch'}")
    
    n_estimators, min_samples_split, class_weight = hyperparameters_array_generator(
        train_frame, 4, 1.3, 4)

    outcomes = train_frame[target].unique()

    print(len(outcomes))

    class_weight = np.linspace(1.0, 1.2, 2)
    __class_weight = []

    for combination in product(class_weight, repeat=int(2)):
        res = {key: round(x, 3) for key, x in zip(outcomes, combination)}
        __class_weight.append(res)

    class_weight = __class_weight

    overrides = {}
    
    if model_type in ["RandomForest", "BalancedRandomForestClassifier", "ExtraTrees"]:
        overrides = {
            'n_estimators': n_estimators,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': [1, 4],
            'class_weight': [None],
            'max_features': ['sqrt', 'log2', None],
            'max_depth': [5, 10, 20, None]
        }
    elif model_type in ["GradientBoosting"]:
        overrides = {
            "n_estimators": [250],
            'max_depth': [10, 20, None],
            "min_samples_split": [2, 5],
            "learning_rate": [0.5],
        }
    elif model_type in ["HistGB"]:
        overrides = {
            "learning_rate": [0.1],
        }
    elif model_type == "LogReg":
        overrides = {
            "C": [1, 10],
            "max_iter": [8000],
            "class_weight": [None],
        }

    # Get the default param grid and merge with overrides
    param_grid = get_param_grid(model_type, overrides)

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
            cv=grid_search_n_splits,
            n_iter=20,
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
    y_pred = estimator.predict(X)
    
    totals = len(X)
    class_diffs = []
    
    # Loop over all classes in natural occurrences
    for cls, nat_pct in occurrences.items():
        pred_pct = sum(y_pred == cls) / totals * 100
        diff = abs(pred_pct - nat_pct)
        class_diffs.append(diff)
    
    # Use the maximum deviation or mean deviation
    max_diff = max(class_diffs)
    mean_diff = np.mean(class_diffs)
    
    # Convert to a normalized natural adherence score [0,1]
    adherence_score = 1 - (1.5 * max_diff / 100)
    adherence_score = max(0, adherence_score)  # clip
    
    # F1 score (macro to treat all classes equally)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    # Combine F1 and adherence
    combined_score = 0.5 * f1 + 0.5 * adherence_score
    
    return combined_score
