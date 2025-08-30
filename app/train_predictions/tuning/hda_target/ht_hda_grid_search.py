from sklearn.metrics import f1_score, precision_score, recall_score, make_scorer
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
        train_frame, 10, 2.0, 4)

    # Get dictionary grid for grid search
    param_grid = get_param_grid(model_type, n_estimators, min_samples_split, class_weight=['balanced', 'balanced_subsample'], max_feature=[None, 'sqrt'])

    grid_search_n_splits = 2 if len(train_frame) < 50 else GRID_SEARCH_N_SPLITS
        
    # Fitting grid search to the train data
    if not is_random_search:
        gridsearch = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=StratifiedKFold(n_splits=grid_search_n_splits),
            verbose=GRID_SEARCH_VERBOSE,
            n_jobs=TRAIN_MAX_CORES,
            scoring=lambda estimator, X, y_true: scorer_v2(
                estimator, X, y_true, occurrences),
        ).fit(train_frame[FEATURES], train_frame[target])
    else:
        gridsearch = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            n_iter=10,
            cv=grid_search_n_splits,
            verbose=GRID_SEARCH_VERBOSE,
            n_jobs=TRAIN_MAX_CORES,
            scoring=lambda estimator, X, y_true: scorer_v2(
                estimator, X, y_true, occurrences),
        ).fit(train_frame[FEATURES], train_frame[target])

    # Extract and print the best class weight and score
    best_params = gridsearch.best_params_
    best_score = gridsearch.best_score_
    print(f"Best params: {best_params}")
    print(f"Best draw F1 score: {best_score:.4f}")

    return {
        "best_estimator": gridsearch.best_estimator_,   # fitted model
        "best_params": best_params,                     # dict of best hyperparameters
        "best_score": best_score,                       # best CV score
        "cv_results": gridsearch.cv_results_,           # full results from all runs
    }

def scorer_v2(estimator, X, y_true, occurrences):
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
    rec_score = recall_score(y_true, y_pred, labels=[1], average='macro', zero_division=0)
    f1_draw = f1_score(y_true, y_pred, labels=[1], average='macro')

    # Weighted combination
    combined_score = (
        0.20 * precision +
        0.20 * natural_score +
        0.25 * f1_macro +
        0.15 * f1_weighted +
        0.15 * rec_score +
        0.05 * f1_draw   # keep small, not dominant
    )
    return combined_score
