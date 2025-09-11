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
        train_frame, 10, 2.0, 4)
    
    overrides= {}
    
    if model_type in ["RandomForest", "BalancedRandomForestClassifier", "ExtraTrees"]:
        overrides= {
            'n_estimators': n_estimators,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': [1, 4],
            'class_weight': ['balanced', 'balanced_subsample'],
            'max_features': ['sqrt', 'log2'],
            'max_depth': [10, 20]
        }

    # Get the default param grid and merge with overrides
    param_grid = get_param_grid(model_type, overrides)

    # Set estimator to run single-threaded
    if hasattr(model, "n_jobs"):
        model.set_params(n_jobs=1)

    grid_search_n_splits = 2 if len(train_frame) < 50 else GRID_SEARCH_N_SPLITS
    
    # Fitting grid search to the train data
    if not is_random_search:
        gridsearch = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=StratifiedKFold(n_splits=grid_search_n_splits),
            verbose=GRID_SEARCH_VERBOSE,
            n_jobs=TRAIN_MAX_CORES,
            scoring=lambda estimator, X, y_true: scorer(
                estimator, X, y_true, occurrences),
        ).fit(train_frame[FEATURES], train_frame[target])
    else:
        gridsearch = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            cv=grid_search_n_splits,
            n_iter=20,
            verbose=GRID_SEARCH_VERBOSE,
            n_jobs=TRAIN_MAX_CORES,
            scoring=lambda estimator, X, y_true: scorer(
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

def scorer_v3(estimator, X, y_true, occurrences):
    y_pred = estimator.predict(X)
    totals = len(X)

    # True distribution percentages
    true_dist = np.array([occurrences.get(i, 0) for i in [0, 1, 2]])
    true_dist = true_dist / true_dist.sum() * 100

    # Predicted distribution percentages
    pred_counts = [sum(p == i for p in y_pred) for i in [0, 1, 2]]
    pred_dist = np.array(pred_counts) / totals * 100

    # Hybrid natural score: median + mean
    median_diff = np.median(np.abs(pred_dist - true_dist))
    mean_diff = np.mean(np.abs(pred_dist - true_dist))
    dist_penalty = 0.7 * median_diff + 0.3 * mean_diff
    natural_score = max(0, 1 - dist_penalty / 100)

    # Standard metrics
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_draw = f1_score(y_true, y_pred, labels=[1], average="macro", zero_division=0)

    # Coverage penalty: missing classes
    missing_classes = sum(
        1 for i in [0, 1, 2] if occurrences.get(i, 0) > 0 and pred_counts[i] == 0
    )
    coverage_penalty = 1.0 - (0.15 * missing_classes)
    coverage_penalty = max(0.5, coverage_penalty)  # never drop below 0.5

    # Weighted combination: slightly increased draw weight for FT
    combined_score = (
        0.25 * f1_macro +
        0.25 * f1_weighted +
        0.20 * f1_draw +      # increase to help FT draws
        0.30 * natural_score
    )

    # Apply coverage penalty
    return combined_score * coverage_penalty

def scorer(estimator, X, y_true, occurrences):
    y_pred = estimator.predict(X)
    totals = len(X)

    # True distribution (percentages)
    true_dist = np.array([occurrences.get(i, 0) for i in [0, 1, 2]])
    true_dist = true_dist / true_dist.sum() * 100

    # Predicted distribution (percentages)
    pred_counts = [sum(p == i for p in y_pred) for i in [0, 1, 2]]
    pred_dist = np.array(pred_counts) / totals * 100

    # Natural score (median absolute difference → smoother than mean/max)
    dist_penalty = np.median(np.abs(pred_dist - true_dist)) / 100
    natural_score = max(0, 1 - dist_penalty)

    # Standard metrics
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_draw = f1_score(y_true, y_pred, labels=[1], average="macro", zero_division=0)

    # Softer coverage penalty
    missing_classes = sum(
        1 for i in [0, 1, 2] if occurrences.get(i, 0) > 0 and pred_counts[i] == 0
    )
    coverage_penalty = 1.0 - (0.15 * missing_classes)
    coverage_penalty = max(0.5, coverage_penalty)  # never drop below 0.5

    # Weighted combination (moderate draw weight: 15%)
    combined_score = (
        0.25 * f1_macro +
        0.25 * f1_weighted +
        0.15 * f1_draw +
        0.35 * natural_score
    )

    return combined_score * coverage_penalty

def scorer_v2(estimator, X, y_true, occurrences):
    y_pred = estimator.predict(X)
    totals = len(X)

    # True distribution (percentages)
    true_dist = np.array([occurrences.get(i, 0) for i in [0, 1, 2]])
    true_dist = true_dist / true_dist.sum() * 100

    # Predicted distribution (percentages)
    pred_counts = [sum(p == i for p in y_pred) for i in [0, 1, 2]]
    pred_dist = np.array(pred_counts) / totals * 100

    # Distribution penalty (mean absolute difference across all classes)
    dist_penalty = np.mean(np.abs(pred_dist - true_dist)) / 100
    natural_score = max(0, 1 - dist_penalty)

    # Standard metrics
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_draw = f1_score(y_true, y_pred, labels=[1], average="macro", zero_division=0)

    # Coverage penalty: punish ignoring existing classes
    coverage_penalty = 1.0
    for i in [0, 1, 2]:
        if occurrences.get(i, 0) > 0 and pred_counts[i] == 0:
            coverage_penalty -= 0.3

    coverage_penalty = max(0, coverage_penalty)

    # Weighted combination
    combined_score = (
        0.25 * f1_macro +
        0.20 * f1_weighted +
        0.20 * f1_draw +      # increased weight on draws
        0.35 * natural_score
    )

    return combined_score * coverage_penalty

def scorer_v2(estimator, X, y_true, occurrences):
    y_pred = estimator.predict(X)

    totals = len(X)
    occ_0, occ_1, occ_2 = [occurrences.get(i, 0) for i in [0, 1, 2]]
    pred_counts = [sum(p == i for p in y_pred) for i in [0, 1, 2]]
    pred_perc = [round(c / totals * 100) for c in pred_counts]

    # Natural score = how close predicted distribution is to actual
    max_diff = max(abs(pred_perc[i] - (occurrences.get(i, 0))) for i in [0, 1, 2])
    natural_score = max(0, 1 - (max_diff / 100))

    # Standard metrics
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_draw = f1_score(y_true, y_pred, labels=[1], average="macro", zero_division=0)

    # Coverage penalty: if model never predicts a class that exists in y_true
    coverage_penalty = 1.0
    for i in [0, 1, 2]:
        if occurrences.get(i, 0) > 0 and pred_counts[i] == 0:
            coverage_penalty -= 0.3  # subtract penalty for each ignored class

    coverage_penalty = max(0, coverage_penalty)  # don’t go negative

    # Weighted combination
    combined_score = (
        0.25 * f1_macro +
        0.25 * f1_weighted +
        0.10 * f1_draw +
        0.40 * natural_score
    )

    # Apply penalty
    combined_score *= coverage_penalty

    return combined_score
