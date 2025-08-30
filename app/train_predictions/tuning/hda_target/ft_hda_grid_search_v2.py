from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, make_scorer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, StratifiedKFold, RandomizedSearchCV
from app.train_predictions.hyperparameters.hyperparameters import hyperparameters_array_generator, get_param_grid
import numpy as np
from app.configs.settings import GRID_SEARCH_N_SPLITS, GRID_SEARCH_VERBOSE, TRAIN_MAX_CORES

# Set a random seed for reproducibility
np.random.seed(42)

# Custom scorer function to optimize for draw predictions (class 1)
def draw_precision_scorer(y_true, y_pred):
    """Custom scorer that focuses on precision for draw class (class 1)"""
    return precision_score(y_true, y_pred, labels=[1], average='macro', zero_division=0)

def draw_recall_scorer(y_true, y_pred):
    """Custom scorer that focuses on recall for draw class (class 1)"""
    return recall_score(y_true, y_pred, labels=[1], average='macro', zero_division=0)

def draw_f1_scorer(y_true, y_pred):
    """Custom scorer that focuses on F1 score for draw class (class 1)"""
    return f1_score(y_true, y_pred, labels=[1], average='macro', zero_division=0)

def grid_search(model, train_frame, FEATURES, target, occurrences, is_random_search=False, model_type="RandomForest"):
    print(
        f"SearchCV Strategy: {'Randomized' if is_random_search else 'GridSearch'}")

    n_estimators, min_samples_split, class_weight = hyperparameters_array_generator(
        train_frame, 10, 2.0, 4)
    
    # Get dictionary grid for grid search
    param_grid = get_param_grid(model_type, n_estimators, min_samples_split, class_weight=['balanced', 'balanced_subsample'], max_feature=[None, 'sqrt'])

    grid_search_n_splits = 2 if len(train_frame) < 50 else GRID_SEARCH_N_SPLITS
    
    # Create custom scorer for draw optimization
    draw_scorer = make_scorer(draw_f1_scorer)  # Using F1 for balanced precision/recall
    
    # Fitting grid search to the train data
    if not is_random_search:
        gridsearch = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=StratifiedKFold(n_splits=grid_search_n_splits),
            verbose=GRID_SEARCH_VERBOSE,
            n_jobs=TRAIN_MAX_CORES,
            scoring=draw_scorer  # Use custom scorer for draw optimization
        ).fit(train_frame[FEATURES], train_frame[target])
    else:
        gridsearch = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            n_iter=10,
            cv=grid_search_n_splits,
            verbose=GRID_SEARCH_VERBOSE,
            n_jobs=TRAIN_MAX_CORES,
            scoring=draw_scorer  # Use custom scorer for draw optimization
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