from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
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

    # Get dictionary grid for grid search
    param_grid = get_param_grid(model_type, n_estimators, min_samples_split, class_weight=['balanced', 'balanced_subsample'], max_feature=[None, 'sqrt'])

    grid_search_n_splits = 2 if len(train_frame) < 50 else GRID_SEARCH_N_SPLITS
    # Fitting grid search to the train data
    if not is_random_search:
        gridsearch = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=StratifiedKFold(n_splits=grid_search_n_splits),
            scoring=lambda estimator, X, y_true: scorer_v1(
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
            scoring=lambda estimator, X, y_true: scorer_v1(
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

def scorer_v4(estimator, X, y_true, occurrences):
    """
    Improved scorer that prioritizes draw prediction accuracy
    """
    y_pred = estimator.predict(X)
    totals = len(X)
    
    # Get actual and predicted distributions
    actual_dist = [occurrences.get(i, 0) for i in [0, 1, 2]]
    pred_counts = [sum(p == i for p in y_pred) for i in [0, 1, 2]]
    pred_dist = [round(count / totals * 100) if totals > 0 else 0 for count in pred_counts]
    
    # Distribution similarity
    dist_diffs = [abs(pred_dist[i] - actual_dist[i]) for i in range(3)]
    dist_penalty = sum(dist_diffs) / 300  # Normalized penalty
    dist_score = max(0, 1 - dist_penalty * 1.5)
    
    # Class-specific metrics
    try:
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    except:
        # Fallback if not all classes are present
        precision_per_class = [0, 0, 0]
        recall_per_class = [0, 0, 0]
        f1_per_class = [0, 0, 0]
    
    # Extract draw-specific metrics
    draw_precision = precision_per_class[1] if len(precision_per_class) > 1 else 0
    draw_recall = recall_per_class[1] if len(recall_per_class) > 1 else 0
    draw_f1 = f1_per_class[1] if len(f1_per_class) > 1 else 0
    
    # Standard metrics
    precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Combined score with heavy emphasis on draw performance
    combined_score = (
        0.10 * precision_weighted +
        0.08 * recall_weighted +
        0.07 * f1_weighted +
        0.20 * draw_precision +
        0.25 * draw_recall +  # Highest weight for draw recall
        0.10 * draw_f1 +
        0.10 * dist_score +
        0.10 * accuracy_score(y_true, y_pred)
    )
    
    return combined_score

def scorer_v3(estimator, X, y_true, occurrences):
    y_pred = estimator.predict(X)
    totals = len(X)

    # --- 1. Distribution similarity ---
    occ = [occurrences.get(i, 0) for i in [0, 1, 2]]
    pred = [
        round(sum(p == i for p in y_pred) / totals * 100) for i in [0, 1, 2]
    ]
    dist_diffs = [abs(pred[i] - occ[i]) for i in range(3)]
    dist_penalty = sum(dist_diffs) / (3 * 100)
    dist_score = max(0, 1 - dist_penalty * 2.0)  # harsher penalty

    # --- 2. Standard metrics ---
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')

    # --- 3. Class-specific (with focus on draws) ---
    f1_home = f1_score(y_true, y_pred, labels=[0], average='macro', zero_division=0)
    f1_draw = f1_score(y_true, y_pred, labels=[1], average='macro', zero_division=0)
    f1_away = f1_score(y_true, y_pred, labels=[2], average='macro', zero_division=0)

    # Balance all classes, but give draws slightly higher emphasis
    per_class_score = (0.45 * f1_home + 0.35 * f1_draw + 0.2 * f1_away)

    # --- 4. Final weighted combo ---
    combined_score = (
        0.15 * precision +
        0.20 * f1_macro +
        0.15 * f1_weighted +
        0.25 * per_class_score +   # explicit per-class balancing
        0.25 * dist_score          # prevent collapse
    )
    return combined_score

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
    f1_draw = f1_score(y_true, y_pred, labels=[1], average='macro')

    # Weighted combination
    combined_score = (
        0.25 * precision +
        0.25 * natural_score +
        0.20 * f1_macro +
        0.20 * f1_weighted +
        0.10 * f1_draw
    )
    return combined_score

def scorer_v1(estimator, X, y_true, occurrences):
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

    # print(f"NATURAL: 0: {occurance_outcome_0}, 1: {occurance_outcome_1}, 2: {occurance_outcome_2}")
    # print(f"PREDICT: 0: {y_pred_0}, 1: {y_pred_1}, 2: {y_pred_2}")
    # print(f"TOTALS: {totals}, Ntrl score: {round(natural_score, 3)}")

    # Calculate other required scorers and combine
    precision = precision_score(
        y_true, y_pred, average='weighted', zero_division=0)

    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    combined_score = 0.3 * precision + 0.4 * f1 + 0.3 * natural_score

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