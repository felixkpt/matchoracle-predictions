import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from app.configs.logger import Logger
from app.helpers.functions import get_features, feature_importance
from app.train_predictions.hyperparameters.hyperparameters import get_hyperparameters
from app.helpers.print_results import print_preds_hyperparams

# Import grid_search and normalizer
# Grid search imports
from app.train_predictions.tuning.hda_target.ft_hda_grid_search import grid_search as ft_hda_grid_search
from app.train_predictions.tuning.hda_target.ht_hda_grid_search import grid_search as ht_hda_grid_search
from app.train_predictions.tuning.bts_target.bts_grid_search import grid_search as bts_grid_search
from app.train_predictions.tuning.over15_target.over15_grid_search import grid_search as over15_grid_search
from app.train_predictions.tuning.over25_target.over25_grid_search import grid_search as over25_grid_search
from app.train_predictions.tuning.over35_target.over35_grid_search import grid_search as over35_grid_search
from app.train_predictions.tuning.cs_target.cs_grid_search import grid_search as cs_grid_search
# Normalizer imports
from app.predictions_normalizers.hda_normalizer import normalizer as hda_normalizer
from app.predictions_normalizers.bts_normalizer import normalizer as bts_normalizer
from app.predictions_normalizers.over_normalizer import normalizer as over_normalizer
from app.predictions_normalizers.cs_normalizer import normalizer as cs_normalizer

# Set a random seed for reproducibility
np.random.seed(42)


def train_predictions(train_frame, test_frame, compe_data, target, outcomes, occurrences, is_grid_search=False, is_random_search=False, model_type="LogReg"):

    features, has_features = get_features(model_type, compe_data, target)
    FEATURES = features
    print(f"Has filtered features: {'Yes' if has_features else 'No'}")


    if occurrences[0] == 0:
        print(f'No results in this category.')
        return None

    # Select the appropriate class weight dictionary based on the target
    hyper_params, has_weights = get_hyperparameters(model_type, compe_data, target, outcomes)

    if model_type == "RandomForest":
        model = RandomForestClassifier()
    elif model_type == "BalancedRandomForestClassifier":
        model = BalancedRandomForestClassifier()
    elif model_type == "ExtraTrees":
        model = ExtraTreesClassifier()
    elif model_type == "GradientBoosting":
        model = GradientBoostingClassifier()
    elif model_type == "HistGB":
        model = HistGradientBoostingClassifier()
    elif model_type == "LogReg":
        model = LogisticRegression()
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    best_params = None
     
    if is_grid_search or not has_weights:
        is_grid_search = True
        if target == 'ft_hda_target':
            grid_search_function = ft_hda_grid_search
        elif target == 'ht_hda_target':
            grid_search_function = ht_hda_grid_search
        elif target == 'bts_target':
            grid_search_function = bts_grid_search
        elif target == 'over15_target':
            grid_search_function = over15_grid_search
        elif target == 'over25_target':
            grid_search_function = over25_grid_search
        elif target == 'over35_target':
            grid_search_function = over35_grid_search
        elif target == 'cs_target':
            grid_search_function = cs_grid_search
        else:
            grid_search_function = None  # Default grid search function

        # Start timing
        start_time = time.time()

        # should handle case when is_grid_search is True and grid_search_function is missing
        if grid_search_function:
            print('Model type: ', model_type)
            grid_result = grid_search_function(
                model, train_frame, FEATURES, target, occurrences, is_random_search, model_type)

            best_params = grid_result["best_params"]
            hyper_params = best_params

            # Apply only valid hyperparameters if you want to reuse model
            hyper_params = {k: v for k, v in best_params.items() if k in model.get_params()}
            model.set_params(**hyper_params)

        log_elapsed_time(start_time, "Grid Search")

    Logger.info(
        f"Hyper Params {'(default)' if not has_weights else ''}: {hyper_params}\n")
    
    if len(FEATURES) == 0: return f'No features for {target}'
    
    # First fit on all features
    model.fit(train_frame[FEATURES], train_frame[target])

    # Select top features
    FEATURES = feature_importance(model, model_type, compe_data, target, FEATURES, False, 0.008)

    # Refit with reduced features
    model.fit(train_frame[FEATURES], train_frame[target])
    
    # Make predictions on the test data
    preds = model.predict(test_frame[FEATURES])
    predict_proba = model.predict_proba(test_frame[FEATURES])

    # Choose normalizer based on target
    if target == 'ft_hda_target' or target == 'ht_hda_target':
        normalizer_function = hda_normalizer
    elif target == 'bts_target':
        normalizer_function = bts_normalizer
    elif target == 'over15_target':
        normalizer_function = over_normalizer
    elif target == 'over25_target':
        normalizer_function = over_normalizer
    elif target == 'over35_target':
        normalizer_function = over_normalizer
    elif target == 'cs_target':
        normalizer_function = cs_normalizer
    else:
        normalizer_function = None  # Default normalizer function

    if normalizer_function:
        predict_proba = normalizer_function(predict_proba)

    compe_data['is_training'] = is_grid_search
    compe_data['occurrences'] = occurrences
    compe_data['best_params'] = best_params

    print_preds_hyperparams(target, model_type, compe_data, preds, predict_proba, test_frame, print_minimal=True)

    print(f'*** End training for model: {model_type}, target: {target} ***\n')

    # F1 score on test set
    f1 = f1_score(test_frame[target], preds, average='weighted')  # or 'macro' as needed

    # Include features used and the fitted model itself
    return {
        "model": model,
        "model_type": model_type,
        "preds": preds,
        "predict_proba": predict_proba,
        "train_frame": train_frame,
        "test_frame": test_frame,
        "occurrences": occurrences,
        "features": FEATURES,
        "f1": f1,
        "best_params": best_params,
        "best_score": grid_result["best_score"] if 'grid_result' in locals() else None,
        "cv_results": grid_result["cv_results"] if 'grid_result' in locals() else None,
    }

def check_missing_values(dataframe, features, target):
    features_missing = dataframe[features].isnull().sum()
    target_missing = dataframe[target].isnull().sum()
    
    print("Features missing values:")
    print(features_missing[features_missing > 0] if features_missing.sum() > 0 else "No missing values in features.")
    
    print("\nTarget missing values:")
    print(f"{target_missing} missing values." if target_missing > 0 else "No missing values in target.")

def log_elapsed_time(start_time, task_name="Task"):
    """
    Logs the elapsed time since start_time in a human-readable format.
    
    Args:
        start_time (float): The start time (from time.time()).
        task_name (str): Optional name of the task being timed.
    """
    import time
    elapsed = time.time() - start_time
    hours, rem = divmod(elapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"{task_name} completed in {int(hours)}h {int(minutes)}m {seconds:.2f}s")
