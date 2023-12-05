from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from configs.logger import Logger
from app.predictions_normalizers.hda_normalizer import normalizer
from app.train_predictions.tuning.hda_target.hda_grid_search import grid_search
from app.helpers.functions import natural_occurrences, save_model, get_features, feature_importance
from app.train_predictions.hyperparameters.hyperparameters import get_hyperparameters
from app.helpers.print_results import print_preds_update_hyperparams
import numpy as np

def all_predictions(user_token, train_matches, test_matches, compe_data, is_grid_search=False, is_random_search=False, update_model=False):
    targets = ['hda_target', 'bts_target', 'over25_target', 'cs_target']
    Logger.info(f"Prediction Targets: {targets}")

    features, has_features = get_features(compe_data, targets, is_grid_search)
    FEATURES = features
    print(f"Has filtered features: {'Yes' if has_features else 'No'}")

    # Create train and test DataFrames
    train_frame = pd.DataFrame(train_matches)
    test_frame = pd.DataFrame(test_matches)

    # Extract features and labels
    X_train = train_frame[FEATURES]
    y_train = train_frame[targets]

    X_test = test_frame[FEATURES]
    y_test = test_frame[targets]

    # Create a multi-output classifier (RandomForestClassifier)
    classifier = MultiOutputClassifier(RandomForestClassifier())

    # Train the model
    classifier.fit(X_train, y_train)

    # Make predictions
    predictions = classifier.predict(X_test)

    # Evaluate accuracy for each target
    for i, target in enumerate(targets):
        accuracy = accuracy_score(y_test[target], predictions[:, i])
        print(f'Accuracy for {target}: {accuracy:.4f}')

    # You can also print an overall accuracy if you consider all targets together
    # overall_accuracy = accuracy_score(y_test, predictions)
    # print(f'Overall Accuracy: {overall_accuracy:.4f}')

    # Flatten the multi-class labels and predictions
    y_true_flat = y_test.values.reshape(-1)
    y_pred_flat = predictions.reshape(-1)

    overall_accuracy = accuracy_score(y_true_flat, y_pred_flat)
    print(f'Overall Accuracy: {overall_accuracy:.4f}')