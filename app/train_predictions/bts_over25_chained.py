from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from configs.logger import Logger
from app.predictions_normalizers.bts_normalizer import normalizer
from app.helpers.functions import natural_occurrences, save_model, get_features, feature_importance
from app.train_predictions.hyperparameters.hyperparameters import get_hyperparameters
from app.helpers.print_results import print_preds_update_hyperparams

from sklearn.multioutput import ClassifierChain
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from configs.logger import Logger
from app.helpers.functions import get_features
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from app.train_predictions.tuning.run_random_search import grid_search

from sklearn.metrics import classification_report


def bts_over25_chained(user_token, train_matches, test_matches, compe_data, do_grid_search=False, is_random_search=False, update_model=False):

    target = 'bts_over25_target'
    targets = ['bts_target', 'over25_target']

    Logger.info(f"Prediction Target: {target}")

    features, has_features = get_features(compe_data, target, do_grid_search)
    FEATURES = features
    print(f"Has filtered features: {'Yes' if has_features else 'No'}")

    # Create train and test DataFrames
    train_frame = pd.DataFrame(train_matches)
    test_frame = pd.DataFrame(test_matches)

    outcomes = [0, 1]
    occurrences = natural_occurrences(
        outcomes, train_frame, test_frame, targets[0])

    occurrences2 = natural_occurrences(
        outcomes, train_frame, test_frame, targets[1])

   # Select the appropriate class weight dictionary based on the target
    hyper_params, has_weights = get_hyperparameters(
        compe_data, targets, outcomes)

    # begin
    # Create a RandomForestClassifier (you can replace this with any other classifier)
    model = RandomForestClassifier(
        random_state=1, n_estimators=250, min_samples_split=50, min_samples_leaf=5, class_weight={0: 1.3, 1: 1.15})

    best_params = None
    if do_grid_search or not has_weights:
        do_grid_search = True
        best_params = grid_search(
            model, train_frame, FEATURES, targets, occurrences, is_random_search)

        hyper_params = best_params
        model.set_params(**hyper_params)

    Logger.info(
        f"Hyper Params {'(default)' if not has_weights else ''}: {hyper_params}\n")

    # Create a ClassifierChain, Adjust the order based on your column order
    classifier_chain = ClassifierChain(model, order=[0, 1])

    model = classifier_chain

    # Save model if update_model is set
    if update_model:
        save_model(model, train_frame, test_frame,
                   FEATURES, targets, compe_data)

    model.fit(train_frame[FEATURES], train_frame[targets])

    # Make predictions on the test data
    preds = model.predict(test_frame[FEATURES])
    predict_proba = model.predict_proba(test_frame[FEATURES])

    # Evaluate the performance
    report = classification_report(
        test_frame[targets], preds, zero_division=0)
    print("Classification Report:\n", report)

    # end
    print('End')
    return



    feature_importance(model, compe_data, target, FEATURES, False, 0.007)

    predict_proba = normalizer(predict_proba)

    compe_data['is_training'] = do_grid_search
    compe_data['occurrences'] = occurrences
    compe_data['best_params'] = best_params
    compe_data['from_date'] = train_matches[0]['utc_date']
    compe_data['to_date'] = test_matches[-1]['utc_date']

    print_preds_update_hyperparams(user_token, target, compe_data,
                                   preds, predict_proba, train_frame, test_frame, print_minimal=False)

    return [preds, predict_proba, occurrences]


# from sklearn.multioutput import ClassifierChain
# from sklearn.ensemble import RandomForestClassifier
# import pandas as pd
# from configs.logger import Logger
# from app.helpers.functions import get_features
# from sklearn.utils.class_weight import compute_class_weight
# import numpy as np
# from app.train_predictions.tuning.run_random_search import grid_search

# from sklearn.metrics import classification_report


# def bts_over25_chained(user_token, train_matches, test_matches, compe_data, do_grid_search=False, is_random_search=False, update_model=False):

#     target = 'bts_over25_target'
#     targets = ['bts_target', 'over25_target']

#     Logger.info(f"Prediction Target: {target}")

#     features, has_features = get_features(compe_data, target, do_grid_search)
#     FEATURES = features
#     print(f"Has filtered features: {'Yes' if has_features else 'No'}")


#     # Extract features and targets from train_matches
#     train_frame = pd.DataFrame(train_matches)
#     test_frame = pd.DataFrame(test_matches)

#     # Call the grid search function
#     if do_grid_search:
#         best_model, best_params = grid_search(RandomForestClassifier(random_state=1,), train_frame, FEATURES, targets, None, is_random_search=False)
#         return 0

#     # Compute class weights
#     class_weights = compute_class_weight('balanced', classes=np.unique(
#         train_frame['bts_target'].values), y=train_frame['bts_target'].values.flatten())


#     # print(class_weights)
#     # return
#     class_weight_dict = dict(enumerate(class_weights))
#     # print(class_weight_dict)
#     # return

#     # Create a RandomForestClassifier (you can replace this with any other classifier)
#     base_classifier = RandomForestClassifier(
#         random_state=1, n_estimators=250, min_samples_split=50, min_samples_leaf=5, class_weight={0:1.3, 1:1.15}

#         )

#     # Create a ClassifierChain
#     # Adjust the order based on your column order
#     classifier_chain = ClassifierChain(base_classifier, order=[0, 1])

#     # Train the model
#     classifier_chain.fit(train_frame[FEATURES], train_frame[targets])

#     # Make predictions on the test set
#     y_pred = classifier_chain.predict(test_frame[FEATURES])

#     # Evaluate the performance
#     report = classification_report(test_frame[targets], y_pred, zero_division=0)
#     print("Classification Report:\n", report)

#     # Extract predictions for 'bts_target'
#     bts_predictions = y_pred[:, 0]

#     # Extract predictions for 'over25_target'
#     over25_predictions = y_pred[:, 1]

#     # print(f"BTS preds:\n")
#     # print(f"{bts_predictions}\n")

#     # print(f"OVER25 preds:\n")
#     # print(f"{over25_predictions}\n")

#     # for true_labels, bts_pred, over25_pred in zip(y_test.values, bts_predictions, over25_predictions):
#     #     print(f"True Labels: {true_labels}, Predicted BTS: {bts_pred}, Predicted Over25: {over25_pred}")

#     # Get predicted probabilities on the test set
#     y_pred_proba = classifier_chain.predict_proba(test_frame[FEATURES])

#     # Print the predicted probabilities
#     # print("Predicted Probabilities:\n", y_pred_proba)

#     # Extract predicted probabilities for 'bts_target' (class 0) and 'over25_target' (class 1)
#     # Probabilities for 'bts_target' (class 0)
#     bts_pred_proba = y_pred_proba[:, 0]
#     # Probabilities for 'over25_target' (class 1)
#     over25_pred_proba = y_pred_proba[:, 1]

#     # Print the predicted probabilities for each class
#     # print("BTS Predicted Probabilities:\n", bts_pred_proba)
#     # print("OVER25 Predicted Probabilities:\n", over25_pred_proba)
