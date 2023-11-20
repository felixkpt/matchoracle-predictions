from sklearn.metrics import f1_score, brier_score_loss, balanced_accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from configs.settings import PREDICTORS
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from app.train_predictions.includes.functions import hyperparameters_array_generator, save_hyperparameters


def grid_search(model, train_frame, target, occurances, COMPETITION_ID, save_params=False):

    n_estimators, min_samples_split, class_weight = hyperparameters_array_generator(
        train_frame, 16, 6, 2)

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

    # Creating a dictionary grid for grid search
    param_grid = {
        'random_state': [1],
        'n_estimators': n_estimators,
        'min_samples_split': min_samples_split,
        'class_weight': class_weight,
    }

    # Fitting grid search to the train data with 5 folds
    gridsearch = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=StratifiedKFold(n_splits=int(train_frame / 100)),
        scoring=lambda estimator, X, y_true: scorer(
            estimator, X, y_true, occurances),
        verbose=3,
        n_jobs=-1,
    ).fit(train_frame[PREDICTORS], train_frame[target])

    # Extract and print the best class weight and score
    best_params = gridsearch.best_params_
    best_score = gridsearch.best_score_
    print(f"Best params: {best_params}")
    print(f"Best score: {best_score}")

    if save_params:
        save_hyperparameters(COMPETITION_ID, target, best_params)

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


def scorer(estimator, X, y_true, occurances):
    occurance_outcome_0 = occurances.get(0, 0)
    occurance_outcome_1 = occurances.get(1, 0)

    # Assuming estimator is a RandomForestClassifier
    y_pred = estimator.predict(X)

    # Calculate the proportions of predicted occurrences
    totals = len(X)
    y_pred_0 = round(sum(1 for p in y_pred if p == 0) / totals * 100)
    y_pred_1 = round(sum(1 for p in y_pred if p == 1) / totals * 100)

    # Define a threshold for penalization
    max_diff_threshold = 25

    # Penalize if predicted occurrences exceed specific natural occurrences by more than 10%
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
    natural_score = 1 - (1.5 * max_diff / 100)  # Normalize to [0, 1]
    natural_score = natural_score if natural_score > 0 else 0

    print(f"NATURAL: Under: {occurance_outcome_0}, Over: {occurance_outcome_1}")
    print(f"PREDICT: Under: {y_pred_0}, Over: {y_pred_1}")
    print(f"TOTALS: {totals}, Ntrl score: {round(natural_score, 3)}")

    combined_score = 1/2 * brier_score_loss(y_true, y_prob=y_pred) + 1/2 * balanced_accuracy_score(y_true, y_pred=y_pred)

    return combined_score
