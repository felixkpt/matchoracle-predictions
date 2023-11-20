from sklearn.metrics import precision_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from configs.settings import PREDICTORS
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from app.train_predictions.includes.functions import hyperparameters_array_generator, save_hyperparameters


def grid_search(model, train_frame, target, occurances, COMPETITION_ID, save_params=False):

    n_estimators, min_samples_split, class_weight = hyperparameters_array_generator(
        train_frame, 4)

    class_weight = []

    curr_val = 1
    increment = 0.1

    for i in range(6):  # Assuming you have 6 classes
        inner_list = []
        for j in range(6):
            class_weight_dict = {k: curr_val if k <=
                                 j else curr_val - increment for k in range(6)}
            inner_list.append(class_weight_dict)

        class_weight.append(inner_list)

        curr_val += increment
        curr_val = round(curr_val, 1) if curr_val <= 2 else 1

    # Creating a dictionary grid for grid search
    param_grid = {
        'n_estimators': n_estimators,
        'min_samples_split': min_samples_split,
        'class_weight': class_weight,
    }

    # Fitting grid search to the train data with 5 folds
    gridsearch = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=lambda estimator, X, y_true: scorer(
            estimator, X, y_true, occurances),
        verbose=2,
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
    y_pred_0 = round((sum(1 for p in y_pred if p == 0) / totals) * 100)
    y_pred_1 = round((sum(1 for p in y_pred if p == 1) / totals) * 100)

    # If none of the penalization conditions are met, return a combined score
    # Calculate the absolute differences between predicted and actual occurrences
    diff_outcome_0 = abs(y_pred_0 - occurance_outcome_0)
    diff_outcome_1 = abs(y_pred_1 - occurance_outcome_1)

    # Calculate a similarity score based on the differences
    max_diff = max(diff_outcome_0, diff_outcome_1)
    natural_score = 1 - (max_diff / 100)  # Normalize to [0, 1]
    natural_score = natural_score if natural_score >= 0 else 0

    print(f"NATURAL: NO: {occurance_outcome_0}, YES: {occurance_outcome_1}")
    print(f"PREDICT: NO: {y_pred_0}, YES: {y_pred_1}")
    print(f"TOTALS: {totals}, Ntrl score: {round(natural_score, 3)}")

    # Calculate other required scorers and combine
    precision = precision_score(
        y_true, y_pred, average='weighted', zero_division=0)

    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    combined_score = 0.1 * precision + 0.2 * f1 + 0.7 * natural_score

    return combined_score
