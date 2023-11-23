from sklearn.metrics import f1_score, accuracy_score, precision_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, StratifiedKFold, RandomizedSearchCV
from app.train_predictions.hyperparameters.hyperparameters import hyperparameters_array_generator
import numpy as np

# Set a random seed for reproducibility
np.random.seed(42)


def grid_search(model, train_frame, PREDICTORS, target, occurrences, is_random_search=False):

    n_estimators, min_samples_split, class_weight = hyperparameters_array_generator(
        train_frame, 12, 2.0, 4)

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
        if x[0] < 1.3 and x[1] > 1.4 and x[2] < 1.8:
            class_weight.append(x)

    # Creating a dictionary grid for grid search
    param_grid = {
        'random_state': [1],
        'n_estimators': n_estimators,
        'min_samples_split': min_samples_split,
        'class_weight': class_weight,
        'min_samples_leaf': [1, 2, 3],
        'max_features': [None]
    }

    # Fitting grid search to the train data
    if not is_random_search:
        gridsearch = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=StratifiedKFold(n_splits=5),
            scoring=lambda estimator, X, y_true: scorer(
                estimator, X, y_true, occurrences),
            verbose=2,
            n_jobs=-1,
        ).fit(train_frame[PREDICTORS], train_frame[target])
    else:
        gridsearch = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            n_iter=10,
            cv=5,
            scoring=lambda estimator, X, y_true: scorer(
                estimator, X, y_true, occurrences),
            random_state=42,
            verbose=3,
        ).fit(train_frame[PREDICTORS], train_frame[target])

    # Extract and print the best class weight and score
    best_params = gridsearch.best_params_
    best_score = gridsearch.best_score_
    print(f"Best params: {best_params}")
    print(f"Best score: {best_score}")

    return best_params

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
